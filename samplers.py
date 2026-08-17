import numpy as np
from observables import Observable
from params import Params

class BaseSampler:
    """Base class for samplers"""
    def __init__(self, likelihood):
        self.params = likelihood.params
        self.likelihood = likelihood

    def log_prior(self, param_values):
        """Calculate the log prior probability for the given parameter values"""
        log_prior = 0.0
        for name, value in param_values.items():
            param = self.params.parameters[name]
            if param.prior is not None:
                if param.prior_type == "uniform":
                    if not (param.prior[0] <= value <= param.prior[1]):
                        return -np.inf 
                    else:
                        # add the actual log prior contribution for uniform prior
                        # to ensure proper normalization (important when estimating profile errors?)
                        log_prior += np.log(1.0 / (param.prior[1] - param.prior[0]))

                elif param.prior_type == "gaussian":
                    mean, std = param.prior
                    # Gaussian contribution: -0.5 * chi2_prior
                    log_prior += -0.5 * ((value - mean) / std) ** 2 
                    log_prior += np.log(1.0 / (std * np.sqrt(2 * np.pi)))  # normalization term 
        return log_prior



class NautilusSampler(BaseSampler):
    """Sampler using the Nautilus algorithm"""
    def __init__(self, likelihood, **kwargs):
        super().__init__(likelihood)
        from nautilus import Sampler
        
        # 1. Build the Prior object using our Params helper
        self.prior = self.params.build_nautilus_prior()
        self.require_blobs = len(self.params.exported_derived_names) > 0
        
        # 2. Define the likelihood wrapper
        # Nautilus passes a dictionary of arguments if the prior was built with names
        # def likelihood_wrapper(param_dict):
        #     full_dict = self.params.get_full_dict(param_dict)
        #     loglike = self.likelihood.get_loglike(full_dict) 
        #     if self.require_blobs:
        #         blobs = [full_dict[name] for name in self.params.exported_derived_names]
        #         return loglike, blobs
            
        #     return loglike

        # 3. Initialize Nautilus Sampler
        self.sampler = Sampler(
            self.prior, 
            self.likelihood_wrapper,
            **kwargs,
        )

    def likelihood_wrapper(self, param_dict):
        full_dict = self.params.get_full_dict(param_dict)
        loglike = self.likelihood.get_loglike(full_dict) 
        if self.require_blobs:
            blobs = [full_dict[name] for name in self.params.exported_derived_names]
            return loglike, blobs
        
        return loglike

    def sample(self, **kwargs):
        """Run the Nautilus sampling algorithm"""
        self.sampler.run(**kwargs)
        
    def save(self, filename):
        """Save posterior samples to a file"""

        if self.require_blobs:
            points, log_w, log_l, blobs = self.sampler.posterior(return_blobs=True)
            if blobs.ndim == 1:
                blobs = blobs[:, None]  # Ensure blobs is 2D for hstack
            points = np.hstack([points, blobs])
            names = self.prior.keys + self.params.exported_derived_names
        else:
            points, log_w, log_l = self.sampler.posterior()
            names = self.prior.keys
        latex_names = [self.params.parameters[n].latex for n in names] 

        np.savez(filename, points=points, log_weights=log_w, log_likelihoods=log_l,
                 names=names, latex_names=latex_names)
        

class MinuitMinimizer(BaseSampler):
    """Wrapper for the iMinuit minimizer"""
    def __init__(self, likelihood, initial_step=0.1, seed_init=None, verbose=False):
        super().__init__(likelihood)
        from iminuit import Minuit
            
        # 1. Define the cost function (Total Chi2 = Chi2_data + Chi2_prior)
        # Minuit will pass the parameters as positional arguments in the order of names
        def cost_function(*args):
            # Convert positional args to dictionary
            param_dict = dict(zip(self.params.sampled_param_names, args))
            full_dict = self.params.get_full_dict(param_dict)
            lp = self.log_prior(param_dict)
            if not np.isfinite(lp):
                return np.inf

            
            chi2_prior = -2.0 * lp
            # Get Data Chi2
            # Note: We use get_chi2 directly, not get_loglike 
            chi2_data = self.likelihood.get_chi2(full_dict)
            return chi2_data + chi2_prior

        # 2. Setup Initial Values

        if seed_init is not None:
            np.random.seed(seed_init)
        self.sampled_param_names = self.params.sampled_param_names
        # init_values = [self.params.parameters[n].value for n in self.sampled_param_names]
        init_values = []
        for n in self.sampled_param_names:
            p = self.params.parameters[n]
            if p.prior_type == 'uniform':
                # Start at the midpoint of the uniform prior
                if seed_init is None:
                    init_values.append(0.5 * (p.prior[0] + p.prior[1]))
                else:
                    # draw a random starting point within the uniform prior range
                    init_values.append(np.random.uniform(p.prior[0], p.prior[1]))
            elif p.prior_type == 'gaussian':
                # Start at the mean of the Gaussian prior
                if seed_init is None:
                    init_values.append(p.prior[0])
                else:
                    # draw a random starting point from the Gaussian prior
                    init_values.append(np.random.normal(p.prior[0], p.prior[1]))
            else:
                # Fallback to the current value if no prior is defined
                print(f"Warning: Parameter {n} has no prior defined. Using current value {p.value} as starting point.")
                print("This should almost never happen, as all sampled parameters should have a prior.")
                init_values.append(p.value)

        # 3. Initialize Minuit
        # We pass the cost function, the starting values, and the names
        self.m = Minuit(cost_function, *init_values, name=self.params.sampled_param_names)
        
        # 4. Configure Limits and Steps
        self.m.errordef = Minuit.LEAST_SQUARES # = 1.0 (for Chi2 minimization)
        
        for name in self.sampled_param_names:
            p = self.params.parameters[name]
            
            # Set Limits (Critical for Uniform priors)
            if p.prior_type == 'uniform' and p.prior is not None:
                self.m.limits[name] = p.prior

            # hardcode limits for comet parameters
            _comet_limits = {'wc': (0.08, 0.16), 'wb': (0.01930, 0.02535), 'ns': (0.9, 1.03),
                             'As': (1.0, 3.5), 'Mnu': (0.0, 1.0), 'sigma_12': (0.2, 1.0),
                             'f': (0.5, 1.05), 'log10As': (np.log(1e10 * 1e-9), np.log(1e10 * 3.5 * 1e-9))}
            if name in _comet_limits:
                if p.prior_type == 'gaussian' or (p.prior_type == 'uniform' and p.prior[0] < _comet_limits[name][0] or p.prior[1] > _comet_limits[name][1]):
                    print(f"Warning: Overriding limits for {name} to {_comet_limits[name]} based on COMET constraints.")
                    self.m.limits[name] = _comet_limits[name]
            
            # Set Initial Step Size (Heuristic)
            # Prefer prior-based step so that zero-initialised nuisance params
            # get a meaningful scale rather than a hard-coded constant.
            if p.value != 0:
                step = abs(p.value) * initial_step
            elif p.prior is not None:
                if p.prior_type == 'gaussian':
                    # sigma of the Gaussian prior is the natural scale
                    step = p.prior[1] * initial_step
                elif p.prior_type == 'uniform':
                    # half-width of the uniform interval
                    step = (p.prior[1] - p.prior[0]) * initial_step
                else:
                    step = initial_step
            else:
                step = initial_step
            # Guard against zero or tiny steps
            if step == 0 or not np.isfinite(step):
                step = initial_step
            self.m.errors[name] = step
        
        if verbose:
            print(f"Initialized Minuit with {len(self.params.sampled_param_names)} free parameters.")

    def set_starting_point(self, param_dict, errors_dict=None, reset_errors=True):
        """Seed Minuit's starting values from a dictionary of parameter values.

        Parameters
        ----------
        param_dict : dict
            Mapping of parameter name -> value.  Only names that are free
            (i.e. in ``self.sampled_param_names``) will be applied; extra
            keys are silently ignored.
        errors_dict : dict, optional
            Mapping of parameter name -> step size / uncertainty.  When
            provided, these override the automatic ``reset_errors`` heuristic
            for the corresponding parameters.
        reset_errors : bool
            If True, reset each step-size to 10 % of the absolute starting
            value for parameters not covered by ``errors_dict`` (falls back
            to the current error if the value is 0).
        """
        for name in self.sampled_param_names:
            if name in param_dict:
                val = float(param_dict[name])
                self.m.values[name] = val
                if errors_dict is not None and name in errors_dict:
                    err = float(errors_dict[name])
                    if np.isfinite(err) and err > 0:
                        self.m.errors[name] = err
                elif reset_errors and val != 0:
                    self.m.errors[name] = abs(val) * 0.1

    def run(self, hesse=False, strategy=1, tol=0.1, max_calls=(200000, 800000, 2000000),
            simplex_on_retry=True, two_phase=False, verbose=True):
        """
        Run the minimization with robust retries.

        Parameters
        ----------
        hesse : bool
            If True, run HESSE after a successful MIGRAD call.
        strategy : int
            Minuit strategy level. Use 2 for a stringent convergence strategy.
        tol : float
            EDM tolerance. Smaller values enforce tighter convergence.
        max_calls : tuple[int, ...]
            Sequence of ncall values to try for MIGRAD. Each element is one retry.
        simplex_on_retry : bool
            If True, run SIMPLEX before MIGRAD on retries to improve robustness.
        two_phase : bool
            If True, run a cheap loose first pass (strategy=1, tol=1.0) before the
            main minimization with the requested strategy/tol.  Helps locate the
            basin of attraction cheaply when starting far from the minimum.
        verbose : bool
            If True, print retry/convergence status.
        """
        # --- optional cheap first pass to locate the basin -----------------
        if two_phase:
            if verbose:
                print("two_phase=True: running loose first pass (strategy=1, tol=1.0) …")
            self.m.strategy = 1
            self.m.tol = 1.0
            self.m.migrad(ncall=max_calls[0])
            _print_fmin_diagnostics = lambda prefix: None  # placeholder; real one defined below
            if verbose:
                fmin = self.m.fmin
                print(
                    f"Loose pass: valid={self.m.valid}, "
                    f"edm={fmin.edm:.3e}, nfcn={fmin.nfcn}"
                )
        # --- main minimization -----------------------------------------------
        self.m.strategy = strategy
        self.m.tol = tol

        def _fmin_flag(fmin, attr, default="n/a"):
            return getattr(fmin, attr, default)

        def _fmt_sci(value):
            return f"{value:.3e}" if isinstance(value, (int, float, np.floating)) else value

        def _print_fmin_diagnostics(prefix):
            fmin = self.m.fmin
            if not verbose:
                return
            print(
                f"{prefix}: "
                f"valid={self.m.valid}, "
                f"fval={_fmt_sci(_fmin_flag(fmin, 'fval'))}, "
                f"edm={_fmt_sci(_fmin_flag(fmin, 'edm'))}, "
                f"edm_goal={_fmt_sci(_fmin_flag(fmin, 'edm_goal'))}, "
                f"above_max_edm={_fmin_flag(fmin, 'is_above_max_edm')}, "
                f"call_limit={_fmin_flag(fmin, 'has_reached_call_limit')}, "
                f"at_limit={_fmin_flag(fmin, 'has_parameters_at_limit')}, "
                f"hesse_failed={_fmin_flag(fmin, 'hesse_failed')}, "
                f"cov_posdef={_fmin_flag(fmin, 'has_posdef_covar')}, "
                f"nfcn={_fmin_flag(fmin, 'nfcn')}, "
                f"ngrad={_fmin_flag(fmin, 'ngrad')}"
            )

        for i, ncall in enumerate(max_calls):
            if i > 0 and simplex_on_retry:
                simplex_ncall = max(2000, ncall // 5)
                if verbose:
                    print(f"Retry {i}: running SIMPLEX with ncall={simplex_ncall} before MIGRAD")
                self.m.simplex(ncall=simplex_ncall)
                print(f"SIMPLEX attempt {i} completed, valid={self.m.valid}")

            if verbose:
                print(f"Running MIGRAD attempt {i+1}/{len(max_calls)} with ncall={ncall}, strategy={strategy}, tol={tol}")
            self.m.migrad(ncall=ncall)
            _print_fmin_diagnostics(prefix=f"MIGRAD attempt {i+1} status")

            fmin = self.m.fmin
            if self.m.valid and not fmin.has_reached_call_limit:
                if verbose:
                    print("MIGRAD converged.")
                break

            if verbose:
                print(
                    "MIGRAD did not fully converge "
                    f"(valid={self.m.valid}, call_limit={fmin.has_reached_call_limit}, edm={fmin.edm:.3e})."
                )

        # Optionally run HESSE only after a valid minimum is found.
        if hesse and self.m.valid:
            self.m.hesse()
            _print_fmin_diagnostics(prefix="Post-HESSE status")

        return self.m
    
    def get_map(self, return_am=True):
        """Return the best-fit parameters and their uncertainties.

        Parameters
        ----------
        return_am : bool
            If True and the likelihood uses analytical marginalisation, also
            return the conditional MAP values *and* formal uncertainties
            (``sqrt(diag(cond_cov))``) for the analytically-marginalised
            parameters.
        """
        if not self.m.valid:
            import warnings
            edm = getattr(self.m.fmin, 'edm', float('nan'))
            warnings.warn(
                f"Minimization did not fully converge (valid=False, edm={edm:.3e}). "
                f"Returning best-fit values anyway — verify that edm is acceptably small."
            )
        
        best_fit = {name: self.m.values[name] for name in self.sampled_param_names}
        uncertainties = {name: self.m.errors[name] for name in self.sampled_param_names}
        if return_am and self.likelihood.do_am:
            full_dict = self.params.get_full_dict(best_fit)
            self.likelihood.am_sample_mode = 'map'
            self.likelihood.get_chi2(full_dict)  # Update AM params and cache cond_cov
            all_am_params = [name for am_params_iz in self.likelihood.am_params for name in am_params_iz]
            best_fit.update({name: full_dict[name] for name in all_am_params})
            # Extract formal uncertainties from the cached conditional covariances
            for iz, am_params_iz in enumerate(self.likelihood.am_params):
                cond_cov = getattr(self.likelihood, '_last_am_cond_covs', [None] * (iz + 1))[iz]
                if cond_cov is not None:
                    am_errors = np.sqrt(np.diag(np.atleast_2d(cond_cov)))
                    for j, name in enumerate(am_params_iz):
                        uncertainties[name] = float(am_errors[j])
        return best_fit, uncertainties