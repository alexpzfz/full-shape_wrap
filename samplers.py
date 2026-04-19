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
                    # For minimization, we handle uniform bounds via Minuit limits,
                    # but we keep this check for completeness.
                    if not (param.prior[0] <= value <= param.prior[1]):
                        return -np.inf 
                    # Constant log_prior for uniform can be ignored for minimization
                elif param.prior_type == "gaussian":
                    mean, std = param.prior
                    # Gaussian contribution: -0.5 * chi2_prior
                    log_prior += -0.5 * ((value - mean) / std) ** 2 
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
    def __init__(self, likelihood, initial_step=0.1, verbose=False):
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
        self.sampled_param_names = self.params.sampled_param_names
        init_values = [self.params.parameters[n].value for n in self.sampled_param_names]

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
            # If value is non-zero, take fraction, else take absolute step
            step = abs(p.value) * initial_step if p.value != 0 else initial_step
            self.m.errors[name] = step
        
        if verbose:
            print(f"Initialized Minuit with {len(self.params.sampled_param_names)} free parameters.")

    def run(self, hesse=False, strategy=1, tol=0.1, max_calls=(200000, 800000, 2000000),
            simplex_on_retry=True, verbose=True):
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
        verbose : bool
            If True, print retry/convergence status.
        """
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
        """Return the best-fit parameters and their uncertainties"""
        if not self.m.valid:
            raise RuntimeError("Minimization did not converge. Check the fit status.")
        
        best_fit = {name: self.m.values[name] for name in self.sampled_param_names}
        uncertainties = {name: self.m.errors[name] for name in self.sampled_param_names}
        if return_am and self.likelihood.do_am:
            full_dict = self.params.get_full_dict(best_fit)
            self.likelihood.am_sample_mode = 'map'
            self.likelihood.get_chi2(full_dict)  # Update AM params to best-fit values
            all_am_params = [name for am_params_iz in self.likelihood.am_params for name in am_params_iz]
            best_fit.update({name: full_dict[name] for name in all_am_params})
        return best_fit, uncertainties