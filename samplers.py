import numpy as np
from observables import Observable
from params import Params

class BaseSampler:
    """Base class for samplers"""
    def __init__(self, params: Params, likelihood):
        self.params = params
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
    def __init__(self, params: Params, likelihood, **kwargs):
        super().__init__(params, likelihood)
        from nautilus import Sampler
        
        # 1. Build the Prior object using our Params helper
        self.prior = self.params.build_nautilus_prior()
        self.require_blobs = len(self.params.exported_derived_names) > 0
        
        # 2. Define the likelihood wrapper
        # Nautilus passes a dictionary of arguments if the prior was built with names
        def likelihood_wrapper(param_dict):
            full_dict = self.params.get_full_dict(param_dict)
            emu_dict = self.params.get_comet_dict(full_dict)

            loglike = self.likelihood.get_loglike(emu_dict) 
            if self.require_blobs:
                blobs = [full_dict[name] for name in self.params.exported_derived_names]
                return loglike, blobs
            
            return loglike

        # 3. Initialize Nautilus Sampler
        self.sampler = Sampler(
            self.prior, 
            likelihood_wrapper,
            **kwargs,
        )

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
    def __init__(self, params: Params, likelihood, initial_step=0.1, verbose=False):
        super().__init__(params, likelihood)
        from iminuit import Minuit
            
        # 1. Define the cost function (Total Chi2 = Chi2_data + Chi2_prior)
        # Minuit will pass the parameters as positional arguments in the order of names
        def cost_function(*args):
            # Convert positional args to dictionary
            param_dict = dict(zip(self.params.sampled_param_names, args))
            full_dict = self.params.get_full_dict(param_dict)
            lp = self.log_prior(full_dict)
            if not np.isfinite(lp):
                return np.inf

            
            chi2_prior = -2.0 * lp
            # Get Data Chi2
            # Note: We use get_chi2 directly, not get_loglike
            emu_dict = self.params.get_comet_dict(full_dict)
            chi2_data = self.likelihood.get_chi2(emu_dict)
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

    def run(self, hesse=True):
        """
        Run the minimization.
        hesse: If True, runs HESSE after MIGRAD to estimate covariance/errors.
        """
        # Run MIGRAD (Gradient descent)
        self.m.migrad()
        
        # Optionally run HESSE (Hessian calculation for accurate errors)
        if hesse:
            self.m.hesse()
            
        return self.m