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
        
        # 2. Define the likelihood wrapper
        # Nautilus passes a dictionary of arguments if the prior was built with names
        def likelihood_wrapper(param_dict):
            # Convert the distinct free params (dict) into the full dictionary 
            # required by the emulator/likelihood
            full_cosmo_dict = self.params.get_full_dict(param_dict)
            full_cosmo_dict['z'] = self.likelihood.observable.cosmo_fid['z']
            
            # Call the likelihood class
            return self.likelihood.get_loglike(full_cosmo_dict)

        # 3. Initialize Nautilus Sampler
        self.sampler = Sampler(
            self.prior, 
            likelihood_wrapper,
            **kwargs,
        )

    def sample(self, verbose=True):
        """Run the Nautilus sampling algorithm"""
        self.sampler.run(verbose=verbose)
        
    def save(self, filename):
        """Save posterior samples to a file"""
        points, log_w, log_l = self.sampler.posterior()
        np.savez(filename, points=points, log_weights=log_w, log_likelihoods=log_l,
                 names=self.prior.keys, latex_names=[self.params.parameters[n].latex for n in self.prior.keys])
        

class MinuitMinimizer(BaseSampler):
    """Wrapper for the iMinuit minimizer"""
    def __init__(self, params: Params, likelihood, initial_step=0.1, verbose=False):
        super().__init__(params, likelihood)
        from iminuit import Minuit
            
        # 1. Define the cost function (Total Chi2 = Chi2_data + Chi2_prior)
        # Minuit will pass the parameters as positional arguments in the order of names
        def cost_function(*args):
            # Convert positional args to dictionary
            param_dict = dict(zip(self.params.free_param_names, args))
            full_dict = self.params.get_full_dict(param_dict)
            
            # Get Data Chi2
            # Note: We use get_chi2 directly, not get_loglike
            chi2_data = self.likelihood.get_chi2(full_dict | {'z': self.likelihood.observable.cosmo_fid['z']})
            
            # Get Prior Penalty
            # log_prior returns ln(P). We need -2*ln(P) to convert to Chi2 scale
            # If log_prior is -inf (out of bounds), we return infinity
            lp = self.log_prior(full_dict)
            if not np.isfinite(lp):
                return np.inf
                
            chi2_prior = -2.0 * lp
            
            return chi2_data + chi2_prior

        # 2. Setup Initial Values
        self.free_names = self.params.free_param_names
        init_values = [self.params.parameters[n].value for n in self.free_names]

        # 3. Initialize Minuit
        # We pass the cost function, the starting values, and the names
        self.m = Minuit(cost_function, *init_values, name=self.free_names)
        
        # 4. Configure Limits and Steps
        self.m.errordef = Minuit.LEAST_SQUARES # = 1.0 (for Chi2 minimization)
        
        for name in self.free_names:
            p = self.params.parameters[name]
            
            # Set Limits (Critical for Uniform priors)
            if p.prior_type == 'uniform' and p.prior is not None:
                self.m.limits[name] = p.prior
            
            # Set Initial Step Size (Heuristic)
            # If value is non-zero, take fraction, else take absolute step
            step = abs(p.value) * initial_step if p.value != 0 else initial_step
            self.m.errors[name] = step
        
        if verbose:
            print(f"Initialized Minuit with {len(self.free_names)} free parameters.")

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