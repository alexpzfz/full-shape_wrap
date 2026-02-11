import numpy as np
from observables import Observable, PowerSpectrumMultipoles

class Likelihood:
    """Base class for likelihoods"""
    def __init__(self, observable, emu, params, am_params=None):
        self.observable = observable
        self.icov = np.linalg.inv(observable.cov)
        self.nmocks_cov = observable.nmocks_cov
        self.emu = emu
        self.params = params
        self.params.z = observable.cosmo_fid['z']  # Set redshift in params for use in derived parameters if needed
        self.de_model = self.params.de_model
        self.x = observable.x
        self.y = observable.get_flatten()
        self.emu.define_fiducial_cosmology(observable.cosmo_fid)
        if getattr(self.observable, 'nbar', None) is not None:
            self.nbar = observable.nbar
            self.emu.define_nbar(self.nbar)
        
        if self.nmocks_cov is not None:
            self._rescale_covariance(mode='Hartlap')
 

        self.do_am = False
        self.am_params = am_params
        self.am_priors = None
        if self.am_params is not None:
            if not isinstance(self.am_params, list):
                self.am_params = [self.am_params]
             # check that all am_params are in bias, counterterms or stochastic
            for am_param in self.am_params:
                if am_param not in self.params.bias_params and am_param not in self.params.counterterm_params and am_param not in self.params.stochastic_params:
                    raise ValueError(f"AM parameter '{am_param}' not found in bias, counterterm or stochastic parameters.")
                # check that gaussian priors are set for all am_params
                if self.params.parameters[am_param].prior is None or self.params.parameters[am_param].prior_type != 'gaussian':
                    raise ValueError(f"AM parameter '{am_param}' must have a Gaussian prior defined.")
            # for the moment we use comet's chi2 function for AM
            n_realizations = observable.nmocks_cov if observable.nmocks_cov is not None else None
            theory_cov = True if self.nmocks_cov is None else False
            emu.define_data_set(obs_id='pk', bins=observable._k, signal=observable._Pell.T, cov=observable._cov,
                                theory_cov=theory_cov, n_realizations=n_realizations, zeff=observable.cosmo_fid['z'],
                                fiducial_cosmology=observable.cosmo_fid)
            self.do_am = True
            self.am_priors = {am_param: list(self.params.parameters[am_param].prior) for am_param in self.am_params}

            
        if observable.__class__ == PowerSpectrumMultipoles:
            self.get_chi2 = self._get_chi2_powerspectrum
            if self.do_am:
                self.get_chi2 = self._get_chi2_am_from_comet

        # if self.params.fixed_cosmo:
        #     print("All cosmological parameters are fixed. Likelihood will only depend on nuisance parameters.")
        #     self.pell_func = emu.Pell_fixed_cosmo_boost
        # else:
        #     self.pell_func = emu.Pell

    def _rescale_covariance(self, mode='Hartlap'):
        n_data = self.observable.n_data
        n_mocks = self.nmocks_cov
        if mode == 'Hartlap':
            factor = (n_mocks - n_data - 2) / (n_mocks - 1)
        else:
            raise NotImplementedError(f"Covariance rescaling mode '{mode}' not implemented")
        print(f"Rescaling covariance by factor {factor:.3f} using {mode} correction: n_mocks={n_mocks}, n_data={n_data}")
        self.icov *= factor

    def _get_chi2_powerspectrum(self, params):
        pred = self.emu.Pell(self.x, params=params, ell=self.observable.ell, de_model=self.de_model)
        y_model = np.concatenate([pred[f'ell{l}'] for l in self.observable.ell])
        delta = self.y - y_model
        chi2 = np.dot(delta, np.dot(self.icov, delta))
        return chi2

    def _get_chi2_am_from_comet(self, params):
        chi2 = self.emu.chi2(obs_id='pk', params=params, kmax=self.observable._kmax, de_model=self.de_model, AM_priors=self.am_priors)
        chi2 = float(chi2)  # Ensure chi2 is a scalar float, not a 0-dim array
        return chi2

    def get_loglike(self, params):
        chi2 = self.get_chi2(params)
        loglike = -0.5 * chi2
        return loglike
    