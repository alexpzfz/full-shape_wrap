import numpy as np
from observables import Observable, PowerSpectrumMultipoles

class Likelihood:
    """Base class for likelihoods"""
    def __init__(self, observable, emu, params):
        self.observable = observable
        self.icov = np.linalg.inv(observable.cov)
        self.nmocks_cov = observable.nmocks_cov
        self.emu = emu
        self.params = params
        self.de_model = self.params.de_model
        self.x = observable.x
        self.y = observable.get_flatten()
        self.emu.define_fiducial_cosmology(observable.cosmo_fid)
        if getattr(self.observable, 'nbar', None) is not None:
            self.nbar = observable.nbar
            self.emu.define_nbar(self.nbar)
        
        if self.nmocks_cov is not None:
            self._rescale_covariance(mode='Hartlap')

        if observable.__class__ == PowerSpectrumMultipoles:
            self.get_chi2 = self._get_chi2_powerspectrum

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

    def get_loglike(self, params):
        chi2 = self.get_chi2(params)
        loglike = -0.5 * chi2
        return loglike
    