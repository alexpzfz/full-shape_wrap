from comet import comet
import numpy as np
from observables import PowerSpectrumMultipoles, BispectrumScoccimarroMultipoles, JointObservable
from bispectrum import bispectrum_scoccimarro_proj
from scipy.special import eval_legendre

class BaseModel:
    """Base class for models"""
    def __init__(self, **kwargs):
        pass
    
    def predict(self, observable, params, **kwargs):
        """Predict the observable given the parameters
           This method should return a 1D array of the same length as the data vector of the observable
        """
        if isinstance(observable, PowerSpectrumMultipoles):
            return self.predict_power_spectrum_multipoles(observable, params, **kwargs)
        elif isinstance(observable, BispectrumScoccimarroMultipoles):
            return self.predict_bispectrum_scoccimarro_multipoles(observable, params, **kwargs)
        elif isinstance(observable, JointObservable):
            pred_list = []
            for obs in observable.observables:
                pred_list.append(self.predict(obs, params, **kwargs))
            return np.concatenate(pred_list)
        

class COMET(comet, BaseModel):
    """COMET model for power spectrum and bispectrum"""
    # weird hack: wrapper is an emu instance itself
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extra_diagrams = ['Pctr_a0', 'Pctr_a2', 'Pctr_a4']
        self._extra_diagrams_to_marg = {'a0': 'Pctr_a0', 'a2': 'Pctr_a2', 'a4': 'Pctr_a4'}

    def predict_power_spectrum_multipoles(self, observable, params, de_model):
        k = observable.k if observable.kwin is None else observable.kwin
        ell = observable.ell if observable.ellwin is None else observable.ellwin
        pell = self.Pell(k, params, ell, de_model=de_model)
        pell = np.concatenate([pell[f'ell{ell}'] for ell in observable.ell])
        if observable.xwin is not None:
            pell = observable.wmat @ pell
        return pell
    
    def predict_bispectrum_scoccimarro_multipoles(self, observable, params, de_model):
        # for the moment, no window support for bispectrum
        k1 = observable.tri[:, 0]
        k2 = observable.tri[:, 1]
        k3 = observable.tri[:, 2]
        ell = observable.ell
        bscocc = bispectrum_scoccimarro_proj(k1, k2, k3, self, params, ell=ell, de_model=de_model) #shape (ntri, n_ell)
        bscocc = bscocc.flatten()
        return bscocc

    def PX_ell_extra(self, k, params, ell, diagram, de_model):
        mu = self.gl_x
        mu2 = self.gl_x2
        APfac = np.sqrt(
            np.divide.outer(mu2, self.params['q_lo']**2) \
            + np.divide.outer(1.0 - mu2, self.params['q_tr']**2))
        kp = np.multiply.outer(k, APfac)
        mup = np.divide.outer(mu, self.params['q_lo'])/APfac

        p2d = -0.5 * self.PX_2d(kp, mup, params, 'Pctr_c0', de_model=de_model)
        q3 = self.params['q_lo'] * self.params['q_tr']**2

        prefact_dict = {'Pctr_a0': self.params['b1'], 'Pctr_a2': self.params['f'] * mup**2, 'Pctr_a4': self.params['f'] * mup**4}
        kaiser_fact = (self.params['b1'] + self.params['f'] * mup**2)
        wdamping = self._W_kurt(kp, mup)[:, None, ...]
        res = {}
        for ll in ell:
            integrand = kaiser_fact * prefact_dict[diagram] * p2d * wdamping
            legendre = eval_legendre.outer(ll, mu)[None, :]
            r_ = 0.5 * np.einsum("aebc,db,b->adec", integrand, legendre,
                                   self.gl_weights) 
            res[f'ell{ll}'] = (2 * ll + 1)/q3 * r_[0, 0, :, 0]
        return res