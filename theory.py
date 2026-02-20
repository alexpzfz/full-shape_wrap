from comet import comet
from .params import Params
import numpy as np
from observables import PowerSpectrumMultipoles, BispectrumScoccimarroMultipoles, JointObservable
from samplers import NautilusSampler
from bispectrum import bispectrum_scoccimarro_proj

class BaseModel:
    """Base class for models"""
    def __init__(self, **kwargs):
        pass
    
    def _predict(self, observable, params):
        """Predict the observable given the parameters
           This method should return a 1D array of the same length as the data vector of the observable
        """
        if isinstance(observable, PowerSpectrumMultipoles):
            return self.predict_power_spectrum_multipoles(observable, params)
        elif isinstance(observable, BispectrumScoccimarroMultipoles):
            return self.predict_bispectrum_scoccimarro_multipoles(observable, params)
        elif isinstance(observable, JointObservable):
            pred_list = []
            for obs in observable.observables:
                pred_list.append(self._predict(obs, params))
            return np.concatenate(pred_list)
        

class COMET(BaseModel, comet):
    """COMET model for power spectrum and bispectrum"""
    # weird hack: wrapper is an emu instance itself
    def __init__(self, **kwargs):
        super().__init__(**kwargs) 

    def predict_power_spectrum_multipoles(self, observable, params, de_model):
        k = observable.k if observable.kwin is None else observable.kwin
        ell = observable.ell if observable.ellwin is None else observable.ellwin
        pell = self.Pell(k, ell, params, de_model=de_model)
        pell = np.concatenate([pell[f'ell={ell}'] for ell in observable.ell])
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





            

    


