from comet import comet
import numpy as np
from observables import PowerSpectrumMultipoles, BispectrumScoccimarroMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from bispectrum import bispectrum_scoccimarro_proj, bispectrum_sugiyama_proj, bX_5d, bX_ell_scoccimarro, bX_ell_sugiyama
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
        elif isinstance(observable, BispectrumSugiyamaMultipoles):
            return self.predict_bispectrum_sugiyama_multipoles(observable, params, **kwargs)
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
        bk = self.Bell_scoccimarro(observable.tri, params, observable.ell, de_model=de_model)
        bk = np.concatenate([bk[f'ell{ell}'] for ell in observable.ell])
        return bk

    def predict_bispectrum_sugiyama_multipoles(self, observable, params, de_model):
        ell = observable.ell if observable.ellwin is None else observable.ellwin
        bsugi = self.Bell_sugiyama(observable.pair, params, ell, de_model=de_model) #shape (npair, n_ell)
        bsugi = np.concatenate([bsugi[f'{ell}'] for ell in observable.ell])
        if observable.xwin is not None:
            bsugi = observable.wmat @ bsugi 
        return bsugi

    def PX_ell_extra(self, k, params, ell, diagram, de_model):
        if not isinstance(k, list):
            k_all = k
            k = len(ell) * [k]
            idx_inverse = None
        else:
            k_all = np.concatenate(k)
            k_all, idx_inverse = np.unique(k_all, return_inverse=True) 
            idx_ell = [int(np.sum([len(kk) for kk in k[:i]])) for i in range(len(k)+1)]
        mu = self.gl_x
        mu2 = self.gl_x2
        APfac = np.sqrt(
            np.divide.outer(mu2, self.params['q_lo']**2) \
            + np.divide.outer(1.0 - mu2, self.params['q_tr']**2))
        kp = np.multiply.outer(k_all, APfac) # shape (nk, nmu, 1)
        mup = np.divide.outer(mu, self.params['q_lo'])/APfac

        p2d = -0.5 * self.PX_2d(kp, mup, params, 'Pctr_c0', de_model=de_model)
        p2d = p2d[0] # shape (nk, nmu, 1)
        q3 = self.params['q_lo'] * self.params['q_tr']**2

        prefact_dict = {'Pctr_a0': self.params['b1'], 'Pctr_a2': self.params['f'] * mup**2, 'Pctr_a4': self.params['f'] * mup**4}
        kaiser_fact = (self.params['b1'] + self.params['f'] * mup**2)
        wdamping = self._W_kurt(kp, mup)
        res = {}
        integrand = kaiser_fact * prefact_dict[diagram] * p2d * wdamping
        legendre = eval_legendre.outer(ell, mu) # shape (n_ell, nmu)
        r_ = 0.5 * np.einsum("ebc,db,b->dec", integrand, legendre,
                                self.gl_weights) 
        for i, ll in enumerate(ell):
            res[f'ell{ll}'] = (2 * ll + 1)/q3 * r_[i, :, 0]
            if idx_inverse is not None:
                res[f'ell{ll}'] = res[f'ell{ll}'][idx_inverse][idx_ell[i]:idx_ell[i]+len(k[i])]
        return res

    def Bell_scoccimarro(self, tri, params, ell, de_model):
        if not isinstance(tri, list):
            tri_all = tri
            tri = len(ell) * [tri]
            idx_inverse = None
        # tri can be different for each ell
        else:
        # use only the unique values, but keep track of the indices to put the results back in the right order
        # keep indices for each ell
            tri_all = np.concatenate(tri) # tri is a list of arrays of shape (ntri_ell, 3), tri_all is an array of shape (sum(ntri_ell), 3)
            tri_all, idx_inverse = np.unique(tri_all, axis=0, return_inverse=True) 
            idx_ell = [np.sum([len(t) for t in tri[:i]]) for i in range(len(tri)+1)] # idx_ell[i] is the starting index of tri[i] in tri_all

        k1, k2, k3 = tri_all[:, 0], tri_all[:, 1], tri_all[:, 2]
        bscocc = bispectrum_scoccimarro_proj(k1, k2, k3, self, params, ell=ell, de_model=de_model) #shape (ntri, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[f'ell{ll}'] = bscocc[f'ell{ll}'][idx_inverse[idx_ell[i]:idx_ell[i]+len(tri[i])]] if idx_inverse is not None else bscocc[f'ell{ll}']
        return res
    
    def Bell_sugiyama(self, pair, params, ell, de_model):
        # same as above..
        if not isinstance(pair, list):
            pair_all = pair
            pair = len(ell) * [pair]
            idx_inverse = None
        else:
            pair_all = np.concatenate(pair)
            pair_all, idx_inverse = np.unique(pair_all, axis=0, return_inverse=True) 
            idx_ell = [int(np.sum([len(p) for p in pair[:i]])) for i in range(len(pair)+1)]
        k1, k2 = pair_all[:, 0], pair_all[:, 1] 
        bsugi = bispectrum_sugiyama_proj(k1, k2, self, params, ell=ell, de_model=de_model) #shape (npair, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[f'{ll}'] = bsugi[f'{ll}'][idx_inverse][idx_ell[i]:idx_ell[i+1]] if idx_inverse is not None else bsugi[f'{ll}']
        return res

    def BX_ell_scoccimarro(self, tri, params, ell, diagram, de_model):
        if not isinstance(tri, list):
            tri_all = tri
            tri = len(ell) * [tri]
            idx_inverse = None
        # tri can be different for each ell
        else:
            tri_all = np.concatenate(tri) # tri is a list of arrays of shape (ntri_ell, 3), tri_all is an array of shape (sum(ntri_ell), 3)
            tri_all, idx_inverse = np.unique(tri_all, axis=0, return_inverse=True) 
            idx_ell = [np.sum([len(t) for t in tri[:i]]) for i in range(len(tri)+1)] # idx_ell[i] is the starting index of tri[i] in tri_all
        k1, k2, k3 = tri_all[:, 0], tri_all[:, 1], tri_all[:, 2]
        bX_scocc = bX_ell_scoccimarro(k1, k2, k3, self, params, ell=ell, diagram=diagram, de_model=de_model) #shape (ntri, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[f'ell{ll}'] = bX_scocc[f'ell{ll}'][idx_inverse][idx_ell[i]:idx_ell[i]+len(tri[i])] if idx_inverse is not None else bX_scocc[f'ell{ll}']
        return res
    
    def BX_ell_sugiyama(self, pair, params, ell, diagram, de_model):
        if not isinstance(pair, list):
            pair_all = pair
            pair = len(ell) * [pair]
            idx_inverse = None
        else:
            pair_all = np.concatenate(pair)
            pair_all, idx_inverse = np.unique(pair_all, axis=0, return_inverse=True) 
            idx_ell = [int(np.sum([len(p) for p in pair[:i]])) for i in range(len(pair)+1)]
        k1, k2 = pair_all[:, 0], pair_all[:, 1] 
        bX_sugi = bX_ell_sugiyama(k1, k2, self, params, ell=ell, diagram=diagram, de_model=de_model) #shape (npair, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[f'{ll}'] = bX_sugi[f'{ll}'][idx_inverse][idx_ell[i]:idx_ell[i+1]] if idx_inverse is not None else bX_sugi[f'{ll}']
        return res
        
