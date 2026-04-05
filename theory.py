from comet import comet
import numpy as np
from observables import PowerSpectrumMultipoles, BispectrumScoccimarroMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from bispectrum import bispectrum_scoccimarro_proj, bispectrum_sugiyama_proj, bX_5d, bX_ell_scoccimarro, bX_ell_sugiyama
from scipy.special import eval_legendre

class BaseModel:
    """Base class for models"""
    def __init__(self, **kwargs):
        pass
    
    def predict(self, observables, params, **kwargs):
        """Predict the observable given the parameters
           This method should return a 1D array of the same length as the data vector of the observable
        """
        if not isinstance(observables, list):
            observables = [observables]
        if isinstance(observables[0], PowerSpectrumMultipoles):
            return self.predict_power_spectrum_multipoles(observables, params, **kwargs)
        elif isinstance(observables[0], BispectrumScoccimarroMultipoles):
            return self.predict_bispectrum_scoccimarro_multipoles(observables, params, **kwargs)
        elif isinstance(observables[0], BispectrumSugiyamaMultipoles):
            return self.predict_bispectrum_sugiyama_multipoles(observables, params, **kwargs)
        elif isinstance(observables[0], JointObservable):
            obs1_list = [o.observables[0] for o in observables]
            obs2_list = [o.observables[1] for o in observables]
            preds1 = self.predict(obs1_list, params, **kwargs)
            preds2 = self.predict(obs2_list, params, **kwargs)
            return [np.concatenate((p1, p2)) for p1, p2 in zip(preds1, preds2)]
        

class COMET(comet, BaseModel):
    """COMET model for power spectrum and bispectrum"""
    # weird hack: wrapper is an emu instance itself
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._extra_diagrams = ['Pctr_a0', 'Pctr_a2', 'Pctr_a4']
        self._extra_diagrams_to_marg = {'a0': 'Pctr_a0', 'a2': 'Pctr_a2', 'a4': 'Pctr_a4'}

    def predict_power_spectrum_multipoles(self, observables, params, de_model):
        ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)]))
        k_arrays = [np.concatenate(obs.k) if obs.kwin is None else np.concatenate(obs.kwin) for obs in observables]
        k_all = np.unique(np.concatenate(k_arrays))
        pell_batched = self.Pell(k_all, params, ell_all, de_model=de_model)
        preds = [] 
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        #print(f"DEBUG: predict_power_spectrum_multipoles: is_batched={is_batched}, z={params.get('z', None)}")
        for obs in observables:
            iz = list(params['z']).index(obs.cosmo_fid['z']) if is_batched else None
            k = obs.k if obs.kwin is None else obs.kwin
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            pell_z = []
            for i, ll in enumerate(ell):
                idx = np.searchsorted(k_all, k[i])
                p = pell_batched[f'ell{ll}'][idx, iz] if is_batched else pell_batched[f'ell{ll}'][idx]
                pell_z.append(p)
            pell_z = np.concatenate(pell_z)
        
            if obs.xwin is not None:
                pell_z = obs.wmat @ pell_z
            preds.append(pell_z)

        return preds

    def predict_PX_ell(self, observables, params, diagram, de_model, is_extra=False):
        ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)]))
        k_arrays = [np.concatenate(obs.k) if obs.kwin is None else np.concatenate(obs.kwin) for obs in observables]
        k_all = np.unique(np.concatenate(k_arrays))
        
        if is_extra:
            px_batched = self.PX_ell_extra(k_all, params, ell_all, diagram, de_model=de_model)
        else:
            px_batched = self.PX_ell(k_all, params, ell_all, diagram, de_model=de_model)
            
        preds = [] 
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        for obs in observables:
            iz = list(params['z']).index(obs.cosmo_fid['z']) if is_batched else None
            if is_batched:
                for kkx in px_batched:
                    print(f"DEBUG PX_ell: key={kkx}, shape={px_batched[kkx].shape}, iz={iz}, len(z)={len(params['z'])}")
            k = obs.k if obs.kwin is None else obs.kwin
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            
            px_z_dict = {}
            for i, ll in enumerate(ell):
                idx = np.searchsorted(k_all, k[i])
                # We return the dictionary for this observable to build the design matrix
                px_z_dict[f'ell{ll}'] = px_batched[f'ell{ll}'][idx, iz] if is_batched else px_batched[f'ell{ll}'][idx]
                
            preds.append(px_z_dict)
            
        return preds
    
    def predict_bispectrum_scoccimarro_multipoles(self, observables, params, de_model):
        ell_all = list(set([ll for obs in observables for ll in obs.ell]))
        tri_concat = np.concatenate([obs.tri for obs in observables], axis=0)
        tri_unique, idx_inverse = np.unique(tri_concat, axis=0, return_inverse=True)

        bk_batched = self.Bell_scoccimarro(tri_unique, params, ell_all, de_model=de_model)
        
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        
        idx_start = 0
        for obs in observables:
            iz = list(params['z']).index(obs.cosmo_fid['z']) if is_batched else None
            ntri = len(obs.tri)
            idx = idx_inverse[idx_start:idx_start+ntri]
            idx_start += ntri
            
            bk_z = []
            for i, ell in enumerate(obs.ell):
                bk_slice = bk_batched[f'ell{ell}'][idx]
                if is_batched:
                    bk_slice = bk_slice[:, iz]
                bk_z.append(bk_slice)
            preds.append(np.concatenate(bk_z))
            
        return preds

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
        # print("DEBUG q_lo len", len(self.params.get('q_lo', [])))
        if 'q_lo' in self.params:
            print("DEBUG: q_lo shape:", getattr(self.params['q_lo'], 'shape', type(self.params['q_lo'])), "len:", len(self.params['q_lo']))
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
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        for i, ll in enumerate(ell):
            res[f'ell{ll}'] = (2 * ll + 1)/q3 * (r_[i] if is_batched else r_[i, :, 0])
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
        
