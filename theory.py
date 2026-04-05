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
           This method should return a  list 1D arrayy of the same length as the data vector of the observables.
        """
        if not isinstance(observables, list):
            observables = [observables]

        grouped = {
            'pk': [],
            'bk_scocc': [],
            'bk_sugiyama': [],
        }

        def add(group_name, observable):
            grouped[group_name].append(observable)
            return len(grouped[group_name]) - 1

        assembly_plan = []
        for obs in observables:
            if isinstance(obs, PowerSpectrumMultipoles):
                assembly_plan.append(('single', 'pk', add('pk', obs)))
            elif isinstance(obs, BispectrumScoccimarroMultipoles):
                assembly_plan.append(('single', 'bk_scocc', add('bk_scocc', obs)))
            elif isinstance(obs, BispectrumSugiyamaMultipoles):
                assembly_plan.append(('single', 'bk_sugiyama', add('bk_sugiyama', obs)))
            elif isinstance(obs, JointObservable):
                pk_idx = add('pk', obs.observables[0])
                bk_obs = obs.observables[1]
                if isinstance(bk_obs, BispectrumScoccimarroMultipoles):
                    bk_group = 'bk_scocc'
                elif isinstance(bk_obs, BispectrumSugiyamaMultipoles):
                    bk_group = 'bk_sugiyama'
                else:
                    raise ValueError(f"Unsupported observable type in JointObservable: {type(bk_obs)}")
                bk_idx = add(bk_group, bk_obs)
                assembly_plan.append(('joint', pk_idx, bk_group, bk_idx))
            else:
                raise ValueError(f"Unsupported observable type: {type(obs)}")

        predictions = {
            'pk': self.predict_power_spectrum_multipoles(grouped['pk'], params, **kwargs) if grouped['pk'] else [],
            'bk_scocc': self.predict_bispectrum_scoccimarro_multipoles(grouped['bk_scocc'], params, **kwargs) if grouped['bk_scocc'] else [],
            'bk_sugiyama': self.predict_bispectrum_sugiyama_multipoles(grouped['bk_sugiyama'], params, **kwargs) if grouped['bk_sugiyama'] else [],
        }

        pred = []
        for item in assembly_plan:
            if item[0] == 'single':
                _, group_name, idx = item
                pred.append(predictions[group_name][idx])
            else:
                _, pk_idx, bk_group, bk_idx = item
                pred.append(np.concatenate([predictions['pk'][pk_idx], predictions[bk_group][bk_idx]]))
        return pred
        
    def predict_power_spectrum_multipoles(self, observables, params, **kwargs):
        raise NotImplementedError
    def predict_bispectrum_scoccimarro_multipoles(self, observables, params, **kwargs):
        raise NotImplementedError
    def predict_bispectrum_sugiyama_multipoles(self, observables, params, **kwargs):
        raise NotImplementedError
        

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
    
    def predict_power_spectrum_X_multipoles(self, observables, params, diagram, de_model):
        ell_all =list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)]))
        k_arrys = [np.concatenate(obs.k) if obs.kwin is None else np.concatenate(obs.kwin) for obs in observables]
        k_all = np.unique(np.concatenate(k_arrys))
        if  'a0' in diagram or 'a2' in diagram or 'a4' in diagram:
            pX_batched = self.PX_ell_extra(k_all, params, ell_all, diagram, de_model=de_model)
        else:
            pX_batched = self.PX_ell(k_all, params, ell_all, diagram, de_model=de_model) 
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        for obs in observables:
            iz = list(params['z']).index(obs.cosmo_fid['z']) if is_batched else None
            k = obs.k if obs.kwin is None else obs.kwin
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            pX_z = []
            for i, ll in enumerate(ell):
                idx = np.searchsorted(k_all, k[i])
                # px_batched has shape (nk, ndiag, nz) if is_batched else (nk, ndiag) if diagram is a list of diagrams, otherwise (nk, nz) or (nk,)
                pX_slice = pX_batched[f'ell{ll}'][idx,...,iz] if is_batched else pX_batched[f'ell{ll}'][idx]
                # here pX_slice has shape (nk, ndiag) if diagram is a list of diagrams, otherwise (nk,)
                pX_z.append(pX_slice)
            pX_z = np.concatenate(pX_z, axis=0) # shape (sum(nk_ell), ndiag) or (sum(nk_ell),)
        
            if obs.xwin is not None:
                pX_z = np.einsum('ij,jk->ik', obs.wmat, pX_z) if pX_z.ndim == 2 else obs.wmat @ pX_z
            preds.append(pX_z)
        return preds
    
    def predict_bispectrum_scoccimarro_multipoles(self, observables, params, de_model):
        ell_all = list(set([ll for obs in observables for ll in (obs.ell if obs.ellwin is not None else obs.ell)])) 
        tri_arrays = [np.concatenate(obs.tri) if obs.triwin is None else np.concatenate(obs.triwin) for obs in observables]
        tri_all = np.unique(np.concatenate(tri_arrays), axis=0)
        bell_batched = self.Bell_scoccimarro(tri_all, params, ell_all, de_model=de_model)
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        for obs in observables:
            iz = list(params['z']).index(obs.cosmo_fid['z']) if is_batched else None
            tri = obs.tri if obs.triwin is None else obs.triwin
            ell = obs.ell if obs.ellwin is None else obs.ellwin 
            bk_z = []
            for i, ell in enumerate(obs.ell):
                # Use lexsort for robust row-wise comparison
                idx = np.array([np.where((tri_all == t).all(axis=1))[0][0] for t in tri[i]])
                bell_slice = bell_batched[f'ell{ell}'][idx]
                if is_batched:
                    bell_slice = bell_slice[:, iz]
                bk_z.append(bell_slice)
            bell_z = np.concatenate(bk_z)

            if obs.xwin is not None:
                bell_z = obs.wmat @ bell_z
            preds.append(bell_z)
            
        return preds

    def predict_bispectrum_sugiyama_multipoles(self, observables, params, de_model):
        ell_all = list(set([ll for obs in observables for ll in (obs.ell if obs.ellwin is not None else obs.ell)])) 
        pair_arrays = [np.concatenate(obs.pair) if obs.xwin is None else np.concatenate(obs.pairwin) for obs in observables]
        pair_all = np.unique(np.concatenate(pair_arrays), axis=0)
        bell_batched = self.Bell_sugiyama(pair_all, params, ell_all, de_model=de_model)
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        for obs in observables:
            iz = list(params['z']).index(obs.cosmo_fid['z']) if is_batched else None
            pair = obs.pair if obs.xwin is None else obs.pairwin
            ell = obs.ell if obs.ellwin is None else obs.ellwin 
            bk_z = []
            for i, ell in enumerate(obs.ell):
                # Use lexsort for robust row-wise comparison
                idx = np.array([np.where((pair_all == p).all(axis=1))[0][0] for p in pair[i]])
                bell_slice = bell_batched[f'{ell}'][idx]
                if is_batched:
                    bell_slice = bell_slice[:, iz]
                bk_z.append(bell_slice)
            bell_z = np.concatenate(bk_z)

            if obs.xwin is not None:
                bell_z = obs.wmat @ bell_z
            preds.append(bell_z)

        return preds

    def predict_bispectrum_X_multipoles(self, observables, params, diagram, de_model):
        ell_all = list(set([ll for obs in observables for ll in (obs.ell if obs.ellwin is not None else obs.ell)])) 
        is_scoccimarro = hasattr(observables[0], 'tri')
        
        if is_scoccimarro:
            coord_arragys = [np.concatenate(obs.tri) if obs.xwin is None else np.concatenate(obs.triwin) for obs in observables]
        else:
            coord_arragys = [np.concatenate(obs.pair) if obs.xwin is None else np.concatenate(obs.pairwin) for obs in observables]

        coord_all = np.unique(np.concatenate(coord_arragys), axis=0)     
        
        if is_scoccimarro:
            bX_batched = self.BX_ell_scoccimarro(coord_all, params, ell_all, diagram, de_model=de_model)
        else:
            bX_batched = self.BX_ell_sugiyama(coord_all, params, ell_all, diagram, de_model=de_model)
            
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        
        for obs in observables:
            iz = list(params['z']).index(obs.cosmo_fid['z']) if is_batched else None
            coord = obs.tri if is_scoccimarro and obs.xwin is None else (obs.triwin if is_scoccimarro else (obs.pair if obs.xwin is None else obs.pairwin))
            ell = obs.ell if obs.ellwin is None else obs.ellwin 
            bX_z = []
            for i, ell in enumerate(obs.ell):
                ell_key = f'ell{ell}' if is_scoccimarro else f'{ell}'
                idx = np.array([np.where((coord_all == c).all(axis=1))[0][0] for c in coord[i]])
                bX_slice = bX_batched[ell_key][idx]
                if is_batched:
                    bX_slice = bX_slice[:, iz]
                bX_z.append(bX_slice) 
            bX_z = np.concatenate(bX_z)
            
            if obs.xwin is not None:
                bX_z = obs.wmat @ bX_z
            preds.append(bX_z)
            
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
        mu2 = self.gl_x2
        APfac = np.sqrt(
            np.divide.outer(mu2, self.params['q_lo']**2) \
            + np.divide.outer(1.0 - mu2, self.params['q_tr']**2))
        kp = np.multiply.outer(k_all, APfac) # shape (nk, nmu, 1)
        mup = np.divide.outer(mu, self.params['q_lo'])/APfac
        #print(f"APfac shape: {APfac.shape}, kp shape: {kp.shape}, mup shape: {mup.shape}")

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
        
