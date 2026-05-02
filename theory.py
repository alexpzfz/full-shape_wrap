from comet import comet
import numpy as np
from observables import PowerSpectrumMultipoles, BispectrumScoccimarroMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from bispectrum import bispectrum_scoccimarro_proj, bispectrum_sugiyama_proj, bX_5d, bX_ell_scoccimarro, bX_ell_sugiyama
from scipy.special import eval_legendre
from scipy.interpolate import UnivariateSpline, make_interp_spline

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

        def add(group_name, observable, parent=None):
            if parent is not None and hasattr(parent, '_batch_iz'):
                observable._batch_iz = parent._batch_iz
            elif not hasattr(observable, '_batch_iz'):
                observable._batch_iz = None
            grouped[group_name].append(observable)
            return len(grouped[group_name]) - 1

        assembly_plan = []
        for obs in observables:
            if isinstance(obs, PowerSpectrumMultipoles):
                assembly_plan.append(('single', 'pk', add('pk', obs, obs)))
            elif isinstance(obs, BispectrumScoccimarroMultipoles):
                assembly_plan.append(('single', 'bk_scocc', add('bk_scocc', obs, obs)))
            elif isinstance(obs, BispectrumSugiyamaMultipoles):
                assembly_plan.append(('single', 'bk_sugiyama', add('bk_sugiyama', obs, obs)))
            elif isinstance(obs, JointObservable):
                pk_idx = add('pk', obs.observables[0], obs)
                bk_obs = obs.observables[1]
                if isinstance(bk_obs, BispectrumScoccimarroMultipoles):
                    bk_group = 'bk_scocc'
                elif isinstance(bk_obs, BispectrumSugiyamaMultipoles):
                    bk_group = 'bk_sugiyama'
                else:
                    raise ValueError(f"Unsupported observable type in JointObservable: {type(bk_obs)}")
                bk_idx = add(bk_group, bk_obs, obs)
                assembly_plan.append(('joint', pk_idx, bk_group, bk_idx))
            else:
                raise ValueError(f"Unsupported observable type: {type(obs)}")

        pk = self.predict_power_spectrum_multipoles(grouped['pk'], params, **kwargs) if grouped['pk'] else []
        bk_scocc = self.predict_bispectrum_scoccimarro_multipoles(grouped['bk_scocc'], params, **kwargs) if grouped['bk_scocc'] else []
        bk_sugiyama = self.predict_bispectrum_sugiyama_multipoles(grouped['bk_sugiyama'], params, **kwargs) if grouped['bk_sugiyama'] else []
        predictions = {
            'pk': pk,
            'bk_scocc': bk_scocc,
            'bk_sugiyama': bk_sugiyama,}

        # predictions = {
        #     'pk': self.predict_power_spectrum_multipoles(grouped['pk'], params, **kwargs) if grouped['pk'] else [],
        #     'bk_scocc': self.predict_bispectrum_scoccimarro_multipoles(grouped['bk_scocc'], params, **kwargs) if grouped['bk_scocc'] else [],
        #     'bk_sugiyama': self.predict_bispectrum_sugiyama_multipoles(grouped['bk_sugiyama'], params, **kwargs) if grouped['bk_sugiyama'] else [],
        # }

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
        self.bispec_kwargs = {'soccimarro': {'nmu': 5, 'nphi': 5}, 
                              'sugiyama': {'nmu1': 5, 'nmu12': 12, 'nphi': 5, 'mu12_transform': 'quadratic'}}
        self.use_interp_kwin = False

    def predict_power_spectrum_multipoles(self, observables, params, de_model):
        cache_key = tuple(id(obs) for obs in observables)
        if not hasattr(self, '_pk_cache'):
            self._pk_cache = {}
            
        if cache_key not in self._pk_cache:
            ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)]))
            
            # Collect all k values and track which observable/ell they belong to
            k_segments = []  # List of (k_array, obs_idx, ell_idx)
            for obs_idx, obs in enumerate(observables):
                k = obs.k if obs.kwin is None else obs.kwin
                ell = obs.ell if obs.ellwin is None else obs.ellwin
                for ell_idx, k_ell in enumerate(k):
                    k_segments.append((k_ell, obs_idx, ell_idx))
            
            # Get unique k values and inverse indices (avoids searchsorted)
            k_all_concat = np.concatenate([ks[0] for ks in k_segments])
            k_all, inverse_indices = np.unique(k_all_concat, return_inverse=True)
            
            # Map inverse indices back to each segment
            inverse_idx_offset = 0
            segment_indices = []
            for k_ell, _, _ in k_segments:
                n_k = len(k_ell)
                segment_indices.append(inverse_indices[inverse_idx_offset:inverse_idx_offset + n_k])
                inverse_idx_offset += n_k
                
            self._pk_cache[cache_key] = (ell_all, k_all, segment_indices)
            
        ell_all, k_all, segment_indices = self._pk_cache[cache_key]

        if (obs.kwin is not None for obs in observables) and self.use_interp_kwin:
            k_eval = self.get_kvec_compression(min(k_all), max(k_all))
            pell_eval = self.Pell(k_eval, params, ell_all, de_model=de_model)
            pell_list = np.stack([pell_eval[f'ell{ll}'] for ll in ell_all], axis=1)
            spline = make_interp_spline(k_eval, pell_list, axis=0)(k_all)
            spline = spline.reshape((spline.shape[0] * spline.shape[1],)  + spline.shape[2:], order='F')
            pell_batched = {f'ell{ll}': spline[:, i, ...] for i, ll in enumerate(ell_all)} 

        
        # Evaluate model only at unique k values
        else:
            pell_batched = self.Pell(k_all, params, ell_all, de_model=de_model)
        
        # Extract predictions for each observable
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        segment_idx = 0
        
        for obs in observables:
            iz = getattr(obs, '_batch_iz', None) if is_batched else None
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            pell_z = []
            
            for ll in ell:
                idx = segment_indices[segment_idx]
                p = pell_batched[f'ell{ll}'][idx, iz] if is_batched else pell_batched[f'ell{ll}'][idx]
                pell_z.append(p)
                segment_idx += 1
            
            pell_z = np.concatenate(pell_z)
            if obs.xwin is not None:
                pell_z = obs.wmat @ pell_z
            preds.append(pell_z)

        return preds
    
    def predict_power_spectrum_X_multipoles(self, observables, params, diagram, de_model):
        cache_key = tuple(id(obs) for obs in observables)
        if not hasattr(self, '_pk_X_cache'):
            self._pk_X_cache = {}
            
        if cache_key not in self._pk_X_cache:
            ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)]))
            
            # Collect all k values and track which observable/ell they belong to
            k_segments = []  # List of (k_array, obs_idx, ell_idx)
            for obs_idx, obs in enumerate(observables):
                k = obs.k if obs.kwin is None else obs.kwin
                ell = obs.ell if obs.ellwin is None else obs.ellwin
                for ell_idx, k_ell in enumerate(k):
                    k_segments.append((k_ell, obs_idx, ell_idx))
            
            # Get unique k values and inverse indices
            k_all_concat = np.concatenate([ks[0] for ks in k_segments])
            k_all, inverse_indices = np.unique(k_all_concat, return_inverse=True)
            
            # Map inverse indices back to each segment
            inverse_idx_offset = 0
            segment_indices = []
            for k_ell, _, _ in k_segments:
                n_k = len(k_ell)
                segment_indices.append(inverse_indices[inverse_idx_offset:inverse_idx_offset + n_k])
                inverse_idx_offset += n_k
                
            self._pk_X_cache[cache_key] = (ell_all, k_all, segment_indices)
            
        ell_all, k_all, segment_indices = self._pk_X_cache[cache_key]

        px_ell_func = self.PX_ell
        if 'a0' in diagram or 'a2' in diagram or 'a4' in diagram:
            px_ell_func = self.PX_ell_extra

        if (obs.kwin is not None for obs in observables) and self.use_interp_kwin:
            k_eval = self.get_kvec_compression(min(k_all), max(k_all))
            pX_eval = px_ell_func(k_eval, params, ell_all, de_model=de_model)
            pX_list = np.stack([pX_eval[f'ell{ll}'] for ll in ell_all], axis=1)
            spline = make_interp_spline(k_eval, pX_list, axis=0)(k_all)
            spline = spline.reshape((spline.shape[0] * spline.shape[1],)  + spline.shape[2:], order='F')
            pX_batched = {f'ell{ll}': spline[:, i, ...] for i, ll in enumerate(ell_all)} 

        
        # Evaluate model only at unique k values
        else:
            pX_batched = self.px_ell_func(k_all, params, ell_all, de_model=de_model)
         
        # # Determine which X prediction method to use
        # if 'a0' in diagram or 'a2' in diagram or 'a4' in diagram:
        #     pX_batched = self.PX_ell_extra(k_all, params, ell_all, diagram, de_model=de_model)
        # else:
        #     pX_batched = self.PX_ell(k_all, params, ell_all, diagram, de_model=de_model)
        
        # Extract predictions for each observable
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        segment_idx = 0
        
        for obs in observables:
            iz = getattr(obs, '_batch_iz', None) if is_batched else None
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            pX_z = []
            
            for ll in ell:
                idx = segment_indices[segment_idx]
                pX_slice = pX_batched[f'ell{ll}'][idx, ..., iz] if is_batched else pX_batched[f'ell{ll}'][idx]
                pX_z.append(pX_slice)
                segment_idx += 1
            
            pX_z = np.concatenate(pX_z, axis=0)
            if obs.xwin is not None:
                pX_z = np.einsum('ij,jk->ik', obs.wmat, pX_z) if pX_z.ndim == 2 else obs.wmat @ pX_z
            preds.append(pX_z)
            
        return preds
    
    def predict_bispectrum_scoccimarro_multipoles(self, observables, params, de_model):
        cache_key = tuple(id(obs) for obs in observables)
        if not hasattr(self, '_bk_scocc_cache'):
            self._bk_scocc_cache = {}
            
        if cache_key not in self._bk_scocc_cache:
            ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)])) 
            
            # Collect all triangles and track which observable/ell they belong to
            tri_segments = []  # List of (tri_array, obs_idx, ell_idx)
            for obs_idx, obs in enumerate(observables):
                tri = obs.tri if obs.triwin is None else obs.triwin
                ell = obs.ell if obs.ellwin is None else obs.ellwin
                for ell_idx, tri_ell in enumerate(tri):
                    tri_segments.append((tri_ell, obs_idx, ell_idx))
            
            # Get unique triangles and inverse indices
            tri_all_concat = np.concatenate([tri_seg[0] for tri_seg in tri_segments])
            tri_all, inverse_indices = np.unique(tri_all_concat, axis=0, return_inverse=True)
            
            # Map inverse indices back to each segment
            inverse_idx_offset = 0
            segment_indices = []
            for tri_ell, _, _ in tri_segments:
                n_tri = len(tri_ell)
                segment_indices.append(inverse_indices[inverse_idx_offset:inverse_idx_offset + n_tri])
                inverse_idx_offset += n_tri
                
            self._bk_scocc_cache[cache_key] = (ell_all, tri_all, segment_indices)
            
        ell_all, tri_all, segment_indices = self._bk_scocc_cache[cache_key]
        
        # Evaluate model only at unique triangles
        bell_batched = self.Bell_scoccimarro(tri_all, params, ell_all, de_model=de_model)
        
        # Extract predictions for each observable
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        segment_idx = 0
        
        for obs in observables:
            iz = getattr(obs, '_batch_iz', None) if is_batched else None
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            bk_z = []
            
            for ll in ell:
                idx = segment_indices[segment_idx]
                bell_slice = bell_batched[ll][idx]
                if is_batched:
                    bell_slice = bell_slice[:, iz]
                bk_z.append(bell_slice)
                segment_idx += 1
            
            bell_z = np.concatenate(bk_z)
            if obs.xwin is not None:
                bell_z = obs.wmat @ bell_z
            preds.append(bell_z)
            
        return preds

    def predict_bispectrum_sugiyama_multipoles(self, observables, params, de_model):
        cache_key = tuple(id(obs) for obs in observables)
        if not hasattr(self, '_bk_sugi_cache'):
            self._bk_sugi_cache = {}
            
        if cache_key not in self._bk_sugi_cache:
            ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)])) 
            
            # Collect all pairs and track which observable/ell they belong to
            pair_segments = []  # List of (pair_array, obs_idx, ell_idx)
            for obs_idx, obs in enumerate(observables):
                pair = obs.pair if obs.xwin is None else obs.pairwin
                ell = obs.ell if obs.ellwin is None else obs.ellwin
                for ell_idx, pair_ell in enumerate(pair):
                    pair_segments.append((pair_ell, obs_idx, ell_idx))
            
            # Get unique pairs and inverse indices
            pair_all_concat = np.concatenate([pair_seg[0] for pair_seg in pair_segments])
            pair_all, inverse_indices = np.unique(pair_all_concat, axis=0, return_inverse=True)
            
            # Map inverse indices back to each segment
            inverse_idx_offset = 0
            segment_indices = []
            for pair_ell, _, _ in pair_segments:
                n_pair = len(pair_ell)
                segment_indices.append(inverse_indices[inverse_idx_offset:inverse_idx_offset + n_pair])
                inverse_idx_offset += n_pair
                
            self._bk_sugi_cache[cache_key] = (ell_all, pair_all, segment_indices)
            
        ell_all, pair_all, segment_indices = self._bk_sugi_cache[cache_key]
        
        # Evaluate model only at unique pairs
        bell_batched = self.Bell_sugiyama(pair_all, params, ell_all, de_model=de_model)
        
        # Extract predictions for each observable
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        segment_idx = 0
        
        for obs in observables:
            iz = getattr(obs, '_batch_iz', None) if is_batched else None
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            bk_z = []
            
            for ll in ell:
                idx = segment_indices[segment_idx]
                bell_slice = bell_batched[ll][idx]
                if is_batched:
                    bell_slice = bell_slice[:, iz]
                bk_z.append(bell_slice)
                segment_idx += 1
            
            bell_z = np.concatenate(bk_z)
            if obs.xwin is not None:
                bell_z = obs.wmat @ bell_z
            preds.append(bell_z)

        return preds

    def predict_bispectrum_X_multipoles(self, observables, params, diagram, de_model):
        cache_key = tuple(id(obs) for obs in observables)
        if not hasattr(self, '_bk_X_cache'):
            self._bk_X_cache = {}
            
        if cache_key not in self._bk_X_cache:
            ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)])) 
            is_scoccimarro = hasattr(observables[0], 'tri')
            
            # Collect coordinate segments (tri or pair)
            coord_segments = []
            if is_scoccimarro:
                for obs_idx, obs in enumerate(observables):
                    tri = obs.tri if obs.xwin is None else obs.triwin
                    ell = obs.ell if obs.ellwin is None else obs.ellwin
                    for ell_idx, tri_ell in enumerate(tri):
                        coord_segments.append((tri_ell, obs_idx, ell_idx, True))  # True = scoccimarro
            else:
                for obs_idx, obs in enumerate(observables):
                    pair = obs.pair if obs.xwin is None else obs.pairwin
                    ell = obs.ell if obs.ellwin is None else obs.ellwin
                    for ell_idx, pair_ell in enumerate(pair):
                        coord_segments.append((pair_ell, obs_idx, ell_idx, False))  # False = sugiyama
            
            # Get unique coordinates and inverse indices
            coord_all_concat = np.concatenate([seg[0] for seg in coord_segments])
            coord_all, inverse_indices = np.unique(coord_all_concat, axis=0, return_inverse=True)
            
            # Map inverse indices back to each segment
            inverse_idx_offset = 0
            segment_indices = []
            for coord_ell, _, _, _ in coord_segments:
                n_coord = len(coord_ell)
                segment_indices.append(inverse_indices[inverse_idx_offset:inverse_idx_offset + n_coord])
                inverse_idx_offset += n_coord
                
            self._bk_X_cache[cache_key] = (ell_all, is_scoccimarro, coord_all, segment_indices)
            
        ell_all, is_scoccimarro, coord_all, segment_indices = self._bk_X_cache[cache_key]
        
        # Evaluate model only at unique coordinates
        if is_scoccimarro:
            bX_batched = self.BX_ell_scoccimarro(coord_all, params, ell_all, diagram, de_model=de_model)
        else:
            bX_batched = self.BX_ell_sugiyama(coord_all, params, ell_all, diagram, de_model=de_model)
        
        # Extract predictions for each observable
        preds = []
        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1
        segment_idx = 0
        
        for obs in observables:
            iz = getattr(obs, '_batch_iz', None) if is_batched else None
            ell = obs.ell if obs.ellwin is None else obs.ellwin
            bX_z = []
            
            for ll in ell:
                idx = segment_indices[segment_idx]
                bX_slice = bX_batched[ll][idx]
                if is_batched:
                    bX_slice = bX_slice[:, iz]
                bX_z.append(bX_slice)
                segment_idx += 1
            
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
        bscocc = bispectrum_scoccimarro_proj(k1, k2, k3, self, params, ell=ell, de_model=de_model, **self.bispec_kwargs['soccimarro']) #shape (ntri, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[ll] = bscocc[ll][idx_inverse[idx_ell[i]:idx_ell[i]+len(tri[i])]] if idx_inverse is not None else bscocc[ll]
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
        bsugi = bispectrum_sugiyama_proj(k1, k2, self, params, ell=ell, de_model=de_model, **self.bispec_kwargs['sugiyama']) #shape (npair, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[ll] = bsugi[ll][idx_inverse][idx_ell[i]:idx_ell[i+1]] if idx_inverse is not None else bsugi[ll]
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
        bX_scocc = bX_ell_scoccimarro(k1, k2, k3, self, params, ell=ell, diagram=diagram, de_model=de_model, **self.bispec_kwargs['soccimarro']) #shape (ntri, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[ll] = bX_scocc[ll][idx_inverse][idx_ell[i]:idx_ell[i]+len(tri[i])] if idx_inverse is not None else bX_scocc[ll]
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
        bX_sugi = bX_ell_sugiyama(k1, k2, self, params, ell=ell, diagram=diagram, de_model=de_model, **self.bispec_kwargs['sugiyama']) #shape (npair, n_ell)
        res = {}
        for i, ll in enumerate(ell):
            res[ll] = bX_sugi[ll][idx_inverse][idx_ell[i]:idx_ell[i+1]] if idx_inverse is not None else bX_sugi[ll]
        return res
        


    def get_kvec_compression(self, kmin, kmax, nk=100):
        def croot(x, p):
            return np.sign(x) * np.abs(x)**(1.0 / p)

        kcenter = 0.65
        power = 1.5
        qmin = np.log10(kmin)
        qmax = np.log10(kmax)
        qmin = croot(qmin + kcenter, power)
        qmax = croot(qmax + kcenter, power)

        kvec = np.zeros(nk, dtype=float)
        for i in range(nk):
            k = (qmax - qmin) * (i / (nk - 1)) + qmin
            k = np.sign(k) * np.abs(k)**power - kcenter
            kvec[i] = k

        kvec = 10.0**kvec
        return kvec