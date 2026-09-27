from comet import comet
import numpy as np
from observables import PowerSpectrumMultipoles, BispectrumScoccimarroMultipoles, BispectrumSugiyamaMultipoles, JointObservable
from scipy.special import eval_legendre
from scipy.interpolate import UnivariateSpline, make_interp_spline
from utils import check_finite

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
        self.bispec_kwargs = {'scoccimarro': {'quad_deg': (5, 5), 'norm': 'legendre'},
                              'sugiyama': {'quad_deg': (7, 16, 5), 'mu12_transform': 'quadratic'}}
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

        if self.use_interp_kwin and all(obs.kwin is not None for obs in observables):
            k_eval = self.get_kvec_compression(min(k_all), max(k_all))
            pell_eval = self.Pell(k_eval, params, ell_all, de_model=de_model)
            pell_list = np.stack([pell_eval[f'ell{ll}'] for ll in ell_all], axis=1)
            # make_interp_spline raises an opaque ValueError on non-finite input;
            # flag it here instead so the likelihood can reject the point.
            check_finite(pell_list, 'Pell', self.params)
            spline = make_interp_spline(k_eval, pell_list, axis=0)(k_all) #shapke nk nell
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

        if self.use_interp_kwin and all(obs.kwin is not None for obs in observables):
            k_eval = self.get_kvec_compression(min(k_all), max(k_all))
            pX_eval = px_ell_func(k_eval, params, ell_all, diagram, de_model=de_model)
            pX_list = np.stack([pX_eval[f'ell{ll}'] for ll in ell_all], axis=1)
            check_finite(pX_list, f'PX_ell[{diagram}]', self.params)
            spline = make_interp_spline(k_eval, pX_list, axis=0)(k_all)
            pX_batched = {f'ell{ll}': spline[:, i, ...] for i, ll in enumerate(ell_all)}
            

        
        # Evaluate model only at unique k values
        else:
            pX_batched = px_ell_func(k_all, params, ell_all, diagram, de_model=de_model)
         
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
                tri = obs.tri if obs.xwin is None else obs.triwin
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
        """Back-compat single-label shim around `predict_bispectrum_X_multipoles_batch`."""
        return self.predict_bispectrum_X_multipoles_batch(
            observables, params, [diagram], de_model)[diagram]

    def predict_bispectrum_X_multipoles_batch(self, observables, params, am_diagrams, de_model):
        """Evaluate multiple AM bispectrum design-matrix contributions in one shot.

        Each entry in `am_diagrams` (e.g. 'B_NP0', 'B_MB0', 'B_NB0') maps via
        `_native_bx_recipe` to a list of native Bnoise_* kernels plus a combine
        closure. We take the *union* of those kernels, run a single
        `BX_ell_Scocc`/`BX_ell_Sugi` call (the expensive 5D bispectrum eval
        happens once), then slice + combine per label.

        Returns ``{label: [pred_per_obs_array]}`` matching the single-label
        method's per-observable shape.
        """
        if not am_diagrams:
            return {}
        if not observables:
            return {d: [] for d in am_diagrams}

        cache_key = tuple(id(obs) for obs in observables)
        if not hasattr(self, '_bk_X_cache'):
            self._bk_X_cache = {}

        if cache_key not in self._bk_X_cache:
            ell_all = list(set([ll for obs in observables for ll in (obs.ellwin if obs.ellwin is not None else obs.ell)]))
            is_scoccimarro = hasattr(observables[0], 'tri')

            coord_segments = []
            if is_scoccimarro:
                for obs_idx, obs in enumerate(observables):
                    tri = obs.tri if obs.xwin is None else obs.triwin
                    ell = obs.ell if obs.ellwin is None else obs.ellwin
                    for ell_idx, tri_ell in enumerate(tri):
                        coord_segments.append((tri_ell, obs_idx, ell_idx, True))
            else:
                for obs_idx, obs in enumerate(observables):
                    pair = obs.pair if obs.xwin is None else obs.pairwin
                    ell = obs.ell if obs.ellwin is None else obs.ellwin
                    for ell_idx, pair_ell in enumerate(pair):
                        coord_segments.append((pair_ell, obs_idx, ell_idx, False))

            coord_all_concat = np.concatenate([seg[0] for seg in coord_segments])
            coord_all, inverse_indices = np.unique(coord_all_concat, axis=0, return_inverse=True)

            inverse_idx_offset = 0
            segment_indices = []
            for coord_ell, _, _, _ in coord_segments:
                n_coord = len(coord_ell)
                segment_indices.append(inverse_indices[inverse_idx_offset:inverse_idx_offset + n_coord])
                inverse_idx_offset += n_coord

            self._bk_X_cache[cache_key] = (ell_all, is_scoccimarro, coord_all, segment_indices)

        ell_all, is_scoccimarro, coord_all, segment_indices = self._bk_X_cache[cache_key]

        # Build deduplicated union of native diagrams across all requested AM labels.
        recipes = [(d, *self._native_bx_recipe(d)) for d in am_diagrams]
        native_idx = {}
        for _, names, _ in recipes:
            for n in names:
                if n not in native_idx:
                    native_idx[n] = len(native_idx)
        native_names = list(native_idx.keys())

        ell_tuple = tuple(tuple(ll) for ll in ell_all)
        if is_scoccimarro:
            bx_native = self.BX_ell_Scocc(coord_all, params, ell=ell_tuple, X_list=native_names,
                                          de_model=de_model, **self.bispec_kwargs['scoccimarro'])
        else:
            bx_native = self.BX_ell_Sugi(coord_all, params, ell=ell_tuple, X_list=native_names,
                                         de_model=de_model, **self.bispec_kwargs['sugiyama'])
        # bx_native[ll] shape: (n_coord, nx_native) single-z, (n_coord, nx_native, nz) batched.

        is_batched = isinstance(params.get('z'), (list, np.ndarray)) and len(params['z']) > 1

        # Per AM label, slice the union stack to its recipe-ordered columns and combine.
        bx_per_diag = {}
        for d, names, combine in recipes:
            col_idx = [native_idx[n] for n in names]
            per_ll = {}
            for ll in ell_all:
                full_stack = bx_native[tuple(ll)]
                sub = full_stack[:, col_idx] if not is_batched else full_stack[:, col_idx, :]
                per_ll[ll] = combine(sub)
            bx_per_diag[d] = per_ll

        # Assemble per-observable predictions for each AM label.
        result = {d: [] for d in am_diagrams}
        for d in am_diagrams:
            bx_batched = bx_per_diag[d]
            segment_idx = 0
            for obs in observables:
                iz = getattr(obs, '_batch_iz', None) if is_batched else None
                ell = obs.ell if obs.ellwin is None else obs.ellwin
                bX_z = []
                for ll in ell:
                    idx = segment_indices[segment_idx]
                    bX_slice = bx_batched[ll][idx]
                    if is_batched:
                        bX_slice = bX_slice[:, iz]
                    bX_z.append(bX_slice)
                    segment_idx += 1
                bX_z = np.concatenate(bX_z)
                if obs.xwin is not None:
                    bX_z = obs.wmat @ bX_z
                result[d].append(bX_z)

        return result

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
        # The large-scale damping only belongs to the VDG model (EFT has no avir/sv).
        wdamping = self._W_kurt(kp, mup) if 'VDG_infty' in self.model else 1.0
        res = {}
        # integrand = kaiser_fact * prefact_dict[diagram] * p2d * wdamping
        integrand = kaiser_fact * prefact_dict[diagram]
        if diagram == 'Pctr_a4':
            # Remove octopole contribution from the integrand, since it is not mappable to the c0, c2, c4 basis.
            integrand = integrand - 16/231 * self.params['f']**2 * eval_legendre(6, mup)
        integrand = integrand * p2d * wdamping
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
        ell_tuple = tuple(tuple(ll) for ll in ell)
        bscocc = self.Bell_Scocc(tri, params, ell=ell_tuple, de_model=de_model,
                                 **self.bispec_kwargs['scoccimarro'])
        return {ll: bscocc[tuple(ll)] for ll in ell}

    def Bell_sugiyama(self, pair, params, ell, de_model):
        ell_tuple = tuple(tuple(ll) for ll in ell)
        bsugi = self.Bell_Sugi(pair, params, ell=ell_tuple, de_model=de_model,
                               **self.bispec_kwargs['sugiyama'])
        return {ll: bsugi[tuple(ll)] for ll in ell}

    def _native_bx_recipe(self, diagram):
        """Map local diagram label (B_NP0/B_MB0/B_NB0) to the native Bnoise_*
        diagram list plus a closure that combines the stripped kernels into the
        design-matrix contribution dB/dparam.

        Native bias decomposition (from `BispNum.__stoch_bias_coeffs`):
            coeff(Bnoise_MB0b1b1) = b1**2 * MB0 / nbar
            coeff(Bnoise_MB0b1)   = b1 * (MB0 + NP0) / nbar
            coeff(Bnoise_NP0)     = NP0 / nbar
            coeff(Bnoise_NB0)     = NB0 / nbar**2  (kernel is 1/qiso6 at the
                                                   monopole, 0 elsewhere)

        The closure reads `self.params['b1']`, which the upstream native call
        will have updated to match the input `params` dict.
        """
        if diagram == 'B_NP0':
            names = ['Bnoise_MB0b1', 'Bnoise_NP0']
            def combine(stack):
                b1 = np.atleast_1d(self.params['b1'])
                nbar = np.atleast_1d(self.nbar)
                if stack.ndim == 3:  # (n_coord, nx, nparams)
                    return (b1 * stack[:, 0] + stack[:, 1]) / nbar
                return (b1[0] * stack[:, 0] + stack[:, 1]) / nbar[0]
            return names, combine
        if diagram == 'B_MB0':
            names = ['Bnoise_MB0b1b1', 'Bnoise_MB0b1']
            def combine(stack):
                b1 = np.atleast_1d(self.params['b1'])
                nbar = np.atleast_1d(self.nbar)
                if stack.ndim == 3:
                    return (b1**2 * stack[:, 0] + b1 * stack[:, 1]) / nbar
                return (b1[0]**2 * stack[:, 0] + b1[0] * stack[:, 1]) / nbar[0]
            return names, combine
        if diagram == 'B_NB0':
            names = ['Bnoise_NB0']
            def combine(stack):
                nbar = np.atleast_1d(self.nbar)
                if stack.ndim == 3:
                    return stack[:, 0] / nbar**2
                return stack[:, 0] / nbar[0]**2
            return names, combine
        raise ValueError(f"Unknown bispectrum X diagram label: {diagram!r}")
        


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