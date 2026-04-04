import numpy as np
from observables import Observable, PowerSpectrumMultipoles
import params

class Likelihood:
    """Base class for likelihoods"""
    def __init__(self, observable, emu, params, am_params=None,
                 am_sample = True, am_from_comet=False, conditional_prior=None):
        self.observable = observable
        if self.observable.__class__.__name__ == 'JointObservable':
            self.observables = observable.observables
        else:
            self.observables = [observable]
        self.nobservables = len(self.observables)
    
        self.cov = self.observable.cov
        self.nmocks_cov = observable.nmocks_cov
        if self.nmocks_cov is not None:
            self._rescale_covariance(mode='Hartlap')
        self.icov = np.linalg.inv(self.cov)
        self.lcov = np.linalg.cholesky(self.cov)
        self.emu = emu
        self.params = params
        self.params.z = observable.cosmo_fid['z']  # Set redshift in params for use in derived parameters if needed
        self.de_model = self.params.de_model
        # self.x = observable.x
        # self.xwin = observable.xwin
        self.y = observable.get_flatten()
        self.emu.define_fiducial_cosmology(observable.cosmo_fid)
        if getattr(self.observable, 'nbar', None) is not None:
            self.nbar = observable.nbar
            self.emu.define_nbar(self.nbar)
        
        
        self.conditional_prior = None
        if conditional_prior is not None:
            self.add_conditional_prior(conditional_prior)

        self.do_am = False
        self.am_params = am_params
        self.am_sample = am_sample
        self.am_sample_mode = None if not self.am_sample else 'sample' # 'sample' or 'map'
        if self.am_params is not None:
            if not isinstance(self.am_params, list):
                self.am_params = [self.am_params]
             # check that all am_params are in bias, counterterms or stochastic
            for am_param in self.am_params:
                am_param_base = am_param.replace('_r', '') if am_param.endswith('_r') else am_param
                if am_param_base not in self.params.bias_params and am_param_base not in self.params.counterterm_params and am_param_base not in self.params.stochastic_params:
                    raise ValueError(f"AM parameter '{am_param}' not found in bias, counterterm or stochastic parameters.")
                # check that gaussian priors are set for all am_params
                if self.params.parameters[am_param].prior is None or self.params.parameters[am_param].prior_type != 'gaussian':
                    raise ValueError(f"AM parameter '{am_param}' must have a Gaussian prior defined.")
                
                # make sure these params are initialized to zero and now keep them fixed to zero in the sampler
                self.params.parameters[am_param].value = 0.0
                self.params.parameters[am_param].fixed = True
                if self.am_sample:
                    self.params.parameters[am_param].derived_am = True
                    self.params.parameters[am_param].exported = True

                if am_param.endswith('_r'):
                    self.params.parameters[am_param_base].fixed = True
                    self.params.parameters[am_param_base].value = 0.0
                    self.params.parameters[am_param_base].derived = False
                    
        
            # # for the moment we use comet's chi2 function for AM
            # if am_from_comet:
            #     n_realizations = observable.nmocks_cov if observable.nmocks_cov is not None else None
            #     theory_cov = True if self.nmocks_cov is None else False
            #     emu.define_data_set(obs_id='pk', bins=observable._k, signal=observable._Pell.T, cov=observable._cov,
            #                         theory_cov=theory_cov, n_realizations=n_realizations, zeff=observable.cosmo_fid['z'],
            #                         fiducial_cosmology=observable.cosmo_fid)
            #     self.am_priors = {am_param: self.params.parameters[am_param].prior for am_param in self.am_params}

            # self.do_am = True
            self.do_am = True
            self.am_params_0 = np.array([self.params.parameters[am_param].prior[0] for am_param in self.am_params])
            prior_sigmas = np.array([self.params.parameters[am_param].prior[1] for am_param in self.am_params], dtype=float)
            if not np.all(np.isfinite(prior_sigmas)):
                bad = np.array(self.am_params, dtype=object)[~np.isfinite(prior_sigmas)]
                raise ValueError(f"Non-finite AM prior sigma for parameters: {list(bad)}")
            if np.any(prior_sigmas <= 0.0):
                bad = np.array(self.am_params, dtype=object)[prior_sigmas <= 0.0]
                raise ValueError(f"AM prior sigma must be > 0 for parameters: {list(bad)}")
            prior_var = prior_sigmas**2
            self.am_inv_cov = np.diag(1.0 / prior_var)
            self.am_logdet_cov = np.sum(np.log(prior_var))
            
        # if observable.__class__ == PowerSpectrumMultipoles:
        #     self.get_chi2 = self._get_chi2_powerspectrum
        #     if self.do_am:
        #         #self.get_chi2 = self._get_chi2_am_from_comet
        #         self.get_chi2 = self._get_chi2_powerspectrum_am
        #         if am_from_comet:
        #             self.get_chi2 = self._get_chi2_am_from_comet

        if self.conditional_prior is not None:
            old_get_chi2 = self.get_chi2
            def get_chi2_with_prior(params):
                if not self.conditional_prior(params):
                    return np.inf  # Return infinite chi2 if prior condition is not satisfied
                return old_get_chi2(params)
            self.get_chi2 = get_chi2_with_prior

     
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
        print(f"Rescaling covariance by factor {1/factor:.3f} using {mode} correction: n_mocks={n_mocks}, n_data={n_data}")
        #self.icov *= factor
        self.cov /= factor

    # def _get_chi2_powerspectrum(self, params):
    #     comet_params = self.params.get_comet_dict(params)
    #     pred = self.emu.Pell(self.x, params=comet_params, ell=self.observable.ell, de_model=self.de_model)
    #     y_model = np.concatenate([pred[f'ell{l}'] for l in self.observable.ell])
    #     delta = self.y - y_model
    #     chi2 = np.dot(delta, np.dot(self.icov, delta))
    #     return chi2

    # def _get_chi2_powerspectrum_am(self, params):
    #     comet_params = self.params.get_comet_dict(params)
    #     pred = self.emu.Pell(self.x, params=comet_params, ell=self.observable.ell, de_model=self.de_model)
    #     y_model = np.concatenate([pred[f'ell{l}'] for l in self.observable.ell])
    #     delta = self.y - y_model
    #     dm = self.get_design_matrix_ps(params)
    #     if not self.am_sample:
    #         chi2 = self.marg_chi2(delta, self.icov, self.am_params_0, self.am_inv_cov, self.am_det_cov, dm)
    #     else:
    #         chi2, cond_mean, cond_cov = self.marg_chi2(delta, self.icov, self.am_params_0, self.am_inv_cov,
    #                                                     self.am_det_cov, dm, return_cond_mean_cov=True)
    #         self.sample_cond_am(params, cond_mean, cond_cov, mode=self.am_sample_mode) 

    #     return chi2

    # def _get_chi2_am_from_comet(self, params):
    #     comet_params = self.params.get_comet_dict(params)
    #     chi2 = self.emu.chi2(obs_id='pk', params=comet_params, kmax=self.observable._kmax, de_model=self.de_model, AM_priors=self.am_priors)
    #     chi2 = float(chi2)  # Ensure chi2 is a scalar float, not a 0-dim array
    #     return chi2
    
    def get_chi2(self, params):
        comet_params = self.params.get_comet_dict(params)
        pred = self.emu.predict(self.observable, comet_params, de_model=self.de_model)
        delta = self.y - pred
        if not self.do_am:
            # chi2 = delta.T @ self.icov @ delta
            chi2 = get_bCib(self.lcov, delta)
        else:
            dm = self.get_design_matrix(params)
            if not self.am_sample:
                chi2 = self.marg_chi2(delta, self.lcov, self.am_params_0, self.am_inv_cov, self.am_logdet_cov, dm)
            else:
                chi2, cond_mean, cond_cov = self.marg_chi2(delta, self.lcov, self.am_params_0, self.am_inv_cov,
                                                            self.am_logdet_cov, dm, return_cond_mean_cov=True)
                self.sample_cond_am(params, cond_mean, cond_cov, mode=self.am_sample_mode)
        return chi2

    def get_loglike(self, params):
        chi2 = self.get_chi2(params)
        loglike = -0.5 * chi2
        if np.isnan(loglike):
            loglike = -np.inf
        return loglike

    @staticmethod
    def marg_chi2(diff, dcov_chol, p0_vec, pcov_inv, logdetpcov, design_mat,
                  return_cond_mean_cov=False):
        if not np.all(np.isfinite(diff)):
            raise np.linalg.LinAlgError("diff contains NaN or Inf values in marg_chi2")
        if not np.all(np.isfinite(dcov_chol)):
            raise np.linalg.LinAlgError("dcov_chol contains NaN or Inf values in marg_chi2")
        if not np.all(np.isfinite(design_mat)):
            raise np.linalg.LinAlgError("design_mat contains NaN or Inf values in marg_chi2")
        if not np.all(np.isfinite(pcov_inv)):
            raise np.linalg.LinAlgError("pcov_inv contains NaN or Inf values in marg_chi2")
        if not np.isfinite(logdetpcov):
            raise np.linalg.LinAlgError("logdetpcov is NaN or Inf in marg_chi2")

        diag_dcov = np.diag(dcov_chol)
        if np.any(diag_dcov <= 0.0):
            raise np.linalg.LinAlgError(
                f"dcov_chol has non-positive diagonal entries in marg_chi2 (min={diag_dcov.min():.3e})"
            )

        res = diff - design_mat @ p0_vec
        #lamb = design_mat.T @ dcov_inv @ design_mat + pcov_inv
        dt_cinv_d = get_bCib(dcov_chol, design_mat)
        if not np.all(np.isfinite(dt_cinv_d)):
            raise np.linalg.LinAlgError(
                "D^T C^-1 D contains NaN or Inf in marg_chi2; "
                f"max|D|={np.max(np.abs(design_mat)):.3e}, min diag(L_C)={np.min(diag_dcov):.3e}"
            )
        lamb = dt_cinv_d + pcov_inv
        if not np.all(np.isfinite(lamb)):
            raise np.linalg.LinAlgError(
                "lambda contains NaN or Inf in marg_chi2 after adding prior precision; "
                f"max|D^T C^-1 D|={np.max(np.abs(dt_cinv_d)):.3e}, max|P^-1|={np.max(np.abs(pcov_inv)):.3e}"
            )
        lamb = make_posdef(lamb, matrix_name='lambda')
        lamb_chol = np.linalg.cholesky(lamb) 
        # lamb_inv = np.linalg.inv(lamb) if lamb.shape[0] > 1 else 1/lamb
        # detlamb = np.linalg.det(lamb) if lamb.shape[0] > 1 else lamb
        # compute log(det(lamb)) from the Cholesky decomposition for numerical stability
        logdetlamb = 2.0 * np.sum(np.log(np.diag(lamb_chol)))
        b = design_mat.T @ get_Cib(dcov_chol, res)
        chi2 =  get_bCib(dcov_chol, res)
        chi2 = chi2  - get_bCib(lamb_chol, b)
        chi2 = chi2 + logdetlamb + logdetpcov
        chi2 = chi2[0][0] if lamb.shape[0] == 1 else chi2  # If lamb is 1D, return scalar chi2 
        if return_cond_mean_cov:
            # Here res is already centered on p0_vec, so mean is p0_vec + lamb^{-1} b.
            # cond_cov = lamb_inv
            cond_mean = p0_vec + get_Cib(lamb_chol, b)
            lamb_inv = get_inv_chol(lamb_chol)
            return chi2, cond_mean, lamb_inv
        return chi2
    

    def get_design_matrix(self, params):
        if self.observable.__class__.__name__ == 'PowerSpectrumMultipoles':
            return self.get_design_matrix_pk(params, self.observable)
        elif 'Bispectrum' in self.observable.__class__.__name__:
            return self.get_design_matrix_bk(params, self.observable)
        elif self.observable.__class__.__name__ == 'JointObservable':
            dm_pk = self.get_design_matrix_pk(params, self.observables[0])
            dm_bk = self.get_design_matrix_bk(params, self.observables[1])
            return self.join_design_matrices(dm_pk, dm_bk)
    
    def get_design_matrix_pk(self, params, observable):
        _bispec_only_params = ['NB0', 'MB0']
        am_params = [param for param in self.am_params if param not in _bispec_only_params and param.replace('_r', '') not in _bispec_only_params]
        comet_params = self.params.get_comet_dict(params)
        design_mat = np.zeros((observable.n_data, len(am_params)))
        xeval = observable.xwin if observable.xwin is not None else observable.x
        elleval = observable.ell if observable.xwin is None else observable.ellwin
        convol = observable.xwin is not None
        for i, param in enumerate(am_params):
            if param.endswith('_r'):
                param_base = param.replace('_r', '')
                factor = self.params.get_reparam_factor(params, param)
            else:
                param_base = param
                factor = 1.0

            if param_base not in ['a0', 'a2', 'a4']: 
                diag_to_marg = self.emu.diagrams_to_marg[param_base]
                bx = self.emu._get_bias_coeff_for_AM(diag_to_marg)
                
                bx = bx * factor# Apply reparametrization factor if needed
                px_ell = self.emu.PX_ell(xeval, comet_params, elleval, diag_to_marg, de_model=self.de_model)
            else:
                diag_to_marg = self.emu._extra_diagrams_to_marg[param_base]
                bx = factor # Apply reparametrization factor if needed
                px_ell = self.emu.PX_ell_extra(xeval, comet_params, elleval, diag_to_marg, de_model=self.de_model)
            # if not np.all(np.isfinite(bx)):
            #     raise FloatingPointError(
            #         f"Non-finite bias coefficient while building PK design matrix for AM parameter '{param}' "
            #         f"(base='{param_base}', diagram='{diag_to_marg}', params={comet_params})."
            #     )
            # for l in elleval:
            #     arr = np.asarray(px_ell[f'ell{l}'])
            #     if not np.all(np.isfinite(arr)):
            #         bad = np.size(arr) - np.count_nonzero(np.isfinite(arr))
            #         raise FloatingPointError(
            #             f"Non-finite emulator output in PK design matrix for AM parameter '{param}' "
            #             f"(base='{param_base}', diagram='{diag_to_marg}', ell={l}, bad={bad}, shape={arr.shape}, params={comet_params})."
            #         )
            nx = px_ell[f'ell0'].ndim
            if nx == 1:
                m_list = [bx * px_ell[f'ell{l}'] for l in elleval]
            elif nx > 1:
                m_list = [np.sum(bx * px_ell[f'ell{l}'], axis=1) for l in elleval] 
            m_vec = np.concatenate(m_list)
            if convol:
                m_vec = observable.wmat @ m_vec
            # if not np.all(np.isfinite(m_vec)):
            #     bad = np.size(m_vec) - np.count_nonzero(np.isfinite(m_vec))
            #     raise FloatingPointError(
            #         f"Non-finite model vector in PK design matrix for AM parameter '{param}' "
            #         f"(base='{param_base}', diagram='{diag_to_marg}', bad={bad}, params={comet_params})."
            #     )
            design_mat[:, i] = m_vec
        # if not np.all(np.isfinite(design_mat)):
        #     bad = np.size(design_mat) - np.count_nonzero(np.isfinite(design_mat))
        #     raise FloatingPointError(f"PK design matrix contains non-finite entries (bad={bad}, params={comet_params}).")
        return design_mat
   
    def get_design_matrix_bk(self, params, observable):
        _allowed_params = ['NP0', 'NB0', 'MB0']
        am_params = [param for param in self.am_params if param in _allowed_params or param.replace('_r', '') in _allowed_params]
        comet_params = self.params.get_comet_dict(params)
        design_mat = np.zeros((observable.n_data, len(am_params)))
        xeval = observable.xwin if observable.xwin is not None else observable.x
        elleval = observable.ell if observable.xwin is None else observable.ellwin
        convol = observable.xwin is not None
    
        for i, param in enumerate(am_params):
            if param.endswith('_r'):
                param_base = param.replace('_r', '')
                factor = self.params.get_reparam_factor(params, param)
            else:
                param_base = param
                factor = 1.0

            diag_to_marg = 'B_' + param_base
            if observable.__class__.__name__ == 'BispectrumScoccimarroMultipoles':
                bx_ell = self.emu.BX_ell_scoccimarro(xeval, comet_params, elleval, diagram=diag_to_marg, de_model=self.de_model)
                m_list = [factor * bx_ell[f'ell{l}'] for l in elleval]
            elif observable.__class__.__name__ == 'BispectrumSugiyamaMultipoles':
                bx_ell = self.emu.BX_ell_sugiyama(xeval, comet_params, ell=elleval, diagram=diag_to_marg, de_model=self.de_model)
                m_list = [factor * bx_ell[f'{l}'] for l in elleval]
            # for l in elleval:
            #     key = f'ell{l}' if observable.__class__.__name__ == 'BispectrumScoccimarroMultipoles' else f'{l}'
            #     arr = np.asarray(bx_ell[key])
            #     if not np.all(np.isfinite(arr)):
            #         bad = np.size(arr) - np.count_nonzero(np.isfinite(arr))
            #         raise FloatingPointError(
            #             f"Non-finite emulator output in BK design matrix for AM parameter '{param}' "
            #             f"(base='{param_base}', diagram='{diag_to_marg}', ell={l}, bad={bad}, shape={arr.shape})."
            #         )
            m_vec = np.concatenate(m_list)
            if convol:
                m_vec = observable.wmat @ m_vec
            # if not np.all(np.isfinite(m_vec)):
            #     bad = np.size(m_vec) - np.count_nonzero(np.isfinite(m_vec))
            #     raise FloatingPointError(
            #         f"Non-finite model vector in BK design matrix for AM parameter '{param}' "
            #         f"(base='{param_base}', diagram='{diag_to_marg}', bad={bad})."
            #     )
            design_mat[:, i] = m_vec
        # if not np.all(np.isfinite(design_mat)):
        #     bad = np.size(design_mat) - np.count_nonzero(np.isfinite(design_mat))
        #     raise FloatingPointError(f"BK design matrix contains non-finite entries (bad={bad}).")
        return design_mat

    def join_design_matrices(self, dm_pk, dm_bk):
        _bispec_only_params = ['NB0', 'MB0']
        am_pk = [param for param in self.am_params if  param not in _bispec_only_params and param.replace('_r', '') not in _bispec_only_params]
        NP0_pk_idx = None
        if 'NP0' in am_pk:
            NP0_pk_idx = am_pk.index('NP0')
        elif 'NP0_r' in am_pk:
            NP0_pk_idx = am_pk.index('NP0_r')
        am_bk = [param for param in self.am_params if param not in am_pk or (param.replace('_r', '') == 'NP0')]
        NP0_bk_idx = None
        if 'NP0' in am_bk:
            NP0_bk_idx = am_bk.index('NP0')
        elif 'NP0_r' in am_bk:
            NP0_bk_idx = am_bk.index('NP0_r')
         
        ny_pk = dm_pk.shape[0]
        ny_bk = dm_bk.shape[0]
        ny = ny_pk + ny_bk
        nam_pk = dm_pk.shape[1]
        nam_bk = dm_bk.shape[1]
        if NP0_pk_idx is None and NP0_bk_idx is None:
            dm = np.zeros((ny, nam_pk + nam_bk))
            dm[:ny_pk, :nam_pk] = dm_pk
            dm[ny_pk:, nam_pk:] = dm_bk
        else:
            if NP0_pk_idx is None or NP0_bk_idx is None:
                raise ValueError("Both NP0_pk_idx and NP0_bk_idx must be provided together.")
            dm = np.zeros((ny, nam_pk + nam_bk - 1))
            NP0_idx = NP0_pk_idx  # Use NP0_pk_idx as the index for the shared parameter in the combined design matrix
            # Extract the dm_bk column corresponding to NP0
            dm_bk_NP0 = dm_bk[:, NP0_bk_idx] # shape (ny_bk,)
            dm_bk = np.delete(dm_bk, NP0_bk_idx, axis=1) # shape (ny_bk, nam_bk - 1)
            # Fill pk part of the design matrix
            dm[:ny_pk, :nam_pk] = dm_pk
            # Fill bk part of the design matrix
            dm[ny_pk:, nam_pk:] = dm_bk
            # Fill the bk part to the NP0 column of the design matrix
            dm[ny_pk:, NP0_idx] = dm_bk_NP0
        return dm


    def add_conditional_prior(self, conditional_prior):
        """Add a conditional prior to the likelihood. The conditional_prior should be a function that takes the full parameter vector
          and returns a boolean indicating whether the parameters satisfy the prior condition or not."""
        if self.add_conditional_prior is None:
            self.conditional_prior = conditional_prior
        else:
            old_prior = self.conditional_prior
            def combined_prior(params):
                return old_prior(params) and conditional_prior(params)
            self.conditional_prior = combined_prior

    def sample_cond_am(self, params, mean, cov, mode='sample'):
        if len(self.am_params) == 1:
            am_param = self.am_params[0]
            if mode == 'sample':
                value = np.random.normal(mean, np.sqrt(cov))  # Sample from the conditional distribution
            else:
                value = mean  # MAP estimate
            params[am_param] = value
        else:
            if mode == 'sample':
                # Sample from a normal distribution with mean 0 and cov 1.
                z = np.random.normal(size=len(self.am_params))
                # Transform to the desired mean and covariance using the Cholesky decomposition.
                cov_chol = np.linalg.cholesky(cov)
                value = mean + cov_chol @ z    
            else:
                value = mean  # MAP estimate
            for i, am_param in enumerate(self.am_params):
                params[am_param] = value[i]


def get_Cib(Lchol, b):
    # Solve L y = b for y
    y = np.linalg.solve(Lchol, b)
    # Solve L^T x = y for x
    x = np.linalg.solve(Lchol.T, y)
    return x


def make_posdef(matrix, matrix_name='matrix', base_jitter=1e-12, max_tries=8):
    """Return a symmetric positive-definite version of ``matrix``.

    Strategy:
    1) Symmetrize to remove tiny numerical asymmetries.
    2) Try Cholesky directly.
    3) If needed, add diagonal jitter with increasing amplitude.
    4) If eigen-decomposition fails to converge, use an SVD-based symmetric projection.
    5) Final safeguard: keep adding jitter until Cholesky succeeds or fail loudly.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{matrix_name} must be a square 2D array, got shape={matrix.shape}.")

    matrix = 0.5 * (matrix + matrix.T)

    if not np.all(np.isfinite(matrix)):
        raise np.linalg.LinAlgError(
            f"{matrix_name} contains NaN or Inf entries; cannot repair to positive definite."
        )

    try:
        np.linalg.cholesky(matrix)
        return matrix
    except np.linalg.LinAlgError:
        pass

    diag_scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
    jitter = base_jitter * diag_scale
    eye = np.eye(matrix.shape[0], dtype=matrix.dtype)

    for _ in range(max_tries):
        candidate = matrix + jitter * eye
        try:
            np.linalg.cholesky(candidate)
            print(f"Warning: {matrix_name} was not positive definite. Added jitter={jitter:.3e}.")
            return candidate
        except np.linalg.LinAlgError:
            jitter *= 10.0

    floor = base_jitter * diag_scale
    try:
        eigvals, eigvecs = np.linalg.eigh(matrix)
        eigvals = np.clip(eigvals, floor, None)
        repaired = (eigvecs * eigvals) @ eigvecs.T
        repair_mode = "eigenvalue floor"
    except np.linalg.LinAlgError:
        # Fallback when eigensolver does not converge: project with SVD to a nearby symmetric matrix.
        _, singvals, vt = np.linalg.svd(matrix, full_matrices=False)
        hmat = (vt.T * singvals) @ vt
        repaired = 0.5 * (matrix + hmat)
        repair_mode = "SVD projection"

    repaired = 0.5 * (repaired + repaired.T)

    jitter = floor
    for _ in range(max_tries + 4):
        candidate = repaired + jitter * eye
        try:
            np.linalg.cholesky(candidate)
            print(
                f"Warning: {matrix_name} was not positive definite. "
                f"Applied {repair_mode} with final jitter={jitter:.3e}."
            )
            return candidate
        except np.linalg.LinAlgError:
            jitter *= 10.0

    raise np.linalg.LinAlgError(
        f"Unable to make {matrix_name} positive definite after jitter/eigenvalue/SVD repair."
    )

def get_bCib(Lchol, b):
    # Solve L y = b for y
    y = np.linalg.solve(Lchol, b)
    # Compute b^T C^-1 b = y^T y
    bCib = y.T @ y
    return bCib

def get_inv_chol(Lchol):
    # Compute the inverse of the covariance matrix given its Cholesky decomposition
    # Solve L y = I for y (where I is the identity matrix)
    identity = np.eye(Lchol.shape[0])
    y = np.linalg.solve(Lchol, identity)
    # Solve L^T x = y for x
    cov_inv = np.linalg.solve(Lchol.T, y)
    return cov_inv