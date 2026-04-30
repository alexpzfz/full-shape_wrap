import numpy as np
from observables import Observable, PowerSpectrumMultipoles
import params

class Likelihood:
    """Base class for likelihoods
    
    Parameters
    ----------
    observables : Observable or list of Observable
        The observable(s) to use for the likelihood
    params : Params
        Parameter configuration object (which holds the emulator)
    am_params : str or list, optional
        Analytical marginalization parameters
    am_sample : bool, default True
        Whether to sample AM parameters conditionally
    conditional_prior : callable, optional
        A function that checks prior conditions on the full parameter dict
    """
    def __init__(self, observables, params, am_params=None,
                 am_sample=True, conditional_prior=None):
        self.observables = observables if isinstance(observables, list) else [observables]
        # sort observables by redshift
        if len(self.observables) > 1:
            print("Sorting observables by redshift...")
            self.observables.sort(key=lambda obs: obs.cosmo_fid['z'])

        for i, obs in enumerate(self.observables):
            obs._batch_iz = i

        self.nobservables = len(self.observables)
    

        
        self.params = params
        self.emu = self.params.emu  # Access emu through params
        self.de_model = self.params.de_model
        # self.x = observable.x
        # self.xwin = observable.xwin
        self.ys = [obs.get_flatten() for obs in self.observables]
        self.z_list = [obs.cosmo_fid['z'] for obs in self.observables]
        cosmo_fid = {p: np.array([v]*self.nobservables) for p, v in self.observables[0].cosmo_fid.items()}
        cosmo_fid['z'] = np.array(self.z_list)
        self.emu.define_fiducial_cosmology(cosmo_fid)
        self.nbar_list = [getattr(obs, 'nbar', None) for obs in self.observables]
        if any(nbar is not None for nbar in self.nbar_list):
            self.emu.define_nbar(self.nbar_list)
        
        
        self.conditional_prior = conditional_prior
        # if conditional_prior is not None:
        #     self.add_conditional_prior(conditional_prior)

        self.do_am = False
        #self.am_params = am_params
        self.am_sample = am_sample
        self.am_sample_mode = None if not self.am_sample else 'sample' # 'sample' or 'map'
        if am_params is not None:
            self.do_am = True
            self.am_params = []
            self.am_params_0 = []
            self.am_inv_cov = []
            self.am_det_cov = []

            def _am_base_param_name(name):
                if '_r_' in name:
                    return name.split('_r_')[0]
                elif name.endswith('_r'):
                    return name[:-2]
                return name.rsplit('_', 1)[0] if '_' in name else name

            _bispec_only_params = {'NB0', 'MB0'}

            base_am_params = am_params if isinstance(am_params, list) else [am_params]
            for iz in range(self.nobservables):
                am_iz = [f"{param}_{iz}" if self.nobservables > 1 else param for param in base_am_params]
                if self.observables[iz].__class__.__name__ == 'PowerSpectrumMultipoles':
                    am_iz = [p for p in am_iz if _am_base_param_name(p) not in _bispec_only_params]
                self.am_params.append(am_iz)
                for param in am_iz:
                    self.params.parameters[param].value = 0.0
                    self.params.parameters[param].fixed = True
                    self.params.parameters[param].derived_am = True
                    self.params.parameters[param].exported = True
                    if '_r_' in param or param.endswith('_r'):
                        base_param = param.replace('_r_', '_') if '_r_' in param else param.replace('_r', '')
                        self.params.parameters[base_param].value = 0.0
                        self.params.parameters[base_param].fixed = True
                        self.params.parameters[base_param].derived = False
                
                p0 = np.array([self.params.parameters[param].prior[0] for param in am_iz])
                inv_cov = np.diag([1/self.params.parameters[param].prior[1]**2 for param in am_iz])
                det_cov = np.prod([self.params.parameters[param].prior[1]**2 for param in am_iz])
                self.am_params_0.append(p0)
                self.am_inv_cov.append(inv_cov)
                self.am_det_cov.append(det_cov)


            #  # check that all am_params are in bias, counterterms or stochastic
            # for am_param in self.am_params:
            #     am_param_base = am_param.replace('_r', '') if am_param.endswith('_r') else am_param
            #     if am_param_base not in self.params.bias_params and am_param_base not in self.params.counterterm_params and am_param_base not in self.params.stochastic_params:
            #         raise ValueError(f"AM parameter '{am_param}' not found in bias, counterterm or stochastic parameters.")
            #     # check that gaussian priors are set for all am_params
            #     if self.params.parameters[am_param].prior is None or self.params.parameters[am_param].prior_type != 'gaussian':
            #         raise ValueError(f"AM parameter '{am_param}' must have a Gaussian prior defined.")
                
            #     # make sure these params are initialized to zero and now keep them fixed to zero in the sampler
            #     self.params.parameters[am_param].value = 0.0
            #     self.params.parameters[am_param].fixed = True
            #     if self.am_sample:
            #         self.params.parameters[am_param].derived_am = True
            #         self.params.parameters[am_param].exported = True

            #     if am_param.endswith('_r'):
            #         self.params.parameters[am_param_base].fixed = True
            #         self.params.parameters[am_param_base].value = 0.0
            #         self.params.parameters[am_param_base].derived = False
                    
        
            # self.do_am = True
            # self.do_am = True
            # self.am_params_0 = np.array([self.params.parameters[am_param].prior[0] for am_param in self.am_params])
            # self.am_inv_cov = np.diag([1/self.params.parameters[am_param].prior[1]**2 for am_param in self.am_params])
            # self.am_det_cov = np.prod([self.params.parameters[am_param].prior[1]**2 for am_param in self.am_params])
        self.covs = [obs.cov.copy() for obs in self.observables]
        self.nmocks_covs = [obs.nmocks_cov for obs in self.observables]
        self._rescale_covariance(mode='Percival')
        # self.icov = np.linalg.inv(self.cov)
        self.lcovs = [np.linalg.cholesky(obs.cov) for obs in self.observables]
    
 
    def _rescale_covariance(self, mode='Percival'):
        for i, obs in enumerate(self.observables):
            n_data = obs.n_data
            n_mocks = self.nmocks_covs[i]
            n_params = self.params.n_free_params

            if n_mocks is not None:
                hartlap_factor = (n_mocks - n_data - 2) / (n_mocks - 1)

                b = (n_mocks - n_data - 2) / ((n_mocks - n_data - 1) * (n_mocks - n_data - 4))
                percival_factor = (n_mocks - n_data + n_params - 1) / ((n_mocks - 1) * (1 + b * (n_data - n_params)))
                print('Rescaling covariance for observable {} with n_data={}, n_mocks={}, n_params={}'.format(i, n_data, n_mocks, n_params))
                print('Hartlap factor: {:.3f}, Percival factor: {:.3f}'.format(hartlap_factor, percival_factor))
                if mode == 'Hartlap':
                    factor = hartlap_factor
                elif mode == 'Percival':
                    factor = percival_factor
                else:
                    raise ValueError("Invalid mode for covariance rescaling. Choose 'Hartlap' or 'Percival'.")
                self.covs[i] *= 1/factor
        #self.icov *= factor

 
    def get_chi2(self, params):
        if self.conditional_prior is not None and not self.conditional_prior(params):
            return np.inf  # Return infinite chi2 if prior condition is not satisfied
        comet_params = self.params.get_comet_dict(params)
        preds = self.emu.predict(self.observables, comet_params, de_model=self.de_model)

        total_chi2 = 0.0
        dm_cache = {} if self.do_am else None
        for i, obs in enumerate(self.observables):
            delta = self.ys[i] - preds[i]
            if not self.do_am:
                chi2 = get_bCib(self.lcovs[i], delta)
            else:

                dm_iz = self.get_design_matrix(params, dm_cache, i)  # This should be modified to get the correct design matrix for each observable if needed
                if not self.am_sample:
                    chi2 = self.marg_chi2(delta, self.lcovs[i], self.am_params_0[i], self.am_inv_cov[i], self.am_det_cov[i], dm_iz)
                else:
                    chi2, cond_mean, cond_cov = self.marg_chi2(delta, self.lcovs[i], self.am_params_0[i], self.am_inv_cov[i],
                                                                self.am_det_cov[i], dm_iz, return_cond_mean_cov=True)
                    self.sample_cond_am(params, cond_mean, cond_cov, iz=i, mode=self.am_sample_mode)
            total_chi2 += chi2
        return total_chi2 

    def get_loglike(self, params):
        chi2 = self.get_chi2(params)
        loglike = -0.5 * chi2
        if np.isnan(loglike):
            loglike = -np.inf
        return loglike

    @staticmethod
    def marg_chi2(diff, dcov_chol, p0_vec, pcov_inv, detpcov, design_mat,
                  return_cond_mean_cov=False):
        # if not np.all(np.isfinite(diff)):
        #     raise np.linalg.LinAlgError("diff contains NaN or Inf values in marg_chi2")
        # if not np.all(np.isfinite(dcov_chol)):
        #     raise np.linalg.LinAlgError("dcov_chol contains NaN or Inf values in marg_chi2")
        # if not np.all(np.isfinite(design_mat)):
        #     raise np.linalg.LinAlgError("design_mat contains NaN or Inf values in marg_chi2")
        # if not np.all(np.isfinite(pcov_inv)):
        #     raise np.linalg.LinAlgError("pcov_inv contains NaN or Inf values in marg_chi2")
        # # if not np.isfinite(logdetpcov):
        #     raise np.linalg.LinAlgError("logdetpcov is NaN or Inf in marg_chi2")

        diag_dcov = np.diag(dcov_chol)
        if np.any(diag_dcov <= 0.0):
            raise np.linalg.LinAlgError(
                f"dcov_chol has non-positive diagonal entries in marg_chi2 (min={diag_dcov.min():.3e})"
            )

        res = diff - design_mat @ p0_vec
        #lamb = design_mat.T @ dcov_inv @ design_mat + pcov_inv
        dt_cinv_d = get_bCib(dcov_chol, design_mat)
        # if not np.all(np.isfinite(dt_cinv_d)):
        #     raise np.linalg.LinAlgError(
        #         "D^T C^-1 D contains NaN or Inf in marg_chi2; "
        #         f"max|D|={np.max(np.abs(design_mat)):.3e}, min diag(L_C)={np.min(diag_dcov):.3e}"
        #     )
        lamb = dt_cinv_d + pcov_inv
        # if not np.all(np.isfinite(lamb)):
        #     raise np.linalg.LinAlgError(
        #         "lambda contains NaN or Inf in marg_chi2 after adding prior precision; "
        #         f"max|D^T C^-1 D|={np.max(np.abs(dt_cinv_d)):.3e}, max|P^-1|={np.max(np.abs(pcov_inv)):.3e}"
        #     )
        lamb = make_posdef(lamb, matrix_name='lambda')
        lamb_chol = np.linalg.cholesky(lamb) 
        # lamb_inv = np.linalg.inv(lamb) if lamb.shape[0] > 1 else 1/lamb
        # detlamb = np.linalg.det(lamb) if lamb.shape[0] > 1 else lamb
        # compute log(det(lamb)) from the Cholesky decomposition for numerical stability
        detlamb = np.prod(np.diag(lamb_chol))**2
        b = design_mat.T @ get_Cib(dcov_chol, res)
        chi2 =  get_bCib(dcov_chol, res)
        chi2 = chi2  - get_bCib(lamb_chol, b)
        chi2 = chi2 + np.log(np.abs(detlamb)) + np.log(np.abs(detpcov))  # Include detpcov in log
        chi2 = float(np.asarray(chi2)) if lamb.shape[0] == 1 else chi2  # If lamb is 1D, return scalar chi2
        if return_cond_mean_cov:
            # Here res is already centered on p0_vec, so mean is p0_vec + lamb^{-1} b.
            # cond_cov = lamb_inv
            cond_mean = p0_vec + get_Cib(lamb_chol, b)
            lamb_inv = get_inv_chol(lamb_chol)
            return chi2, cond_mean, lamb_inv
        return chi2
    

    def get_design_matrix(self, params, cache, iz):
        observable = self.observables[iz]
        if observable.__class__.__name__ == 'PowerSpectrumMultipoles':
            return self.get_design_matrix_pk(params, cache, iz)
        elif 'Bispectrum' in observable.__class__.__name__:
            return self.get_design_matrix_bk(params, cache, iz)
        elif observable.__class__.__name__ == 'JointObservable':
            dm_pk = self.get_design_matrix_pk(params, cache, iz)
            dm_bk = self.get_design_matrix_bk(params, cache, iz)
            return self.join_design_matrices(dm_pk, dm_bk, iz)
    

    def get_design_matrix_pk(self, params, cache, iz):
        _bispec_only_params = ['NB0', 'MB0']
        def _base_param_name(name):
            if '_r_' in name:
                return name.split('_r_')[0]
            elif name.endswith('_r'):
                return name[:-2]
            return name.rsplit('_', 1)[0] if '_' in name else name

        am_params_iz = [p for p in self.am_params[iz] if _base_param_name(p) not in _bispec_only_params]
        full_observable = self.observables[iz]
        observable = full_observable.observables[0] if full_observable.__class__.__name__ == 'JointObservable' else full_observable
        comet_params = self.params.get_comet_dict(params)
        design_mat = np.zeros((observable.n_data, len(am_params_iz)))
        if cache is None:
            cache = {}

        for i, param in enumerate(am_params_iz):
            base_name = _base_param_name(param)
            if '_r_' in param or param.endswith('_r'):
                factor = self.params.get_reparam_factor(params, param)
            else:
                factor = 1.0

            if base_name not in ['a0', 'a2', 'a4']: 
                diag_to_marg = self.emu.diagrams_to_marg[base_name]
                bx = self.emu._get_bias_coeff_for_AM(diag_to_marg)   
                bx = bx * factor# Apply reparametrization factor if needed
                if len(self.z_list) > 1:
                    bx = bx[..., iz] # Get the bias coefficient for the correct redshift bin
            else:
                diag_to_marg = self.emu._extra_diagrams_to_marg[base_name]
                bx = factor # Apply reparametrization factor if needed
            
            if base_name not in cache:
                pk_observables = [obs.observables[0] if obs.__class__.__name__ == 'JointObservable' else obs for obs in self.observables]
                cache[base_name] = self.emu.predict_power_spectrum_X_multipoles(pk_observables, comet_params, diag_to_marg, de_model=self.de_model)
            px_ell = cache[base_name][iz]

            nx = px_ell.ndim
            if nx == 1:
                m_vec = bx * px_ell
            elif nx > 1:
                m_vec = np.sum(bx * px_ell, axis=1)
            design_mat[:, i] = m_vec
        # if not np.all(np.isfinite(design_mat)):
        #     bad = np.size(design_mat) - np.count_nonzero(np.isfinite(design_mat))
        #     raise FloatingPointError(f"PK design matrix contains non-finite entries (bad={bad}, params={comet_params}).")
        return design_mat
    
    def get_design_matrix_bk(self, params, cache, iz):
        _allowed_params = ['NP0', 'NB0', 'MB0']
        def _base_param_name(name):
            if '_r_' in name:
                return name.split('_r_')[0]
            elif name.endswith('_r'):
                return name[:-2]
            return name.rsplit('_', 1)[0] if '_' in name else name

        am_params_iz = [param for param in self.am_params[iz] if _base_param_name(param) in _allowed_params]
        full_observable = self.observables[iz]
        observable = full_observable.observables[1] if full_observable.__class__.__name__ == 'JointObservable' else full_observable

        comet_params = self.params.get_comet_dict(params)
        design_mat = np.zeros((observable.n_data, len(am_params_iz)))
        if cache is None:
            cache = {}
    
        for i, param in enumerate(am_params_iz):
            base_name = _base_param_name(param)
            if '_r_' in param or param.endswith('_r'):
                factor = self.params.get_reparam_factor(params, param)
            else:
                factor = 1.0

            diag_to_marg = 'B_' + base_name
            
            if diag_to_marg not in cache:
                bk_observables = [obs.observables[1] if obs.__class__.__name__ == 'JointObservable' else obs for obs in self.observables]
                cache[diag_to_marg] = self.emu.predict_bispectrum_X_multipoles(bk_observables, comet_params, diag_to_marg, de_model=self.de_model)
            
            m_vec = factor * cache[diag_to_marg][iz]
            design_mat[:, i] = m_vec
            
        return design_mat

    def join_design_matrices(self, dm_pk, dm_bk, iz):
        _bispec_only_params = {'NB0', 'MB0'}
        def _base_param_name(name):
            if '_r_' in name:
                return name.split('_r_')[0]
            elif name.endswith('_r'):
                return name[:-2]
            return name.rsplit('_', 1)[0] if '_' in name else name

        am_params_iz = self.am_params[iz]
        
        # Create output design matrix with shape (n_pk + n_bk, len(am_params_iz))
        ny_pk = dm_pk.shape[0]
        ny_bk = dm_bk.shape[0]
        ny = ny_pk + ny_bk
        nam_params = len(am_params_iz)
        dm = np.zeros((ny, nam_params))
        
        # Build parameter lists that match dm_pk and dm_bk column ordering
        am_pk = [param for param in am_params_iz if _base_param_name(param) not in _bispec_only_params]
        am_bk = [param for param in am_params_iz if _base_param_name(param) in _bispec_only_params or _base_param_name(param) == 'NP0']
        
        # Create index mappings: parameter -> column index in dm_pk/dm_bk
        pk_indices = {param: i for i, param in enumerate(am_pk)}
        bk_indices = {param: i for i, param in enumerate(am_bk)}
        
        # Fill the design matrix
        for idx, param in enumerate(am_params_iz):
            if param in pk_indices:
                # PK parameters: fill pk rows from dm_pk
                dm[:ny_pk, idx] = dm_pk[:, pk_indices[param]]
            if param in bk_indices:
                # BK parameters (includes shared NP0): fill bk rows from dm_bk
                dm[ny_pk:, idx] = dm_bk[:, bk_indices[param]]
        
        return dm


    # def add_conditional_prior(self, conditional_prior):
    #     """Add a conditional prior to the likelihood. The conditional_prior should be a function that takes the full parameter vector
    #       and returns a boolean indicating whether the parameters satisfy the prior condition or not."""
    #     if self.add_conditional_prior is None:
    #         self.conditional_prior = conditional_prior
    #     else:
    #         old_prior = self.conditional_prior
    #         def combined_prior(params):
    #             return old_prior(params) and conditional_prior(params)
    #         self.conditional_prior = combined_prior

    def sample_cond_am(self, params, mean, cov, iz=0, mode='sample'):
        if len(self.am_params[iz]) == 1:
            am_param = self.am_params[iz][0]
            if mode == 'sample':
                value = np.random.normal(mean, np.sqrt(cov))  # Sample from the conditional distribution
            else:
                value = mean  # MAP estimate
            params[am_param] = value
        else:
            if mode == 'sample':
                # Sample from a normal distribution with mean 0 and cov 1.
                z = np.random.normal(size=len(self.am_params[iz]))
                # Transform to the desired mean and covariance using the Cholesky decomposition.
                cov_chol = np.linalg.cholesky(cov)
                value = mean + cov_chol @ z    
            else:
                value = mean  # MAP estimate
            for i, am_param in enumerate(self.am_params[iz]):
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

    # if not np.all(np.isfinite(matrix)):
    #     raise np.linalg.LinAlgError(
    #         f"{matrix_name} contains NaN or Inf entries; cannot repair to positive definite."
    #     )

    try:
        np.linalg.cholesky(matrix)
        return matrix

    # except np.linalg.LinAlgError:
    #     pass

    except:
        # look for infintes
        if np.any(np.isinf(matrix)):
            # substitute inf with NaNs
            matrix = np.where(np.isinf(matrix), np.nan, matrix)
            print(f"Warning: {matrix_name} contains Inf entries; replaced with NaN for repair process.")
            try:
                np.linalg.cholesky(matrix)
                return matrix
            except np.linalg.LinAlgError:
                pass

        else:
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