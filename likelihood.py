import numpy as np
from observables import Observable, PowerSpectrumMultipoles
import params
from bispectrum import BX_ell_scoccimarro, BX_ell_sugiyama

class Likelihood:
    """Base class for likelihoods"""
    def __init__(self, observable, emu, params, am_params=None,
                 am_sample = True, am_from_comet=False, conditional_prior=None):
        self.observable = observable
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
        self.x = observable.x
        self.xwin = observable.xwin
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
            if am_from_comet:
                n_realizations = observable.nmocks_cov if observable.nmocks_cov is not None else None
                theory_cov = True if self.nmocks_cov is None else False
                emu.define_data_set(obs_id='pk', bins=observable._k, signal=observable._Pell.T, cov=observable._cov,
                                    theory_cov=theory_cov, n_realizations=n_realizations, zeff=observable.cosmo_fid['z'],
                                    fiducial_cosmology=observable.cosmo_fid)
                self.am_priors = {am_param: self.params.parameters[am_param].prior for am_param in self.am_params}

            # self.do_am = True
            self.do_am = True
            self.am_params_0 = np.array([self.params.parameters[am_param].prior[0] for am_param in self.am_params])
            self.am_inv_cov = np.diag([1/self.params.parameters[am_param].prior[1]**2 for am_param in self.am_params])
            self.am_det_cov = np.prod([self.params.parameters[am_param].prior[1]**2 for am_param in self.am_params])
            
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
            dm = self.get_design_matrix_pk(params)
            if not self.am_sample:
                chi2 = self.marg_chi2(delta, self.lcov, self.am_params_0, self.am_inv_cov, self.am_det_cov, dm)
            else:
                chi2, cond_mean, cond_cov = self.marg_chi2(delta, self.lcov, self.am_params_0, self.am_inv_cov,
                                                            self.am_det_cov, dm, return_cond_mean_cov=True)
                self.sample_cond_am(params, cond_mean, cond_cov, mode=self.am_sample_mode)
        return chi2

    def get_loglike(self, params):
        chi2 = self.get_chi2(params)
        loglike = -0.5 * chi2
        return loglike

    @staticmethod
    def marg_chi2(diff, dcov_chol, p0_vec, pcov_inv, detpcov, design_mat,
                  return_cond_mean_cov=False):
        res = diff - design_mat @ p0_vec
        #lamb = design_mat.T @ dcov_inv @ design_mat + pcov_inv
        lamb = get_bCib(dcov_chol, design_mat) + pcov_inv
        lamb_chol = np.linalg.cholesky(lamb) 
        # lamb_inv = np.linalg.inv(lamb) if lamb.shape[0] > 1 else 1/lamb
        # detlamb = np.linalg.det(lamb) if lamb.shape[0] > 1 else lamb
        # compute detlmab from the Cholesky decomposition
        detlamb = np.prod(np.diag(lamb_chol))**2
        b = design_mat.T @ get_Cib(dcov_chol, res)
        chi2 =  get_bCib(dcov_chol, res)
        chi2 = chi2  - get_bCib(lamb_chol, b)
        chi2 = chi2 + np.log(np.abs(detlamb)) + np.log(np.abs(detpcov))  # Include detpcov in log
        chi2 = chi2[0][0] if lamb.shape[0] == 1 else chi2  # If lamb is 1D, return scalar chi2 
        if return_cond_mean_cov:
            # cond_mean = lamb_inv @ (b + pcov_inv @ p0_vec)
            # cond_cov = lamb_inv
            cond_mean = get_Cib(lamb_chol, b + pcov_inv @ p0_vec)
            lamb_inv = get_inv_chol(lamb_chol)
            return chi2, cond_mean, lamb_inv
        return chi2
    
    def get_design_matrix_pk(self, params):
        comet_params = self.params.get_comet_dict(params)
        design_mat = np.zeros((len(self.y), len(self.am_params)))
        xeval = self.xwin if self.xwin is not None else self.x
        convol = self.xwin is not None
        for i, param in enumerate(self.am_params):
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
                px_ell = self.emu.PX_ell(xeval, comet_params, self.observable.ell, diag_to_marg, de_model=self.de_model)
            else:
                diag_to_marg = self.emu._extra_diagrams_to_marg[param_base]
                bx = factor # Apply reparametrization factor if needed
                px_ell = self.emu.PX_ell_extra(xeval, comet_params, self.observable.ell, diag_to_marg, de_model=self.de_model)
            nx = px_ell[f'ell0'].ndim
            if nx == 1:
                m_list = [bx * px_ell[f'ell{l}'] for l in self.observable.ell]
            elif nx > 1:
                m_list = [np.sum(bx * px_ell[f'ell{l}'], axis=1) for l in self.observable.ell] 
            m_vec = np.concatenate(m_list)
            if convol:
                m_vec = self.observable.wmat @ m_vec
            design_mat[:, i] = m_vec
        return design_mat
    
    def get_design_matrix_bk(self, params, base='soccimarro'):
        comet_params = self.params.get_comet_dict(params)
        design_mat = np.zeros((len(self.y), len(self.am_params)))
        xeval = self.xwin if self.xwin is not None else self.x
        convol = self.xwin is not None
    
        for i, param in enumerate(self.am_params):
            if param.endswith('_r'):
                param_base = param.replace('_r', '')
                factor = self.params.get_reparam_factor(params, param)
            else:
                param_base = param
                factor = 1.0

            diag_to_marg = 'B_' + param_base
            if base == 'soccimarro':
                bx_ell = BX_ell_scoccimarro(self.observable.tri[:, 0], self.observable.tri[:, 1], self.observable.tri[:, 2],
                                            self.emu, comet_params, ell=self.observable.ell, diagram=diag_to_marg, de_model=self.de_model, nbar=getattr(self.observable, 'nbar', None))
                m_list = [factor * bx_ell[f'ell{l}'] for l in self.observable.ell]
            elif base == 'sugiyama':
                bx_ell = BX_ell_sugiyama(self.observable.k1, self.observable.k2, self.emu, comet_params, ell=self.observable.ell, diagram=diag_to_marg, de_model=self.de_model, nbar=getattr(self.observable, 'nbar', None))
                m_list = [factor * bx_ell[f'{l}'] for l in self.observable.ell]
            if convol:
                m_list = [self.observable.wmat @ mli for mli in m_list]
            m_vec = np.concatenate(m_list)
            design_mat[:, i] = m_vec
        return design_mat

    def join_design_matrices(self, dm_pk, dm_bk, NP0_pk_idx=None, NP0_bk_idx=None):
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