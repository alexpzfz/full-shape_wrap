import numpy as np
from scipy.special import legendre, factorial, lpmv
from scipy.interpolate import interp1d, RegularGridInterpolator, RectBivariateSpline, make_interp_spline, CubicSpline
from sympy.physics.wigner import wigner_3j
from numba import njit

einsum_opt = 'optimal' # 'greedy', True


def _contract_proj(flat_vals, proj_op):
    """Contract flattened angular axis with a BLAS-backed matrix multiply."""
    if flat_vals.ndim == 2:
        return flat_vals @ proj_op

    n = flat_vals.shape[0]
    nang = flat_vals.shape[1]
    trailing = flat_vals.shape[2:]
    view = np.moveaxis(flat_vals, 1, -1).reshape(-1, nang)
    return (view @ proj_op).reshape((n, *trailing))


def _interp_separable_2d(vals, W1, W2, ng):
    """Evaluate separable 2D interpolation from a (ng, ng, ...) grid."""
    npts = W1.shape[0]
    trailing = vals.shape[2:]

    tmp = W1 @ vals.reshape(ng, -1)
    tmp = tmp.reshape(npts, ng, *trailing)

    w2 = W2.reshape(npts, ng, *([1] * len(trailing)))
    return np.sum(tmp * w2, axis=1)


def bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, use_pdw_interp=False, **kwargs):
    params = emu.params
    nbar = emu.nbar
    b1, b2, g2, f = params['b1'], params['b2'], params['g2'], params['f']
    
    is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1
    if is_batched:
        k1, k2, k3 = k1[..., None], k2[..., None], k3[..., None]
        mu1, mu2 = mu1[..., None], mu2[..., None]
        
    # Apply AP effect
    qpar, qperp = params['q_lo'], params['q_tr']
    qiso6 = qpar**2 * qperp**4
    mu3 = np.where(k3 > 0, - (mu1 * k1 + mu2 * k2) / k3, -1.0)
    k1_p, mu1_p = apply_ap(k1, mu1, qpar, qperp) 
    k2_p, mu2_p = apply_ap(k2, mu2, qpar, qperp)
    k3_p, mu3_p = apply_ap(k3, mu3, qpar, qperp)
    # k1, k2, k3 are either arrays of any shape or floats

    # if not use_pdw_interp:
    #     # Avoid np.unique() sort overhead (O(N log N)), but process as a single 
    #     # concatenated block to avoid triple fixed-function-call overhead in Pdw.
    #     k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
    #     pdw_all = emu.Pdw(k_all, comet_params, mu=0.6, **kwargs)
        
    #     n1, n2 = k1_p.size, k2_p.size
    #     is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1

    #     if is_batched:
    #         nz = len(comet_params['z'])
    #         has_z_axis = (k1_p.shape[-1] == nz)
    #         if has_z_axis:
    #             p1 = pdw_all[:n1, :]
    #             pdw1 = p1[np.arange(n1), np.arange(n1) % nz].reshape(k1_p.shape)
    #             p2 = pdw_all[n1:n1+n2, :]
    #             pdw2 = p2[np.arange(n2), np.arange(n2) % nz].reshape(k2_p.shape)
    #             p3 = pdw_all[n1+n2:, :]
    #             n3 = k3_p.size
    #             pdw3 = p3[np.arange(n3), np.arange(n3) % nz].reshape(k3_p.shape)
    #         else:
    #             pdw1 = pdw_all[:n1, :].reshape(*k1_p.shape, nz)
    #             pdw2 = pdw_all[n1:n1+n2, :].reshape(*k2_p.shape, nz)
    #             pdw3 = pdw_all[n1+n2:, :].reshape(*k3_p.shape, nz)
    #     else:
    #         pdw1 = pdw_all[:n1].reshape(k1_p.shape)
    #         pdw2 = pdw_all[n1:n1+n2].reshape(k2_p.shape)
    #         pdw3 = pdw_all[n1+n2:].reshape(k3_p.shape)


    # if is_batched:
    #     nz = len(comet_params['z'])
    #     z_values = np.asarray(comet_params['z'], dtype=float)

    #     k_all_flat = np.concatenate([k1_p.ravel(), k2_p.ravel(), k3_p.ravel()])
    #     kmin, kmax = k_all_flat.min(), k_all_flat.max()
    #     kgrid = get_kvec_compression(kmin, kmax, 100)

    #     #kwargs_ = kwargs.copy()
    #     #kwargs_['de_model'] = 'w0wa'
    #     pdw_grid = emu.Pdw(kgrid, comet_params, mu=0.6, **kwargs)  # shape (100, nz)

    #     sort_idx = np.argsort(z_values)
    #     ky = min(3, nz - 1)
    #     biv_spl = RectBivariateSpline(kgrid, z_values[sort_idx], pdw_grid[:, sort_idx], kx=3, ky=ky, s=0)

    #     z_bcast = np.broadcast_to(z_values, k1_p.shape)
    #     pdw1 = biv_spl.ev(k1_p.ravel(), z_bcast.ravel()).reshape(k1_p.shape)
    #     z_bcast = np.broadcast_to(z_values, k2_p.shape)
    #     pdw2 = biv_spl.ev(k2_p.ravel(), z_bcast.ravel()).reshape(k2_p.shape)
    #     z_bcast = np.broadcast_to(z_values, k3_p.shape)
    #     pdw3 = biv_spl.ev(k3_p.ravel(), z_bcast.ravel()).reshape(k3_p.shape)

    # if is_batched:
    #     nz = len(comet_params['z'])
        
    #     k_all_flat = np.concatenate([k1_p.ravel(), k2_p.ravel(), k3_p.ravel()])
    #     kmin, kmax = k_all_flat.min(), k_all_flat.max()
        
    #     # Add a tiny buffer (1%) to prevent extrapolation errors at the bounds
    #     kgrid = get_kvec_compression(kmin * 0.99, kmax * 1.01, 100)

    #     pdw_grid = emu.Pdw(kgrid, comet_params, mu=0.6, **kwargs)  # shape (100, nz)

    #     # 1D vectorised spline interpolation along the k-axis
    #     spline = make_interp_spline(kgrid, pdw_grid, axis=0, k=3)
        
    #     # k1_p, k2_p, k3_p have a trailing dimension of 1 due to the [..., None] 
    #     # added for `is_batched`. We slice it off with [..., 0] before evaluating.
    #     # The spline automatically returns an ND-array with a trailing `nz` dimension.
    #     pdw1 = spline(k1_p[..., 0])
    #     pdw2 = spline(k2_p[..., 0])
    #     pdw3 = spline(k3_p[..., 0])
    if is_batched:
        nz = len(comet_params['z'])
        
        kmin = min(k1_p.min(), k2_p.min(), k3_p.min())
        kmax = max(k1_p.max(), k2_p.max(), k3_p.max())
        
        kgrid = get_kvec_compression(kmin * 0.99, kmax * 1.01, 100)
        pdw_grid = emu.Pdw(kgrid, comet_params, mu=0.6, **kwargs)  # shape (100, nz)

        pdw1 = np.empty_like(k1_p)
        pdw2 = np.empty_like(k2_p)
        pdw3 = np.empty_like(k3_p)

        for i in range(nz):
            cs = CubicSpline(kgrid, pdw_grid[:, i])
            pdw1[..., i] = cs(k1_p[..., i])
            pdw2[..., i] = cs(k2_p[..., i])
            pdw3[..., i] = cs(k3_p[..., i])
    # if is_batched:
    #     nz = len(comet_params['z'])
        
    #     kmin = min(k1_p.min(), k2_p.min(), k3_p.min())
    #     kmax = max(k1_p.max(), k2_p.max(), k3_p.max())
        
    #     # 1. Evaluate on a 10x denser grid. 
    #     # Linear interp on 1000 points = Cubic interp on 100 points, but 50x faster.
    #     kgrid_dense = get_kvec_compression(kmin * 0.99, kmax * 1.01, 1000)
    #     pdw_grid_dense = emu.Pdw(kgrid_dense, comet_params, mu=0.6, **kwargs)

    #     # 2. Flatten all spatial dimensions down to 1D. 
    #     # We keep 'nz' as the second dimension. Shape becomes (N_elements, nz).
    #     k1_flat = k1_p.reshape(-1, nz)
    #     k2_flat = k2_p.reshape(-1, nz)
    #     k3_flat = k3_p.reshape(-1, nz)

    #     pdw1_flat = np.empty_like(k1_flat)
    #     pdw2_flat = np.empty_like(k2_flat)
    #     pdw3_flat = np.empty_like(k3_flat)

    #     # 3. np.interp runs in raw C and dominates Scipy spline performance.
    #     for i in range(nz):
    #         pdw1_flat[:, i] = np.interp(k1_flat[:, i], kgrid_dense, pdw_grid_dense[:, i])
    #         pdw2_flat[:, i] = np.interp(k2_flat[:, i], kgrid_dense, pdw_grid_dense[:, i])
    #         pdw3_flat[:, i] = np.interp(k3_flat[:, i], kgrid_dense, pdw_grid_dense[:, i])

    #     # 4. Instantly reshape back to the original N-dimensional broadcast shape
    #     pdw1 = pdw1_flat.reshape(k1_p.shape)
    #     pdw2 = pdw2_flat.reshape(k2_p.shape)
    #     pdw3 = pdw3_flat.reshape(k3_p.shape)

    elif not use_pdw_interp:
        k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
        pdw_all = emu.Pdw(k_all, comet_params, mu=0.6, **kwargs)

        n1, n2 = k1_p.size, k2_p.size
        pdw1 = pdw_all[:n1].reshape(k1_p.shape)
        pdw2 = pdw_all[n1:n1+n2].reshape(k2_p.shape)
        pdw3 = pdw_all[n1+n2:].reshape(k3_p.shape)
    else:
        k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
        kmin, kmax = np.min(k_all), np.max(k_all)
        kgrid = get_kvec_compression(kmin, kmax, 100)
        pdw_grid = emu.Pdw(kgrid, comet_params, mu=0.6, **kwargs)

        spline = make_interp_spline(kgrid, pdw_grid, axis=0, k=3)
        pdw1 = spline(k1_p)
        pdw2 = spline(k2_p)
        pdw3 = spline(k3_p)
    # else:
    #     k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
    #     kmin, kmax = np.min(k_all), np.max(k_all)
        
    #     # Add a tiny buffer to avoid edge extrapolation crashes
    #     kgrid = get_kvec_compression(kmin * 0.99, kmax * 1.01, 100)
    #     pdw_grid = emu.Pdw(kgrid, comet_params, mu=0.6, **kwargs)

    #     is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1

    #     spline = make_interp_spline(kgrid, pdw_grid, axis=0, k=3)

    #     if is_batched:
    #         # Using axis1=-2 and axis2=-1 ensures this works even if k1_p has more dimensions (like nmu, nphi)
    #         pdw1 = np.diagonal(spline(k1_p), axis1=-2, axis2=-1)
    #         pdw2 = np.diagonal(spline(k2_p), axis1=-2, axis2=-1)
    #         pdw3 = np.diagonal(spline(k3_p), axis1=-2, axis2=-1)
    #     else:
    #         pdw1 = spline(k1_p)
    #         pdw2 = spline(k2_p)
    #         pdw3 = spline(k3_p)

        # if is_batched:
        #     pdw_interp_list = [interp1d(kgrid, pdw_grid[:, j], axis=0, kind='cubic') for j in range(pdw_grid.shape[1])]
        #     has_z_axis = (k1_p.shape[-1] == len(comet_params['z']))
        #     if has_z_axis:
        #         pdw1 = np.stack([pdw_interp_list[j](k1_p[..., j]) for j in range(pdw_grid.shape[1])], axis=-1)
        #         pdw2 = np.stack([pdw_interp_list[j](k2_p[..., j]) for j in range(pdw_grid.shape[1])], axis=-1)
        #         pdw3 = np.stack([pdw_interp_list[j](k3_p[..., j]) for j in range(pdw_grid.shape[1])], axis=-1)
        #     else:
        #         pdw1 = np.stack([pdw_interp_list[j](k1_p) for j in range(pdw_grid.shape[1])], axis=-1)
        #         pdw2 = np.stack([pdw_interp_list[j](k2_p) for j in range(pdw_grid.shape[1])], axis=-1)
        #         pdw3 = np.stack([pdw_interp_list[j](k3_p) for j in range(pdw_grid.shape[1])], axis=-1)
        # else:
        #     pdw_interp = interp1d(kgrid, pdw_grid, axis=0, kind='cubic')
        #     pdw1 = pdw_interp(k1_p)
        #     pdw2 = pdw_interp(k2_p)
        #     pdw3 = pdw_interp(k3_p)

    # tree level first. Precompute shared algebra and use in-place operators
    mu1_sq, mu2_sq, mu3_sq = mu1_p**2, mu2_p**2, mu3_p**2
    Z1_1 = kernel_Z1(mu1_p, b1, f, mu_sq=mu1_sq)
    Z1_2 = kernel_Z1(mu2_p, b1, f, mu_sq=mu2_sq)
    Z1_3 = kernel_Z1(mu3_p, b1, f, mu_sq=mu3_sq)

    btree = tree_term(k1_p, k2_p, mu1_p, mu2_p, k3_p, mu3_p, b1, b2, g2, f, Z1_1, Z1_2) * pdw1 * pdw2 + \
            tree_term(k2_p, k3_p, mu2_p, mu3_p, k1_p, mu1_p, b1, b2, g2, f, Z1_2, Z1_3) * pdw2 * pdw3 + \
            tree_term(k3_p, k1_p, mu3_p, mu1_p, k2_p, mu2_p, b1, b2, g2, f, Z1_3, Z1_1) * pdw3 * pdw1
    # # btree = btree * pdw1
    # # btree = btree * pdw2
    
    # t2 = tree_term(k2_p, k3_p, mu2_p, mu3_p, k1_p, mu1_p, b1, b2, g2, f, Z1_2, Z1_3)
    # t2 = t2 * pdw2
    # t2 = t2 * pdw3
    # btree = btree + t2

    # t3 = tree_term(k3_p, k1_p, mu3_p, mu1_p, k2_p, mu2_p, b1, b2, g2, f, Z1_3, Z1_1)
    # t3 = t3 * pdw3
    # t3 = t3 * pdw1
    # btree = btree + t3
    
    # now the stochastic part
    NB0, MB0, NP0 = params['NB0'], params['MB0'], params['NP0']
    avir, sv = params['avir'], params['sv']

    kxmu1_sq = (k1_p**2) * mu1_sq
    kxmu2_sq = (k2_p**2) * mu2_sq
    kxmu3_sq = (k3_p**2) * mu3_sq
    
    # avoid call if MB0=NP0=0 (useful for AM)
    if np.all(MB0 == 0) and np.all(NP0 == 0):
        bstoch = np.zeros_like(btree)
    else:
        bstoch = stoch_term(kxmu1_sq, mu1_sq, b1, f, avir, sv, MB0, NP0, Z1_1) * pdw1 +\
                stoch_term(kxmu2_sq, mu2_sq, b1, f, avir, sv, MB0, NP0, Z1_2) * pdw2 +\
                stoch_term(kxmu3_sq, mu3_sq, b1, f, avir, sv, MB0, NP0, Z1_3) * pdw3
    # bstoch = bstoch * pdw1
    
    # s2 = stoch_term(kxmu2_sq, mu2_sq, b1, f, avir, sv, MB0, NP0, Z1_2)
    # s2 = s2 * pdw2
    # bstoch = bstoch + s2

    # s3 = stoch_term(kxmu3_sq, mu3_sq, b1, f, avir, sv, MB0, NP0, Z1_3)
    # s3 = s3 * pdw3
    # bstoch = bstoch + s3

    # bstoch = bstoch * (1.0 / nbar)
    # bstoch = bstoch + (NB0/nbar**2)

    bstoch *= (1.0 / nbar)
    bstoch += (NB0/nbar**2)

    #construct vdg bispectrum
    lambda2 = -0.5 * f**2 * (kxmu1_sq + kxmu2_sq + kxmu3_sq)
    winfty = w_B_infty(lambda2, avir, sv)
    # btree = btree * winfty
    # btree = btree + bstoch
    # btree = btree / qiso6
    bvdg = (btree * winfty + bstoch) / qiso6
    return bvdg



def tree_term(ki, kj, mui, muj, kk, muk, b1, b2, g2, f, Z1_i, Z1_j):
    muij = get_dot_cosine(ki, kj, kk)
    t = (2.0 * Z1_i) * Z1_j
    # Use normal assignment here since t may be smaller broadcast shape than kernel_Z2
    t = t * kernel_Z2(ki, kj, mui, muj, muij, kk, -muk, b1, b2, g2, f, Z1_i, Z1_j)
    return t


def stoch_term(kxmu_sq, mui_sq, b1, f, avir, sv, MB0, NP0, Z1_i):
    t = (b1 * MB0 + (f * NP0) * mui_sq) * Z1_i
    lambda2 = -f**2 * kxmu_sq
    t = t * w_B_infty(lambda2, avir, sv)
    return t

def bispectrum_scoccimarro_proj(k1, k2, k3, emu, comet_params, ell=[(0, 0), (2, 0)], **kwargs):
    nmu = kwargs.pop('nmu', 5)
    nphi = kwargs.pop('nphi', 5)
    phi_quad = kwargs.pop('phi_quad', 'linear')
    mu, w_mu = np.polynomial.legendre.leggauss(nmu)

    if phi_quad == 'linear':
        phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
        cphi = np.cos(phi)
        w_phi = 2 * np.pi / nphi
        w_cphi = w_phi * np.ones_like(cphi) # transform weights to be in terms of phi
    elif phi_quad == 'chebyshev':
        cphi, w_cphi = np.polynomial.chebyshev.chebgauss(nphi) # cphi = cos(phi)
        w_cphi = 2 * w_cphi # transform weights to be in terms of phi
        phi = np.arccos(cphi) # transform cphi to phi
    elif phi_quad == 'legendre':
        phi, w_phi = np.polynomial.legendre.leggauss(nphi) # phi quadrature
        # transform -1, 1 to 0, 2pi
        phi = 0.5 * (phi + 1) * 2 * np.pi
        cphi = np.cos(phi)
        w_cphi = w_phi * np.pi # transform weights to be in terms of phi

    mu1 = mu[None, :, None] # shape (1, nmu, 1)
    cphi = cphi[None, None, :] # shape (1, 1, nphi)
    w_cphi = w_cphi[None, None, :] # shape (1, 1, nphi)

    k1, k2, k3 = k1[:, None, None], k2[:, None, None], k3[:, None, None] # shape (ntri, 1, 1)

    mu12 = get_dot_cosine(k1, k2, k3)
    # ensure mu12 is in the range [-1, 1] to avoid numerical issues with sqrt
    mu12 = np.clip(mu12, -1, 1)
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * cphi
    # reshape everything to be (ntri, nmu, nphi)
    bfull = bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, **kwargs) # shape (ntri, nmu, nphi) or (ntri, nmu, nphi, nz)
    res = {}
    is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1
    n = k1.shape[0]

    if is_batched:
        nz = len(comet_params['z'])
        bfull_flat = bfull.reshape(n, -1, nz)
    else:
        bfull_flat = bfull.reshape(n, -1)
    
    for ll in ell:
        l, m = ll
        m_ = abs(m)
        sign = (-1)**m if m < 0 else 1.
        fact = 1 / np.sqrt(2) if m != 0 else 1.
        ylm = sph_harm_real(l, m_, mu1, phi) # shape (1, nmu, nphi)
        weights = w_mu[None, :, None] * w_cphi
        proj_op = (ylm * weights).ravel()
            
        if is_batched:
            integral = np.einsum('ijk,j->ik', bfull_flat, proj_op, optimize=einsum_opt)
        else:
            integral = np.dot(bfull_flat, proj_op)

        bell = sign * fact * (2*l + 1) * integral / (4 * np.pi)
        res[ll] = bell
    return res

def bispectrum_sugiyama_proj(k1, k2, emu, comet_params, ell=[(0, 0, 0), (2, 0, 2)], use_pdw_interp=False, 
                             interpolate_k1k2=False, **kwargs):
    # let's use Scoccimarro coordinate system!!
    n = k1.shape[0]
    nmu1 = kwargs.pop('nmu1', 5) # cos(\omega)
    nmu12 = kwargs.pop('nmu12', 12) # cos(\theta_{12})
    nphi = kwargs.pop('nphi', 5) # \phi
    k1k2_interp_method = kwargs.pop('k1k2_interp_method', 'cubic')
    k1k2_interp_grid_size = kwargs.pop('k1k2_interp_grid_size', None)
    k1k2_interp_adaptive = kwargs.pop('k1k2_interp_adaptive', True)
    k1k2_interp_scale = kwargs.pop('k1k2_interp_scale', 'log')
    mu12_transform = kwargs.pop('mu12_transform', 'quadratic') # change of variables for mu12 to resolve k3 ~ 0 singularity when k1 ~ k2
    mu1, w_mu1 = np.polynomial.legendre.leggauss(nmu1)
    
    # Change of variables for mu12 to resolve the k3 ~ 0 singularity when k1 ~ k2
    # We substitute mu12 = 0.5 * (x + 1)**2 - 1.0, where x is Gauss-Legendre roots in [-1, 1].
    # This places more integration points near mu12 = -1 and removes the square root 
    x_mu12, w_x_mu12 = np.polynomial.legendre.leggauss(nmu12)
    # cphi, w_cphi = np.polynomial.chebyshev.chebgauss(nphi) # cphi = cos(phi)
    # w_cphi = 2 * w_cphi
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    cphi = np.cos(phi)
    w_cphi = w_phi * np.ones_like(cphi) 

    if mu12_transform == 'linear':
        mu12 = x_mu12
        w_mu12 = w_x_mu12
    elif mu12_transform == 'quadratic':
        mu12 = 0.5 * (x_mu12 + 1)**2 - 1.0
        w_mu12 = w_x_mu12 * (x_mu12 + 1)
    elif mu12_transform == 'quartic':
        mu12 = 0.125 * (x_mu12 + 1)**4 - 1.0
        w_mu12 = w_x_mu12 * 0.5 * (x_mu12 + 1)**3


    mu1 = mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    w_mu1 = w_mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    mu12 = mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    w_mu12 = w_mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    cphi = cphi[None, None, None, :] # shape (1, 1, 1, nphi)
    w_cphi = w_cphi[None, None, None, :] # shape (1, 1, 1, nphi)

    if not interpolate_k1k2:
        k1, k2 = k1[:, None, None, None], k2[:, None, None, None] # shape (n, 1, 1, 1)
    else:
        k1_old, k2_old = k1.copy(), k2.copy()
        n_input = k1_old.shape[0]
        if k1k2_interp_grid_size is not None:
            interp_grid_size = max(4, int(k1k2_interp_grid_size))
        elif k1k2_interp_adaptive:
            # For large n this keeps interpolation accurate while avoiding oversized grids.
            interp_grid_size = int(np.clip(np.sqrt(n_input), 20, 40))
        else:
            interp_grid_size = 30

        if k1k2_interp_scale == 'log':
            k1_grid = np.logspace(np.log10(k1_old.min()*0.99), np.log10(k1_old.max()*1.1), interp_grid_size, endpoint=True)
            k2_grid = np.logspace(np.log10(k2_old.min()*0.99), np.log10(k2_old.max()*1.1), interp_grid_size, endpoint=True)
        elif k1k2_interp_scale == 'linear':
            k1_grid = np.linspace(k1_old.min()*0.99, k1_old.max()*1.1, interp_grid_size, endpoint=True)
            k2_grid = np.linspace(k2_old.min()*0.99, k2_old.max()*1.1, interp_grid_size, endpoint=True)
        elif k1k2_interp_scale == 'hybrid':
            # Logarithmic spacing at low k and linear spacing at high k
            kthresh = 0.02
            log_size = interp_grid_size // 4
            lin_size = interp_grid_size - log_size
            k1_grid_log = np.logspace(np.log10(k1_old.min()*0.99), np.log10(kthresh*0.99), log_size, endpoint=True)
            k1_grid_lin = np.linspace(kthresh*1.05, k1_old.max()*1.1, lin_size, endpoint=True)
            k1_grid = np.concatenate([k1_grid_log, k1_grid_lin])

            k2_grid_log = np.logspace(np.log10(k2_old.min()*0.99), np.log10(kthresh*0.99), log_size, endpoint=True)
            k2_grid_lin = np.linspace(kthresh*1.05, k2_old.max()*1.1, lin_size, endpoint=True)
            k2_grid = np.concatenate([k2_grid_log, k2_grid_lin])
        elif k1k2_interp_scale == 'comet':
            k1_grid = get_kvec_compression(np.min(k1_old), np.max(k1_old), interp_grid_size)
            k2_grid = get_kvec_compression(np.min(k2_old), np.max(k2_old), interp_grid_size)
        else:
            raise ValueError(f"Unsupported k1k2_interp_scale: {k1k2_interp_scale}")
        interp_points = np.column_stack((k1_old, k2_old))

        k1 = k1_grid
        k2 = k2_grid
        k1, k2 = np.meshgrid(k1, k2, indexing='ij') # shape (35, 35)
        k1 = k1.flatten()[:, None, None, None] # shape (n, 1, 1, 1)
        k2 = k2.flatten()[:, None, None, None] # shape (n, 1, 1, 1)
        n = k1.shape[0]
    
    
    # get k3 using the triangle condition
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12) # shape (n, 1, nmu12, 1)
    # get mu2 using the Scoccimarro coordinate system
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * cphi # shape (n, nmu1, nmu12, nphi)
    
    bfull = bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, use_pdw_interp=use_pdw_interp, **kwargs) # shape (n, nmu1, nmu2, nphi) or (n, ..., nz)
    proj_ops = get_cached_proj_operator(nmu1, nmu12, nphi, ell, w_mu1, w_mu12, w_cphi, mu1, mu12, phi, mu12_transform=mu12_transform)
    
    is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1
    res = {}
    
    # if is_batched:
    #     nz = len(comet_params['z'])
    #     bfull_flat = bfull.reshape(n, -1, nz)
    #     for ll in ell:
    #         res[ll] = np.einsum('ijk,j->ik', bfull_flat, proj_ops[ll], optimize=einsum_opt)
    #         if interpolate_k1k2:
    #             interpolated_z = []
    #             for iz in range(nz):
    #                 grid_values = res[ll][:, iz].reshape(interp_grid_size, interp_grid_size)
    #                 if k1k2_interp_method == 'linear':
    #                     interp_func = RegularGridInterpolator((k1_grid, k2_grid), grid_values, method='linear')
    #                     interpolated_z.append(interp_func(interp_points))
    #                 elif k1k2_interp_method == 'cubic':
    #                     interp_func = RectBivariateSpline(k1_grid, k2_grid, grid_values, kx=3, ky=3, s=0)
    #                     interpolated_z.append(interp_func.ev(interp_points[:, 0], interp_points[:, 1]))
    #                 else:
    #                     raise ValueError(f"Unsupported k1k2_interp_method: {k1k2_interp_method}")
    #             res[ll] = np.column_stack(interpolated_z)
    if is_batched:
        nz = len(comet_params['z'])

    if interpolate_k1k2:
        deg = {'linear': 1, 'quadratic': 2, 'cubic': 3, 'quintic': 5}[k1k2_interp_method]
        ng = interp_grid_size
        W1 = make_interp_spline(k1_grid, np.eye(ng), k=deg)(interp_points[:, 0])  # (n_pts, ng)
        W2 = make_interp_spline(k2_grid, np.eye(ng), k=deg)(interp_points[:, 1])  # (n_pts, ng)

    if is_batched:
        bfull_flat = bfull.reshape(n, -1, nz)
        for ll in ell:
            res_ll = _contract_proj(bfull_flat, proj_ops[ll])
            if interpolate_k1k2:
                vals = res_ll.reshape(ng, ng, nz)
                res_ll = _interp_separable_2d(vals, W1, W2, ng)
            res[ll] = res_ll
    else:
        bfull_flat = bfull.reshape(n, -1)
        for ll in ell:
            res_ll = _contract_proj(bfull_flat, proj_ops[ll])
            if interpolate_k1k2:
                vals = res_ll.reshape(ng, ng)
                res_ll = _interp_separable_2d(vals, W1, W2, ng)
            res[ll] = res_ll

    return res


def bispectrum_sugiyama_proj_alt(k1, k2, emu, comet_params, ell=[(0, 0, 0), (2, 0, 2)], **kwargs): 
   # This is a version using a coordinate system where the line of sight is along the z-axis.
    n = k1.shape[0]
    nmu1 = kwargs.pop('nmu1', 5) # cos(\omega)
    nmu2 = kwargs.pop('nmu2', 5) # cos(\omega)
    nphi12 = kwargs.pop('nphi12', 5) # \phi

    mu1, w_mu1 = np.polynomial.legendre.leggauss(nmu1)
    mu2, w_mu2 = np.polynomial.legendre.leggauss(nmu2)
    # cphi12, w_cphi12 = np.polynomial.chebyshev.chebgauss(nphi12) # cphi12 = cos(phi12)
    # cphi12 is cos(phi12), we need to tranfrom the weights to be in terms of phi12
    # w_cphi12 = 2 * w_cphi12

    phi12 = np.linspace(0, 2*np.pi, nphi12, endpoint=False) + np.pi / nphi12 # shift by half a bin to avoid phi12 = 0 where the integrand can be singular
    w_phi12 = 2 * np.pi / nphi12
    cphi12 = np.cos(phi12)
    w_cphi12 = w_phi12 * np.ones_like(cphi12)

    mu1 = mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    w_mu1 = w_mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    mu2 = mu2[None, None, :, None] # shape (1, 1, nmu2, 1)
    w_mu2 = w_mu2[None, None, :, None] # shape (1, 1, nmu2, 1)
    cphi12 = cphi12[None, None, None, :] # shape (1, 1, 1, nphi12)
    w_cphi12 = w_cphi12[None, None, None, :] # shape (1, 1, 1, nphi12)

    k1, k2 = k1[:, None, None, None], k2[:, None, None, None] # shape (n, 1, 1, 1)

    mu12 = mu1 * mu2 + np.sqrt(1 - mu1**2) * np.sqrt(1 - mu2**2) * cphi12

    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12) # shape (n, nmu1, nmu2, nphi12)
    bfull = bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, **kwargs) # shape (n, nmu1, nmu2, nphi12) or (n, ..., nz)
    proj_op = get_cached_proj_operator_alt(nmu1, nmu2, nphi12, ell, w_mu1, w_mu2, w_cphi12, mu1, mu2, cphi12)

    is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1
    res = {}
    if is_batched:
        nz = len(comet_params['z'])
        bfull_flat = bfull.reshape(n, -1, nz)
        for ll in ell:
            res[ll] = np.einsum('ijk,j->ik', bfull_flat, proj_op[ll], optimize=einsum_opt)
    else:
        bfull_flat = bfull.reshape(n, -1)
        for ll in ell:
            res[ll] = bfull_flat @ proj_op[ll]

    return res

_PROJ_CACHE = {}
_PROJ_CACHE_ALT = {}

def get_cached_proj_operator(nmu1, nmu12, nphi, ell, w_mu1, w_mu12, w_phi, mu1, mu12, phi, mu12_transform='quartic', cache=True):
    """Fetches or computes the projection operator for a given grid configuration."""
    cache_key = (nmu1, nmu12, nphi, tuple(ell), mu12_transform)
    
    if cache_key in _PROJ_CACHE and cache:
        return _PROJ_CACHE[cache_key]
        
    res_ops = {}
    for ll in ell:
        l1, l2, L = ll
        proj_operator = np.zeros((1, nmu1, nmu12, nphi), dtype=complex) # Match broadcast shape
        h = float(wigner_3j(l1, l2, L, 0, 0, 0).evalf())
        if h == 0:
            continue

        for M in range(-L, L+1):
            w3j = float(wigner_3j(l1, l2, L, 0, -M, M).evalf()) # Ensure it's a float, not a sympy Rational
            if w3j == 0:
                continue
            y1 = sph_harm(l2, -M, mu12, 0)
            y2 = sph_harm(L, M, mu1, -phi)
            proj_operator = proj_operator + w3j * y1 * y2
            
        # Apply integration weights here to save operations later
        proj_operator = (proj_operator * w_mu1 * w_mu12 * w_phi).squeeze()
        
        # Flatten the operator for faster dot products later
        prefactor = h *(2*l1 + 1) * (2*l2 + 1) * (2*L + 1) / (8 * np.pi)
        res_ops[ll] = np.real(proj_operator.ravel() * prefactor)
        
    _PROJ_CACHE[cache_key] = res_ops
    return res_ops


def get_cached_proj_operator_alt(nmu1, nmu2, nphi12, ell, w_mu1, w_mu2, w_phi12, mu1, mu2, phi12, cache=True):
    """Projection operator for the alternative coordinate system."""
    cache_key = (nmu1, nmu2, nphi12, tuple(ell))
    
    if cache_key in _PROJ_CACHE_ALT and cache:
        return _PROJ_CACHE_ALT[cache_key]
        
    res_ops = {}
    for ll in ell:
        l1, l2, L = ll
        proj_operator = np.zeros((1, nmu1, nmu2, nphi12), dtype=complex) # Match broadcast shape
        h = float(wigner_3j(l1, l2, L, 0, 0, 0).evalf())
        if h == 0:
            continue

        for M in range(-L, L+1):
            w3j = float(wigner_3j(l1, l2, L, M, -M, 0).evalf()) # Ensure it's a float
            if w3j == 0:
                continue
            y1 = sph_harm(l1, M, mu1, 0)
            y2 = sph_harm(l2, -M, mu2, -phi12)
            proj_operator = proj_operator + w3j * y1 * y2
            
        # Apply integration weights here to save operations later
        proj_operator = (proj_operator * w_mu1 * w_mu2 * w_phi12).squeeze()
        
        # Flatten the operator for faster dot products later
        prefactor = h *(2*l1 + 1) * (2*l2 + 1) * (2*L + 1) / (8 * np.pi)
        res_ops[ll] = np.real(proj_operator.ravel() * prefactor)
        
    _PROJ_CACHE_ALT[cache_key] = res_ops
    return res_ops

def bX_5d(k1, k2, k3, mu1, mu2, emu, comet_params, X_list, use_pdw_interp=False, **kwargs):
    if not isinstance(X_list, (list, tuple)):
        X_list = [X_list]
    # only supporting NP0, NB0 and MB0\
    params = emu.params
    nbar = emu.nbar
    b1, f, avir, sv = params['b1'], params['f'], params['avir'], params['sv']
    qpar, qperp = params['q_lo'], params['q_tr']
    qiso6 = qpar**2 * qperp**4

    if len(X_list) == 1 and X_list[0] == 'B_NB0':
        bstoch = np.ones_like(k1) * np.ones_like(mu1) * np.ones_like(mu2)
        if isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1:
            bstoch = bstoch[..., None]
        btosch = bstoch / (qiso6 * nbar**2)
 
        return btosch
    
    # for diagram in diagrams:
    # if diagram == 'B_NP0':
    #     NP0, MB0, NB0 = 1, 0, 0
    # elif diagram == 'B_MB0':
    #     NP0, MB0, NB0 = 0, 1, 0
    # elif diagram == 'B_NB0':
    #     bstoch = np.ones_like(k1) * np.ones_like(mu1) * np.ones_like(mu2)
    #     if isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1:
    #         bstoch = bstoch[..., None]
    #     btosch = bstoch / (qiso6 * nbar**2)
    #     return btosch
         
    is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1
    if is_batched:
        k1, k2, k3 = k1[..., None], k2[..., None], k3[..., None]
        mu1, mu2 = mu1[..., None], mu2[..., None]

    mu3 = np.where(k3 > 0, - (mu1 * k1 + mu2 * k2) / k3, -1.0)
    k1_p, mu1_p = apply_ap(k1, mu1, qpar, qperp) 
    k2_p, mu2_p = apply_ap(k2, mu2, qpar, qperp)
    k3_p, mu3_p = apply_ap(k3, mu3, qpar, qperp)
    # k1, k2, k3 are either arrays of any shape or floats

    if is_batched:
        nz = len(comet_params['z'])
        kmin = min(k1_p.min(), k2_p.min(), k3_p.min())
        kmax = max(k1_p.max(), k2_p.max(), k3_p.max())

        kgrid = get_kvec_compression(kmin * 0.99, kmax * 1.01, 100)
        pdw_grid = emu.Pdw(kgrid, comet_params, mu=0.6, **kwargs)  # shape (100, nz)

        pdw1 = np.empty_like(k1_p)
        pdw2 = np.empty_like(k2_p)
        pdw3 = np.empty_like(k3_p)

        for i in range(nz):
            cs = CubicSpline(kgrid, pdw_grid[:, i])
            pdw1[..., i] = cs(k1_p[..., i])
            pdw2[..., i] = cs(k2_p[..., i])
            pdw3[..., i] = cs(k3_p[..., i])

    elif not use_pdw_interp:
        k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
        pdw_all = emu.Pdw(k_all, comet_params, mu=0.6, **kwargs)

        n1, n2 = k1_p.size, k2_p.size
        pdw1 = pdw_all[:n1].reshape(k1_p.shape)
        pdw2 = pdw_all[n1:n1+n2].reshape(k2_p.shape)
        pdw3 = pdw_all[n1+n2:].reshape(k3_p.shape)
    else:
        k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
        kmin, kmax = np.min(k_all), np.max(k_all)
        kgrid = get_kvec_compression(kmin * 0.99, kmax * 1.01, 100)
        pdw_grid = emu.Pdw(kgrid, comet_params, mu=0.6, **kwargs)

        spline = make_interp_spline(kgrid, pdw_grid, axis=0, k=3)
        pdw1 = spline(k1_p)
        pdw2 = spline(k2_p)
        pdw3 = spline(k3_p)

    # Cache squares
    mu1_sq, mu2_sq, mu3_sq = mu1_p**2, mu2_p**2, mu3_p**2
    Z1_1 = kernel_Z1(mu1_p, b1, f, mu_sq=mu1_sq)
    Z1_2 = kernel_Z1(mu2_p, b1, f, mu_sq=mu2_sq)
    Z1_3 = kernel_Z1(mu3_p, b1, f, mu_sq=mu3_sq)

    kxmu1_sq = (k1_p**2) * mu1_sq
    kxmu2_sq = (k2_p**2) * mu2_sq
    kxmu3_sq = (k3_p**2) * mu3_sq

    res = []
    for X in X_list:
        if X == 'B_NP0':
            NP0, MB0, NB0 = 1, 0, 0
        elif X == 'B_MB0':
            NP0, MB0, NB0 = 0, 1, 0
        elif X == 'B_NB0':
            bstoch = np.ones_like(k1) * np.ones_like(mu1) * np.ones_like(mu2)
            if is_batched:
                bstoch = bstoch[..., None]
            btosch = bstoch / (qiso6 * nbar**2)
            res.append(btosch)
            continue

        bstoch = stoch_term(kxmu1_sq, mu1_sq, b1, f, avir, sv, MB0, NP0, Z1_1)* pdw1 + \
                 stoch_term(kxmu2_sq, mu2_sq, b1, f, avir, sv, MB0, NP0, Z1_2)* pdw2 + \
                 stoch_term(kxmu3_sq, mu3_sq, b1, f, avir, sv, MB0, NP0, Z1_3)* pdw3
        bstoch *= (1.0 / nbar)
        bstoch /= qiso6
        res.append(bstoch)
    return np.stack(res, axis=-1) if len(res) > 1 else res[0]

def bX_ell_scoccimarro(k1, k2, k3, emu, comet_params, ell, X_list, **kwargs):
    nmu, nphi = kwargs.pop('nmu', 20), kwargs.pop('nphi', 20)
    mu, w_mu = np.polynomial.legendre.leggauss(nmu)
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    mu1 = mu[None, :, None] # shape (1, nmu, 1)
    phi = phi[None, None, :] # shape (1, 1, nphi)
    k1, k2, k3 = k1[:, None, None], k2[:, None, None], k3[:, None, None] # shape (ntri, 1, 1)
    mu12 = get_dot_cosine(k1, k2, k3)
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12)
    mu12 = np.clip(mu12, -1, 1)
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * np.cos(phi)
    bfull = bX_5d(k1, k2, k3, mu1, mu2, emu, comet_params, X_list, **kwargs) # shape (ntri, nmu, nphi)
    
    is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1
    is_single_nb0 = (
        (isinstance(X_list, (list, tuple, np.ndarray)) and len(X_list) == 1 and X_list[0] == 'B_NB0')
        or (not isinstance(X_list, (list, tuple, np.ndarray)) and X_list == 'B_NB0')
    )
    has_x_axis = bfull.ndim == (5 if is_batched else 4)
    
    if is_single_nb0:
        if is_batched:
            b0 = np.ones((k1.shape[0], len(comet_params['z']))) * bfull[0, 0, 0, :]
        else:
            b0 = np.ones(k1.shape[0]) * bfull[0, 0, 0]
        res = {ll: b0 if ll == (0, 0) else np.zeros_like(b0) for ll in ell}
        return res
        
    n = k1.shape[0]
    if is_batched:
        nz = len(comet_params['z'])
        if has_x_axis:
            nx = bfull.shape[-1]
            bfull_flat = bfull.reshape(n, -1, nz, nx)
        else:
            bfull_flat = bfull.reshape(n, -1, nz)
    else:
        if has_x_axis:
            nx = bfull.shape[-1]
            bfull_flat = bfull.reshape(n, -1, nx)
        else:
            bfull_flat = bfull.reshape(n, -1)

    res = {}
    for ll in ell:
        # lell = legendre(ll)(mu1) # shape (1, nmu, 1)
        l, m = int(ll[0]), int(ll[1])
        ylm = sph_harm(l, m, mu1, phi) # shape (1, nmu, nphi)
        weights = w_mu[None, :, None] * w_phi
        proj_op = (ylm * weights).ravel()
            
        if is_batched:
            if has_x_axis:
                integral = np.einsum('ijkx,j->ikx', bfull_flat, proj_op, optimize=einsum_opt)
            else:
                integral = np.einsum('ijk,j->ik', bfull_flat, proj_op, optimize=einsum_opt)
        else:
            if has_x_axis:
                integral = np.einsum('ijx,j->ix', bfull_flat, proj_op, optimize=einsum_opt)
            else:
                integral = np.dot(bfull_flat, proj_op)
            
        bell = (2*l + 1) * integral / (4 * np.pi)
        res[ll] = bell
    return res


def bX_ell_sugiyama(k1, k2, emu, comet_params, ell, X_list, use_pdw_interp=False, 
                    interpolate_k1k2=False, **kwargs):
    n = k1.shape[0]
    nmu1 = kwargs.pop('nmu1', 5) # cos(\omega)
    nmu12 = kwargs.pop('nmu12', 12) # cos(\theta_{12})
    nphi = kwargs.pop('nphi', 5) # \phi
    k1k2_interp_method = kwargs.pop('k1k2_interp_method', 'cubic')
    k1k2_interp_grid_size = kwargs.pop('k1k2_interp_grid_size', None)
    k1k2_interp_adaptive = kwargs.pop('k1k2_interp_adaptive', True)
    k1k2_interp_scale = kwargs.pop('k1k2_interp_scale', 'log')
    mu12_transform = kwargs.pop('mu12_transform', 'quadratic')

    mu1, w_mu1 = np.polynomial.legendre.leggauss(nmu1)
    
    # Change of variables for mu12 to resolve the k3 ~ 0 singularity when k1 ~ k2
    x_mu12, w_x_mu12 = np.polynomial.legendre.leggauss(nmu12)
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    cphi = np.cos(phi)
    w_cphi = w_phi * np.ones_like(cphi) 

    if mu12_transform == 'linear':
        mu12 = x_mu12
        w_mu12 = w_x_mu12
    elif mu12_transform == 'quadratic':
        mu12 = 0.5 * (x_mu12 + 1)**2 - 1.0
        w_mu12 = w_x_mu12 * (x_mu12 + 1)
    elif mu12_transform == 'quartic':
        mu12 = 0.125 * (x_mu12 + 1)**4 - 1.0
        w_mu12 = w_x_mu12 * 0.5 * (x_mu12 + 1)**3


    mu1 = mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    w_mu1 = w_mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    mu12 = mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    w_mu12 = w_mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    cphi = cphi[None, None, None, :] # shape (1, 1, 1, nphi)
    w_cphi = w_cphi[None, None, None, :] # shape (1, 1, 1, nphi)
    phi = phi[None, None, None, :] # shape (1, 1, 1, nphi)

    if not interpolate_k1k2:
        k1, k2 = k1[:, None, None, None], k2[:, None, None, None] # shape (n, 1, 1, 1)
    else:
        k1_old, k2_old = k1.copy(), k2.copy()
        n_input = k1_old.shape[0]
        if k1k2_interp_grid_size is not None:
            interp_grid_size = max(4, int(k1k2_interp_grid_size))
        elif k1k2_interp_adaptive:
            # For large n this keeps interpolation accurate while avoiding oversized grids.
            interp_grid_size = int(np.clip(np.sqrt(n_input), 20, 40))
        else:
            interp_grid_size = 30

        if k1k2_interp_scale == 'log':
            k1_grid = np.logspace(np.log10(k1_old.min()*0.99), np.log10(k1_old.max()*1.1), interp_grid_size, endpoint=True)
            k2_grid = np.logspace(np.log10(k2_old.min()*0.99), np.log10(k2_old.max()*1.1), interp_grid_size, endpoint=True)
        elif k1k2_interp_scale == 'linear':
            k1_grid = np.linspace(k1_old.min()*0.99, k1_old.max()*1.1, interp_grid_size, endpoint=True)
            k2_grid = np.linspace(k2_old.min()*0.99, k2_old.max()*1.1, interp_grid_size, endpoint=True)
        elif k1k2_interp_scale == 'hybrid':
            # Logarithmic spacing at low k and linear spacing at high k
            kthresh = 0.02
            log_size = interp_grid_size // 4
            lin_size = interp_grid_size - log_size
            k1_grid_log = np.logspace(np.log10(k1_old.min()*0.99), np.log10(kthresh*0.99), log_size, endpoint=True)
            k1_grid_lin = np.linspace(kthresh*1.05, k1_old.max()*1.1, lin_size, endpoint=True)
            k1_grid = np.concatenate([k1_grid_log, k1_grid_lin])

            k2_grid_log = np.logspace(np.log10(k2_old.min()*0.99), np.log10(kthresh*0.99), log_size, endpoint=True)
            k2_grid_lin = np.linspace(kthresh*1.05, k2_old.max()*1.1, lin_size, endpoint=True)
            k2_grid = np.concatenate([k2_grid_log, k2_grid_lin])
        elif k1k2_interp_scale == 'comet':
            k1_grid = get_kvec_compression(np.min(k1_old), np.max(k1_old), interp_grid_size)
            k2_grid = get_kvec_compression(np.min(k2_old), np.max(k2_old), interp_grid_size)
        else:
            raise ValueError(f"Unsupported k1k2_interp_scale: {k1k2_interp_scale}")

        interp_points = np.column_stack((k1_old, k2_old))

        k1 = k1_grid
        k2 = k2_grid
        k1, k2 = np.meshgrid(k1, k2, indexing='ij') # shape (35, 35)
        k1 = k1.flatten()[:, None, None, None] # shape (n, 1, 1, 1)
        k2 = k2.flatten()[:, None, None, None] # shape (n, 1, 1, 1)
        n = k1.shape[0]
    
    # get k3 using the triangle condition
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12) # shape (n, 1, nmu12, 1)
    # get mu2 using the Scoccimarro coordinate system
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * cphi # shape (n, nmu1, nmu12, nphi)
    
    bfull = bX_5d(k1, k2, k3, mu1, mu2, emu, comet_params, X_list=X_list,
                  use_pdw_interp=use_pdw_interp, **kwargs) # shape (n, nmu1, nmu12, nphi)
                  
    is_batched = isinstance(comet_params.get('z'), (list, np.ndarray)) and len(comet_params['z']) > 1
    is_single_nb0 = (
        (isinstance(X_list, (list, tuple, np.ndarray)) and len(X_list) == 1 and X_list[0] == 'B_NB0')
        or (not isinstance(X_list, (list, tuple, np.ndarray)) and X_list == 'B_NB0')
    )
    has_x_axis = bfull.ndim == (6 if is_batched else 5)
    x_labels = list(X_list) if isinstance(X_list, (list, tuple, np.ndarray)) else [X_list]
    nb0_idx = np.array([i for i, x in enumerate(x_labels) if x == 'B_NB0'], dtype=int)
    
    if is_single_nb0:
        n_out = n_input if interpolate_k1k2 else k1.shape[0]
        if is_batched:
            b0 = np.ones((n_out, len(comet_params['z']))) * bfull[0, 0, 0, 0, :]
        else:
            b0 = np.ones(n_out) * bfull[0, 0, 0, 0]
        res = {ll: b0 if ll == (0, 0, 0) else np.zeros_like(b0) for ll in ell}
        return res

    proj_ops = get_cached_proj_operator(nmu1=nmu1,
                                       nmu12=nmu12,
                                       nphi=nphi,
                                       ell=ell,
                                       w_mu1=w_mu1,
                                       w_mu12=w_mu12,
                                       w_phi=w_phi,
                                       mu1=mu1,
                                       mu12=mu12,
                                       phi=phi,
                                       mu12_transform=mu12_transform)
                                       
    res = {}
    if is_batched:
        nz = len(comet_params['z'])

    if interpolate_k1k2:
        deg = {'linear': 1, 'quadratic': 2, 'cubic': 3, 'quintic': 5}[k1k2_interp_method]
        ng = interp_grid_size
        W1 = make_interp_spline(k1_grid, np.eye(ng), k=deg)(interp_points[:, 0])  # (n_pts, ng)
        W2 = make_interp_spline(k2_grid, np.eye(ng), k=deg)(interp_points[:, 1])  # (n_pts, ng)

    if is_batched:
        if has_x_axis:
            nx = bfull.shape[-1]
            bfull_flat = bfull.reshape(n, -1, nz, nx)
            use_nb0_fastpath = nb0_idx.size > 0
            if use_nb0_fastpath:
                non_nb0_idx = np.array([i for i in range(nx) if i not in set(nb0_idx.tolist())], dtype=int)
                nb0_monopole = bfull[0, 0, 0, 0, :, nb0_idx]
        else:
            bfull_flat = bfull.reshape(n, -1, nz)
            use_nb0_fastpath = False
        for ll in ell:
            if use_nb0_fastpath:
                if non_nb0_idx.size > 0:
                    res_non = _contract_proj(bfull_flat[..., non_nb0_idx], proj_ops[ll])
                    if interpolate_k1k2:
                        vals_non = res_non.reshape(ng, ng, nz, non_nb0_idx.size)
                        res_non = _interp_separable_2d(vals_non, W1, W2, ng)
                    n_out = res_non.shape[0]
                    res_ll = np.zeros((n_out, nz, nx), dtype=res_non.dtype)
                    res_ll[..., non_nb0_idx] = res_non
                else:
                    n_out = interp_points.shape[0] if interpolate_k1k2 else n
                    res_ll = np.zeros((n_out, nz, nx), dtype=bfull.dtype)

                if ll == (0, 0, 0):
                    res_ll[..., nb0_idx] = nb0_monopole
            else:
                res_ll = _contract_proj(bfull_flat, proj_ops[ll])
                if interpolate_k1k2:
                    if has_x_axis:
                        vals = res_ll.reshape(ng, ng, nz, nx)
                    else:
                        vals = res_ll.reshape(ng, ng, nz)
                    res_ll = _interp_separable_2d(vals, W1, W2, ng)
            res[ll] = res_ll
    else:
        if has_x_axis:
            nx = bfull.shape[-1]
            bfull_flat = bfull.reshape(n, -1, nx)
            use_nb0_fastpath = nb0_idx.size > 0
            if use_nb0_fastpath:
                non_nb0_idx = np.array([i for i in range(nx) if i not in set(nb0_idx.tolist())], dtype=int)
                nb0_monopole = bfull[0, 0, 0, 0, nb0_idx]
        else:
            bfull_flat = bfull.reshape(n, -1)
            use_nb0_fastpath = False
        for ll in ell:
            if use_nb0_fastpath:
                if non_nb0_idx.size > 0:
                    res_non = _contract_proj(bfull_flat[..., non_nb0_idx], proj_ops[ll])
                    if interpolate_k1k2:
                        vals_non = res_non.reshape(ng, ng, non_nb0_idx.size)
                        res_non = _interp_separable_2d(vals_non, W1, W2, ng)
                    n_out = res_non.shape[0]
                    res_ll = np.zeros((n_out, nx), dtype=res_non.dtype)
                    res_ll[..., non_nb0_idx] = res_non
                else:
                    n_out = interp_points.shape[0] if interpolate_k1k2 else n
                    res_ll = np.zeros((n_out, nx), dtype=bfull.dtype)

                if ll == (0, 0, 0):
                    res_ll[..., nb0_idx] = nb0_monopole
            else:
                res_ll = _contract_proj(bfull_flat, proj_ops[ll])
                if interpolate_k1k2:
                    if has_x_axis:
                        vals = res_ll.reshape(ng, ng, nx)
                    else:
                        vals = res_ll.reshape(ng, ng)
                    res_ll = _interp_separable_2d(vals, W1, W2, ng)
            res[ll] = res_ll

    return res


def kernel_Z1(mu, b1, f, mu_sq=None):
    if mu_sq is None:
        mu_sq = mu**2
    return b1 + f * mu_sq

def get_dot_cosine(k1, k2, k3):
    """Calculate (k1 . k2) / (|k1| |k2|) using the triangle condition"""
    return (k3**2 - k1**2 - k2**2) / (2 * k1 * k2)


def kernel_G2(k1, k2, mu12):
    return 3./7. + 4./7. * mu12**2 + 0.5 * (k1/k2 + k2/k1) * mu12


def kernel_F2(k1, k2, mu12):
    return 5./7. + 2./7. * mu12**2 + 0.5 * (k1/k2 + k2/k1) * mu12


def kernel_K(mu12):
    return mu12**2 -1.


def kernel_K2(k1, k2, mu12, b1, b2, g2):
    return b1 * kernel_F2(k1, k2, mu12) + b2/2. + g2 * kernel_K(mu12)


def kernel_Z2(k1, k2, mu1, mu2, mu12, k, mu, b1, b2, g2, f, Z1_1=None, Z1_2=None):
    # k = k3 = k1^2 + k2^2 - 2 k1 k2 mu12
    # mu = (k1 mu1 + k2 mu2) / k = -mu3
    if Z1_1 is None:
        Z1_1 = kernel_Z1(mu1, b1, f)
    if Z1_2 is None:
        Z1_2 = kernel_Z1(mu2, b1, f)
        
    return kernel_K2(k1, k2, mu12, b1, b2, g2) + \
           f * mu**2 * kernel_G2(k1, k2, mu12) + \
           0.5 * f * k * mu * ((mu1/k1) * Z1_2 + (mu2/k2) * Z1_1)


def w_B_infty(lamb2, avir, sv):
    return 1./(1 - lamb2 * avir**2)**(3./2.) * \
          np.exp(lamb2 * sv**2/(1 - lamb2 * avir**2))

def w_12_0_0():
    return 1.0

def w_12_infty_0(lamb2, avir, sv):
    return 1./(1 - lamb2 * avir**2)**(3./2.) * \
            np.exp(lamb2 * sv**2/(1 - lamb2 * avir**2)) 

def apply_ap(k, mu, qpar, qperp):
    # calculate real coordinates
    F = qpar / qperp
    k_p = k / qperp * np.sqrt(1 + mu**2 * (1/F**2 - 1))
    mu_p = mu / F / np.sqrt(1 + mu**2 * (1/F**2 - 1))
    return k_p, mu_p

def sph_harm(l, m, costheta, phi, normalized=False):
    norm = np.sqrt(factorial(l - abs(m)) / factorial(l + abs(m)))
    norm = norm * (-1)**(0.5 * (m - abs(m)))
    if normalized:
        norm = norm * np.sqrt((2*l + 1)/(4*np.pi))
    return norm * lpmv(abs(m), l, costheta) * np.exp(1j * m * phi) 

def sph_harm_real(l, m, costheta, phi, normalized=False):
    norm = np.sqrt(factorial(l - abs(m)) / factorial(l + abs(m)))
    norm = norm * (-1)**(0.5 * (m - abs(m)))
    if normalized:
        norm = norm * np.sqrt((2*l + 1)/(4*np.pi))
    if m > 0:
        return np.sqrt(2) * norm * lpmv(m, l, costheta) * np.cos(m * phi)
    elif m < 0:
        return np.sqrt(2) * norm * lpmv(-m, l, costheta) * np.sin(-m * phi)
    else:
        return norm * lpmv(0, l, costheta)
    

def get_kvec_compression(kmin, kmax, nk=100):
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