import numpy as np
from scipy.special import legendre, factorial, lpmv
from sympy.physics.wigner import wigner_3j

def bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, nbar=1.0, **kwargs):
    params = emu.params
    b1, b2, g2, f = params['b1'], params['b2'], params['g2'], params['f']
    # Apply AP effect
    qpar, qperp = params['q_lo'], params['q_tr']
    qiso6 = qpar**2 * qperp**4
    mu3 = np.where(k3 > 0, - (mu1 * k1 + mu2 * k2) / k3, -1.0)
    k1_p, mu1_p = apply_ap(k1, mu1, qpar, qperp) 
    k2_p, mu2_p = apply_ap(k2, mu2, qpar, qperp)
    k3_p, mu3_p = apply_ap(k3, mu3, qpar, qperp)
    # k1, k2, k3 are either arrays of any shape or floats
    k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
    kunique = np.unique(k_all)
    pdw = emu.Pdw(kunique, comet_params, mu=0.6, **kwargs)
    pdw1 = pdw[np.searchsorted(kunique, k1_p)]
    pdw2 = pdw[np.searchsorted(kunique, k2_p)]
    pdw3 = pdw[np.searchsorted(kunique, k3_p)]
    
    # tree level first
    btree = tree_term(k1_p, k2_p, mu1_p, mu2_p, k3_p, mu3_p, b1, b2, g2, f) * pdw1 * pdw2 + \
            tree_term(k2_p, k3_p, mu2_p, mu3_p, k1_p, mu1_p, b1, b2, g2, f) * pdw2 * pdw3 + \
            tree_term(k3_p, k1_p, mu3_p, mu1_p, k2_p, mu2_p, b1, b2, g2, f) * pdw3 * pdw1
    
    # now the stochastic part
    NB0, MB0, NP0 = params['NB0'], params['MB0'], params['NP0']
    avir, sv = params['avir'], params['sv']

    bstoch = stoch_term(k1_p, mu1_p, b1, f, avir, sv, MB0, NP0) * pdw1 + \
             stoch_term(k2_p, mu2_p, b1, f, avir, sv, MB0, NP0) * pdw2 + \
             stoch_term(k3_p, mu3_p, b1, f, avir, sv, MB0, NP0) * pdw3
    bstoch = bstoch * 1/nbar
    bstoch = bstoch + NB0/nbar**2

    #construct vdg bispectrum
    lambda2 = -0.5 * f**2 * (k1_p**2 * mu1_p**2 + k2_p**2 * mu2_p**2 + k3_p**2 * mu3_p**2)
    winfty = w_B_infty(lambda2, avir, sv)
    bvdg = winfty * btree + bstoch
    bvdg = bvdg / qiso6
    return bvdg



def tree_term(ki, kj, mui, muj, kk, muk, b1, b2, g2, f):
    muij = get_dot_cosine(ki, kj, kk)
    t = 2 * kernel_Z1(mui, b1, f) * kernel_Z1(muj, b1, f)
    t = t * kernel_Z2(ki, kj, mui, muj, muij, kk, -muk, b1, b2, g2, f)
    return t

def stoch_term(ki, mui, b1, f, avir, sv, MB0, NP0):
    t = (b1 * MB0 + f * mui**2 * NP0) * kernel_Z1(mui, b1, f)
    lambda2 = -f**2 * ki**2 * mui**2
    t = t * w_B_infty(lambda2, avir, sv)
    return t

def bispectrum_scoccimarro_proj(k1, k2, k3, emu, comet_params, ell=[0, 2], nbar=1.0, **kwargs):
    nmu = kwargs.pop('nmu', 20)
    nphi = kwargs.pop('nphi', 20)
    mu, w_mu = np.polynomial.legendre.leggauss(nmu)
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    mu1 = mu[None, :, None] # shape (1, nmu, 1)
    phi = phi[None, None, :] # shape (1, 1, nphi)

    k1, k2, k3 = k1[:, None, None], k2[:, None, None], k3[:, None, None] # shape (ntri, 1, 1)

    mu12 = get_dot_cosine(k1, k2, k3)
    # ensure mu12 is in the range [-1, 1] to avoid numerical issues with sqrt
    mu12 = np.clip(mu12, -1, 1)
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * np.cos(phi)
    # reshape everything to be (ntri, nmu, nphi)
    bfull = bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, nbar=nbar, **kwargs) # shape (ntri, nmu, nphi)
    res = {}
    for ll in ell:
        lell = legendre(ll)(mu1) # shape (1, nmu, 1)
        integral = np.sum(bfull * lell * w_mu[None, :, None] * w_phi, axis=(1, 2)) # shape (ntri,)
        bell = (2*ll + 1) * integral / (4 * np.pi)
        res[f'ell{ll}'] = bell
    return res

def bispectrum_sugiyama_proj(k1, k2, emu, comet_params, ell=['000'], nbar=1.0, **kwargs):
    # let's use Scoccimarro coordinate system!!
    n = k1.shape[0]
    nmu1 = kwargs.pop('nmu1', 20) # cos(\omega)
    nmu12 = kwargs.pop('nmu12', 20) # cos(\theta_{12})
    nphi = kwargs.pop('nphi', 20) # \phi
    mu1, w_mu1 = np.polynomial.legendre.leggauss(nmu1)
    mu12, w_mu12 = np.polynomial.legendre.leggauss(nmu12)
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    mu1 = mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    w_mu1 = w_mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    mu12 = mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    w_mu12 = w_mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    phi = phi[None, None, None, :] # shape (1, 1, 1, nphi)

    k1, k2 = k1[:, None, None, None], k2[:, None, None, None] # shape (n, 1, 1, 1)
    
    # get k3 using the triangle condition
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12) # shape (n, 1, nmu12, 1)
    # get mu2 using the Scoccimarro coordinate system
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * np.cos(phi) # shape (n, nmu1, nmu12, nphi)
    
    bfull = bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, nbar=nbar, **kwargs) # shape (n, nmu1, nmu2, nphi)
    proj_ops = get_cached_proj_operator(nmu1, nmu12, nphi, ell, w_mu1, w_mu12, w_phi, mu1, mu12, phi)
    
    # Reshape bfull to (n, nmu1 * nmu12 * nphi) for a blazing fast BLAS matrix-vector product
    bfull_flat = bfull.reshape(n, -1)
    res = {}
    for ll in ell:
        res[f'{ll}'] = bfull_flat @ proj_ops[f'{ll}']

    return res

_PROJ_CACHE = {}

def get_cached_proj_operator(nmu1, nmu12, nphi, ell, w_mu1, w_mu12, w_phi, mu1, mu12, phi, cache=True):
    """Fetches or computes the projection operator for a given grid configuration."""
    cache_key = (nmu1, nmu12, nphi, tuple(ell))
    
    if cache_key in _PROJ_CACHE and cache:
        return _PROJ_CACHE[cache_key]
        
    res_ops = {}
    for ll in ell:
        l1, l2, L = map(int, ll)
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
        res_ops[f'{ll}'] = np.real(proj_operator.ravel() * prefactor)
        
    _PROJ_CACHE[cache_key] = res_ops
    return res_ops

def BX_5d(k1, k2, k3, mu1, mu2, emu, comet_params, diagram, nbar=1.0, **kwargs):
    # only supporting NP0, NB0 and MB0\
    params = emu.params
    b1, f, avir, sv = params['b1'], params['f'], params['avir'], params['sv']
    qpar, qperp = params['q_lo'], params['q_tr']
    qiso6 = qpar**2 * qperp**4
    if diagram == 'B_NP0':
        NP0, MB0, NB0 = 1, 0, 0
    elif diagram == 'B_MB0':
        NP0, MB0, NB0 = 0, 1, 0
    elif diagram == 'B_NB0':
        bstoch = np.ones_like(k1) * np.ones_like(mu1) * np.ones_like(mu2)
        btosch = bstoch / (qiso6 * nbar**2)
        return btosch
     
    mu3 = np.where(k3 > 0, - (mu1 * k1 + mu2 * k2) / k3, -1.0)
    k1_p, mu1_p = apply_ap(k1, mu1, qpar, qperp) 
    k2_p, mu2_p = apply_ap(k2, mu2, qpar, qperp)
    k3_p, mu3_p = apply_ap(k3, mu3, qpar, qperp)
    # k1, k2, k3 are either arrays of any shape or floats
    k_all = np.concatenate([np.ravel(k1_p), np.ravel(k2_p), np.ravel(k3_p)])
    kunique = np.unique(k_all)
    pdw = emu.Pdw(kunique, comet_params, mu=0.6, **kwargs)
    pdw1 = pdw[np.searchsorted(kunique, k1_p)]
    pdw2 = pdw[np.searchsorted(kunique, k2_p)]
    pdw3 = pdw[np.searchsorted(kunique, k3_p)]
    
    bstoch = stoch_term(k1_p, mu1_p, b1, f, avir, sv, MB0, NP0) * pdw1 + \
             stoch_term(k2_p, mu2_p, b1, f, avir, sv, MB0, NP0) * pdw2 + \
             stoch_term(k3_p, mu3_p, b1, f, avir, sv, MB0, NP0) * pdw3
    bstoch = bstoch * 1/nbar
    bstoch = bstoch / qiso6
    return bstoch

def BX_ell_scoccimarro(k1, k2, k3, emu, comet_params, ell, diagram, nbar=1.0, **kwargs):
    nmu, nphi = kwargs.pop('nmu', 20), kwargs.pop('nphi', 20)
    mu, w_mu = np.polynomial.legendre.leggauss(nmu)
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    mu1 = mu[None, :, None] # shape (1, nmu, 1)
    phi = phi[None, None, :] # shape (1, 1, nphi)
    k1, k2, k3 = k1[:, None, None], k2[:, None, None], k3[:, None, None] # shape (ntri, 1, 1)
    mu12 = get_dot_cosine(k1, k2, k3)
    mu12 = np.clip(mu12, -1, 1)
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * np.cos(phi)
    bfull = BX_5d(k1, k2, k3, mu1, mu2, emu, comet_params, diagram, nbar=nbar, **kwargs) # shape (ntri, nmu, nphi)
    if diagram == 'B_NB0':
        b0 = np.ones(k1.shape[0]) * bfull[0, 0, 0]
        res = {f'ell{ll}': b0 if ll == 0 else np.zeros_like(b0) for ll in ell}
        return res
    res = {}
    for ll in ell:
        lell = legendre(ll)(mu1) # shape (1, nmu, 1)
        integral = np.sum(bfull * lell * w_mu[None, :, None] * w_phi, axis=(1, 2)) # shape (ntri,)
        bell = (2*ll + 1) * integral / (4 * np.pi)
        res[f'ell{ll}'] = bell
    return res


def BX_ell_sugiyama(k1, k2, emu, comet_params, ell, diagram, nbar=1.0, **kwargs):
    nmu1, nmu12, nphi = kwargs.pop('nmu1', 20), kwargs.pop('nmu12', 20), kwargs.pop('nphi', 20)
    mu1, w_mu1 = np.polynomial.legendre.leggauss(nmu1)
    mu12, w_mu12 = np.polynomial.legendre.leggauss(nmu12)
    phi = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    mu1 = mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    w_mu1 = w_mu1[None, :, None, None] # shape (1, nmu1, 1, 1)
    mu12 = mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    w_mu12 = w_mu12[None, None, :, None] # shape (1, 1, nmu12, 1)
    phi = phi[None, None, None, :] # shape (1, 1, 1, nphi)

    k1, k2 = k1[:, None, None, None], k2[:, None, None, None] # shape (ntri, 1, 1, 1)
    
    k3 = np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu12) # shape (ntri, 1, nmu12, 1)
    mu2 = mu12 * mu1 + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1**2) * np.cos(phi) # shape (ntri, nmu1, nmu12, nphi)
    
    bfull = BX_5d(k1, k2, k3, mu1, mu2, emu, comet_params, diagram=diagram,
                  nbar=nbar,
                  **kwargs) # shape (ntri, nmu1, nmu12, nphi)
    if diagram == 'B_NB0':
        b0 = np.ones(k1.shape[0]) * bfull[0, 0, 0, 0]
        res = {f'{ll}': b0 if ll == '000' else np.zeros_like(b0) for ll in ell}
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
                                       phi=phi)
    bfull_flat = bfull.reshape(bfull.shape[0], -1)
    res = {}
    for ll in ell:
        res[f'{ll}'] = bfull_flat @ proj_ops[f'{ll}']
    return res


def kernel_Z1(mu, b1, f):
    return b1 + f * mu**2

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

def kernel_Z2(k1, k2, mu1, mu2, mu12, k, mu, b1, b2, g2, f):
    # k = k3 = k1^2 + k2^2 - 2 k1 k2 mu12
    # mu = (k1 mu1 + k2 mu2) / k = -mu3
    return kernel_K2(k1, k2, mu12, b1, b2, g2) + \
           f * mu**2 * kernel_G2(k1, k2, mu12) + \
           0.5 * f * k * mu * (mu1/k1 * kernel_Z1(mu2, b1, f) \
                                + mu2/k2 * kernel_Z1(mu1, b1, f))

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

def sph_harm(l, m, costheta, phi):
    norm = np.sqrt(factorial(l - abs(m)) / factorial(l + abs(m)))
    norm = norm * (-1)**(m - abs(m)/2)
    return norm * lpmv(abs(m), l, costheta) * np.exp(1j * m * phi)