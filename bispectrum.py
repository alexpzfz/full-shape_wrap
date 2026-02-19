import numpy as np
from scipy.special import legendre, sph_harm
from sympy.physics.wigner import wigner_3j


def bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, nbar=1.0,**kwargs):
    # k1, k2, k3 are either arrays of any shape or floats
    k_all = np.concatenate([np.ravel(k1), np.ravel(k2), np.ravel(k3)])
    kunique = np.unique(k_all)
    pdw = emu.Pdw(kunique, comet_params, **kwargs)
    params = emu.params
    pdw1 = pdw[np.searchsorted(kunique, k1)]
    pdw2 = pdw[np.searchsorted(kunique, k2)]
    pdw3 = pdw[np.searchsorted(kunique, k3)]

    b1, b2, g2, f = params['b1'], params['b2'], params['g2'], params['f']
    mu3 = - (mu1 * k1 + mu2 * k2) / k3

    # Apply AP effect
    qpar, qperp = params['q_lo'], params['q_tr']
    k1_p, mu1_p = apply_ap(k1, mu1, qpar, qperp) # shape (ntri, m, n)
    k2_p, mu2_p = apply_ap(k2, mu2, qpar, qperp) # shape (ntri, m, n)
    k3_p, mu3_p = apply_ap(k3, mu3, qpar, qperp) #
    
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
    mus, w_mu = np.polynomial.legendre.leggauss(nmu)
    phis = np.linspace(0, 2*np.pi, nphi, endpoint=False)
    w_phi = 2 * np.pi / nphi
    mu, phi = np.meshgrid(mus, phis, indexing='ij')
    mu1_grid = mu 

    k1, k2, k3 = k1[:, None, None], k2[:, None, None], k3[:, None, None] # shape (ntri, 1, 1)
    mu1_grid = mu1_grid[None, :, :] # shape (1, nmu, nphi)
    phi_grid = phi[None, :, :] # shape (1, nmu, nphi)

    mu12 = get_dot_cosine(k1, k2, k3)
    # ensure mu12 is in the range [-1, 1] to avoid numerical issues with sqrt
    mu12 = np.clip(mu12, -1, 1)
    mu2_grid = mu12 * mu1_grid + np.sqrt(1 - mu12**2) * np.sqrt(1 - mu1_grid**2) * np.cos(phi_grid)
    # reshape everything to be (ntri, nmu, nphi)
    bfull = bispectrum_vdg(k1, k2, k3, mu1_grid, mu2_grid, emu, comet_params, nbar=nbar, **kwargs) # shape (ntri, nmu, nphi)
    res = {}
    for ll in ell:
        lell = legendre(ll)(mu1_grid) # shape (1, nmu, nphi)
        integral = np.sum(bfull * lell * w_mu[None, :, None] * w_phi, axis=(1, 2)) # shape (ntri,)
        bell = (2*ll + 1) * integral / (4 * np.pi)
        res[f'ell{ll}'] = bell
    return res

# def bispectrum_sugiyama_proj(k1k2, emu, comet_params, ell=['000'], nbar=1.0, **kwargs):
#     # k1k2 must be of shape (n, 2) where n is the number of triangles, and the two columns are k1 and k2. We will reconstruct k3 using the triangle condition.
#     nmu1 = kwargs.pop('nmu1', 20)
#     nmu2 = kwargs.pop('nmu2', 20)
#     nphi = kwargs.pop('nphi', 20)
#     mu1, w_mu1 = np.polynomial.legendre.leggauss(nmu1)
#     mu2, w_mu2 = np.polynomial.legendre.leggauss(nmu2)
#     phis = np.linspace(0, 2*np.pi, nphi, endpoint=False)
#     w_phi = 2 * np.pi / nphi
#     mu1_grid, mu2_grid, phi_grid = np.meshgrid(mu1, mu2, phis, indexing='ij')
#     mu1_grid = mu1_grid[None, :, :, :] # shape (1, nmu1, nmu2, nphi)
#     mu2_grid = mu2_grid[None, :, :, :] # shape (1, nmu1, nmu2, nphi)
#     phi_grid = phi_grid[None, :, :, :] # shape (1, nmu1, nmu2, nphi)
#     # \vec{k1} = (k1sin(theta1), 0, k1cos(theta1))
#     # \vec{k2} = (k2sin(theta2)cos(phi), k2sin(theta2)sin(phi), k2cos(theta2))
#     k1 = k1k2[:, 0][:, None, None, None] # shape (n, 1, 1, 1)
#     k2 = k1k2[:, 1][:, None, None, None] # shape (n, 1, 1, 1)
#     # reconstruc k3 from the triangle condition
#     # mu12 = dot(k1, k2) / (|k1| |k2|)
#     mu12 = np.sqrt(1 - mu1_grid**2) * np.sqrt(1 - mu2_grid**2) * np.cos(phi_grid) +  mu1_grid * mu2_grid # shape (nmu1, nmu2, nphi) 
#     mu12 = np.clip(mu12, -1, 1) # ensure mu12 is in the range [-1, 1] to avoid numerical issues with sqrt
#     k3 = np.sqrt(k1**2 + k2**2 - 2 * k1 * k2 * mu12)
#     tri = np.zeros((len(k1k2)*nmu1*nmu2*nphi, 3))
#     tri[:, 0] = np.repeat(k1k2[:, 0], nmu1*nmu2*nphi)
#     tri[:, 1] = np.repeat(k1k2[:, 1], nmu1*nmu2*nphi)
#     tri[:, 2] = k3.flatten()

#     #bfull = bispectrum_vdg(tri, mu1_grid.reshape(nmu1*nphi, nmu1*nphi), mu2_grid.reshape, emu, comet_params, nbar=nbar, **kwargs) # shape (n*nmu1*nmu2*nphi, nmu1, nmu2, nphi)
#     #bfull = bispectrum_vdg(tri, mu1_grid, mu2_grid, emu, comet_params, nbar=nbar, **kwargs) # shape (n*nmu1*nmu2*nphi, nmu1, nmu2, nphi)
#     bfull = bfull.reshape(len(k1k2), nmu1*nmu2*nphi, nmu1, nmu2, nphi) # shape (n, nmu1*nmu2*nphi, nmu1, nmu2, nphi)
#     basis = np.zeros_like(bfull, dtype=complex) # shape (ntri, nmu, nphi)
#     res = {}
#     for b in ell:
#         l1, l2, L = map(int, b)
#         for m in range(-min(l1, l2), min(l1, l2)+1):
#             w3j = float(wigner_3j(l1, l2, L, m, -m, 0))
#             if w3j == 0:
#                 continue
#             Y1 = sph_harm(m, l1, phi_grid, np.arccos(mu1_grid)) # shape (1, nmu, nphi)
#             Y2 = sph_harm(-m, l2, phi_grid, np.arccos(mu2_grid)) # shape (1, nmu, nphi)
#             cg_weight = (-1.0)**(l1 - l2) * np.sqrt(2 * L + 1) * w3j
#             basis = basis + cg_weight * Y1 * Y2
#             integrand = bfull * np.conj(basis) * w_mu1[None, :, None] * w_mu2[None, None, :] * w_phi
#             integral = np.sum(integrand, axis=(1, 2)) # shape (ntri,)
#             norm =(2*l1 + 1) * (2*l2 + 1) / (4 * np.pi)
#             bell = norm * integral
#             res[f'{b}'] = bell.real
#     return res


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