import numpy as np

class Observable:
    """Base class for observables"""
    def __init__(self, x, y, cov=None, nbar=None, cosmo_fid=None, wmat=None, xwin=None,
                 xmin=None, xmax=None, xwinmin=None, xwinmax=None, nmocks_cov=None):
        self.x = x.copy()
        self.y = y.copy()
        if not isinstance(self.y, list):
            self.y = [self.y]
        if not isinstance(self.x, list) or len(self.x) == 1:
            self.x = [self.x] * len(self.y)
        self.cosmo_fid = cosmo_fid
        self.h_fid = cosmo_fid['h'] if cosmo_fid is not None else None
        self.cov = cov.copy() if cov is not None else None
        self.n_obs = len(self.y)
        self.nbar = nbar
        self.nmocks_cov = nmocks_cov
        self.wmat = wmat
        self.xwin = xwin
        if self.xwin is not None and not isinstance(self.xwin, list):
            self.xwin = [self.xwin] * self.n_obs
        self.nobswin = None
        if wmat is not None:
            assert self.xwin is not None, "xwin must be provided if wmat is provided"
            if isinstance(self.xwin, float):
                self.nobswin = self.n_obs
                self.xwin = [self.xwin] * self.nobswin
            else:
                self.nobswin = len(self.xwin)
            assert self.wmat.shape == (self.n_data, np.sum([len(xwini) for xwini in self.xwin])), "wmat has wrong shape"

        if xmin is not None or xmax is not None:
            self._cut_scales(xmin, xmax, xwinmin, xwinmax)

    def _cut_scales(self, xmin, xmax, xwinmin=None, xwinmax=None):
        if isinstance(xmin, float):
            xmin = [xmin] * self.n_obs
        if isinstance(xmax, float):
            xmax = [xmax] * self.n_obs
        if xmin is None:
            xmin = [-np.inf] * self.n_obs
        if xmax is None:
            xmax = [np.inf] * self.n_obs
        masks = []
        for i in range(self.n_obs):
            # depending on the shape of x[i]
            if self.x[i].ndim == 1:
                mask = (self.x[i] >= xmin[i]) & (self.x[i] <= xmax[i])
            else:
                mask = np.all((self.x[i] >= xmin[i]) & (self.x[i] <= xmax[i]), axis=1)
            self.x[i] = self.x[i][mask]
            self.y[i] = self.y[i][mask]
            masks.append(mask)
        cov_mask = np.concatenate(masks)
        if self.cov is not None:
            self.cov = self.cov[np.ix_(cov_mask, cov_mask)]
        if self.wmat is not None:
            self.wmat = self.wmat[cov_mask, :] # shape (n_data_cut, n_win)
        if xwinmin is not None or xwinmax is not None:
            if isinstance(xwinmin, float):
                xwinmin = [xwinmin] * self.nobswin
            if isinstance(xwinmax, float):
                xwinmax = [xwinmax] * self.nobswin
            if xwinmin is None:
                xwinmin = [-np.inf] * self.nobswin
            if xwinmax is None:
                xwinmax = [np.inf] * self.nobswin
            win_masks = []
            for i in range(self.nobswin):
                if self.xwin[i].ndim == 1:
                    win_mask = (self.xwin[i] >= xwinmin[i]) & (self.xwin[i] <= xwinmax[i])
                else:
                    win_mask = np.all((self.xwin[i] >= xwinmin[i]) & (self.xwin[i] <= xwinmax[i]), axis=1)
                self.xwin[i] = self.xwin[i][win_mask]
                win_masks.append(win_mask)
            wmat_mask = np.concatenate(win_masks)
            self.wmat = self.wmat[:, wmat_mask]
        
    def get_flatten(self):
        y = np.concatenate(self.y)
        return y
    
    @property
    def n_data(self):
        return sum(len(yi) for yi in self.y)

class PowerSpectrumMultipoles(Observable):
    """Power spectrum multipoles"""
    def __init__(self, k, Pell, cov=None, nbar=None, cosmo_fid=None, Mpc_units=False, kmin=None, kmax=None,
                 wmat=None, kwin=None, kwinmin=None, kwinmax=None, nmocks_cov=None):
        super().__init__(k, Pell, cov, nbar, cosmo_fid, xmin=kmin, xmax=kmax, nmocks_cov=nmocks_cov,
                         wmat=wmat, xwin=kwin, xwinmin=kwinmin, xwinmax=kwinmax)
        # Internatlly, everything is done in Mpc units
        if not Mpc_units:
            assert getattr(self, 'h_fid') is not None, "h value is required to convert to Mpc units"
            self.x = [xi * self.h_fid for xi in self.x]
            self.y = [yi / self.h_fid**3 for yi in self.y]
            self.nbar = self.nbar * self.h_fid**3 if self.nbar is not None else None
            if cov is not None:
                self.cov = self.cov / self.h_fid**6
            if self.xwin is not None:
                self.xwin = [xwini * self.h_fid for xwini in self.xwin]
        self.k = self.x
        self.Pell = self.y
        self.kwin = self.xwin
        self.ell = [2*i for i in range(self.n_obs)]
        self.ellwin = [2*i for i in range(self.nobswin)] if self.nobswin is not None else None

        # # for the moment, store unformated data to use for comet AM chi2 function
        # self._k = k * self.h_fid if not Mpc_units else k
        # self._Pell = np.array(Pell) / self.h_fid**3 if not Mpc_units else np.array(Pell)
        # self._cov = cov / self.h_fid**6 if cov is not None and not Mpc_units else cov
        # self._kmax = kmax
        # if self._kmax is not None and not Mpc_units:
        #     if isinstance(self._kmax, list):
        #         self._kmax = [km * self.h_fid for km in self._kmax] 
        #     else:
        #         self._kmax = self._kmax * self.h_fid
    
    def plot(self, ax=None, h_units=False,**kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        factor = 1.0
        if h_units and getattr(self, 'h_fid') is not None:
            factor = self.h_fid
            ax.set_xlabel(r'$k ~ [h ~ \mathrm{Mpc}^{-1}]$')
            ax.set_ylabel(r'$k P_\ell(k) ~ [h^{-2} ~ \mathrm{Mpc}^2]$')
        else:
            ax.set_xlabel(r'$k ~ [\mathrm{Mpc}^{-1}]$')
            ax.set_ylabel(r'$k P_\ell(k) ~ [\mathrm{Mpc}^2]$')
        for i in range(self.n_obs):
            err = np.sqrt(np.diag(self.cov))[sum(len(yi) for yi in self.y[:i]):sum(len(yi) for yi in self.y[:i+1])]
            ax.errorbar(self.x[i]/factor, self.x[i] * self.y[i] * factor**2, yerr=self.x[i] * err * factor**2, label=fr'$\ell = {{{2*i}}}$', fmt='o', **kwargs)
        
        ax.legend()
        return ax

class BispectrumScoccimarroMultipoles(Observable):
    def __init__(self, tri, Bell, cov=None, nbar=None, cosmo_fid=None, Mpc_units=False, kmin=None, kmax=None, nmocks_cov=None):
        super().__init__(tri, Bell, cov, nbar, cosmo_fid, xmin=kmin, xmax=kmax, nmocks_cov=nmocks_cov)
        if not Mpc_units:
            assert getattr(self, 'h_fid') is not None, "h value is required to convert to Mpc units"
            self.x = [xi * self.h_fid for xi in self.x]
            self.y = [yi / self.h_fid**6 for yi in self.y]
            self.nbar = self.nbar * self.h_fid**3 if self.nbar is not None else None
            if cov is not None:
                self.cov = self.cov / self.h_fid**12
        self.tri = self.x
        self.Bell = self.y
        self.ell = [2*i for i in range(self.n_obs)]


    def plot(self, ax=None, h_units=False, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        factor = 1.0
        if h_units and getattr(self, 'h_fid') is not None:
            factor = self.h_fid
            ax.set_xlabel(r'Triangle index')
            ax.set_ylabel(r'$B_\ell(k_1, k_2, k_3) ~ [h^{-6} ~ \mathrm{Mpc}^6]$')
        else:
            ax.set_xlabel(r'Triangle index')
            ax.set_ylabel(r'$B_\ell(k_1, k_2, k_3) ~ [\mathrm{Mpc}^6]$')
        for i in range(self.n_obs):
            err = np.sqrt(np.diag(self.cov))[sum(len(yi) for yi in self.y[:i]):sum(len(yi) for yi in self.y[:i+1])]
            tindex = np.arange(len(self.y[i]))
            ax.errorbar(tindex, self.y[i] * factor**6, yerr=err * factor**6, label=fr'$\ell = {{{2*i}}}$', fmt='o', **kwargs)
        
        ax.legend()
        return ax
        

class JointObservable(Observable):
    def __init__(self, obs1, obs2, cov=None, cov_Mpc_units=False):
        assert obs1.cosmo_fid == obs2.cosmo_fid, "Observables must have the same fiducial cosmology"
        self.obs1 = obs1
        self.obs2 = obs2
        x = obs1.x + obs2.x
        y = obs1.y + obs2.y
        self.h_fid = obs1.h_fid
        hpower_dict = {'PowerSpectrumMultipoles': 3.0, 'BispectrumScoccimarroMultipoles': 6.0}

        if cov is None:
            print("No covariance matrix provided for joint observable, constructing block diagonal covariance matrix")
            cov = self.get_block_cov(obs1.cov, obs2.cov)
        else:
            assert cov.shape == (obs1.n_data + obs2.n_data, obs1.n_data + obs2.n_data), "Covariance matrix has wrong shape" 
            if not cov_Mpc_units:
                hpower1 = hpower_dict.get(type(obs1).__name__, None)
                hpower2 = hpower_dict.get(type(obs2).__name__, None)
                hfact1 = self.h_fid**hpower1
                hfact2 = self.h_fid**hpower2
                cov[:obs1.n_data, :obs1.n_data] /= hfact1**2
                cov[obs1.n_data:, obs1.n_data:] /= hfact2**2
                cov[:obs1.n_data, obs1.n_data:] /= hfact1 * hfact2
                cov[obs1.n_data:, :obs1.n_data] /= hfact1 * hfact2
                
        super().__init__(x, y, cov=cov, nbar=obs1.nbar, cosmo_fid=obs1.cosmo_fid)

    def get_block_cov(self, cov1, cov2):
        return np.block([[cov1, np.zeros((cov1.shape[0], cov2.shape[1]))], [np.zeros((cov2.shape[0], cov1.shape[1])), cov2]])