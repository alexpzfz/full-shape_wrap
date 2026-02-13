import numpy as np

class Observable:
    """Base class for observables"""
    def __init__(self, x, y, cov=None, nbar=None, cosmo_fid=None, xmin=None, xmax=None, nmocks_cov=None):
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
        if xmin is not None or xmax is not None:
            self._cut_scales(xmin, xmax)

    def _cut_scales(self, xmin, xmax):
        if isinstance(xmin, float):
            xmin = [xmin] * self.n_obs
        if isinstance(xmax, float):
            xmax = [xmax] * self.n_obs
        masks = []
        for i in range(self.n_obs):
            mask = (self.x[i] >= xmin[i]) & (self.x[i] <= xmax[i])
            self.x[i] = self.x[i][mask]
            self.y[i] = self.y[i][mask]
            masks.append(mask)
        cov_mask = np.concatenate(masks)
        if self.cov is not None:
            self.cov = self.cov[np.ix_(cov_mask, cov_mask)]
         
    def get_flatten(self):
        y = np.concatenate(self.y)
        return y
    
    @property
    def n_data(self):
        return sum(len(yi) for yi in self.y)

class PowerSpectrumMultipoles(Observable):
    """Power spectrum multipoles"""
    def __init__(self, k, Pell, cov=None, nbar=None, cosmo_fid=None, Mpc_units=False, kmin=None, kmax=None, nmocks_cov=None):
        super().__init__(k, Pell, cov, nbar, cosmo_fid, xmin=kmin, xmax=kmax, nmocks_cov=nmocks_cov)
        # Internatlly, everything is done in Mpc units
        if not Mpc_units:
            assert getattr(self, 'h_fid') is not None, "h value is required to convert to Mpc units"
            self.x = [xi * self.h_fid for xi in self.x]
            self.y = [yi / self.h_fid**3 for yi in self.y]
            self.nbar = self.nbar * self.h_fid**3 if self.nbar is not None else None
            if cov is not None:
                self.cov = self.cov / self.h_fid**6
        self.k = self.x
        self.Pell = self.y
        self.ell = [2*i for i in range(self.n_obs)]

        # for the moment, store unformated data to use for comet AM chi2 function
        self._k = k * self.h_fid if not Mpc_units else k
        self._Pell = np.array(Pell) / self.h_fid**3 if not Mpc_units else np.array(Pell)
        self._cov = cov / self.h_fid**6 if cov is not None and not Mpc_units else cov
        self._kmax = kmax
        if self._kmax is not None and not Mpc_units:
            if isinstance(self._kmax, list):
                self._kmax = [km * self.h_fid for km in self._kmax] 
            else:
                self._kmax = self._kmax * self.h_fid
    
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