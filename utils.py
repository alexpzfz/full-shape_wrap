import numpy as np

def cut_cov(cov, x, ell, xmin=-np.inf, xmax=np.inf, ell_select=None):
    if not isinstance(ell, list):
        ell= [ell]
    if not isinstance(x, list):
        x = [x] * len(ell)
    
    assert len(x) == len(ell), "Length of x must match length of ell"
    
    if ell_select is None:
        ell_select = ell
    if not isinstance(ell_select, list):
        ell_select = [ell_select]
    
    if not isinstance(xmin, list):
        xmin = [xmin] * len(ell_select)
    if not isinstance(xmax, list):
        xmax = [xmax] * len(ell_select)
    
    assert len(xmin) == len(ell_select), "Length of xmin must match length of ell_select"
    assert len(xmax) == len(ell_select), "Length of xmax must match length of ell_select"

    nx = int(np.sum([len(xval) for xval in x]))
    assert cov.shape == (nx, nx), "Covariance matrix shape does not match the total number of x values." 

    ell_select_indices = {ll: i for i, ll in enumerate(ell_select)}

    # Create a mask for the x values to keep
    mask = np.zeros(nx, dtype=bool)
    idx_start = 0
    for xval, ll in zip(x, ell):
        idx_end = idx_start + len(xval)
        if ll in ell_select:
            j = ell_select_indices[ll]
            xmask = (xval >= xmin[j]) & (xval <= xmax[j]) if xval.ndim == 1 else np.all((xval >= xmin[j]) & (xval <= xmax[j]), axis=1)
            mask[idx_start:idx_end] = xmask
        idx_start = idx_end
    # Apply the mask to the covariance matrix
    cov_cut = cov[mask][:, mask]
    return cov_cut



def cut_window(win, x, ell, xwin, ellwin, xmin=-np.inf, xmax=np.inf, xwinmin=-np.inf, xwinmax=np.inf, ell_select=None, ellwin_select=None):
    if not isinstance(ell, list):
        ell= [ell]
    if not isinstance(x, list):
        x = [x] * len(ell)
    
    assert len(x) == len(ell), "Length of x must match length of ell"
    
    if ell_select is None:
        ell_select = ell
    if not isinstance(ell_select, list):
        ell_select = [ell_select]
    
    if not isinstance(xmin, list):
        xmin = [xmin] * len(ell_select)
    if not isinstance(xmax, list):
        xmax = [xmax] * len(ell_select)
    
    assert len(xmin) == len(ell_select), "Length of xmin must match length of ell_select"
    assert len(xmax) == len(ell_select), "Length of xmax must match length of ell_select"

    if not isinstance(ellwin, list):
        ellwin= [ellwin]
    if not isinstance(xwin, list):
        xwin = [xwin] * len(ellwin)
    
    assert len(xwin) == len(ellwin), "Length of xwin must match length of ellwin"
    
    if ellwin_select is None:
        ellwin_select = ellwin
    if not isinstance(ellwin_select, list):
        ellwin_select = [ellwin_select]
    
    if not isinstance(xwinmin, list):
        xwinmin = [xwinmin] * len(ellwin_select)
    if not isinstance(xwinmax, list):
        xwinmax = [xwinmax] * len(ellwin_select)
    
    assert len(xwinmin) == len(ellwin_select), "Length of xwinmin must match length of ellwin_select"
    assert len(xwinmax) == len(ellwin_select), "Length of xwinmax must match length of ellwin_select"


    nout = int(np.sum([len(xval) for xval in x]))
    nin = int(np.sum([len(xval) for xval in xwin]))
    assert win.shape == (nout, nin), "Window matrix shape does not match the total number of x values in the output and input."

    ell_select_indices = {ll: i for i, ll in enumerate(ell_select)}
    ellwin_select_indices = {ll: i for i, ll in enumerate(ellwin_select)}

    # Create a mask for the x values to keep
    mask_out = np.zeros(win.shape[0], dtype=bool)
    mask_in = np.zeros(win.shape[1], dtype=bool)
    idx_start_out = 0
    for xval, ll in zip(x, ell):
        idx_end_out = idx_start_out + len(xval)
        if ll in ell_select:
            j = ell_select_indices[ll]
            xmask = (xval >= xmin[j]) & (xval <= xmax[j]) if xval.ndim == 1 else np.all((xval >= xmin[j]) & (xval <= xmax[j]), axis=1)
            mask_out[idx_start_out:idx_end_out] = xmask
        idx_start_out = idx_end_out
    win_cut = win[mask_out, :]
    idx_start_in = 0
    for xval, ll in zip(xwin, ellwin):
        idx_end_in = idx_start_in + len(xval)
        if ll in ellwin_select:
            j = ellwin_select_indices[ll]
            xmask = (xval >= xwinmin[j]) & (xval <= xwinmax[j]) if xval.ndim == 1 else np.all((xval >= xwinmin[j]) & (xval <= xwinmax[j]), axis=1)
            mask_in[idx_start_in:idx_end_in] = xmask
        idx_start_in = idx_end_in
    win_cut = win_cut[:, mask_in]
    
    return win_cut

class NonFiniteTheoryError(ValueError):
    """Raised when the theory prediction is not finite at the requested point.

    The emulator is only calibrated over a finite range of its input
    parameters (see ``emu.params_ranges``); far outside it the emulated
    ingredients can over/underflow and return inf/nan. Downstream code then
    either raises an opaque error (``make_interp_spline`` rejects non-finite
    input) or silently produces a nan chi2, so the condition is detected
    explicitly here and turned into a single, catchable exception that the
    likelihood converts into a rejected point (log-likelihood = -inf).
    """


def check_finite(arrays, what, params=None):
    """Raise NonFiniteTheoryError if any entry of ``arrays`` is not finite.

    Parameters
    ----------
    arrays : array_like or sequence/dict of array_like
        Quantities to validate.
    what : str
        Short label of what is being checked, used in the error message.
    params : dict, optional
        Parameters the prediction was evaluated at; a few cosmological ones
        are added to the message to make the offending point identifiable.
    """
    if isinstance(arrays, dict):
        arrays = list(arrays.values())
    elif not isinstance(arrays, (list, tuple)):
        arrays = [arrays]

    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if np.all(np.isfinite(arr)):
            continue
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        msg = f"{what}: {n_bad}/{arr.size} non-finite entries"
        if params is not None:
            shown = {p: np.atleast_1d(params[p]).tolist() for p in
                     ('wc', 'wb', 'ns', 'h', 'As', 'w0', 'wa', 's12', 'f',
                      'q_lo', 'q_tr', 'b1', 'avir') if p in params}
            msg += f" at {shown}"
        raise NonFiniteTheoryError(msg)
