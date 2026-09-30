import numpy as np
from observables import Observable
from likelihood import THEORY_FAILURES
from params import Params

class BaseSampler:
    """Base class for samplers"""
    def __init__(self, likelihood):
        self.params = likelihood.params
        self.likelihood = likelihood

    def log_prior(self, param_values):
        """Calculate the log prior probability for the given parameter values"""
        log_prior = 0.0
        for name, value in param_values.items():
            param = self.params.parameters[name]
            if param.prior is not None:
                if param.prior_type == "uniform":
                    if not (param.prior[0] <= value <= param.prior[1]):
                        return -np.inf 
                    else:
                        # add the actual log prior contribution for uniform prior
                        # to ensure proper normalization (important when estimating profile errors?)
                        log_prior += np.log(1.0 / (param.prior[1] - param.prior[0]))

                elif param.prior_type == "gaussian":
                    mean, std = param.prior
                    # Gaussian contribution: -0.5 * chi2_prior
                    log_prior += -0.5 * ((value - mean) / std) ** 2 
                    log_prior += np.log(1.0 / (std * np.sqrt(2 * np.pi)))  # normalization term 
        return log_prior



class NautilusSampler(BaseSampler):
    """Sampler using the Nautilus algorithm"""
    def __init__(self, likelihood, **kwargs):
        super().__init__(likelihood)
        from nautilus import Sampler
        
        self.prior = self.params.build_nautilus_prior()
        self.require_blobs = len(self.params.exported_derived_names) > 0
        
        self.sampler = Sampler(
            self.prior, 
            self.likelihood_wrapper,
            **kwargs,
        )

    def likelihood_wrapper(self, param_dict):
        full_dict = self.params.get_full_dict(param_dict)
        loglike = self.likelihood.get_loglike(full_dict) 
        if self.require_blobs:
            blobs = [full_dict[name] for name in self.params.exported_derived_names]
            return loglike, blobs
        
        return loglike

    def sample(self, **kwargs):
        """Run the Nautilus sampling algorithm"""
        self.sampler.run(**kwargs)
        
    def save(self, filename, metadata=None, save_txt=True):
        """Save posterior samples to an HDF5 file.

        Parameters
        ----------
        filename : str
            Output path. A '.h5' extension is appended if not already present.
        metadata : dict, optional
            Extra scalar/string run info (e.g. sampler or fit settings) to
            store as file attributes for provenance.
        save_txt : bool
            If True, also write a '.txt' file (same basename) with a summary
            table of the weighted posterior (see `summary_table`).
        """
        import h5py

        if self.require_blobs:
            points, log_w, log_l, blobs = self.sampler.posterior(return_blobs=True)
            if blobs.ndim == 1:
                blobs = blobs[:, None]  # Ensure blobs is 2D for hstack
            points = np.hstack([points, blobs])
            names = self.prior.keys + self.params.exported_derived_names
        else:
            points, log_w, log_l = self.sampler.posterior()
            names = self.prior.keys
        latex_names = [self.params.parameters[n].latex for n in names]

        if not filename.endswith('.h5'):
            filename = filename + '.h5'

        if save_txt:
            txt_filename = filename[:-len('.h5')] + '.txt'
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(self.summary_table(points, log_w, log_l, names))

        str_dtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('points', data=points)
            f.create_dataset('log_weights', data=log_w)
            f.create_dataset('log_likelihoods', data=log_l)
            f.create_dataset('names', data=names, dtype=str_dtype)
            f.create_dataset('latex_names', data=latex_names, dtype=str_dtype)
            for key, value in (metadata or {}).items():
                f.attrs[key] = _sanitize_attr(value)

    def summary_table(self, points, log_w, log_l, names):
        """Return a text table of weighted posterior statistics per parameter.

        Columns: argmax (sample with the highest log-posterior, i.e. log-likelihood
        plus log-prior of the sampled parameters), mean, std, median, and the
        lower/upper 1sigma and 2sigma errors, given as offsets from the median to
        the (16, 84) and (2.5, 97.5) weighted percentiles, i.e. the edges of the
        central 68% and 95% credible intervals.
        """
        w = np.exp(log_w - np.max(log_w))
        w /= np.sum(w)

        n_sampled = len(self.prior.keys)
        log_post = np.array([
            ll + self.log_prior(dict(zip(self.prior.keys, pt[:n_sampled])))
            for pt, ll in zip(points, log_l)
        ])
        i_max = np.argmax(log_post)

        header = ['param', 'argmax', 'mean', 'std', 'median', '-1σ', '+1σ', '-2σ', '+2σ']
        rows = []
        for j, name in enumerate(names):
            x = points[:, j]
            mean = np.sum(w * x)
            std = np.sqrt(np.sum(w * (x - mean) ** 2))
            q025, q16, q50, q84, q975 = _weighted_quantile(x, w, [0.025, 0.16, 0.5, 0.84, 0.975])
            # Show every value of a row to the precision of its 1sigma error
            # (3 significant figures of the smaller side).
            err = min(q50 - q16, q84 - q50)
            if np.isfinite(err) and err > 0:
                dec = int(np.clip(2 - np.floor(np.log10(err)), 0, 12))
                fmt_v = lambda v, sign='': f'{v:{sign}.{dec}f}'
            else:
                fmt_v = lambda v, sign='': f'{v:{sign}.6g}'
            rows.append([name] + [fmt_v(v) for v in (x[i_max], mean, std, q50)]
                        + [fmt_v(v, '+') for v in (q16 - q50, q84 - q50, q025 - q50, q975 - q50)])

        widths = [max(len(r[k]) for r in [header] + rows) for k in range(len(header))]
        # The +/- columns of each interval are drawn as one cell under a group header.
        n_single = 5
        groups = [('68% (1σ)', widths[5] + widths[6] + 2), ('95% (2σ)', widths[7] + widths[8] + 2)]
        for g, (label, wd) in enumerate(groups):
            if len(label) > wd:  # widen the '+' column so the group label fits
                widths[6 + 2 * g] += len(label) - wd
                groups[g] = (label, len(label))

        cell_w = widths[:n_single] + [wd for _, wd in groups]
        rule = lambda l, m, r: l + m.join('─' * (wd + 2) for wd in cell_w) + r

        def fmt(r):
            cells = [c.ljust(wd) if k == 0 else c.rjust(wd)
                     for k, (c, wd) in enumerate(zip(r[:n_single], widths))]
            cells += [f'{r[5 + 2 * g].rjust(widths[5 + 2 * g])}  {r[6 + 2 * g].rjust(widths[6 + 2 * g])}'
                      for g in range(len(groups))]
            return '│ ' + ' │ '.join(cells) + ' │'

        group_row = ['' for _ in range(n_single)] + [label.center(wd) for label, wd in groups]
        group_row = '│ ' + ' │ '.join(c.ljust(wd) for c, wd in zip(group_row, cell_w)) + ' │'

        lines = ['Posterior summary',
                 f'  max log-posterior      = {log_post[i_max]:.6g}  '
                 f'(log-likelihood = {log_l[i_max]:.6g})',
                 f'  log-evidence           = {self.sampler.log_z:.6g}',
                 f'  effective sample size  = {self.sampler.n_eff:.1f}',
                 '  argmax: sample with the highest log-posterior',
                 '  ±1σ, ±2σ: offsets from the median to the edges of the central 68% and 95% '
                 'credible intervals',
                 '',
                 rule('┌', '┬', '┐'),
                 group_row,
                 fmt(header),
                 rule('├', '┼', '┤')]
        lines += [fmt(r) for r in rows]
        lines.append(rule('└', '┴', '┘'))
        return '\n'.join(lines) + '\n'


class MinuitMinimizer(BaseSampler):
    """Wrapper for the iMinuit minimizer"""
    def __init__(self, likelihood, initial_step=0.1, seed_init=None, verbose=False):
        super().__init__(likelihood)
        from iminuit import Minuit
            
        def cost_function(*args):
            # Convert positional args to dictionary
            param_dict = dict(zip(self.params.sampled_param_names, args))
            full_dict = self.params.get_full_dict(param_dict)
            lp = self.log_prior(param_dict)
            if not np.isfinite(lp):
                return np.inf
            chi2_prior = -2.0 * lp
            # Get Data Chi2
            # Note: We use get_chi2 directly, not get_loglike
            # A point where the model cannot be evaluated (emulator far outside
            # its range) is reported as infinitely bad rather than crashing the
            # minimization.
            try:
                chi2_data = self.likelihood.get_chi2(full_dict)
            except THEORY_FAILURES as exc:
                self.likelihood._report_failure(exc)
                return np.inf
            if not np.isfinite(chi2_data):
                return np.inf
            return chi2_data + chi2_prior


        if seed_init is not None:
            np.random.seed(seed_init)
        self.sampled_param_names = self.params.sampled_param_names
        # init_values = [self.params.parameters[n].value for n in self.sampled_param_names]
        init_values = []
        for n in self.sampled_param_names:
            p = self.params.parameters[n]
            if p.prior_type == 'uniform':
                # Start at the midpoint of the uniform prior
                if seed_init is None:
                    init_values.append(0.5 * (p.prior[0] + p.prior[1]))
                else:
                    # draw a random starting point within the uniform prior range
                    init_values.append(np.random.uniform(p.prior[0], p.prior[1]))
            elif p.prior_type == 'gaussian':
                # Start at the mean of the Gaussian prior
                if seed_init is None:
                    init_values.append(p.prior[0])
                else:
                    # draw a random starting point from the Gaussian prior
                    init_values.append(np.random.normal(p.prior[0], p.prior[1]))
            else:
                # Fallback to the current value if no prior is defined
                print(f"Warning: Parameter {n} has no prior defined. Using current value {p.value} as starting point.")
                print("This should almost never happen, as all sampled parameters should have a prior.")
                init_values.append(p.value)

        self.m = Minuit(cost_function, *init_values, name=self.params.sampled_param_names)
        
        self.m.errordef = Minuit.LEAST_SQUARES # = 1.0 (for Chi2 minimization)
        
        for name in self.sampled_param_names:
            p = self.params.parameters[name]
            
            if p.prior_type == 'uniform' and p.prior is not None:
                self.m.limits[name] = p.prior

            # hardcode limits for comet parameters
            _comet_limits = {'wc': (0.08, 0.16), 'wb': (0.01930, 0.02535), 'ns': (0.9, 1.03),
                             'As': (1.0, 3.5), 'Mnu': (0.0, 1.0), 'sigma_12': (0.2, 1.0),
                             'f': (0.5, 1.05), 'log10As': (np.log(1e10 * 1e-9), np.log(1e10 * 3.5 * 1e-9))}
            if name in _comet_limits:
                if p.prior_type == 'gaussian' or (p.prior_type == 'uniform' and p.prior[0] < _comet_limits[name][0] or p.prior[1] > _comet_limits[name][1]):
                    print(f"Warning: Overriding limits for {name} to {_comet_limits[name]} based on COMET constraints.")
                    self.m.limits[name] = _comet_limits[name]
            
            # Set Initial Step Size (Heuristic)
            # Prefer prior-based step so that zero-initialised nuisance params
            # get a meaningful scale rather than a hard-coded constant.
            if p.value != 0:
                step = abs(p.value) * initial_step
            elif p.prior is not None:
                if p.prior_type == 'gaussian':
                    # sigma of the Gaussian prior is the natural scale
                    step = p.prior[1] * initial_step
                elif p.prior_type == 'uniform':
                    # half-width of the uniform interval
                    step = (p.prior[1] - p.prior[0]) * initial_step
                else:
                    step = initial_step
            else:
                step = initial_step
            # Guard against zero or tiny steps
            if step == 0 or not np.isfinite(step):
                step = initial_step
            self.m.errors[name] = step
        
        if verbose:
            print(f"Initialized Minuit with {len(self.params.sampled_param_names)} free parameters.")

    def set_starting_point(self, param_dict, errors_dict=None, reset_errors=True):
        """Seed Minuit's starting values from a dictionary of parameter values.

        Parameters
        ----------
        param_dict : dict
            Mapping of parameter name -> value.  Only names that are free
            (i.e. in ``self.sampled_param_names``) will be applied; extra
            keys are silently ignored.
        errors_dict : dict, optional
            Mapping of parameter name -> step size / uncertainty.  When
            provided, these override the automatic ``reset_errors`` heuristic
            for the corresponding parameters.
        reset_errors : bool
            If True, reset each step-size to 10 % of the absolute starting
            value for parameters not covered by ``errors_dict`` (falls back
            to the current error if the value is 0).
        """
        for name in self.sampled_param_names:
            if name in param_dict:
                val = float(param_dict[name])
                self.m.values[name] = val
                if errors_dict is not None and name in errors_dict:
                    err = float(errors_dict[name])
                    if np.isfinite(err) and err > 0:
                        self.m.errors[name] = err
                elif reset_errors and val != 0:
                    self.m.errors[name] = abs(val) * 0.1

    def run(self, hesse=False, strategy=2, tol=0.1, ncall=1000000, iterate=20,
            use_simplex=True, print_level=1, pre_simplex=False, verbose=True):
        """
        Run the minimization.

        Parameters
        ----------
        hesse : bool
            If True, run HESSE after a successful MIGRAD call.
        strategy : int
            Minuit strategy level. Use 2 for a stringent convergence strategy.
        tol : float
            EDM tolerance. Smaller values enforce tighter convergence.
        ncall : int or None
            Approximate maximum number of calls per MIGRAD attempt. If None,
            iminuit uses its adaptive heuristic.
        iterate : int
            Number of times Minuit.migrad will automatically retry if
            convergence was not reached (see iminuit's `migrad` docs).
        use_simplex : bool
            If retrying, run SIMPLEX before each MIGRAD retry (see iminuit's
            `migrad` docs).
        print_level : int
            Minuit's own verbosity (0-3) while it runs; at 2 it prints one
            line per MIGRAD/SIMPLEX iteration, which makes retries and
            simplex fallbacks visible live. Only applied when verbose=True.
            Note this sets a process-wide iminuit setting, not just for this
            instance.
        pre_simplex: bool
            Start the run with a SIMPLEX call.
        verbose : bool
            If True, print convergence diagnostics.
        """
        self.m.strategy = strategy
        self.m.tol = tol
        self.m.print_level = print_level if verbose else 0

        if pre_simplex:
            self.m.simplex()
            if verbose:
                print(self.m.fmin)

        self.m.migrad(ncall=ncall, iterate=iterate, use_simplex=use_simplex)
        if verbose:
            print(self.m.fmin)

        # Optionally run HESSE only after a valid minimum is found.
        if hesse and self.m.valid:
            self.m.hesse()
            if verbose:
                print(self.m.fmin)

        return self.m
    
    def get_map(self, return_am=True):
        """Return the best-fit parameters and their uncertainties.

        Parameters
        ----------
        return_am : bool
            If True and the likelihood uses analytical marginalisation, also
            return the conditional MAP values *and* formal uncertainties
            (``sqrt(diag(cond_cov))``) for the analytically-marginalised
            parameters.
        """
        if not self.m.valid:
            import warnings
            edm = getattr(self.m.fmin, 'edm', float('nan'))
            warnings.warn(
                f"Minimization did not fully converge (valid=False, edm={edm:.3e}). "
                f"Returning best-fit values anyway — verify that edm is acceptably small."
            )
        
        best_fit = {name: self.m.values[name] for name in self.sampled_param_names}
        uncertainties = {name: self.m.errors[name] for name in self.sampled_param_names}
        if return_am and self.likelihood.do_am:
            full_dict = self.params.get_full_dict(best_fit)
            self.likelihood.am_sample_mode = 'map'
            self.likelihood.get_chi2(full_dict)  # Update AM params and cache cond_cov
            all_am_params = [name for am_params_iz in self.likelihood.am_params for name in am_params_iz]
            best_fit.update({name: full_dict[name] for name in all_am_params})
            # Extract formal uncertainties from the cached conditional covariances
            for iz, am_params_iz in enumerate(self.likelihood.am_params):
                cond_cov = getattr(self.likelihood, '_last_am_cond_covs', [None] * (iz + 1))[iz]
                if cond_cov is not None:
                    am_errors = np.sqrt(np.diag(np.atleast_2d(cond_cov)))
                    for j, name in enumerate(am_params_iz):
                        uncertainties[name] = float(am_errors[j])
        return best_fit, uncertainties

    def save(self, filename, best_fit=None, uncertainties=None, return_am=True, metadata=None, save_txt=True):
        """Save the best-fit result to an HDF5 file.

        Parameters
        ----------
        filename : str
            Output path. A '.h5' extension is appended if not already present.
        best_fit, uncertainties : dict, optional
            Precomputed results from `get_map` to save as-is (avoids
            recomputing the AM conditional covariance). If either is None,
            `get_map(return_am=return_am)` is called to obtain both.
        return_am : bool
            Passed to `get_map` when `best_fit`/`uncertainties` are not
            supplied. Ignored otherwise.
        metadata : dict, optional
            Extra scalar/string run info (e.g. fit settings) to store as
            file attributes for provenance.
        save_txt : bool
            If True, also write a '.txt' file (same basename) with the
            Minuit fmin/params tables, i.e. the same summary Minuit prints
            to stdout when verbose=True.
        """
        import h5py

        if best_fit is None or uncertainties is None:
            best_fit, uncertainties = self.get_map(return_am=return_am)

        names = list(best_fit.keys())
        values = np.array([best_fit[n] for n in names])
        errors = np.array([uncertainties[n] for n in names])
        latex_names = [self.params.parameters[n].latex for n in names]

        if not filename.endswith('.h5'):
            filename = filename + '.h5'

        if save_txt:
            txt_filename = filename[:-len('.h5')] + '.txt'
            with open(txt_filename, 'w') as f:
                f.write(str(self.m.fmin))
                f.write('\n\n')
                f.write(str(self.m.params))
                f.write('\n')

        str_dtype = h5py.string_dtype(encoding='utf-8')
        with h5py.File(filename, 'w') as f:
            f.create_dataset('names', data=names, dtype=str_dtype)
            f.create_dataset('latex_names', data=latex_names, dtype=str_dtype)
            f.create_dataset('best_fit', data=values)
            f.create_dataset('uncertainties', data=errors)

            if self.m.covariance is not None:
                # Only covers the free (sampled) parameters, not analytically
                # marginalised ones — those only have per-z conditional
                # covariances, not a single joint matrix with the free params.
                f.create_dataset('covariance', data=np.array(self.m.covariance))
                f.create_dataset('covariance_names', data=self.sampled_param_names, dtype=str_dtype)

            fmin = self.m.fmin
            f.attrs['valid'] = bool(self.m.valid)
            f.attrs['edm'] = float(getattr(fmin, 'edm', np.nan))
            f.attrs['fval'] = float(getattr(fmin, 'fval', np.nan))
            f.attrs['nfcn'] = int(getattr(fmin, 'nfcn', -1))

            for key, value in (metadata or {}).items():
                f.attrs[key] = _sanitize_attr(value)

def _weighted_quantile(x, w, q):
    """Quantiles `q` of samples `x` with normalised weights `w`."""
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) - 0.5 * w
    return np.interp(q, cdf, x)

def _sanitize_attr(value):
    """Coerce a Python value into something h5py can store as an attribute.

    argparse.Namespace values routinely include None, tuples, and lists of
    tuples (e.g. --ellB), none of which h5py.attrs accepts directly.
    """
    import h5py

    if value is None:
        return 'None'
    if isinstance(value, (str, bytes, bool, int, float, np.integer, np.floating)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.array(value)
            if arr.dtype == object:
                return str(value)
            if arr.dtype.kind in ('U', 'S'):
                # h5py can't map numpy fixed-width string dtypes to an HDF5
                # type directly; use its variable-length string dtype instead.
                return np.array(value, dtype=h5py.string_dtype(encoding='utf-8'))
            return arr
        except Exception:
            return str(value)
    return str(value)