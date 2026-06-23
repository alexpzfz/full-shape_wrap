# full-shape_wrap

A lightweight likelihood/inference layer built around the [COMET](https://gitlab.com/aegge/comet-emu) emulator for full-shape analysis of galaxy clustering data — power spectrum multipoles, bispectrum multipoles (Scoccimarro and Sugiyama projections), and joint power spectrum + bispectrum fits.

## Overview

The package is organized around four stages:

1. **Observable** ([observables.py](observables.py)) — wraps a measured data vector (k or triangle/pair coordinates, multipole values, covariance, window function) into `PowerSpectrumMultipoles`, `BispectrumScoccimarroMultipoles`, `BispectrumSugiyamaMultipoles`, or a `JointObservable` combining a power spectrum and a bispectrum with their cross-covariance.
2. **Params** ([params.py](params.py)) — builds the full set of cosmological, bias, counterterm, and stochastic parameters for a given COMET model/basis (loaded from [params.yaml](params.yaml)), handles fixing/freeing parameters, co-evolution relations for higher-order bias, multi-redshift parameter duplication, and an optional AP/σ₁₂ reparametrization of nuisance parameters.
3. **Likelihood** ([likelihood.py](likelihood.py)) — computes the Gaussian χ² between data and the COMET prediction (via [theory.py](theory.py)) for one or more observables/redshift bins, with covariance rescaling (Hartlap/Percival factors) and optional analytical marginalization (AM) over linear nuisance parameters (shot-noise, counterterms, etc.).
4. **Samplers** ([samplers.py](samplers.py)) — `NautilusSampler` for nested sampling (via [nautilus](https://github.com/johannesulf/nautilus)) and `MinuitMinimizer` for MAP/best-fit estimation (via [iminuit](https://github.com/scikit-hep/iminuit)).

[theory.py](theory.py) wraps the COMET emulator to batch-evaluate multipole predictions across observables and redshift bins (caching unique k/triangle/pair evaluations), and to build the per-parameter design matrix used for analytical marginalization. [utils.py](utils.py) has helpers for cutting covariance/window matrices to a given scale range.

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `pyyaml`
- [`comet-emu`](https://gitlab.com/aegge/comet-emu) — the underlying emulator (required by [theory.py](theory.py))
- `nautilus-sampler` — for `NautilusSampler`
- `iminuit` — for `MinuitMinimizer`
- `matplotlib` — for `Observable.plot()`
- `sympy`, `numba` — only needed for the standalone PT model in [bispectrum.py](bispectrum.py)

## Quick start

```python
import numpy as np
from theory import COMET
from observables import PowerSpectrumMultipoles
from params import Params
from likelihood import Likelihood
from samplers import MinuitMinimizer, NautilusSampler

emu = COMET(model='EFT')

cosmo_fid = {'wc': 0.1212, 'wb': 0.022, 'Mnu': 0.06, 'ns': 0.96,
             'h': 0.67, 'As': 2.1, 'w0': -1.0, 'wa': 0.0, 'Ok': 0.0, 'z': 1.0}

k, p0, p2, p4 = np.loadtxt('pk_data.txt', unpack=True)
cov = np.loadtxt('pk_cov.txt')

obs = PowerSpectrumMultipoles(k, [p0, p2, p4], ell=[0, 2, 4], cov=cov,
                               cosmo_fid=cosmo_fid, nbar=2e-3, kmax=[0.25, 0.2, 0.2])

pars = Params(emu, z_array=[1.0])
pars.update_parameter('wc', 0.12, prior=(0.08, 0.16), prior_type='uniform')
# ... configure remaining sampled/fixed parameters ...

lik = Likelihood(obs, pars, am_params=['c0', 'c2', 'c4', 'NP0'])

# Best fit
mini = MinuitMinimizer(lik)
mini.run()
best_fit, errors = mini.get_map()

# Posterior via nested sampling
sampler = NautilusSampler(lik, n_live=2000)
sampler.sample(verbose=True)
sampler.save('chain.npz')
```

## Analytical marginalization

Passing `am_params` to `Likelihood` marginalizes those nuisance parameters analytically (assuming they enter the model linearly, with a Gaussian prior) instead of sampling them directly, which significantly speeds up posterior exploration. The marginalized parameters can still be reconstructed afterward (sampled from their conditional posterior, or read off as a MAP estimate) via `Likelihood.sample_cond_am`.
