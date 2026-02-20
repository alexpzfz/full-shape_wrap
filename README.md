# full-shape_wrap

A Python wrapper for full-shape galaxy clustering analysis using the [COMET](https://github.com/comet-emu) emulator. This package provides modular building blocks for computing galaxy power spectrum multipoles and Scoccimarro bispectrum multipoles, managing model parameters, evaluating likelihoods, and running posterior sampling or optimization.

## Overview

`full-shape_wrap` provides:

- **Observables**: data containers for power spectrum multipoles, Scoccimarro bispectrum multipoles, and joint datasets, with optional window-matrix convolution and scale cuts.
- **Parameters**: flexible management of cosmological, bias, counterterm, stochastic, and derived parameters with support for co-evolution relations and reparametrizations.
- **Theory**: a `COMET` model class that wraps the COMET emulator and dispatches predictions to power spectrum or bispectrum routines.
- **Likelihood**: Gaussian likelihood evaluation with optional analytical marginalization (AM) over nuisance parameters and Hartlap covariance correction.
- **Samplers**: nested-sampling wrapper (`NautilusSampler`) and gradient-based minimizer (`MinuitMinimizer`).
- **Bispectrum**: numerical projection of the vdG bispectrum model onto multipoles.

## Dependencies

- Python ≥ 3.9
- [COMET](https://github.com/comet-emu/comet) (`comet`)
- `numpy`
- `scipy`
- For sampling: [`nautilus`](https://nautilus-sampler.readthedocs.io/)
- For minimization: [`iminuit`](https://iminuit.readthedocs.io/)
- For plotting: `matplotlib` (optional)

## Module descriptions

### `observables.py`

Defines data containers for observational data vectors.

| Class | Description |
|---|---|
| `Observable` | Base class. Stores data vectors `x`, `y`, covariance `cov`, optional window matrix `wmat`, and supports scale cuts via `xmin`/`xmax`. |
| `PowerSpectrumMultipoles` | Power spectrum multipoles $P_\ell(k)$. Handles unit conversion between $h$-units and Mpc units. Supports window-matrix convolution. |
| `BispectrumScoccimarroMultipoles` | Scoccimarro bispectrum multipoles $B_\ell(k_1, k_2, k_3)$. Stores triangle configurations. |
| `JointObservable` | Combines two observables into a single joint data vector with a block-diagonal or user-supplied covariance matrix. |

**Example:**
```python
from observables import PowerSpectrumMultipoles

obs = PowerSpectrumMultipoles(
    k=k_arr,              # wavenumbers in h/Mpc
    Pell=[P0, P2],        # monopole and quadrupole
    cov=cov_matrix,
    cosmo_fid={'h': 0.676, 'z': 0.5},
    kmin=0.01, kmax=0.25,
    Mpc_units=False,
)
```

### `params.py`

Manages the full set of model parameters. The `Params` class is initialized from a COMET emulator instance and a chosen bias basis.

**Cosmological parameters** (`wc`, `wb`, `h`, `ns`, `As`, `Mnu`, `w0`, `wa`, `Ok`).

**Bias bases:**

| Key | Parameters |
|---|---|
| `EggScoSmi` | `b1`, `b2`, `g2`, `g21` |
| `AssBauGre` | `b1`, `b2`, `bG2`, `bGam3` |
| `DesJeoSch` | `b1`, `b2t`, `bK2`, `btd` |

**Counterterm bases:** `Comet` (`c0`, `c2`, `c4`), `DESI` (`a0`, `a2`, `a4`).

**Stochastic parameters:** `NP0`, `NP20`, `NP22`.

Key methods:

| Method | Description |
|---|---|
| `set_derived_param(name, func)` | Mark a parameter as derived, computed from other parameter values. |
| `set_and_fix_param(name, value)` | Fix a parameter to a given value. |
| `free_param(name)` | Un-fix a parameter. |
| `update_prior(name, prior)` | Change the prior bounds or type. |
| `get_full_dict(free_values)` | Merge sampled values with fixed/derived parameter values into a full dictionary. |
| `build_nautilus_prior()` | Construct a `nautilus.Prior` object from the current free parameters. |
| `use_reparametrization(...)` | Enable AP-effect and/or $\sigma_{12}$ reparametrization of bias, counterterms, and stochastic parameters. |

**Co-evolution relations** for `bG2`, `bGam3`, `bK2`, `btd` can be enabled at initialization:
```python
params = Params(emu, coev_params=['bG2', 'bGam3'])
```

### `theory.py`

Wraps the COMET emulator in a `COMET` class that inherits from both `comet.comet` and `BaseModel`.

```python
from theory import COMET
emu = COMET(bias_basis='AssBauGre', counterterm_basis='DESI')
```

The `predict(observable, params, de_model)` method dispatches to:
- `predict_power_spectrum_multipoles` — calls `emu.Pell` and applies the window matrix if present.
- `predict_bispectrum_scoccimarro_multipoles` — calls `bispectrum_scoccimarro_proj`.

Supported dark-energy models (`de_model`): `"lambda"`, `"w0"`, `"w0wa"` (determined automatically from the parameter state).

### `likelihood.py`

`Likelihood` evaluates a Gaussian log-likelihood:

$$\ln \mathcal{L} = -\frac{1}{2} \Delta^T C^{-1} \Delta$$

where $\Delta = d - m(\theta)$.

**Analytical marginalization (AM)** over linear nuisance parameters (bias, counterterms, stochastic) is supported. Pass a list of parameter names to `am_params`; the marginalized $\chi^2$ is computed using the Woodbury identity.

| Constructor argument | Description |
|---|---|
| `observable` | An `Observable` instance containing data and covariance. |
| `emu` | A `COMET` emulator instance. |
| `params` | A `Params` instance. |
| `am_params` | List of parameter names to marginalize over analytically. |
| `am_sample` | If `True`, draw AM parameters from their conditional posterior at each evaluation. |
| `conditional_prior` | Optional boolean-valued function applied as a hard prior cut. |

The Hartlap correction is applied automatically when `observable.nmocks_cov` is set.

**Example:**
```python
from likelihood import Likelihood

like = Likelihood(observable=obs, emu=emu, params=params)
chi2 = like.get_chi2(full_param_dict)
loglike = like.get_loglike(full_param_dict)
```

### `samplers.py`

Two backends are provided.

#### `NautilusSampler`

Wraps the [Nautilus](https://nautilus-sampler.readthedocs.io/) nested sampler.

```python
from samplers import NautilusSampler

sampler = NautilusSampler(params, likelihood, filepath='chain.hdf5', n_live=500)
sampler.sample(verbose=True)
sampler.save('posterior.npz')
```

The saved `.npz` file contains `points`, `log_weights`, `log_likelihoods`, `names`, and `latex_names`.

#### `MinuitMinimizer`

Wraps [iMinuit](https://iminuit.readthedocs.io/) for MAP estimation.

```python
from samplers import MinuitMinimizer

minimizer = MinuitMinimizer(params, likelihood, verbose=True)
result = minimizer.run(hesse=True)
print(result)
```

### `bispectrum.py`

Implements the vdG bispectrum model and its projection onto Legendre multipoles.

| Function | Description |
|---|---|
| `bispectrum_vdg(k1, k2, k3, mu1, mu2, emu, comet_params, ...)` | Full redshift-space vdG bispectrum at given triangle and angle configurations. |
| `bispectrum_scoccimarro_proj(k1, k2, k3, emu, comet_params, ell, ...)` | Numerically integrates `bispectrum_vdg` over angles to produce Scoccimarro multipoles $B_\ell$. |

Integration is performed by Gauss–Legendre quadrature in $\mu_1$ and a uniform grid in $\phi$ (defaults: `nmu=20`, `nphi=20`).

## Quick-start example

```python
import numpy as np
from theory import COMET
from params import Params
from observables import PowerSpectrumMultipoles
from likelihood import Likelihood
from samplers import NautilusSampler

# 1. Initialize the emulator
emu = COMET(bias_basis='AssBauGre', counterterm_basis='DESI')

# 2. Set up parameters (free cosmology + co-evolved bG2, bGam3)
params = Params(emu, coev_params=['bG2', 'bGam3'])
params.free_param('b2')           # free b2
params.set_and_fix_param('Mnu', 0.06)

# 3. Load data
k = np.loadtxt('k.txt')
P0, P2 = np.loadtxt('Pell.txt', unpack=True)
cov = np.loadtxt('cov.txt')
cosmo_fid = {'h': 0.676, 'z': 0.5}

obs = PowerSpectrumMultipoles(k, [P0, P2], cov=cov, cosmo_fid=cosmo_fid,
                              kmin=0.02, kmax=0.20, Mpc_units=False)

# 4. Build the likelihood
like = Likelihood(obs, emu, params)

# 5. Run nested sampling
sampler = NautilusSampler(params, like, filepath='chain.hdf5', n_live=500)
sampler.sample(verbose=True)
sampler.save('posterior.npz')
```

## License

See [LICENSE](LICENSE) for details.
