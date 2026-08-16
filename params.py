import numpy as np
import dataclasses
from typing import Callable, Optional, Union
from functools import partial
import yaml
import os

def _load_default_parameters():
    """Load default parameter configurations from YAML file."""
    yaml_path = os.path.join(os.path.dirname(__file__), 'params.yaml')
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def _create_parameter_from_dict(name, param_dict):
    """Create a Parameter object from a dictionary loaded from YAML."""
    return Parameter(
        name=name,
        value=param_dict.get('value', 0.0),
        prior=tuple(param_dict.get('prior', [None, None])) if param_dict.get('prior') else None,
        prior_type=param_dict.get('prior_type', 'uniform'),
        fixed=param_dict.get('fixed', False),
        latex=param_dict.get('latex', '')
    )

@dataclasses.dataclass
class Parameter:
    name: str
    value: float
    base_name: str = ""
    prior: tuple = None  # (min, max)
    prior_type: str = "uniform"  # 'uniform' or 'gaussian'
    fixed: bool = False
    derived: bool = False  # True if this parameter is derived from others (e.g. co-evolution)
    derived_func: Optional[Union[Callable, str]] = None  # Function to compute derived parameter, or name from emu params dict
    requires_emu_eval: bool = False # Whether this derived parameter requires an emulator evaluation (e.g. depends on PLin)
    derived_am: bool = False # Only relevant for analytical marginalization
    exported: bool = False # Only relevant if derived=True, whether to include this parameter in the output samples
    latex: str = ""


_default_params = _load_default_parameters()

# Load cosmological parameters from YAML
_cosmo_params = [_create_parameter_from_dict(name, param_dict) 
                 for name, param_dict in _default_params['cosmological'].items()]

# Load bias parameters from YAML
_bias_params = {}
for basis_name, basis_params in _default_params['bias'].items():
    _bias_params[basis_name] = [_create_parameter_from_dict(name, param_dict) 
                                for name, param_dict in basis_params.items()]

# Create DesJeoSch_r variant
_bias_params["DESI_r"] = [dataclasses.replace(p, name=p.name + "r", latex=p.latex + "^r") 
                                for p in _bias_params["DESI"]]

# Load extra parameters from YAML
_extra_params = {}
for extra_type, extra_params in _default_params['extra'].items():
    _extra_params[extra_type] = [_create_parameter_from_dict(name, param_dict) 
                                 for name, param_dict in extra_params.items()]

# Load counterterm parameters from YAML
_counterterm_params = {}
for basis_name, basis_params in _default_params['counterterms'].items():
    _counterterm_params[basis_name] = [_create_parameter_from_dict(name, param_dict) 
                                       for name, param_dict in basis_params.items()]

# Create DESI_r variant
_counterterm_params["DESIct_r"] = [dataclasses.replace(p, name=p.name + "r", latex=p.latex + "^r") 
                                  for p in _counterterm_params["DESIct"]]

# Load stochastic parameters from YAML
_stochastic_params = [_create_parameter_from_dict(name, param_dict) 
                      for name, param_dict in _default_params['stochastic'].items()]




class Params:
    """Class to handle model parameters"""
    def __init__(self, emu, coev_params=None, z_array=None):
        self.emu = emu
        self.z_array = z_array if z_array is not None else [None]
        self.nz = len(self.z_array)
        self.cosmo_params = {p.name: dataclasses.replace(p, base_name=p.name) 
                             for p in _cosmo_params}

        def duplicate(param_list):
            res = {}
            for iz in range(self.nz):
                for p in param_list:
                    new_name = f"{p.name}_{iz}" if self.nz > 1 else p.name
                    new_latex = add_iz_to_latex(p.latex, iz) if self.nz > 1 and p.latex else (p.latex if p.latex else "")
                    res[new_name] = dataclasses.replace(p, name=new_name, base_name=p.name, latex=new_latex)
            return res
        
        self.bias_params = duplicate(_bias_params[emu.bias_basis])
        self.counterterm_params = duplicate(_counterterm_params[emu.counterterm_basis])
        self.extra_params = {}
        if 'VDG' in emu.model:
            self.extra_params.update(duplicate(_extra_params['VDG_infty']))
        elif 'EFT' in emu.model:
            self.counterterm_params.update(duplicate(_extra_params['EFT']))

        self.stochastic_params = duplicate(_stochastic_params)

        self.parameters = {**self.cosmo_params, **self.bias_params, **self.counterterm_params, **self.stochastic_params, **self.extra_params}
        self.comet_keys = list(set(p.base_name for p in self.parameters.values()))
        
        self.derived_order = [] 
        self.coev_params = None
        if coev_params is not None:
            self.coev_params = coev_params if isinstance(coev_params, list) else [coev_params]
            for name in self.coev_params:
                for iz in range(self.nz):
                    suffixed_name = f"{name}_{iz}"
                    if name == "bG2": self.set_derived_param(suffixed_name, partial(self.bG2_coev, iz=iz))
                    elif name == "bGam3": self.set_derived_param(suffixed_name, partial(self.bGam3_coev, iz=iz))
                    elif name == "bk2" or name == "bK2t": self.set_derived_param(suffixed_name, partial(self.bK2_coev, iz=iz))
                    elif name == "btd" or name == "btdt": self.set_derived_param(suffixed_name, partial(self.btd_coev, iz=iz))
                    else:
                        raise ValueError(f"Co-evolution for {name} not implemented.")
        # self.z = None # Placeholder for redshift, can be set externally if needed for derived parameters
        self.use_reparam = False
        self.reparam_bias_mode = None
        self.reparam_counterterms_mode = None
        self.reparam_stochastic_mode = None
        self.sigmaR_ref = None
        

    # def set_reference_sigmaR(self, sigmaR):
    #     assert 'DESI_DR2' in self.emu.bias_basis, "Reference sigmaR is only relevant for DESI_DR2 bias basis."
    #     self.sigmaR_ref = sigmaR

    def add_sampled_param(self, name, value, prior, prior_type="uniform", latex=""):
        """Helper to add a new sampled parameter on the fly"""
        if name in self.parameters:
            raise KeyError(f"Parameter {name} already exists.")
        new_param = Parameter(name=name, value=value, prior=prior, prior_type=prior_type, fixed=False, derived=False, latex=latex)
        self.parameters[name] = new_param

    def set_derived_param(self, name, deriv_func, latex="", exported=False, requires_emu_eval=False):
        if name not in self.parameters:
            self.parameters[name] = Parameter(name=name, value=None, prior=None, prior_type=None, fixed=True, derived=True, derived_func=deriv_func, requires_emu_eval=requires_emu_eval, latex=latex, exported=exported)
        
        self.parameters[name].derived = True
        self.parameters[name].fixed = False
        self.parameters[name].derived_func = deriv_func
        self.parameters[name].requires_emu_eval = requires_emu_eval
        self.parameters[name].exported = exported
        if latex != "":
            self.parameters[name].latex = latex

        if name not in self.derived_order:
            self.derived_order.append(name)

    def export_param(self, name, exported=True):
        """Helper to set whether a derived parameter should be included in output samples"""
        if name in self.parameters and self.parameters[name].derived:
            self.parameters[name].exported = exported
        else:
            raise KeyError(f"Parameter {name} not found or not a derived parameter.")

    @property
    def sampled_param_names(self):
        return [name for name, p in self.parameters.items() if not p.fixed and not p.derived]
    
    @property
    def exported_derived_names(self):
        return [name for name, p in self.parameters.items() if (p.derived or p.derived_am) and p.exported]
    
    @property
    def derived_am_names(self):
        return [name for name, p in self.parameters.items() if p.derived_am]
        

    @property
    def fixed_params_dict(self):
        """Dynamically get dictionary of fixed parameters"""
        return {name: p.value for name, p in self.parameters.items() if p.fixed}
    
    @property
    def de_model(self):
        if not self.parameters["wa"].fixed or (self.parameters["wa"].fixed and self.parameters["wa"].value != 0.0):
            return "w0wa" 
        elif not self.parameters["w0"].fixed or (self.parameters["w0"].fixed and self.parameters["w0"].value != -1.0): 
            return "w0"
        else:
            return "lambda"
    @property
    def n_sampled_params(self):
        """Dynamically count number of free parameters"""
        return len(self.sampled_param_names)
    
    @property
    def n_free_params(self):
        return self.n_sampled_params + len(self.derived_am_names)
    

    def get_iz_param_names(self, iz):
        """Returns a list of parameters that determine the observable iz"""
        iz_param_names = set()
        for name in self.sampled_param_names:
            if name.endswith(f"_{iz}"):
                iz_param_names.add(name)
            if name in self.cosmo_params:
                iz_param_names.add(name)
        for name in self.derived_order:
            if name in self.cosmo_params:
                iz_param_names.add(name)
        for name in self.derived_am_names:
            if name.endswith(f"_{iz}"):
                iz_param_names.add(name)

        return list(iz_param_names)
        
    def n_free_params_iz(self, iz):
        """Returns the number of free parameters that determine the observable at redshift iz"""
        return len(self.get_iz_param_names(iz))



    @property
    def fixed_cosmo(self):
        """Returns True if all cosmological parameters are fixed, False otherwise"""
        return all(self.parameters[name].fixed for name in self.cosmo_params)
    
    @staticmethod
    def bG2_coev(p, iz=0):
        b1 = p.get(f"b1_{iz}", p.get("b1"))
        return 0.524 - 0.547*b1 + 0.046*b1**2

    @staticmethod
    def bGam3_coev(p, iz=0):
        b1 = p.get(f"b1_{iz}", p.get("b1"))
        bG2 = p.get(f"bG2_{iz}", p.get("bG2"))
        return -1./6.*(b1-1.) -5./2.*bG2

    @staticmethod
    def bK2_coev(p, iz=0):
        b1 = p.get(f"b1_{iz}", p.get("b1"))
        return -2./7.*(b1 - 1.)
    
    @staticmethod
    def btd_coev(p, iz=0):
        b1 = p.get(f"b1_{iz}", p.get("b1"))
        return 23./42.*(b1 - 1.)


    def get_sampled_params(self):
        """Return the actual Parameter objects for sampled parameters"""
        return [self.parameters[name] for name in self.sampled_param_names]

    def build_nautilus_prior(self):
        """
        Constructs a nautilus.Prior object based on the CURRENT free parameters.
        """
        from nautilus import Prior
        from scipy.stats import norm
        prior = Prior()
        
        # Now this iterates over the dynamic property, so it sees your updates
        for name in self.sampled_param_names:
            p = self.parameters[name]
            if p.prior_type == "uniform":
                prior.add_parameter(name, dist=p.prior)
            elif p.prior_type == "gaussian":
                prior.add_parameter(name, dist=norm(loc=p.prior[0], scale=p.prior[1]))
            else:
                raise ValueError(f"Unknown prior type {p.prior_type} for {name}")
        return prior

    def get_full_dict(self, free_values_dict_or_list):
        """
        Merges free parameter values with the current fixed parameters.
        """
        # This now fetches the up-to-date fixed params
        full_dict = self.fixed_params_dict.copy()
        
        if isinstance(free_values_dict_or_list, dict):
            full_dict.update(free_values_dict_or_list)
        else:
            # Assume list/array in correct order of self.sampled_param_names
            names = self.sampled_param_names
            if len(free_values_dict_or_list) != len(names):
                raise ValueError(f"Input length {len(free_values_dict_or_list)} does not match "
                                 f"number of free params {len(names)}.")
            for name, val in zip(names, free_values_dict_or_list):
                full_dict[name] = val

        plin_evaluated = False
        # Compute derived parameters on the fly based on current free and fixed values
        for name in self.derived_order:
            param = self.parameters[name]
            if param.derived_func is not None:
                if not param.requires_emu_eval and callable(param.derived_func):
                    full_dict[name] = param.derived_func(full_dict)

                else:
                    if not plin_evaluated:
                        cosmo_dict = self.cosmo_dict(full_dict)
                        self.emu.PL(0.1, cosmo_dict, de_model=self.de_model)  # Ensure PLin is evaluated for current cosmology
                        plin_evaluated = True
                    if isinstance(param.derived_func, str):
                        full_dict[name] = self.emu.params[param.derived_func].item() # This will only work when 1 dataset is used, need to fix this in the future
                    elif callable(param.derived_func):
                        full_dict[name] = param.derived_func(full_dict)


        # if self.emu.bias_basis == "DESI_DR2" and self.sigmaR_ref is not None:
        #     full_dict['b1t'] = full_dict['b1t'] * self.sigmaR_ref
        #     full_dict['b2t'] = full_dict['b2t'] * self.sigmaR_ref**2
        #     full_dict['bK2t'] = full_dict['bK2t'] * self.sigmaR_ref**2
        #     full_dict['btdt'] = full_dict['btdt'] * self.sigmaR_ref**4
        # if self.emu.counterterm_basis == "DESI_DR2" and self.sigmaR_ref is not None:
        #     full_dict['a0'] = full_dict['a0'] * self.sigmaR_ref**2
        #     full_dict['a2'] = full_dict['a2'] * self.sigmaR_ref**2
        #     full_dict['a4'] = full_dict['a4'] * self.sigmaR_ref**2
        return full_dict
    
    # def get_comet_dict(self, full_dict):
    #     """Extracts the parameters needed for the comet emulator from the full dict, including derived parameters"""
    #     comet_dict = {key: full_dict[key] for key in self.comet_keys if key in full_dict}
    #     comet_dict['z'] = self.z
    #     return comet_dict
    
    def get_comet_dict(self, full_dict):
        """Extracts and batches parameters for the COMET emulator."""
        comet_dict = {}
        # Map shared cosmology
        for name, p in self.cosmo_params.items():
            if name in full_dict:
                comet_dict[p.base_name] = np.array([full_dict[name]]* self.nz) if self.nz > 1 else full_dict[name]

        # Map batched nuisance parameters (arrays)
        for param_group in [self.bias_params, self.counterterm_params, self.stochastic_params, self.extra_params]:
            base_names = list(set(p.base_name for p in param_group.values()))
            for base in base_names:
                if self.nz > 1:
                    val = [full_dict[f"{base}_{i}"] for i in range(self.nz) if f"{base}_{i}" in full_dict]
                    if len(val) == self.nz:
                        comet_dict[base] = np.array(val)
                else:
                    val = full_dict.get(base, None)
                    if val is not None:
                        comet_dict[base] = val
                    
        comet_dict['z'] = np.array(self.z_array) if self.nz > 1 else self.z_array[0]
        return comet_dict
    
    def cosmo_dict(self, full_dict):
        cosmo_dict = {key: np.array([full_dict[key]]*self.nz) for key in self.cosmo_params if key in full_dict}
        cosmo_dict['z'] = np.array(self.z_array)
        return cosmo_dict

    def set_param_value(self, name, value):
        if name in self.parameters:
            self.parameters[name].value = value
        else:
            raise KeyError(f"Parameter {name} not found.")

    def set_and_fix_param(self, name, value):
        if name in self.parameters:
            self.parameters[name].value = value
            self.parameters[name].fixed = True # This change is now immediately reflected in properties
            self.parameters[name].derived = False # Ensure it's not treated as derived
        else:
            raise KeyError(f"Parameter {name} not found.")
            
    def free_param(self, name):
        """Helper to un-fix a parameter if needed"""
        if name in self.parameters:
            self.parameters[name].fixed = False
            self.parameters[name].derived = False # Ensure it's not treated as derived
        else:
            raise KeyError(f"Parameter {name} not found.")
    
    def update_prior(self, name, prior, prior_type="uniform"):
        """Helper to update the prior of a parameter"""
        if name in self.parameters:
            self.parameters[name].prior = prior
            self.parameters[name].prior_type = prior_type
        else:
            raise KeyError(f"Parameter {name} not found.")

    def update_parameter(self, name, value=None, prior=None, prior_type=None, fixed=None):
        """Helper to update multiple attributes of a parameter at once"""
        if name in self.parameters:
            param = self.parameters[name]
            if value is not None:
                param.value = value
            if prior is not None:
                param.prior = prior
            if prior_type is not None:
                param.prior_type = prior_type
            if fixed is not None:
                param.fixed = fixed
        else:
            raise KeyError(f"Parameter {name} not found.")


    def get_AP_parameters(self, basis='par_perp', iz=0):
        q_par = self.emu.H_fid[iz] / self.emu.cosmo.Hz(np.array(self.z_array))[iz]
        q_perp = self.emu.cosmo.comoving_transverse_distance(np.array(self.z_array))[iz] / self.emu.Dm_fid[iz]
        q_par = float(q_par) # Need to change this for multiz
        q_perp = float(q_perp)
        if not self.emu.use_Mpc:
            q_par *= self.emu.params['h'][0]
            q_perp *= self.emu.params['h'][0]
        if basis == 'par_perp':
            return q_par, q_perp
        elif basis == 'iso_ap':
            q_iso = (q_par * q_perp**2)**(1/3)
            q_ap = q_par / q_perp
            return q_iso, q_ap

    def get_qiso3(self, p, iz=0):
        q_iso = (self.get_AP_parameters(basis='iso_ap', iz=iz)[0])**3
        return q_iso
    
    def get_sigma_12(self, p, iz=0):
        s12_out = self.emu.params['s12']
        return s12_out[iz] if isinstance(s12_out, (list, np.ndarray)) else s12_out
    
    def get_sigma_R(self, p, R, iz=0):
        if not hasattr(self, '_sigmaR_cache'):
            self._sigmaR_cache = {}
            
        cosmo_dict = self.cosmo_dict(p)
        cosmo_tup = tuple((k, tuple(v) if isinstance(v, (list, np.ndarray)) else v) for k, v in sorted(cosmo_dict.items()))
        
        # Cache the full array/scalar independent of iz so multiz evaluates it only once
        cache_key = (R, cosmo_tup)
        
        if cache_key in self._sigmaR_cache:
            sR_out = self._sigmaR_cache[cache_key]
        else:
            comet_dict = self.get_comet_dict(p)
            sR_out = self.emu.sigmaR(R, comet_dict, de_model=self.de_model)
            self._sigmaR_cache[cache_key] = sR_out
            
        val = sR_out[iz] if isinstance(sR_out, (list, np.ndarray)) else sR_out
        return val

    def get_sigma_8(self, p, iz=0):
        h = np.asarray(p.get('h', self.emu.params['h'])).flat[0]
        R = 8.0 if not self.emu.use_Mpc else 8.0 / h
        return self.get_sigma_R(p, R=R, iz=iz)

    def _sigma_reparam_mode(self):
        for mode in (self.reparam_bias_mode, self.reparam_counterterms_mode, self.reparam_stochastic_mode):
            if mode is None:
                continue
            if 'sigma_8' in mode:
                return 'sigma_8'
            if 'sigma_12' in mode:
                return 'sigma_12'
        return None

    def _sigma_reparam_name(self, iz=0):
        sigma_mode = self._sigma_reparam_mode()
        if sigma_mode is None:
            return None
        s_iz = f"_{iz}" if self.nz > 1 else ""
        return f"{sigma_mode}{s_iz}"

    def _sigma_reparam_factor(self, p, power, iz=0):
        sigma_name = self._sigma_reparam_name(iz)
        if sigma_name is None:
            return 1.0
        return self.sigmaR_ref[iz] ** power / p[sigma_name] ** power

    def get_base_names(self, param_dict):
            if len(self.z_array) > 1:
                return list(set("_".join(k.split("_")[:-1]) for k in param_dict.keys())) 
            else:
                return list(param_dict.keys())

    def _reparam_bias_factor(self, p, name, iz=0):
        factor_ap = 1.0
        s_iz = f"_{iz}" if self.nz > 1 else ""
        if 'ap' in self.reparam_bias_mode:
            factor_ap = np.sqrt(p[f'q_iso3{s_iz}'])
        factor_sigmaR = self._sigma_reparam_factor(p, power=1, iz=iz)

        if name == f'b1_r{s_iz}':
            return factor_sigmaR * factor_ap
        if name in [f'b2_r{s_iz}', f'b2t_r{s_iz}', f'b2d_r{s_iz}', f'g2_r{s_iz}', f'bk2_r{s_iz}', f'bG2_r{s_iz}']:
            return factor_sigmaR**2 * factor_ap
        if name in [f'g21_r{s_iz}', f'bGam3_r{s_iz}', f'btd_r{s_iz}']:
            if self.reparam_3ordbias_power == 3.0:
                return factor_sigmaR**3 * factor_ap
            elif self.reparam_3ordbias_power == 4.0:
                return factor_sigmaR**4 * factor_ap**2

    def _reparam_counterterm_factor(self, p, iz=0):
        factor = 1.0
        s_iz = f"_{iz}" if self.nz > 1 else ""
        if 'ap' in self.reparam_counterterms_mode:
            factor *= p[f'q_iso3{s_iz}']
        factor *= self._sigma_reparam_factor(p, power=2, iz=iz)
        return factor

    # def _reparam_stochastic_factor(self, p, iz=0):
    #     factor = 1.0
    #     s_iz = f"_{iz}" if self.nz > 1 else ""
    #     if 'ap' in self.reparam_stochastic_mode:
    #         factor *= p[f'q_iso3{s_iz}']
    #     return factor

    def _reparam_stochastic_factor(self, p, name, iz=0):
        factor = 1.0
        s_iz = f"_{iz}" if self.nz > 1 else ""
        if 'ap' in self.reparam_stochastic_mode:
            if name == f'NB0_r{s_iz}':
                factor *= p[f'q_iso3{s_iz}']**2
            # elif name == f'MB0_r{s_iz}':
            #     factor *= p[f'q_iso3{s_iz}']**(3/2)
            else:
                factor *= p[f'q_iso3{s_iz}']
        return factor
        
    def get_reparam_factor(self, p, name):
        #remove '_r' from the name to get the base parameter name and the redshift index
        if self.nz > 1:
            base_name = name.rsplit('_', 2)[0]
            iz = int(name.split('_')[-1])
            base_name_iz = f"{base_name}_{iz}"
        else:
            base_name = name.replace('_r', '')
            iz = 0
            base_name_iz = base_name
            
        if base_name_iz in self.bias_params:
            return self._reparam_bias_factor(p, name, iz=iz)
        elif base_name_iz in self.counterterm_params:
            return self._reparam_counterterm_factor(p, iz=iz)
        elif base_name_iz in self.stochastic_params:
            return self._reparam_stochastic_factor(p, name, iz=iz)
        else:
            return None
    
    def _compute_reparam(self, p, name):
        return p[name] * self.get_reparam_factor(p, name)
    
    def _derived_from_name(self, name):
        return partial(self._compute_reparam, name=name)
        

    def use_reparametrization(self, bias_mode='ap+sigma_12', counterterms_mode='ap+sigma_12',
                              stochastic_mode='ap', third_oder_bias_power=4.0, sigmaR_ref=1.0,
                              bias_linear_only=False):

        # verify that the specified modes are valid
        valid_modes = ['ap', 'sigma_12', 'ap+sigma_12', 'none']
        valid_modes += ['sigma_8', 'ap+sigma_8']
        if bias_mode not in valid_modes:
            raise ValueError(f"Invalid bias_mode {bias_mode}. Must be one of {valid_modes}.")
        if counterterms_mode not in valid_modes:
            raise ValueError(f"Invalid counterterms_mode {counterterms_mode}. Must be one of {valid_modes}.")
        if stochastic_mode not in valid_modes:
            raise ValueError(f"Invalid stochastic_mode {stochastic_mode}. Must be one of {valid_modes}.") 
        self.use_reparam = True
        self.reparam_bias_mode = bias_mode
        self.reparam_counterterms_mode = counterterms_mode
        self.reparam_stochastic_mode = stochastic_mode
        self.reparam_3ordbias_power = third_oder_bias_power
        self.sigmaR_ref = sigmaR_ref if isinstance(sigmaR_ref, (list, np.ndarray)) else [sigmaR_ref]*self.nz

        require_ap = 'ap' in bias_mode or 'ap' in counterterms_mode or 'ap' in stochastic_mode
        require_sigma_12 = 'sigma_12' in bias_mode or 'sigma_12' in counterterms_mode or 'sigma_12' in stochastic_mode
        reparam_counterterms = counterterms_mode in ['ap', 'sigma_12', 'ap+sigma_12', 'sigma_8', 'ap+sigma_8']
        reparam_bias = bias_mode in ['ap', 'sigma_12', 'ap+sigma_12', 'sigma_8', 'ap+sigma_8']
        reparam_stochastic = stochastic_mode in ['ap']

        require_sigma_8 = 'sigma_8' in bias_mode or 'sigma_8' in counterterms_mode or 'sigma_8' in stochastic_mode
        if require_sigma_12 and require_sigma_8:
            raise ValueError("Cannot use both sigma_12 and sigma_8 reparametrization at the same time.")
        require_sigmaR = require_sigma_12 or require_sigma_8 

        if require_ap:
            for iz in range(self.nz):
                name = f'q_iso3_{iz}' if self.nz > 1 else 'q_iso3'
                latex = add_iz_to_latex(r"q_{\rm iso}^3", iz) if self.nz > 1 else r"q_{\rm iso}^3"
                self.set_derived_param(name, partial(self.get_qiso3, iz=iz), requires_emu_eval=True, latex=latex, exported=True)

        if require_sigma_12:
            for iz in range(self.nz):
                name = f'sigma_12_{iz}' if self.nz > 1 else 'sigma_12'
                latex = add_iz_to_latex(r"\sigma_{12}", iz) if self.nz > 1 else r"\sigma_{12}"
                self.set_derived_param(name, partial(self.get_sigma_12, iz=iz), requires_emu_eval=True, latex=latex, exported=True)
        if require_sigma_8:
            for iz in range(self.nz):
                name = f'sigma_8_{iz}' if self.nz > 1 else 'sigma_8'
                latex = add_iz_to_latex(r"\sigma_{8}", iz) if self.nz > 1 else r"\sigma_{8}"
                self.set_derived_param(name, partial(self.get_sigma_8, iz=iz), requires_emu_eval=True, latex=latex, exported=True)

        if reparam_counterterms: 
            for base in self.get_base_names(self.counterterm_params):
                for iz in range(self.nz):
                    name_reparam = f"{base}_r_{iz}" if self.nz > 1 else f"{base}_r"
                    target_name = f"{base}_{iz}" if self.nz > 1 else base
                    latex_reparam = add_tilde_to_latex(self.parameters[target_name].latex)
                    self.add_sampled_param(name_reparam, value=0.0, prior=(0, 500), prior_type="gaussian", latex=latex_reparam)
                    self.set_derived_param(target_name, self._derived_from_name(name_reparam), latex=self.parameters[target_name].latex, exported=True)

        if reparam_bias:
            for base in self.get_base_names(self.bias_params):
                if bias_linear_only:
                    if base not in ['g21', 'bGam3', 'btd', 'btdt']:
                        continue
                for iz in range(self.nz):
                    name_reparam = f"{base}_r_{iz}" if self.nz > 1 else f"{base}_r"
                    target_name = f"{base}_{iz}" if self.nz > 1 else base
                    latex_reparam = add_tilde_to_latex(self.parameters[target_name].latex)
                    if base == 'b1':
                        prior_type = 'uniform'
                        prior = (0.5, 4.0)
                    else:
                        prior_type = 'gaussian'
                        prior = (0, 20)

                    self.add_sampled_param(name_reparam, value=0.0, prior=prior, prior_type=prior_type, latex=latex_reparam)
                    self.set_derived_param(target_name, self._derived_from_name(name_reparam), latex=self.parameters[target_name].latex, exported=True)
            
        if reparam_stochastic:
            for base in self.get_base_names(self.stochastic_params):
                for iz in range(self.nz):
                    name_reparam = f"{base}_r_{iz}" if self.nz > 1 else f"{base}_r"
                    target_name = f"{base}_{iz}" if self.nz > 1 else base
                    latex_reparam = add_tilde_to_latex(self.parameters[target_name].latex)
                    self.add_sampled_param(name_reparam, value=0.0, prior=(-1, 1), prior_type="uniform", latex=latex_reparam)
                    self.set_derived_param(target_name, self._derived_from_name(name_reparam), latex=self.parameters[target_name].latex, exported=True)

def add_tilde_to_latex(latex_str):
    # separate base from the rest of the string
    base = ''
    rest = ''
    i = 0
    while i < len(latex_str):
        if latex_str[i] in ['^', '_', '{', '(', '[']:
            rest = latex_str[i:]
            break
        else:
            base += latex_str[i]
        i += 1
    return r"\tilde{" + base + "}" + rest

def add_iz_to_latex(latex_str, iz):
    # separate base from the rest of the string
    zstring = f"(z_{{{iz}}})"
    return latex_str + zstring