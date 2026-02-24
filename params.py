import numpy as np
import dataclasses
from typing import Callable, Optional, Union
from functools import partial

@dataclasses.dataclass
class Parameter:
    name: str
    value: float
    prior: tuple = None  # (min, max)
    prior_type: str = "uniform"  # 'uniform' or 'gaussian'
    fixed: bool = False
    derived: bool = False  # True if this parameter is derived from others (e.g. co-evolution)
    derived_func: Optional[Union[Callable, str]] = None  # Function to compute derived parameter, or name from emu params dict
    requires_emu_eval: bool = False # Whether this derived parameter requires an emulator evaluation (e.g. depends on PLin)
    derived_am: bool = False # Only relevant for analytical marginalization
    exported: bool = False # Only relevant if derived=True, whether to include this parameter in the output samples
    latex: str = ""


_cosmo_params = [
    Parameter(name="wc", value=0.12, prior=(0.085, 0.155), prior_type="uniform", fixed=False, latex=r"\omega_c"),
    Parameter(name="wb", value=0.022, prior=(0.0205, 0.02415), prior_type="uniform", fixed=False, latex=r"\omega_b"),
    Parameter(name="h", value=0.67, prior=(0.55, 0.85), prior_type="uniform", fixed=False, latex=r"h"),
    Parameter(name="ns", value=0.965, prior=(0.92, 1.01), prior_type="uniform", fixed=False, latex=r"n_s"),
    Parameter(name="As", value=2.1, prior=(1., 3.), prior_type="uniform", fixed=False, latex=r"10^9 A_s"),
    Parameter(name="Mnu", value=0.0, prior=(0.0, 0.5), prior_type="uniform", fixed=True, latex=r"\sum m_\nu"),
    Parameter(name="w0", value=-1.0, prior=(-2.0, -0.33), prior_type="uniform", fixed=True, latex=r"w_0"),
    Parameter(name="wa", value=0.0, prior=(-2.0, 2.0), prior_type="uniform", fixed=True, latex=r"w_a"),
    Parameter(name="Ok", value=0.0, prior=(-0.1, 0.1), prior_type="uniform", fixed=True, latex=r"\Omega_k"),
]

_bias_params = {"EggScoSmi": [
    Parameter(name="b1", value=1.0, prior=(0.5, 4.0), prior_type="uniform", fixed=False, latex=r"b_1"),
    Parameter(name="b2", value=0.0, prior=(-2.0, 2.0), prior_type="uniform", fixed=False, latex=r"b_2"),
    Parameter(name='g2', value=0.0, prior=(-5.0, 5.0), prior_type="uniform", fixed=True, latex=r"\gamma_2"),
    Parameter(name='g21', value=0.0, prior=(-5.0, 5.0), prior_type="uniform", fixed=True, latex=r"\gamma_{21}"),],
    
    "AssBauGre": [
    Parameter(name="b1", value=1.0, prior=(0.5, 4.0), prior_type="uniform", fixed=False, latex=r"b_1"),
    Parameter(name="b2", value=0.0, prior=(-2.0, 2.0), prior_type="uniform", fixed=False, latex=r"b_2"),
    Parameter(name="bG2", value=0.0, prior=(-5.0, 5.0), prior_type="uniform", fixed=True, latex=r"b_{G_{2}}"),
    Parameter(name="bGam3", value=0.0, prior=(-5.0, 5.0), prior_type="uniform", fixed=True, latex=r"b_{\Gamma_{3}}"),],

    "DesJeoSch": [
    Parameter(name="b1", value=1.0, prior=(0.1, 8.0), prior_type="uniform", fixed=False, latex=r"b_1"),
    Parameter(name="b2t", value=0.0, prior=(0, 20), prior_type="gaussian", fixed=False, latex=r"b_2"),
    Parameter(name="bK2", value=0.0, prior=(0, 20), prior_type="gaussian", fixed=False, latex=r"b_{K^2}"),
    Parameter(name="btd", value=0.0, prior=(0, 80), prior_type="gaussian", fixed=False, latex=r"b_{\rm td}"),]}

_bias_params["DesJeoSch_r"] = [Parameter(name=p.name + "r", value=p.value, prior=p.prior, prior_type=p.prior_type, fixed=p.fixed, latex=p.latex + "^r") for p in _bias_params["DesJeoSch"]]

_damping_params = [Parameter(name="avir", value=5.0, prior=(0.0, 10.0), prior_type="uniform", fixed=True, latex=r"a_{\rm vir}")]

_counterterm_params = {"Comet": [   
    Parameter(name="c0", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"c_0"),
    Parameter(name="c2", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"c_2"),
    Parameter(name="c4", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"c_4"),],
    
    "DESI": [
    Parameter(name="a0", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"\alpha_0"),
    Parameter(name="a2", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"\alpha_2"),
    Parameter(name="a4", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"\alpha_4"),]}
_counterterm_params["DESI_r"] = [Parameter(name=p.name + "r", value=p.value, prior=p.prior, prior_type=p.prior_type, fixed=p.fixed, latex=p.latex + "^r") for p in _counterterm_params["DESI"]]

_stochastic_params = [
    Parameter(name="NP0", value=0.0, prior=(-1., 3.), prior_type="uniform", fixed=True, latex=r"N^P_0"),
    Parameter(name="NP20", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"N^P_{2,0}"),
    Parameter(name="NP22", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"N^P_{2,2}"),
]



class Params:
    """Class to handle model parameters"""
    def __init__(self, emu, coev_params=None):
        self.emu = emu
        self.cosmo_params = {param.name: param for param in _cosmo_params}
        self.bias_params = {param.name: param for param in _bias_params[emu.bias_basis]}
        self.damping_params = {param.name: param for param in _damping_params}
        self.counterterm_params = {param.name: param for param in _counterterm_params[emu.counterterm_basis]}
        self.stochastic_params = {param.name: param for param in _stochastic_params}
        self.parameters = {**self.cosmo_params, **self.bias_params, **self.damping_params, **self.counterterm_params, **self.stochastic_params}
        self.comet_keys = [p.name for p in self.parameters.values()]
        self.coev_params = None
        if coev_params is not None:
            if not isinstance(coev_params, list):
                coev_params = [coev_params]
            self.coev_params = coev_params
            for name in self.coev_params:
                assert name in self.bias_params, f"Co-evolution parameter {name} not recognized in bias parameters."
                if name == "bG2": self.set_derived_param(name, self.bG2_coev)
                elif name == "bGam3": self.set_derived_param(name, self.bGam3_coev)
                elif name == "bK2" or name == "bK2t": self.set_derived_param(name, self.bK2_coev)
                elif name == "btd" or name == "btdt": self.set_derived_param(name, self.btd_coev)
                else:
                    raise ValueError(f"Co-evolution for {name} not implemented.")
        self.derived_order = []
        self.z = None # Placeholder for redshift, can be set externally if needed for derived parameters
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
    def fixed_cosmo(self):
        """Returns True if all cosmological parameters are fixed, False otherwise"""
        return all(self.parameters[name].fixed for name in self.cosmo_params)
    
    @staticmethod
    def bG2_coev(p):
        b1 = p["b1"]
        return 0.524 - 0.547*b1 + 0.046*b1**2

    @staticmethod
    def bGam3_coev(p):
        b1 = p["b1"]
        bG2 = p["bG2"]
        return -1./6.*(b1-1.) -5./2.*bG2

    @staticmethod
    def bK2_coev(p):
        b1 = p["b1"]
        return -2./7.*(b1 - 1.)
    
    @staticmethod
    def btd_coev(p):
        b1 = p["b1"]
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
                        full_dict[name] = float(self.emu.params[param.derived_func]) # This will only work when 1 dataset is used, need to fix this in the future
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
    
    def get_comet_dict(self, full_dict):
        """Extracts the parameters needed for the comet emulator from the full dict, including derived parameters"""
        comet_dict = {key: full_dict[key] for key in self.comet_keys if key in full_dict}
        comet_dict['z'] = self.z
        return comet_dict
    
    def cosmo_dict(self, full_dict):
        cosmo_dict = {key: full_dict[key] for key in self.cosmo_params if key in full_dict}
        cosmo_dict['z'] = self.z
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


    def get_AP_parameters(self, basis='par_perp'):
        q_par = self.emu.H_fid / self.emu.cosmo.Hz(np.array([self.z]))
        q_perp = self.emu.cosmo.comoving_transverse_distance(np.array([self.z])) / self.emu.Dm_fid
        q_par = float(q_par) # Need to change this for multiz
        q_perp = float(q_perp)
        if basis == 'par_perp':
            return q_par, q_perp
        elif basis == 'iso_ap':
            q_iso = (q_par * q_perp**2)**(1/3)
            q_ap = q_par / q_perp
            return q_iso, q_ap

    def get_qiso3(self, p):
        q_iso = (self.get_AP_parameters(basis='iso_ap')[0])**3
        return q_iso
    
    def _reparam_bias_factor(self, p, name):
        factor_ap = 1.0
        factor_sigmaR = 1.0
        if 'ap' in self.reparam_bias_mode:
            factor_ap = np.sqrt(p['q_iso3'])
        if 'sigma_12' in self.reparam_bias_mode:
            factor_sigmaR = self.sigmaR_ref / p['sigma_12']

        if name == 'b1_r':
            return factor_sigmaR * factor_ap
        if name in ['b2_r', 'b2t_r', 'g2_r', 'bK2_r', 'bG2_r']:
            return factor_sigmaR**2 * factor_ap
        if name in ['g21_r', 'bGam3_r', 'btd_r']:
            if self.reparam_3ordbias_power == 3.0:
                return factor_sigmaR**3 * factor_ap
            elif self.reparam_3ordbias_power == 4.0:
                return factor_sigmaR**4 * factor_ap**2

    def _reparam_counterterm_factor(self, p):
        factor = 1.0
        if 'ap' in self.reparam_counterterms_mode:
            factor *= p['q_iso3']
        if 'sigma_12' in self.reparam_counterterms_mode:
            factor *= self.sigmaR_ref**2 / p['sigma_12']**2
        return factor

    def _reparam_stochastic_factor(self, p):
        factor = 1.0
        if 'ap' in self.reparam_stochastic_mode:
            factor *= p['q_iso3']
        return factor
        
    def get_reparam_factor(self, p, name):
        base_name = name.replace('_r', '')
        if base_name in self.bias_params:
            return self._reparam_bias_factor(p, name)
        elif base_name in self.counterterm_params:
            return self._reparam_counterterm_factor(p)
        elif base_name in self.stochastic_params:
            return self._reparam_stochastic_factor(p)
        else:
            return 1.0
    
    def _compute_reparam(self, p, name):
        return p[name] * self.get_reparam_factor(p, name)
    
    def _derived_from_name(self, name):
        return partial(self._compute_reparam, name=name)
        

    def use_reparametrization(self, bias_mode='ap+sigma_12', counterterms_mode='ap+sigma_12',
                              stochastic_mode='ap', third_oder_bias_power=3.0, sigmaR_ref=1.0):

        # verify that the specified modes are valid
        valid_modes = ['ap', 'sigma_12', 'ap+sigma_12', 'none']
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
        self.sigmaR_ref = sigmaR_ref

        require_ap = 'ap' in bias_mode or 'ap' in counterterms_mode or 'ap' in stochastic_mode
        require_sigma_12 = 'sigma_12' in bias_mode or 'sigma_12' in counterterms_mode or 'sigma_12' in stochastic_mode
        reparam_counterterms = counterterms_mode in ['ap', 'sigma_12', 'ap+sigma_12']
        reparam_bias = bias_mode in ['ap', 'sigma_12', 'ap+sigma_12']
        reparam_stochastic = stochastic_mode in ['ap'] # no sigma_12 required for shot noise

        if require_ap:
            self.set_derived_param('q_iso3', self.get_qiso3, requires_emu_eval=True, latex=r"q_{\rm iso}^3", exported=True)
        
        if require_sigma_12:
            self.set_derived_param('sigma_12', 's12', requires_emu_eval=True, latex=r"\sigma_{12}", exported=True)

        if reparam_counterterms: 
            for name in self.counterterm_params.keys():
                name_reparam = name + '_r'
                latex_reparam = add_tilde_to_latex(self.parameters[name].latex)
                self.add_sampled_param(name_reparam, value=0.0, prior=(0, 500), prior_type="gaussian", latex=latex_reparam)
                self.set_derived_param(name, self._derived_from_name(name_reparam), latex=self.parameters[name].latex, exported=True)

        if reparam_bias:
            for name in self.bias_params.keys():
                name_reparam = name + '_r'
                latex_reparam = add_tilde_to_latex(self.parameters[name].latex)
                if name == 'b1':
                    prior_type = 'uniform'
                    prior = (0.5, 4.0)
                else:
                    prior_type = 'gaussian'
                    prior = (0, 20)

                self.add_sampled_param(name_reparam, value=0.0, prior=prior, prior_type=prior_type, latex=latex_reparam)
                self.set_derived_param(name, self._derived_from_name(name_reparam), latex=self.parameters[name].latex, exported=True)
            
        if reparam_stochastic:
            for name in self.stochastic_params.keys():
                name_reparam = name + '_r'
                latex_reparam = add_tilde_to_latex(self.parameters[name].latex)
                self.add_sampled_param(name_reparam, value=0.0, prior=(-1e4, 1e4), prior_type="uniform", latex=latex_reparam)
                self.set_derived_param(name, self._derived_from_name(name_reparam), latex=self.parameters[name].latex, exported=True)

def add_tilde_to_latex(latex_str):
    # separate base from the rest of the string
    base = ''
    rest = ''
    i = 0
    while i < len(latex_str):
        if latex_str[i] in ['^', '_']:
            rest = latex_str[i:]
            break
        else:
            base += latex_str[i]
        i += 1
    return r"\tilde{" + base + "}" + rest