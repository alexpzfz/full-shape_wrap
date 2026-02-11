import numpy as np
import dataclasses
from typing import Callable, Optional, Union

@dataclasses.dataclass
class Parameter:
    name: str
    value: float
    prior: tuple = None  # (min, max)
    prior_type: str = "uniform"  # 'uniform' or 'gaussian'
    fixed: bool = False
    derived: bool = False  # True if this parameter is derived from others (e.g. co-evolution)
    derived_func: Optional[Union[Callable, str]] = None  # Function to compute derived parameter, or name from emu params dict
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
    Parameter(name="bG2", value=0.0, prior=(-5.0, 5.0), prior_type="uniform", fixed=True, latex=r"b_{G2}"),
    Parameter(name="bGam3", value=0.0, prior=(-5.0, 5.0), prior_type="uniform", fixed=True, latex=r"b_{\Gamma_{3}}"),],

    "DesJeoSch": [
    Parameter(name="b1", value=1.0, prior=(0.1, 8.0), prior_type="uniform", fixed=False, latex=r"b_1"),
    Parameter(name="b2", value=0.0, prior=(0, 20), prior_type="gaussian", fixed=False, latex=r"b_2"),
    Parameter(name="bK2", value=0.0, prior=(0, 20), prior_type="gaussian", fixed=False, latex=r"b_{K^2}"),
    Parameter(name="btd", value=0.0, prior=(0, 80), prior_type="gaussian", fixed=False, latex=r"b_{\rm td}"),
]}

_damping_params = [Parameter(name="avir", value=5.0, prior=(0.0, 10.0), prior_type="uniform", fixed=True, latex=r"a_{\rm vir}")]

_counterterm_params = {"Comet": [   
    Parameter(name="c0", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"c_0"),
    Parameter(name="c2", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"c_2"),
    Parameter(name="c4", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"c_4"),],
    
    "DESI_DR2": [
    Parameter(name="a0", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"\alpha_0"),
    Parameter(name="a2", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"\alpha_2"),
    Parameter(name="a4", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"\alpha_4"),]}

_stochastic_params = [
    Parameter(name="NP0", value=0.0, prior=(-1., 3.), prior_type="uniform", fixed=True, latex=r"N_{P,0}"),
    Parameter(name="NP20", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"N_{P,2}"),
    Parameter(name="NP22", value=0.0, prior=(-1e4, 1e4), prior_type="uniform", fixed=True, latex=r"N_{P,22}"),
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
        self.sigmaR_ref = None  # Only relevant for DESI_DR2 bias basis
        self.derived_order = []
        self.z = None # Placeholder for redshift, can be set externally if needed for derived parameters
        

    def set_reference_sigmaR(self, sigmaR):
        assert 'DESI_DR2' in self.emu.bias_basis, "Reference sigmaR is only relevant for DESI_DR2 bias basis."
        self.sigmaR_ref = sigmaR

    def add_sampled_param(self, name, value, prior, prior_type="uniform", latex=""):
        """Helper to add a new sampled parameter on the fly"""
        if name in self.parameters:
            raise KeyError(f"Parameter {name} already exists.")
        new_param = Parameter(name=name, value=value, prior=prior, prior_type=prior_type, fixed=False, derived=False, latex=latex)
        self.parameters[name] = new_param

    def set_derived_param(self, name, deriv_func, latex="", exported=False):
        if name not in self.parameters:
            self.parameters[name] = Parameter(name=name, value=None, prior=None, prior_type=None, fixed=True, derived=True, derived_func=deriv_func, latex=latex, exported=exported)
        
        self.parameters[name].derived = True
        self.parameters[name].fixed = False
        self.parameters[name].derived_func = deriv_func
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
        return [name for name, p in self.parameters.items() if p.derived and p.exported]
        

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
                if callable(param.derived_func):
                    full_dict[name] = param.derived_func(full_dict)
                elif isinstance(param.derived_func, str):
                    if not plin_evaluated:
                        cosmo_dict = self.cosmo_dict(full_dict)
                        self.emu.PL(0.1, cosmo_dict, de_model=self.de_model)  # Ensure PLin is evaluated for current cosmology
                        plin_evaluated = True
                    full_dict[name] = float(self.emu.params[param.derived_func]) # This will only work when 1 dataset is used, need to fix this in the future
                else:
                    raise ValueError(f"Invalid derived_func for {name}. Must be callable or string key.")


        if self.emu.bias_basis == "DESI_DR2" and self.sigmaR_ref is not None:
            full_dict['b1t'] = full_dict['b1t'] * self.sigmaR_ref
            full_dict['b2t'] = full_dict['b2t'] * self.sigmaR_ref**2
            full_dict['bK2t'] = full_dict['bK2t'] * self.sigmaR_ref**2
            full_dict['btdt'] = full_dict['btdt'] * self.sigmaR_ref**4
        if self.emu.counterterm_basis == "DESI_DR2" and self.sigmaR_ref is not None:
            full_dict['a0'] = full_dict['a0'] * self.sigmaR_ref**2
            full_dict['a2'] = full_dict['a2'] * self.sigmaR_ref**2
            full_dict['a4'] = full_dict['a4'] * self.sigmaR_ref**2
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


    def use_reparametrization(self, reparam_bias=False, reparam_counterterms=True,
                              reparem_stochastic=False, mode='ap', sigma_ref=1.0):
        if 'ap' in mode:
            self.set_derived_param('q_par', 'q_lo', latex=r"q_{\parallel}", exported=False)
            self.set_derived_param('q_perp', 'q_tr', latex=r"q_{\perp}", exported=False)
            self.set_derived_param('q_iso3', lambda p: p['q_par'] * p['q_perp']**2, latex=r"q_{\rm iso}^3", exported=True)
        
        if 'sigma_12' in mode:
            self.set_derived_param('sigma_12', 's12', latex=r"\sigma_{12}", exported=True)

        if reparam_counterterms:
            def reparam_counterterm_func(p, name):
                factor = 1.0
                if 'ap' in mode:
                    factor *= p['q_iso3']
                if 'sigma_12' in mode:
                    factor *= sigma_ref**2 / p['sigma_12']**2
                return p[name] * factor
                

            for name in self.counterterm_params.keys():
                name_reparam = name + '_r'
                latex_reparam = self.parameters[name].latex + "_r"
                self.add_sampled_param(name_reparam, value=0.0, prior=(0, 500), prior_type="gaussian", latex=latex_reparam)
                self.set_derived_param(name, lambda p, n=name_reparam: reparam_counterterm_func(p, n), latex=self.parameters[name].latex, exported=True)

        if reparam_bias:
            def reparam_bias_func(p, name):
                factor_ap = 1.0
                factor_sigmaR = 1.0
                if 'ap' in mode:
                    factor_ap = np.sqrt(p['q_iso3'])
                if 'sigma_12' in mode:
                    factor_sigmaR = sigma_ref / p['sigma_12']

                if name == 'b1':
                    return p[name] * factor_sigmaR * factor_ap
                if name in ['b2', 'b2t', 'g2', 'bK2', 'bG2']:
                    return p[name] * factor_sigmaR**2 * factor_ap
                if name in ['g21', 'bGam3', 'btd']:
                    return p[name] * factor_sigmaR**4 * factor_ap**2

            for name in self.bias_params.keys():
                name_reparam = name + '_r'
                latex_reparam = self.parameters[name].latex + "_r"
                if name == 'b1':
                    prior_type = 'uniform'
                    prior = (0.5, 4.0)
                else:
                    prior_type = 'gaussian'
                    prior = (0, 20)

                self.add_sampled_param(name_reparam, value=0.0, prior=prior, prior_type=prior_type, latex=latex_reparam)
                self.set_derived_param(name, lambda p, n=name_reparam: reparam_bias_func(p, n), latex=self.parameters[name].latex, exported=True)