import numpy as np
import dataclasses

@dataclasses.dataclass
class Parameter:
    name: str
    value: float
    prior: tuple = None  # (min, max)
    prior_type: str = "uniform"  # 'uniform' or 'gaussian'
    fixed: bool = False
    latex: str = ""


_cosmo_params = [
    Parameter(name="wc", value=0.12, prior=(0.085, 0.155), prior_type="uniform", fixed=False, latex=r"\omega_c"),
    Parameter(name="wb", value=0.022, prior=(0.0205, 0.02415), prior_type="uniform", fixed=False, latex=r"\omega_b"),
    Parameter(name="h", value=0.67, prior=(0.55, 0.85), prior_type="uniform", fixed=False, latex=r"h"),
    Parameter(name="ns", value=0.965, prior=(0.92, 1.01), prior_type="uniform", fixed=False, latex=r"n_s"),
    Parameter(name="As", value=2.1, prior=(1., 3.), prior_type="uniform", fixed=False, latex=r"A_s"),
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

    "DESI_DR2": [
    Parameter(name="b1t", value=1.0, prior=(0.1, 8.0), prior_type="uniform", fixed=False, latex=r"\tilde{b}_1"),
    Parameter(name="b2t", value=0.0, prior=(0, 20), prior_type="gaussian", fixed=False, latex=r"\tilde{b}_2"),
    Parameter(name="bK2t", value=0.0, prior=(0, 20), prior_type="gaussian", fixed=False, latex=r"\tilde{b}_{K^2}"),
    Parameter(name="btdt", value=0.0, prior=(0, 80), prior_type="gaussian", fixed=False, latex=r"\tilde{b}_{\rm td}"),
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
        self.coev_params = None
        if coev_params is not None:
            if not isinstance(coev_params, list):
                coev_params = [coev_params]
            self.coev_params = coev_params
            for name in self.coev_params:
                assert name in self.bias_params, f"Co-evolution parameter {name} not recognized in bias parameters."
                self.bias_params[name].fixed = True
        self.sigmaR_ref = None  # Only relevant for DESI_DR2 bias basis

    def set_reference_sigmaR(self, sigmaR):
        assert 'DESI_DR2' in self.emu.bias_basis, "Reference sigmaR is only relevant for DESI_DR2 bias basis."
        self.sigmaR_ref = sigmaR

    @property
    def free_param_names(self):
        """Dynamically get list of free parameter names"""
        return [name for name, p in self.parameters.items() if not p.fixed]

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
    def n_free_params(self):
        """Dynamically count number of free parameters"""
        return len(self.free_param_names)
    
    @property
    def fixed_cosmo(self):
        """Returns True if all cosmological parameters are fixed, False otherwise"""
        return all(self.parameters[name].fixed for name in self.cosmo_params)
    
    @staticmethod
    def bG2_coev(b1):
        return 0.524 - 0.547*b1 + 0.046*b1**2

    @staticmethod
    def bGam3_coev(b1, bG2):
        return -1./6.*(b1-1.) -5./2.*bG2

    @staticmethod
    def bK2_coev(b1):
        return -2./7.*(b1 - 1.)
    
    @staticmethod
    def btd_coev(b1):
        return 23./42.*(b1 - 1.)


    def get_free_params(self):
        """Return the actual Parameter objects for free parameters"""
        return [self.parameters[name] for name in self.free_param_names]

    def build_nautilus_prior(self):
        """
        Constructs a nautilus.Prior object based on the CURRENT free parameters.
        """
        from nautilus import Prior
        from scipy.stats import norm
        prior = Prior()
        
        # Now this iterates over the dynamic property, so it sees your updates
        for name in self.free_param_names:
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
            # Assume list/array in correct order of self.free_param_names
            current_free_names = self.free_param_names
            if len(free_values_dict_or_list) != len(current_free_names):
                raise ValueError(f"Input length {len(free_values_dict_or_list)} does not match "
                                 f"number of free params {len(current_free_names)}.")
            for name, val in zip(current_free_names, free_values_dict_or_list):
                full_dict[name] = val
                
        if self.coev_params is not None:
            for name in self.coev_params:
                if name == "bG2":
                    full_dict[name] = self.bG2_coev(full_dict["b1"])
                elif name == "bGam3":
                    full_dict[name] = self.bGam3_coev(full_dict["b1"], full_dict["bG2"])
                elif name == "bK2" or name == "bK2t":
                    full_dict[name] = self.bK2_coev(full_dict["b1"])
                elif name == "btd" or name == "btdt":
                    full_dict[name] = self.btd_coev(full_dict["b1"])
                else:
                    raise ValueError(f"Co-evolution for {name} not implemented.")
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

    def set_param_value(self, name, value):
        if name in self.parameters:
            self.parameters[name].value = value
        else:
            raise KeyError(f"Parameter {name} not found.")

    def set_and_fix_param(self, name, value):
        if name in self.parameters:
            self.parameters[name].value = value
            self.parameters[name].fixed = True # This change is now immediately reflected in properties
        else:
            raise KeyError(f"Parameter {name} not found.")
            
    def free_param(self, name):
        """Helper to un-fix a parameter if needed"""
        if name in self.parameters:
            self.parameters[name].fixed = False
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

