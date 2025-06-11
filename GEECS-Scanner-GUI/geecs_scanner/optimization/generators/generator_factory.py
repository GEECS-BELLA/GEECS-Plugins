# optimization/generator_factory.py
from xopt.vocs import VOCS
from xopt.generators.random import RandomGenerator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor

from typing import Any, Dict

import torch
from cheetah import ParameterBeam, ParticleBeam, Segment

import sys
# sys.path.append('../2025_pilot_digital_twin_LDRD/simulation_scripts')
sys.path.append(r'C:\Users\loasis.LOASIS\Documents\GitHub\2025_pilot_digital_twin_LDRD\simulation_scripts')
from htu_lattice import get_lattice, current_to_k

# # Optional: Add your own priors here
# from custom_priors import CheetahPrior  # optional

# Explicitly defined generators dictionary
PREDEFINED_GENERATORS = {
    "random": lambda vocs: RandomGenerator(vocs=vocs),

    "bayes_default": lambda vocs: ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=StandardModelConstructor(use_low_noise_prior=False)
    ),

    "bayes_cheetah": lambda vocs: ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor = StandardModelConstructor(
        mean_modules={"f": CheetahPrior()},
        )
    )
    # Add more explicit named generators here if needed
}

def build_generator_from_config(config: Dict[str, Any], vocs: VOCS):
    """
    Build and return an Xopt generator instance based on the predefined name.

    Args:
        config (Dict[str, Any]): Config dictionary with at least a 'name' field.
        vocs (VOCS): The VOCS instance to pass to the generator.

    Returns:
        Generator: An instance of the corresponding generator.

    Raises:
        ValueError: If the specified generator name is not recognized.
    """
    generator_name = config['name']
    try:
        return PREDEFINED_GENERATORS[generator_name](vocs)
    except KeyError:
        raise ValueError(f"Unsupported or undefined generator name: '{generator_name}'")

from scipy.constants import m_e, e, c

class CheetahPrior(torch.nn.Module):
    def forward(self, X):
        # Use a linear model with parameter beam as a prior mean
        # This is needed for it to be fast enough to be used inside Bayesian optimization
        segment = Segment( get_lattice("cheetah", to_element='UC_ALineEBeam3') )
        segment.EMQ1H.k1 = current_to_k(X[..., 0].to(torch.float32), "EMQD-113-394", 100e6)
        segment.EMQ2V.k1 = current_to_k(X[..., 1].to(torch.float32), "EMQD-113-949", 100e6)
        segment.EMQ3H.k1 = current_to_k(X[..., 2].to(torch.float32), "EMQD-113-394", 100e6)
        torch.manual_seed(0)

        beam_energy = torch.tensor(100e6)  # in eV
        gamma = beam_energy * e / (m_e * c ** 2)


        incoming = ParticleBeam.from_twiss(
                beta_x=torch.tensor(0.002), # in m
                alpha_x=torch.tensor(0.0),
                emittance_x=torch.tensor(1.5e-6)/gamma, # in m.rad ; geometric emittance
                beta_y=torch.tensor(0.002), # in m
                alpha_y=torch.tensor(0.0),
                emittance_y=torch.tensor(1.5e-6)/gamma, # in m.rad ; geometric emittance
                sigma_tau=torch.tensor(1e-6), # in m
                sigma_p=torch.tensor(2.5e-2), # dimensionless
                energy=beam_energy, # in eV
                total_charge=torch.tensor(25.0e-12), # in C
                num_particles=5000
        )
        segment.track( incoming=incoming )
        beam = segment.UC_ALineEBeam3.get_read_beam()
        matching = (1e3*beam.sigma_x)**2 + (1e3*beam.sigma_y)**2
        return matching


