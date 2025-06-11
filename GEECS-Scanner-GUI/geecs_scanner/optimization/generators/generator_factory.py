# optimization/generator_factory.py
from xopt.vocs import VOCS
from xopt.generators.random import RandomGenerator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor

from typing import Any, Dict

import sys
# sys.path.append('../2025_pilot_digital_twin_LDRD/simulation_scripts')
sys.path.append(r'C:\Users\loasis.LOASIS\Documents\GitHub\2025_pilot_digital_twin_LDRD\simulation_scripts')
from htu_lattice import get_lattice, current_to_k

# Explicitly defined generators dictionary
PREDEFINED_GENERATORS = {
    "random": lambda vocs: RandomGenerator(vocs=vocs),

    "bayes_default": lambda vocs: ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=StandardModelConstructor(use_low_noise_prior=False)
    ),

    "bayes_cheetah": lambda vocs: _load_cheetah_generator(vocs)
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

def _load_cheetah_generator(vocs):
    try:
        from geecs_data_acquisition.optimization.generators.cheetah_generator import get_cheetah_generator
        return get_cheetah_generator(vocs)
    except ImportError as e:
        raise ImportError("Could not load 'bayes_cheetah' generator. Make sure 'cheetah' and dependencies are installed.") from e



