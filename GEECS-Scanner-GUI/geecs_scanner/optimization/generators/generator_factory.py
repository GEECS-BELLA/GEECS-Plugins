# optimization/generator_factory.py
from xopt.vocs import VOCS
from xopt.generators.random import RandomGenerator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor

from typing import Any, Dict

# # Optional: Add your own priors here
# from custom_priors import CheetahPrior  # optional

# Explicitly defined generators dictionary
PREDEFINED_GENERATORS = {
    "random": lambda vocs: RandomGenerator(vocs=vocs),

    "bayes_default": lambda vocs: ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=StandardModelConstructor(use_low_noise_prior=False)
    ),

    # "bayes_cheetah": lambda vocs: ExpectedImprovementGenerator(
    #     vocs=vocs,
    #     gp_constructor=StandardModelConstructor(
    #         mean_modules={"f": CheetahPrior()},
    #         use_low_noise_prior=True
    #     )
    # ),

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
