"""
Generator factory for optimization algorithms.

This module provides a factory interface for creating Xopt generator instances
used in automated parameter optimization. It supports various optimization
algorithms including random sampling, Bayesian optimization, and specialized
generators for specific use cases.

The factory pattern allows for easy configuration and instantiation of different
optimization algorithms while maintaining a consistent interface for the
optimization framework.

Functions
---------
build_generator_from_config(config, vocs)
    Create generator instance from configuration dictionary.

Constants
---------
PREDEFINED_GENERATORS : dict
    Dictionary mapping generator names to factory functions.

Examples
--------
Creating a random generator:

>>> from xopt import VOCS
>>> vocs = VOCS(variables={"x": [0, 10]}, objectives={"y": "MINIMIZE"})
>>> config = {"name": "random"}
>>> generator = build_generator_from_config(config, vocs)

Creating a Bayesian optimization generator:

>>> config = {"name": "bayes_default"}
>>> generator = build_generator_from_config(config, vocs)

Notes
-----
The factory supports the following predefined generators:
- "random": Random sampling generator
- "bayes_default": Expected improvement Bayesian optimization
- "bayes_cheetah": Cheetah-based Bayesian optimization (requires cheetah package)

New generators can be added by extending the PREDEFINED_GENERATORS dictionary
with appropriate factory functions.
"""

# optimization/generator_factory.py
from xopt.vocs import VOCS
from xopt.generators.random import RandomGenerator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor

from typing import Any, Dict

# Explicitly defined generators dictionary
PREDEFINED_GENERATORS = {
    "random": lambda vocs: RandomGenerator(vocs=vocs),
    "bayes_default": lambda vocs: ExpectedImprovementGenerator(
        vocs=vocs, gp_constructor=StandardModelConstructor(use_low_noise_prior=False)
    ),
    "bayes_cheetah": lambda vocs: _load_cheetah_generator(vocs),
    # Add more explicit named generators here if needed
}


def build_generator_from_config(config: Dict[str, Any], vocs: VOCS):
    """
    Build Xopt generator instance from configuration dictionary.

    Creates and returns an optimization generator based on the specified
    algorithm name in the configuration. This factory function provides
    a unified interface for instantiating different types of optimization
    generators while handling their specific initialization requirements.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing generator settings. Must include
        a 'name' field specifying the generator type.
    vocs : VOCS
        Variables, Objectives, and Constraints Specification defining the
        optimization problem structure.

    Returns
    -------
    Generator
        Configured Xopt generator instance ready for optimization.

    Raises
    ------
    ValueError
        If the specified generator name is not recognized or supported.
    KeyError
        If the configuration dictionary is missing required fields.

    Examples
    --------
    Creating a random sampling generator:

    >>> from xopt import VOCS
    >>> vocs = VOCS(variables={"x": [0, 10]}, objectives={"y": "MINIMIZE"})
    >>> config = {"name": "random"}
    >>> generator = build_generator_from_config(config, vocs)

    Creating a Bayesian optimization generator:

    >>> config = {"name": "bayes_default"}
    >>> generator = build_generator_from_config(config, vocs)

    Notes
    -----
    Supported generator names:
    - "random": Uniform random sampling within variable bounds
    - "bayes_default": Expected improvement Bayesian optimization
    - "bayes_cheetah": Cheetah-based Bayesian optimization (requires cheetah)

    The generator instances are created using lambda functions stored in
    the PREDEFINED_GENERATORS dictionary, allowing for easy extension
    with new generator types.
    """
    generator_name = config["name"]
    try:
        return PREDEFINED_GENERATORS[generator_name](vocs)
    except KeyError:
        raise ValueError(f"Unsupported or undefined generator name: '{generator_name}'")


def _load_cheetah_generator(vocs):
    """
    Load Cheetah-based Bayesian optimization generator.

    Attempts to import and instantiate a Cheetah-based generator for
    Bayesian optimization. This is a specialized generator that may
    provide enhanced performance for certain optimization problems.

    Parameters
    ----------
    vocs : VOCS
        Variables, Objectives, and Constraints Specification for the
        optimization problem.

    Returns
    -------
    Generator
        Cheetah-based Bayesian optimization generator instance.

    Raises
    ------
    ImportError
        If the cheetah package or its dependencies are not installed
        or cannot be imported.

    Notes
    -----
    This function requires the 'cheetah' package and its dependencies
    to be properly installed. The actual generator implementation is
    imported from a separate module that handles Cheetah integration.

    The import path suggests this may be part of a larger GEECS data
    acquisition system with specialized optimization capabilities.
    """
    try:
        from geecs_data_acquisition.optimization.generators.cheetah_generator import (
            get_cheetah_generator,
        )

        return get_cheetah_generator(vocs)
    except ImportError as e:
        raise ImportError(
            "Could not load 'bayes_cheetah' generator. Make sure 'cheetah' and dependencies are installed."
        ) from e
