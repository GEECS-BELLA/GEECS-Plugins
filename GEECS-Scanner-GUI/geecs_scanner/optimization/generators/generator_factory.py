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

Notes
-----
The factory supports the following predefined generators:
- "random": Random sampling generator
- "bayes_default": Expected improvement Bayesian optimization
- "bayes_cheetah": Cheetah-based Bayesian optimization (requires cheetah package)
- "multipoint_bax_alignment": Multipoint BAX alignment (requires configuration overrides)

New generators can be added by extending the PREDEFINED_GENERATORS dictionary
with appropriate factory functions.
"""

# optimization/generator_factory.py

from typing import Any, Dict, Callable

from xopt.vocs import VOCS
from xopt.generators.random import RandomGenerator
from xopt.generators.bayesian import ExpectedImprovementGenerator
from xopt.generators.bayesian.models.standard import StandardModelConstructor
from xopt.generators.bayesian.turbo import OptimizeTurboController

from .bax_multipoint_alignment_simulation import make_bax_multipoint_alignment_generator

# Explicitly defined generators dictionary
PREDEFINED_GENERATORS: dict[str, Callable[[VOCS, Dict[str, Any]], Any]] = {
    "random": lambda vocs, overrides: RandomGenerator(vocs=vocs),
    "bayes_default": lambda vocs, overrides: ExpectedImprovementGenerator(
        vocs=vocs, gp_constructor=StandardModelConstructor(use_low_noise_prior=False)
    ),
    "bayes_cheetah": lambda vocs, overrides: _load_cheetah_generator(vocs),
    "bayes_turbo_standard": lambda vocs, overrides: _make_bayes_turbo(vocs),
    "bayes_turbo_HTU_e_beam_brightness": lambda vocs, overrides: _make_bayes_turbo(
        vocs,
        success_tolerance=2,
        failure_tolerance=2,
        length=0.25,
        length_max=2.0,
        length_min=0.0078125,
        scale_factor=2.0,
    ),
    "multipoint_bax_alignment": lambda vocs,
    overrides: make_bax_multipoint_alignment_generator(vocs, overrides),
    # Add more explicit named generators here if needed
}

# Backwards-compatible alias
PREDEFINED_GENERATORS["bax_multipoint_alignment"] = PREDEFINED_GENERATORS[
    "multipoint_bax_alignment"
]


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

    Notes
    -----
    Supported generator names:
    - "random": Uniform random sampling within variable bounds
    - "bayes_default": Expected improvement Bayesian optimization
    - "bayes_cheetah": Cheetah-based Bayesian optimization (requires cheetah)
    - "multipoint_bax_alignment": Multipoint BAX alignment with custom overrides

    The generator instances are created using lambda functions stored in
    the PREDEFINED_GENERATORS dictionary, allowing for easy extension
    with new generator types. Any keys in `config` other than ``name``
    are passed to the corresponding factory function as generator-specific
    overrides.
    """
    generator_name = config["name"]
    overrides = dict(config)
    overrides.pop("name", None)
    try:
        return PREDEFINED_GENERATORS[generator_name](vocs, overrides)
    except KeyError:
        raise ValueError(f"Unsupported or undefined generator name: '{generator_name}'")


def _make_bayes_turbo(
    vocs: VOCS,
    length: float = 0.30,
    length_min: float = 0.01,
    length_max: float = 1.00,
    success_tolerance: int = 2,
    failure_tolerance: int = 2,
    scale_factor: float = 2.0,
    restrict_model_data: bool = True,
    batch_size: int = 1,
    n_monte_carlo_samples: int = 128,
    use_low_noise_prior: bool = False,
) -> ExpectedImprovementGenerator:
    """
    Build an ExpectedImprovementGenerator with a customized TuRBO trust region.

    Parameters
    ----------
    vocs : VOCS
        VOCS specification for the optimization problem.
    length, length_min, length_max : float
        Trust region bounds.
    success_tolerance, failure_tolerance : int
        Number of successes/failures to expand/shrink TR.
    scale_factor : float
        Trust region expansion factor.
    restrict_model_data : bool
        Whether to fit GP only to points in the TR.
    batch_size : int
        Number of candidates per iteration.
    n_monte_carlo_samples : int
        Number of MC samples for qEI.
    use_low_noise_prior : bool
        Use low noise prior in GP model.
    """
    turbo = OptimizeTurboController(
        vocs=vocs,
        batch_size=batch_size,
        length=length,
        length_min=length_min,
        length_max=length_max,
        success_tolerance=success_tolerance,
        failure_tolerance=failure_tolerance,
        scale_factor=scale_factor,
        restrict_model_data=restrict_model_data,
        name="OptimizeTurboController",
    )

    return ExpectedImprovementGenerator(
        vocs=vocs,
        gp_constructor=StandardModelConstructor(
            use_low_noise_prior=use_low_noise_prior
        ),
        n_monte_carlo_samples=n_monte_carlo_samples,
        turbo_controller=turbo,
    )


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
