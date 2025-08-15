"""
Optimization generators for the GEECS scanner system.

This package provides generator implementations for different optimization
algorithms used in automated parameter tuning. Generators are responsible
for proposing new parameter combinations to evaluate based on previous
results and the chosen optimization strategy.

The package includes both standard optimization algorithms and specialized
generators that incorporate physics-based priors for accelerator optimization.

Available Generators
--------------------
Random Generator
    Uniform random sampling within parameter bounds.
    Accessed via generator_factory with name "random".

Bayesian Optimization (Default)
    Expected improvement acquisition with standard Gaussian Process.
    Accessed via generator_factory with name "bayes_default".

Cheetah-based Bayesian Optimization
    Physics-informed Bayesian optimization using Cheetah simulations.
    Accessed via generator_factory with name "bayes_cheetah".

Modules
-------
generator_factory
    Factory interface for creating generator instances from configuration.
cheeta_generator
    Cheetah-based physics-informed optimization generator.

Examples
--------
Using the generator factory:

>>> from geecs_scanner.optimization.generators.generator_factory import build_generator_from_config
>>> from xopt import VOCS
>>> vocs = VOCS(variables={"x": [0, 10]}, objectives={"y": "MINIMIZE"})
>>> config = {"name": "random"}
>>> generator = build_generator_from_config(config, vocs)

Creating a Cheetah-based generator:

>>> config = {"name": "bayes_cheetah"}
>>> generator = build_generator_from_config(config, vocs)

Notes
-----
All generators in this package:
- Are compatible with the Xopt optimization framework
- Accept VOCS (Variables, Objectives, Constraints Specification) for configuration
- Provide generate() methods for proposing new parameter combinations
- Support both single and batch candidate generation

The generator factory provides a unified interface for creating different
types of generators while hiding the implementation details and dependency
requirements of each algorithm.

For physics-informed optimization, the Cheetah generator uses particle
accelerator simulations to provide better initial guesses and faster
convergence for beam transport optimization problems.
"""
