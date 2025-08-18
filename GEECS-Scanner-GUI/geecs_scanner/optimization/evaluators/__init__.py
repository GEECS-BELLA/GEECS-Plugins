"""
Optimization evaluators for the GEECS scanner system.

This package contains concrete evaluator implementations for different diagnostic
devices and optimization objectives. Each evaluator inherits from BaseEvaluator
and implements specific objective functions for automated parameter optimization.

The evaluators integrate with the GEECS scan analysis framework to process
experimental data and compute objective function values for optimization
algorithms.

Available Evaluators
--------------------
ALine3SizeEval
    Electron beam size optimization using UC_ALineEBeam3 camera.
    Minimizes beam size based on FWHM measurements.

HiResMagCam
    Beam quality optimization using UC_HiResMagCam device.
    Maximizes the ratio of total counts to emittance proxy.

Examples
--------
Using an evaluator in optimization:

>>> from geecs_scanner.optimization.evaluators.ALine3_FWHM import ALine3SizeEval
>>> evaluator = ALine3SizeEval(
...     scan_data_manager=sdm,
...     data_logger=logger
... )
>>> result = evaluator.get_value(input_parameters)

Loading evaluator from configuration:

>>> from geecs_scanner.optimization.base_optimizer import BaseOptimizer
>>> optimizer = BaseOptimizer.from_config_file(
...     "optimization_config.yaml",
...     scan_data_manager=sdm,
...     data_logger=logger
... )

Notes
-----
All evaluators in this package:
- Inherit from BaseEvaluator
- Implement the _get_value method for objective computation
- Support both per-shot and aggregate evaluation modes
- Integrate with GEECS data acquisition and analysis systems
- Provide proper logging and result tracking

To create a new evaluator, inherit from BaseEvaluator and implement
the required abstract methods. See the existing evaluators for examples
of proper implementation patterns.
"""
