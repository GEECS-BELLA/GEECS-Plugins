"""
Optimization evaluators for the GEECS scanner system.

This package contains concrete evaluator implementations for different diagnostic
devices and optimization objectives. Each evaluator inherits from BaseEvaluator
and implements specific objective functions for automated parameter optimization.

The evaluators integrate with the GEECS scan analysis framework to process
experimental data and compute objective function values for optimization
algorithms.

Notes
-----
All evaluators in this package:
- Inherit from BaseEvaluator
- Implement the _get_value method for objective computation
- Support both per-shot and aggregate evaluation modes
- Integrate with GEECS data acquisition and analysis systems
- Provide proper logging and result tracking

To create a new evaluator, inherit from MultiDeviceScanEvaluator and implement
the compute_objective method. See multi_device_scan_evaluator.py for detailed
documentation and examples.
"""

from geecs_scanner.optimization.evaluators.multi_device_scan_evaluator import (
    MultiDeviceScanEvaluator,
)

__all__ = ["MultiDeviceScanEvaluator"]
