"""
Base optimizer module for automated parameter optimization.

This module provides the core optimization infrastructure for the GEECS scanner system,
wrapping the Xopt optimization library to provide a simplified interface for automated
parameter tuning of laser and accelerator systems.

The module supports various optimization algorithms including random sampling, genetic
algorithms, and Bayesian optimization, with built-in support for both per-shot and
aggregate evaluation modes.

Classes
-------
BaseOptimizer
    Main optimizer class that wraps Xopt functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, List, Any, Dict

if TYPE_CHECKING:
    from geecs_scanner.data_acquisition.scan_data_manager import ScanDataManager
    from geecs_scanner.data_acquisition.data_logger import DataLogger

from xopt import Xopt, VOCS

import yaml

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.generators.generator_factory import (
    build_generator_from_config,
)


class BaseOptimizer:
    """
    Wrapper around Xopt for automated parameter optimization.

    This class provides a simplified interface to the Xopt optimization library,
    designed specifically for integration with the GEECS scanner system. It handles
    the generation of candidate parameter sets and evaluation of objective functions
    while maintaining separation from control system logic.

    The optimizer supports various optimization algorithms and evaluation modes,
    with built-in integration for experimental data acquisition and logging.

    Parameters
    ----------
    vocs : VOCS
        Variables, Objectives, and Constraints Specification defining the
        optimization problem structure.
    evaluate_function : callable
        Function that takes a dictionary of variable values and returns
        a dictionary of objective and constraint results.
    generator_name : str
        Name of the Xopt generator algorithm to use (e.g., 'random', 'cnsga', 'upper_confidence_bound').
    evaluation_mode : str, default='per_shot'
        Evaluation mode, either 'per_shot' for individual shot analysis
        or 'aggregate' for combined shot analysis.
    xopt_config_overrides : dict, optional
        Dictionary to override default Xopt configuration parameters.
    evaluator : BaseEvaluator, optional
        Reference to the evaluator object providing the evaluate_function.
    device_requirements : dict, optional
        Dictionary defining required devices and variables for optimization.
    scan_data_manager : ScanDataManager, optional
        Manager instance for accessing saved non-scalar data.
    data_logger : DataLogger, optional
        Logger instance for accessing shot data and bin information.

    Attributes
    ----------
    vocs : VOCS
        The optimization problem specification.
    evaluate_function : callable
        The objective function evaluator.
    evaluation_mode : str
        Current evaluation mode setting.
    generator_name : str
        Name of the optimization algorithm being used.
    evaluator : BaseEvaluator or None
        Reference to the evaluator instance.
    device_requirements : dict
        Required devices and variables configuration.
    xopt : Xopt or None
        The underlying Xopt optimizer instance.
    scan_data_manager : ScanDataManager or None
        Data manager for accessing scan data.
    data_logger : DataLogger or None
        Logger for accessing experimental data.

    Methods
    -------
    initialize(num_initial)
        Run initial random evaluations to seed the optimization.
    generate(n)
        Generate candidate parameter sets for evaluation.
    evaluate(inputs)
        Evaluate candidate points and store results.
    get_results()
        Return the complete optimization results.
    get_best()
        Return the best observed parameter set.
    from_config_file(config_path, scan_data_manager, data_logger)
        Create optimizer instance from YAML configuration file.

    """

    def __init__(
        self,
        vocs: VOCS,
        evaluate_function: Callable[[Dict[str, Any]], Dict[str, Any]],
        generator_name: str,
        evaluation_mode: str = "per_shot",
        xopt_config_overrides: Optional[dict] = None,
        evaluator: Optional[BaseEvaluator] = None,
        device_requirements: Optional[Dict[str, Any]] = None,
        scan_data_manager: Optional[ScanDataManager] = None,
        data_logger: Optional[DataLogger] = None,
    ):
        self.vocs = vocs
        self.evaluate_function = evaluate_function
        self.evaluation_mode = evaluation_mode
        self.generator_name = generator_name
        self.evaluator = evaluator
        self.device_requirements = device_requirements or {}
        self.xopt: Optional[Xopt] = None
        self.scan_data_manager = scan_data_manager
        self.data_logger = data_logger

        self._setup_xopt(xopt_config_overrides or {})

    def _setup_xopt(self, overrides: dict[str, Any]):
        generator = build_generator_from_config(
            config={"name": self.generator_name}, vocs=self.vocs
        )

        self.xopt = Xopt(
            evaluator={"function": self.evaluate_function},
            generator=generator,
            vocs=self.vocs,
        )

    def initialize(self, num_initial: int = 1):
        """
        Run initial random evaluations to seed the optimization.

        Performs random sampling of the parameter space to provide initial
        data points for the optimization algorithm. This is particularly
        important for algorithms that require historical data to function
        effectively (e.g., Bayesian optimization).

        Parameters
        ----------
        num_initial : int, default=1
            Number of random evaluations to perform for initialization.
        """
        self.xopt.random_evaluate(num_initial)

    def generate(self, n: int = 1) -> List[dict]:
        """
        Generate candidate parameter sets for evaluation.

        Uses the configured optimization algorithm to propose new parameter
        combinations that are likely to improve the objective function based
        on previously evaluated points.

        Parameters
        ----------
        n : int, default=1
            Number of candidate parameter sets to generate.

        Returns
        -------
        list of dict
            List of parameter dictionaries, each representing a set of
            control variable values to be evaluated.
        """
        return self.xopt.generator.generate(n)

    def evaluate(self, inputs: List[dict]):
        """
        Evaluate candidate parameter sets and store results.

        Evaluates the provided parameter sets using the configured evaluation
        function and stores the results in the optimization history for use
        by future generation steps.

        Parameters
        ----------
        inputs : list of dict
            List of parameter dictionaries to evaluate, typically generated
            by the `generate()` method.
        """
        self.xopt.evaluate_data(inputs)

    def get_results(self):
        """
        Return complete optimization results.

        Retrieves the full DataFrame containing all evaluated parameter
        sets and their corresponding objective and constraint values.

        Returns
        -------
        pandas.DataFrame
            Complete results DataFrame with columns for all variables,
            objectives, and constraints that have been evaluated.
        """
        return self.xopt.data

    def get_best(self):
        """
        Return the best observed parameter set.

        Identifies and returns the parameter combination that achieved
        the best objective function value according to the optimization
        criteria (minimize or maximize).

        Returns
        -------
        pandas.DataFrame
            Single-row DataFrame containing the best parameter set and
            its corresponding objective and constraint values.

        """
        return self.xopt.data.sort_values(by=list(self.vocs.objectives.keys()))[:1]

    @classmethod
    def from_config_file(
        cls,
        config_path: str,
        scan_data_manager: Optional["ScanDataManager"] = None,
        data_logger: Optional["DataLogger"] = None,
    ) -> "BaseOptimizer":
        """
        Create optimizer instance from YAML configuration file.

        Loads optimizer configuration, evaluator settings, and VOCS specification
        from a YAML file and creates a fully configured BaseOptimizer instance.
        This method provides a convenient way to set up complex optimization
        problems without manual instantiation.


        The evaluator class is dynamically imported based on the module
        and class name specified in the configuration file.

        Parameters
        ----------
        config_path : str
            Path to the YAML configuration file containing optimizer settings.
        scan_data_manager : ScanDataManager, optional
            Instance of ScanDataManager for accessing data during acquisition.
        data_logger : DataLogger, optional
            Instance of DataLogger for accessing shot data and bin information.

        Returns
        -------
        BaseOptimizer
            Fully configured optimizer instance ready for use.
        """
        import importlib
        from xopt import VOCS

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        vocs = VOCS(**config["vocs"])

        evaluator_cfg = config["evaluator"]
        evaluator_module = evaluator_cfg["module"]
        evaluator_class_name = evaluator_cfg["class"]
        evaluator_init_kwargs = evaluator_cfg.get("kwargs", {})
        device_requirements = config.get("device_requirements", {})
        evaluator_init_kwargs["device_requirements"] = device_requirements
        if scan_data_manager:
            evaluator_init_kwargs["scan_data_manager"] = scan_data_manager
        if data_logger:
            evaluator_init_kwargs["data_logger"] = data_logger

        module = importlib.import_module(evaluator_module)
        evaluator_class = getattr(module, evaluator_class_name)
        evaluator = evaluator_class(**evaluator_init_kwargs)

        generator_name = config["generator"]["name"]

        evaluation_mode = config.get("evaluation_mode", "per_shot")

        xopt_config_overrides = config.get("xopt_config_overrides", {})

        return cls(
            vocs=vocs,
            evaluate_function=evaluator.get_value,
            evaluation_mode=evaluation_mode,
            generator_name=generator_name,
            xopt_config_overrides=xopt_config_overrides,
            evaluator=evaluator,
            device_requirements=device_requirements,
            scan_data_manager=scan_data_manager,
            data_logger=data_logger,
        )
