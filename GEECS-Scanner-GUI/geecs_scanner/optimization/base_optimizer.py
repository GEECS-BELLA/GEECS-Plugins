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

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, List, Any, Dict

if TYPE_CHECKING:
    from geecs_scanner.engine.scan_data_manager import ScanDataManager
    from geecs_scanner.engine.data_logger import DataLogger

import pandas as pd
import yaml

from xopt import Xopt, VOCS

from geecs_scanner.optimization.base_evaluator import BaseEvaluator
from geecs_scanner.optimization.generators.generator_factory import (
    build_generator_from_config,
)
from geecs_scanner.optimization.inspection.dump_loader import (
    check_cross_dump_consistency,
    check_vocs_compatible,
    load_xopt_dump,
)

logger = logging.getLogger(__name__)


def _warn_on_duplicate_inputs(df: pd.DataFrame, vocs: VOCS, threshold: int = 5) -> None:
    var_cols = [v for v in vocs.variable_names if v in df.columns]
    if not var_cols:
        return
    counts = df[var_cols].value_counts()
    offenders = counts[counts > threshold]
    if not offenders.empty:
        logger.warning(
            "Seed data contains %d input row(s) appearing >%d times. "
            "This may cause GP conditioning issues; consider deduplicating manually.",
            len(offenders),
            threshold,
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
    seed_dump_files : list of Path, optional
        Paths to prior Xopt dump YAML files whose evaluated data will be
        loaded into the optimizer before the scan begins.  VOCS must be
        compatible (same variable and objective names); differing bounds
        produce warnings only.

    Attributes
    ----------
    vocs : VOCS
        The optimization problem specification.
    evaluate_function : callable
        The objective function evaluator.
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
    seed_from_dumps(paths)
        Load historical data from dump files into the optimizer.
    from_config_file(config_path, scan_data_manager, data_logger)
        Create optimizer instance from YAML configuration file.

    """

    def __init__(
        self,
        vocs: VOCS,
        evaluate_function: Callable[[Dict[str, Any]], Dict[str, Any]],
        generator_name: str,
        xopt_config_overrides: Optional[dict] = None,
        evaluator: Optional[BaseEvaluator] = None,
        device_requirements: Optional[Dict[str, Any]] = None,
        scan_data_manager: Optional["ScanDataManager"] = None,
        data_logger: Optional["DataLogger"] = None,
        seed_dump_files: Optional[List[Path]] = None,
    ):
        self.vocs = vocs
        self.evaluate_function = evaluate_function
        self.generator_name = generator_name
        self.evaluator = evaluator
        self.device_requirements = device_requirements or {}
        self.xopt: Optional[Xopt] = None
        self.scan_data_manager = scan_data_manager
        self.data_logger = data_logger
        self._n_seeded: int = 0

        self.xopt_config_overrides: dict[str, Any] = dict(xopt_config_overrides or {})
        self._setup_xopt(self.xopt_config_overrides)

        if seed_dump_files:
            self.seed_from_dumps(seed_dump_files)

    @property
    def n_seeded(self) -> int:
        """Number of evaluations loaded from dump files before the scan started."""
        return self._n_seeded

    def _setup_xopt(self, overrides: dict[str, Any]):
        generator_config: dict[str, Any] = {"name": self.generator_name}
        generator_config.update(overrides)
        generator = build_generator_from_config(config=generator_config, vocs=self.vocs)

        self.xopt = Xopt(
            evaluator={"function": self.evaluate_function},
            generator=generator,
            vocs=self.vocs,
        )

    def seed_from_dumps(self, dump_paths: List[Path]) -> int:
        """Load historical data from prior Xopt dump files.

        Each file's VOCS is checked for compatibility with this optimizer's
        VOCS before any data is loaded.  Rows where ``xopt_error`` is True or
        any objective column is NaN are filtered out.

        Parameters
        ----------
        dump_paths:
            Paths to ``xopt_dump.yaml`` files written by ``Xopt.dump()``.

        Returns
        -------
        int
            Total number of rows added to the optimizer (after filtering).
        """
        objective_names = list(self.vocs.objectives.keys())
        all_frames: List[pd.DataFrame] = []
        dump_vocs_pairs = []

        for path in dump_paths:
            path = Path(path)
            if not path.exists():
                logger.warning("Seed dump file not found, skipping: %s", path)
                continue

            try:
                source_vocs, df = load_xopt_dump(path)
            except (KeyError, Exception) as exc:
                logger.warning("Failed to parse dump file %s: %s", path, exc)
                continue

            try:
                check_vocs_compatible(self.vocs, source_vocs, path)
            except ValueError as exc:
                logger.error("Incompatible dump file, skipping: %s", exc)
                continue

            dump_vocs_pairs.append((path, source_vocs))

            # Filter error rows
            if "xopt_error" in df.columns:
                n_before = len(df)
                df = df[df["xopt_error"] != True]  # noqa: E712
                n_errors = n_before - len(df)
                if n_errors:
                    logger.info("Filtered %d error row(s) from %s", n_errors, path.name)

            # Filter NaN objective rows
            for obj_name in objective_names:
                if obj_name in df.columns:
                    n_before = len(df)
                    df = df[df[obj_name].notna()]
                    n_nan = n_before - len(df)
                    if n_nan:
                        logger.info(
                            "Filtered %d NaN '%s' row(s) from %s",
                            n_nan,
                            obj_name,
                            path.name,
                        )

            if df.empty:
                logger.warning(
                    "No valid data in %s after filtering; skipping.", path.name
                )
                continue

            all_frames.append(df)
            logger.info("Loaded %d evaluations from %s", len(df), path.name)

        if not all_frames:
            logger.warning("No valid seed data loaded from any dump file.")
            return 0

        # Cross-dump VOCS bounds consistency (log only)
        if len(dump_vocs_pairs) > 1:
            check_cross_dump_consistency(dump_vocs_pairs)

        combined = pd.concat(all_frames, ignore_index=True)

        _warn_on_duplicate_inputs(combined, self.vocs)

        # Populate xopt.data and propagate to the generator.
        # Xopt.add_data calls generator.add_data internally in recent versions,
        # but the explicit call guards against older versions where it did not.
        self.xopt.add_data(combined)
        if self.xopt.generator.data is None or len(self.xopt.generator.data) == 0:
            self.xopt.generator.add_data(combined)

        self._n_seeded = len(combined)
        logger.info(
            "Seeded optimizer with %d total evaluation(s) from %d dump file(s).",
            self._n_seeded,
            len(dump_paths),
        )
        return self._n_seeded

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

        # If the generator provides diagnostic metadata (e.g., BAX), log it
        metadata: Dict[str, float] = {}
        generator = getattr(self.xopt, "generator", None)
        algo_results = getattr(generator, "algorithm_results", None)

        if isinstance(algo_results, dict):
            center = algo_results.get("solution_center")
            if center is not None:
                try:
                    center_values = list(center)
                except TypeError:
                    center_values = [center]

                for name, value in zip(self.vocs.variable_names, center_values):
                    try:
                        metadata[f"BAX_solution_center[{name}]"] = float(value)
                    except (TypeError, ValueError):
                        continue

            entropy = algo_results.get("solution_entropy")
            if entropy is not None:
                try:
                    metadata["BAX_solution_entropy"] = float(entropy)
                except (TypeError, ValueError):
                    pass

        if metadata and self.evaluator is not None:
            self.evaluator.log_results_for_current_bin(metadata)

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
        from geecs_scanner.optimization.config_models import BaseOptimizerConfig

        # Load and validate config using Pydantic model
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # This handles all validation AND auto-generates device_requirements!
        config = BaseOptimizerConfig.model_validate(config_dict)

        # Dynamically import and instantiate evaluator
        evaluator_init_kwargs = config.evaluator.kwargs.copy()
        evaluator_init_kwargs["device_requirements"] = config.device_requirements
        if scan_data_manager:
            evaluator_init_kwargs["scan_data_manager"] = scan_data_manager
        if data_logger:
            evaluator_init_kwargs["data_logger"] = data_logger

        module = importlib.import_module(config.evaluator.module)
        evaluator_class = getattr(module, config.evaluator.class_)
        evaluator = evaluator_class(**evaluator_init_kwargs)

        # Prepare generator overrides, ensuring any relative output paths are rooted in the scan folder
        overrides = dict(config.xopt_config_overrides)
        scan_folder = None
        if scan_data_manager:
            try:
                scan_folder = scan_data_manager.scan_paths.get_folder()
            except AttributeError:
                scan_folder = None
            else:
                scan_folder = Path(scan_folder)
                scan_folder.mkdir(parents=True, exist_ok=True)

                for key, block in list(overrides.items()):
                    if not isinstance(block, dict):
                        continue

                    file_value = block.get("algorithm_results_file")
                    if file_value is None:
                        # Provide a sensible default within the scan directory
                        block["algorithm_results_file"] = str(
                            scan_folder / f"{key}_algo_results"
                        )
                    else:
                        path = Path(file_value)
                        if not path.is_absolute():
                            block["algorithm_results_file"] = str(
                                (scan_folder / path).resolve()
                            )

        # Resolve seed_dump_files paths relative to the config file's directory
        resolved_seed_paths: Optional[List[Path]] = None
        if config.seed_dump_files:
            config_dir = Path(config_path).parent
            resolved_seed_paths = []
            for raw in config.seed_dump_files:
                p = Path(raw)
                if not p.is_absolute():
                    p = (config_dir / p).resolve()
                if not p.exists():
                    logger.warning(
                        "seed_dump_files entry not found (will be skipped): %s", p
                    )
                resolved_seed_paths.append(p)

        # Create optimizer using validated config
        return cls(
            vocs=config.vocs,
            evaluate_function=evaluator.get_value,
            generator_name=config.generator.name,
            xopt_config_overrides=overrides,
            evaluator=evaluator,
            device_requirements=config.device_requirements,
            scan_data_manager=scan_data_manager,
            data_logger=data_logger,
            seed_dump_files=resolved_seed_paths,
        )
