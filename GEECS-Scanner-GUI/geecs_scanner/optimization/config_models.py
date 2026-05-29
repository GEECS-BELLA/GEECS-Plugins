"""Model definitions for setting up optimizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from typing import TYPE_CHECKING

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from xopt import VOCS

from image_analysis.config import load_diagnostic

if TYPE_CHECKING:
    from geecs_scanner.engine.models.save_devices import SaveDeviceConfig
    from image_analysis.config.diagnostic import DiagnosticAnalysisConfig


class OptimizerAnalyzerRef(BaseModel):
    """One analyzer entry inside :class:`MultiDeviceScanEvaluator`.

    The optimizer YAML points at a unified diagnostic by name; the
    wrapper class, image config, file tail, data folder override, etc.
    are all inherited from the diagnostic's ``image:`` / ``scan:``
    sections on disk (the same YAML the scan-side analyzers consume via
    the analysis-group loader). One optional override field —
    ``analysis_mode`` — lets you flip the same diagnostic between
    per-shot scan analysis and per-bin optimizer evaluation without
    forking a separate YAML.

    YAML shape::

        analyzers:
          - diagnostic: UC_TopView                # everything inherited
          - diagnostic: U_FROG_Grenouille-SpectralPhase
            analysis_mode: per_bin                # override scan.mode

    Attributes
    ----------
    diagnostic : str
        Diagnostic ID (filename stem of the YAML under
        ``scan_analysis_configs/analyzers/<namespace>/<stem>.yaml``).
        Resolved via :func:`image_analysis.config.load_diagnostic` at
        validation time so a missing or malformed YAML fails fast.
    analysis_mode : {"per_shot", "per_bin"}, optional
        Per-run override for the diagnostic's ``scan.mode``. ``None``
        (default) means "use whatever the diagnostic declares." The
        :func:`scan_analysis.config.create_scan_analyzer` factory applies
        the override when building the wrapper analyzer.

    Notes
    -----
    This model is deliberately thin — it caches the loaded
    :class:`DiagnosticAnalysisConfig` (so callers don't pay for a
    second on-disk lookup) and exposes the GEECS device name. Anything
    else previously synthesised here (analyzer class, file tail, typed
    image config) is now the diagnostic-factory's job and not duplicated
    in the optimizer surface area.
    """

    diagnostic: str
    analysis_mode: Optional[Literal["per_shot", "per_bin"]] = None

    # Cached resolved diagnostic. Populated by the validator; consumed by
    # the ``diag`` / ``device_name`` accessors. Private so it doesn't
    # appear in ``model_dump`` / serialised YAML round-trips.
    _diag: Optional["DiagnosticAnalysisConfig"] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve_diagnostic(self) -> "OptimizerAnalyzerRef":
        """Load the unified diagnostic and stash the resolved view.

        Runs once at model construction. ``load_diagnostic`` defaults
        to ``ScanPaths.paths_config.scan_analysis_configs_path`` for
        its config-dir lookup — same root the task-queue + analysis-group
        loader use — so the diagnostic name has to be unique across the
        ``analyzers/`` tree but doesn't need a namespace prefix.
        """
        self._diag = load_diagnostic(self.diagnostic)
        return self

    @property
    def diag(self) -> "DiagnosticAnalysisConfig":
        """The resolved diagnostic config.

        Handed straight to :func:`scan_analysis.config.create_scan_analyzer`
        by :class:`MultiDeviceScanEvaluator`; not part of the YAML schema.
        """
        assert self._diag is not None, "validator must run first"
        return self._diag

    @property
    def device_name(self) -> str:
        """GEECS device name; used for device_requirements + auxiliary-data lookups."""
        return self.diag.name

    def to_device_requirement(self) -> dict:
        """Generate the ``device_requirements`` entry for this analyzer.

        Returns
        -------
        dict
            ``{device_name: {add_all_variables, save_nonscalar_data,
            synchronous, variable_list}}``. Merged into the evaluator-level
            ``Devices`` block by :meth:`BaseOptimizerConfig._load_and_check`.
        """
        return {
            self.device_name: {
                "add_all_variables": False,
                "save_nonscalar_data": True,
                "synchronous": True,
                "variable_list": ["acq_timestamp"],
            }
        }


def _build_device_requirements(analyzer_entries: List[dict]) -> dict:
    """Aggregate ``device_requirements`` across a list of analyzer YAML entries.

    Each entry is validated as an :class:`OptimizerAnalyzerRef` (which
    loads its diagnostic), and the resulting per-analyzer device dicts
    are merged into a single ``{"Devices": {...}}`` block. Used by
    :meth:`BaseOptimizerConfig._load_and_check` to fill in
    ``device_requirements`` when the optimizer YAML omits it.

    Parameters
    ----------
    analyzer_entries : list of dict
        Raw ``evaluator.kwargs.analyzers`` entries from the optimizer YAML.

    Returns
    -------
    dict
        ``{"Devices": {<device>: <requirement>, ...}}`` suitable for
        assignment to ``device_requirements`` and consumption by the
        scan data manager.
    """
    devices: Dict[str, dict] = {}
    for entry in analyzer_entries:
        ref = OptimizerAnalyzerRef.model_validate(entry)
        devices.update(ref.to_device_requirement())
    return {"Devices": devices}


class EvaluatorConfig(BaseModel):
    """
    Configuration for an optimization evaluator.

    This model specifies the module and class to import for the evaluator,
    along with any keyword arguments to be passed to the class initializer.

    Attributes
    ----------
    module : str
        Import path to the evaluator module (e.g.,
        ``geecs_scanner.optimization.evaluators.HiResMagCam``).
    class_ : str
        Name of the evaluator class within the module.
    kwargs : dict of str, Any
        Dictionary of keyword arguments to pass to the evaluator constructor.

    Notes
    -----
    The field `class_` is aliased as `class` in YAML/JSON configuration files.
    """

    module: str
    class_: str = Field(..., alias="class")
    kwargs: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True)


class GeneratorConfig(BaseModel):
    """
    Configuration for the optimization generator.

    Attributes
    ----------
    name : str
        Name of the generator algorithm (e.g., ``random``, ``cnsga``,
        ``upper_confidence_bound``).
    """

    name: str


class BaseOptimizerConfig(BaseModel):
    """
    Canonical optimizer configuration schema.

    This model defines the full configuration for a GEECS optimization run.
    It combines the optimization problem specification (VOCS), evaluator
    definition, generator algorithm, and device saving strategy into a
    unified, validated schema.

    For evaluators using MultiDeviceScanEvaluator, device_requirements are
    automatically generated from the ``analyzers`` list in ``evaluator.kwargs``:
    each entry is validated as an :class:`OptimizerAnalyzerRef` (which
    loads its diagnostic), and the per-analyzer ``to_device_requirement``
    blocks are merged.

    Attributes
    ----------
    vocs : VOCS
        Xopt VOCS specification defining variables, objectives, and constraints.
    evaluator : EvaluatorConfig
        Evaluator module/class specification and initialization arguments.
    generator : GeneratorConfig
        Optimization generator configuration.
    xopt_config_overrides : dict of str, Any
        Dictionary of optional overrides for Xopt configuration.
    device_requirements : dict, optional
        Device requirements dictionary. If None and evaluator.kwargs contains
        'analyzers', this will be auto-generated.
    save_devices : SaveDeviceConfig, optional
        Schema defining device saving strategies and workflow actions.
    save_devices_file : str or Path, optional
        Path to an external YAML file containing a SaveDeviceConfig definition.
    name : str, optional
        Optional name for this optimization configuration.
    description : str, optional
        Optional descriptive text for this optimization configuration.

    Notes
    -----
    This model allows arbitrary types such as `VOCS` by setting
    `arbitrary_types_allowed=True` in the Pydantic config.

    The `vocs` field is not validated or serialized by Pydantic and must be
    constructed manually.

    Raises
    ------
    ValueError
        If `vocs.variables` or `vocs.objectives` is empty.
    """

    vocs: VOCS
    evaluator: EvaluatorConfig
    generator: GeneratorConfig

    xopt_config_overrides: Dict[str, Any] = Field(default_factory=dict)

    device_requirements: Optional[Dict] = None
    save_devices: Optional[SaveDeviceConfig] = None
    save_devices_file: Optional[Union[str, Path]] = None

    seed_dump_files: Optional[List[Union[str, Path]]] = None
    move_to_best_on_finish: bool = False

    name: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("save_devices_file")
    @classmethod
    def _expand_path(cls, v):
        """
        Expand and normalize the save_devices_file path.

        Parameters
        ----------
        v : str or Path or None
            Path value from the configuration.

        Returns
        -------
        Path or None
            Expanded absolute path if provided, otherwise None.
        """
        return Path(v).expanduser().resolve() if v is not None else v

    @model_validator(mode="after")
    def _load_and_check(self) -> "BaseOptimizerConfig":
        """
        Post-validation hook for additional checks and defaults.

        Loads a SaveDeviceConfig from an external file if specified,
        auto-generates device_requirements from evaluator kwargs if needed,
        and performs basic sanity checks on the VOCS.

        Returns
        -------
        BaseOptimizerConfig
            The validated and updated configuration object.

        Raises
        ------
        ValueError
            If `vocs.variables` or `vocs.objectives` is empty.
        """
        # Load save_devices from file if specified
        if self.save_devices is None and self.save_devices_file is not None:
            from geecs_scanner.engine.models.save_devices import (
                SaveDeviceConfig,
            )

            with open(self.save_devices_file, "r") as f:
                loaded = yaml.safe_load(f)
            self.save_devices = SaveDeviceConfig.model_validate(loaded)

        # Auto-generate device_requirements from evaluator kwargs if needed.
        # The evaluator YAML lists analyzers by diagnostic name; this loop
        # validates each entry (loading the diagnostic to discover the
        # GEECS device name) and merges the per-analyzer device blocks.
        if self.device_requirements is None:
            evaluator_kwargs = self.evaluator.kwargs
            if "analyzers" in evaluator_kwargs:
                self.device_requirements = _build_device_requirements(
                    evaluator_kwargs["analyzers"]
                )

        # Validate VOCS
        if not self.vocs.variables:
            raise ValueError("vocs.variables must not be empty.")

        # BAX generators don't require objectives (they model observables only)
        # Define known BAX generator names (supports multiple variants)
        BAX_GENERATORS = {
            "multipoint_bax_alignment",
            "multipoint_bax_alignment_simulated",
            "multipoint_bax_alignment_l2",
        }

        # Only validate objectives for non-BAX generators
        if not self.vocs.objectives and self.generator.name not in BAX_GENERATORS:
            raise ValueError(
                "vocs.objectives must not be empty for non-BAX generators. "
                "BAX generators (e.g., multipoint_bax_alignment) model observables only."
            )

        return self

    def evaluator_import_path(self) -> tuple[str, str]:
        """
        Return the evaluator import path.

        Returns
        -------
        tuple of (str, str)
            Module name and class name for the evaluator.
        """
        return self.evaluator.module, self.evaluator.class_


# SaveDeviceConfig is imported under TYPE_CHECKING to avoid pulling the full
# data_acquisition chain in at module load. Pydantic v2 needs the type resolved
# before it can validate BaseOptimizerConfig, so rebuild here once all classes
# are defined.
def _rebuild() -> None:
    from geecs_scanner.engine.models.save_devices import (
        SaveDeviceConfig as _SDC,
    )

    BaseOptimizerConfig.model_rebuild(_types_namespace={"SaveDeviceConfig": _SDC})


_rebuild()
del _rebuild
