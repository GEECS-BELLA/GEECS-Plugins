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
    computed_field,
    field_validator,
    model_validator,
)
from xopt import VOCS

from image_analysis.config import ImageAnalyzerSpec, load_diagnostic

if TYPE_CHECKING:
    from geecs_scanner.engine.models.save_devices import SaveDeviceConfig
    from image_analysis.config.diagnostic import DiagnosticAnalysisConfig
    from scan_analysis.config.diagnostic_models import ScanRuntimeConfig


# Maps the discriminator on ``DiagnosticAnalysisConfig.image.type`` to the
# scan-side wrapper class name the optimizer instantiates.
_IMAGE_TYPE_TO_ANALYZER_TYPE = {
    "camera": "Array2DScanAnalyzer",
    "line": "Array1DScanAnalyzer",
}


class SingleDeviceScanAnalyzerConfig(BaseModel):
    """Configuration for one analyzer inside ``MultiDeviceScanEvaluator``.

    The optimizer YAML points at a unified diagnostic by name; everything
    else is inherited from the diagnostic's ``image:`` / ``scan:`` sections
    on disk (the same YAML the scan-side ``Array1D/2DScanAnalyzer`` consumes
    via the analysis-group loader). Two optional override fields let you
    twiddle the things you actually care about per-optimization-run —
    typically ``analysis_mode`` (per_shot vs per_bin) — without forking a
    separate diagnostic.

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
        validation time.
    analysis_mode : {"per_shot", "per_bin"}, optional
        Per-run override for the diagnostic's ``scan.mode``. ``None``
        (default) means "use whatever the diagnostic says." Useful when
        the same diagnostic is consumed by both per-shot scan analysis
        and per-bin optimization without a fork.

    Computed (derived from the loaded diagnostic; exposed as fields so
    downstream code reads them by attribute access)
    -------------------------------------------------------------------
    device_name : str
        ``diagnostic.name`` — GEECS device name used for communication
        and device requirements.
    analyzer_type : {"Array1DScanAnalyzer", "Array2DScanAnalyzer"}
        Derived from the typed ``image:`` section's discriminator
        (``"camera"`` → 2D, ``"line"`` → 1D).
    file_tail : str
        ``scan.file_tail`` (falls back to ``".png"`` when the diagnostic
        omits it).
    image_analyzer : ImageAnalyzerSpec
        ``diagnostic.image_analyzer`` — the typed analyzer-class spec.
    data_device_name : str, optional
        ``scan.device`` — data subfolder override. ``None`` means "use
        ``device_name``."

    Methods
    -------
    to_device_requirement()
        Generate a ``device_requirements`` entry for this analyzer.
    """

    diagnostic: str
    analysis_mode: Optional[Literal["per_shot", "per_bin"]] = None

    # Cached resolved diagnostic + scan-runtime view. Populated by the
    # ``_resolve_from_diagnostic`` validator; consumed by the computed
    # fields below. Private so they don't appear in ``model_dump`` /
    # serialised YAML round-trips.
    _diag: Optional["DiagnosticAnalysisConfig"] = PrivateAttr(default=None)
    _scan: Optional["ScanRuntimeConfig"] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _resolve_from_diagnostic(self) -> "SingleDeviceScanAnalyzerConfig":
        """Load the unified diagnostic and stash the resolved view.

        Runs once at model construction. ``load_diagnostic`` defaults to
        ``ScanPaths.paths_config.scan_analysis_configs_path`` for its
        config-dir lookup — same root the task-queue + analysis-group
        loader use — so the diagnostic name has to be unique across the
        ``analyzers/`` tree but doesn't need a namespace prefix.

        The ``scan:`` section on the diagnostic is weakly typed at the
        ImageAnalysis layer (``Optional[Dict]``); we re-validate it
        against the canonical
        ``scan_analysis.config.diagnostic_models.ScanRuntimeConfig``
        here so the optimizer reads field-by-field.
        """
        # Local import: scan_analysis.config has a runtime import of
        # geecs_data_utils which can fail on test fixtures that mock
        # the ScanPaths layer — keeping the import lazy makes this
        # validator safe to instantiate in unit tests without a full
        # data-utils environment.
        from scan_analysis.config.diagnostic_models import ScanRuntimeConfig

        diag = load_diagnostic(self.diagnostic)
        scan = ScanRuntimeConfig.model_validate(diag.scan or {})
        self._diag = diag
        self._scan = scan
        # Resolve analysis_mode: explicit override wins; otherwise inherit
        # from the diagnostic's scan.mode.
        if self.analysis_mode is None:
            self.analysis_mode = scan.mode
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def device_name(self) -> str:
        """GEECS device name; used for device_requirements + comms."""
        assert self._diag is not None, "validator must run first"
        return self._diag.name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def analyzer_type(self) -> Literal["Array1DScanAnalyzer", "Array2DScanAnalyzer"]:
        """Which scan-analyzer wrapper to build, from the image-section type."""
        assert self._diag is not None, "validator must run first"
        if self._diag.image is None:
            raise ValueError(
                f"diagnostic {self.diagnostic!r} has no ``image:`` section; "
                "the optimizer only supports image-driven analyzers."
            )
        image_type = self._diag.image.type
        try:
            return _IMAGE_TYPE_TO_ANALYZER_TYPE[image_type]
        except KeyError as exc:
            raise ValueError(
                f"diagnostic {self.diagnostic!r} declares image.type="
                f"{image_type!r}; optimizer only supports "
                f"{sorted(_IMAGE_TYPE_TO_ANALYZER_TYPE)}."
            ) from exc

    @computed_field  # type: ignore[prop-decorator]
    @property
    def file_tail(self) -> str:
        """Filename suffix to match per-shot data files. Defaults to ``.png``."""
        assert self._scan is not None, "validator must run first"
        return self._scan.file_tail or ".png"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def image_analyzer(self) -> ImageAnalyzerSpec:
        """Typed analyzer-class spec from the diagnostic."""
        assert self._diag is not None, "validator must run first"
        return self._diag.image_analyzer

    @property
    def image_config(self) -> Any:
        """Typed image-section from the diagnostic.

        ``CameraConfig`` for 2D / ``Line1DConfig`` for 1D — already
        validated by the diagnostic loader, no need for a second on-disk
        lookup at analyzer-instantiation time. Not exposed via
        ``@computed_field`` because Pydantic can't serialize the
        discriminated-union shape cleanly and we don't need it in
        ``model_dump`` output anyway.
        """
        assert self._diag is not None, "validator must run first"
        return self._diag.image

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_device_name(self) -> Optional[str]:
        """Data subfolder override; None means "use ``device_name``"."""
        assert self._scan is not None, "validator must run first"
        return self._scan.device

    def to_device_requirement(self) -> dict:
        """Generate the ``device_requirements`` entry for this analyzer.

        Returns
        -------
        dict
            Device requirements dictionary with device name as key and
            configuration as value.
        """
        return {
            self.device_name: {
                "add_all_variables": False,
                "save_nonscalar_data": True,
                "synchronous": True,
                "variable_list": ["acq_timestamp"],
            }
        }


class MultiDeviceScanEvaluatorConfig(BaseModel):
    """
    Configuration for evaluators using multiple SingleDeviceScanAnalyzers.

    This model contains a list of SingleDeviceScanAnalyzerConfig instances
    and provides utilities for auto-generating device requirements from
    the analyzer configurations.

    Attributes
    ----------
    analyzers : list of SingleDeviceScanAnalyzerConfig
        List of analyzer configurations. Each analyzer will be instantiated
        and used to collect data during optimization.

    Methods
    -------
    generate_device_requirements()
        Auto-generate device_requirements dict from analyzer configs.
    """

    analyzers: List[SingleDeviceScanAnalyzerConfig]

    def generate_device_requirements(self) -> dict:
        """
        Auto-generate device_requirements from analyzer configs.

        Iterates through all analyzer configurations and combines their
        device requirements into a single dictionary suitable for use
        by the scan data manager.

        Returns
        -------
        dict
            Device requirements dictionary with "Devices" key containing
            all device configurations.
        """
        devices = {}
        for analyzer_config in self.analyzers:
            devices.update(analyzer_config.to_device_requirement())
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
    automatically generated from the analyzer configurations in evaluator.kwargs.

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

        # Auto-generate device_requirements from evaluator kwargs if needed
        if self.device_requirements is None:
            evaluator_kwargs = self.evaluator.kwargs
            if "analyzers" in evaluator_kwargs:
                # Create MultiDeviceScanEvaluatorConfig to generate requirements
                evaluator_config = MultiDeviceScanEvaluatorConfig(
                    analyzers=evaluator_kwargs["analyzers"]
                )
                self.device_requirements = (
                    evaluator_config.generate_device_requirements()
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
