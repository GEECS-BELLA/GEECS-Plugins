"""Model definitions for setting up optimizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from typing import TYPE_CHECKING

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from xopt import VOCS

from image_analysis.config import load_diagnostic

if TYPE_CHECKING:
    from geecs_scanner.engine.models.save_devices import SaveDeviceConfig


# Per-analyzer device-requirements template. Every optimizer-driven analyzer
# subscribes to the same shape — synchronous, non-scalar saving, ``acq_timestamp``
# in the variable list — so the only thing that varies per analyzer is the
# GEECS device name (the dict key). Lifted out as a module constant so the
# validator below and any downstream consumer share one definition.
_OPTIMIZER_DEVICE_REQUIREMENT_TEMPLATE = {
    "add_all_variables": False,
    "save_nonscalar_data": True,
    "synchronous": True,
    "variable_list": ["acq_timestamp"],
}


class OptimizerAnalyzerEntry(BaseModel):
    """Envelope-only model for an entry in the optimizer YAML's ``analyzers`` list.

    The optimizer YAML lists analyzers as either a bare diagnostic stem
    (no per-run customization) or a dict that names a diagnostic and
    carries a patch over its fields::

        analyzers:
          - UC_TopView                          # use diagnostic as-is
          - diagnostic: UC_FROG                 # patch scan block
            scan:
              mode: per_bin

    This model validates the *envelope* — that the entry has a
    ``diagnostic`` key naming a string — and uses ``extra="allow"`` to
    capture every other field as the override patch. The patch dict is
    then passed straight to :func:`image_analysis.config.load_diagnostic`
    via its ``overrides`` kwarg, which deep-merges it into the on-disk
    YAML before Pydantic re-validates the whole thing.

    The model deliberately does *not* enumerate override fields
    (``analysis_mode`` etc.) — that would re-couple the optimizer's
    surface area to the diagnostic schema. ``model_extra`` carries the
    patch generically, so any field on the diagnostic can be overridden
    without code changes here.

    See :class:`MultiDeviceScanEvaluator` for the consumption pattern.
    """

    diagnostic: str
    model_config = ConfigDict(extra="allow")


def _split_analyzer_entry(
    entry: Union[str, Dict[str, Any]],
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Parse one ``analyzers:`` list element into ``(name, overrides)``.

    Used by both :meth:`BaseOptimizerConfig._load_and_check` (for
    auto-generating ``device_requirements``) and
    :class:`MultiDeviceScanEvaluator.__init__` (for building the
    actual analyzers). Centralising the envelope parsing here keeps
    the two call sites in lockstep — anywhere the optimizer YAML's
    analyzers list is consumed, this is the single decoder.

    Bare-string entries return ``(stem, None)``; dict entries are
    validated as :class:`OptimizerAnalyzerEntry` and any fields beyond
    ``diagnostic`` are returned as the overrides patch (or ``None``
    when no overrides were supplied).
    """
    if isinstance(entry, str):
        return entry, None
    parsed = OptimizerAnalyzerEntry.model_validate(entry)
    return parsed.diagnostic, (parsed.model_extra or None)


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
    automatically generated from the ``analyzers`` list in ``evaluator.kwargs``
    by loading each referenced diagnostic and keying the per-analyzer block on
    its GEECS device name.

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
        # Each entry in ``analyzers`` is either a bare diagnostic stem
        # (string) or a dict with ``diagnostic:`` + override patch (see
        # OptimizerAnalyzerEntry). The patch is forwarded to
        # ``load_diagnostic`` so the resolved GEECS device name reflects
        # any ``name``-level override; in practice overrides target
        # ``scan:`` fields and the name comes straight off disk.
        if self.device_requirements is None:
            analyzers = self.evaluator.kwargs.get("analyzers")
            if analyzers:
                devices: Dict[str, dict] = {}
                for entry in analyzers:
                    name, overrides = _split_analyzer_entry(entry)
                    diag = load_diagnostic(name, overrides=overrides)
                    devices[diag.name] = dict(_OPTIMIZER_DEVICE_REQUIREMENT_TEMPLATE)
                self.device_requirements = {"Devices": devices}

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
