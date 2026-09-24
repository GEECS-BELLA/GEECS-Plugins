"""geecs_schemas — versioned Pydantic models for every GEECS scanner config.

Configs are schemas; YAML is just serialization.  This package is the single
home of the models: presets (the saved scan: device group +
plan call), scan requests, scan variables, trigger profiles, action plans,
and gateway derived channels — plus converters from the legacy YAML
dialects still in use (``geecs_schemas.convert``; scan variables, presets
and action libraries have none, GEECS-Plugins#779 / #807) and a Markdown
reference generator (``geecs_schemas.docgen``).

It depends on Pydantic and GEST (the lightweight VOCS model), so anything — engine, GUI, scripts, docs
tooling — can import it without dragging in hardware or analysis stacks.
"""

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.analysis import (
    AnalysisDiagnostic,
    AnalysisGroup,
    AnalysisRecipe,
    AnalyzerRef,
    AnalyzerSpec,
    CameraConfig,
    Line1DConfig,
    RendererOptions,
    ScanRuntime,
)
from geecs_schemas.action_plan import (
    ActionPlan,
    ActionPlanLibrary,
    ActionStep,
    CheckStep,
    RunPlanStep,
    SetStep,
    WaitStep,
)
from geecs_schemas.derived_channels import (
    DerivedChannel,
    DerivedChannels,
    DerivedInput,
)
from geecs_schemas.experiment_defaults import DefaultActions, ExperimentDefaults
from geecs_schemas.preset import PlanCall, Preset, PresetDevice
from geecs_schemas.log_entry import (
    AnalysisPayload,
    Attachment,
    Book,
    EntryKind,
    EntryPayload,
    EntryStatus,
    LogEntry,
    ProblemPayload,
)
from geecs_schemas.scan_request import (
    AcquisitionMode,
    ActionBindings,
    CaptureSettings,
    PositionList,
    PositionRange,
    Positions,
    PreflightCheckResult,
    PreflightOutcome,
    ScanAxis,
    ScanRequest,
    ScanRequestMode,
    SubmissionRecord,
)
from geecs_schemas.shot_offsets import DeviceOffset, ShotOffsets
from geecs_schemas.sweep import (
    AxisSweep,
    FermatSpiralSweep,
    ListAxis,
    LogAxis,
    RangeAxis,
    RelativeSweepAxis,
    SpiralSweep,
    SquareSpiralSweep,
    Sweep,
    SweepAxis,
    X2XSweep,
)
from geecs_schemas.scan_variables import (
    CompositeMode,
    PseudoComponent,
    PseudoScanVariable,
    ScanVariable,
    ScanVariables,
    ScanVariableSpec,
    split_device_variable,
)
from geecs_schemas.trigger_profile import (
    TriggerProfile,
    TriggerState,
    TriggerWrite,
)

__all__ = [
    "SchemaModel",
    "VersionedSchemaModel",
    # scan_request
    "ScanRequest",
    "ScanRequestMode",
    "AcquisitionMode",
    "ActionBindings",
    "CaptureSettings",
    "PositionRange",
    "ScanAxis",
    "PositionList",
    "Positions",
    "SubmissionRecord",
    "PreflightOutcome",
    "PreflightCheckResult",
    # derived_channels
    "DerivedChannels",
    "DerivedChannel",
    "DerivedInput",
    # preset
    "Preset",
    "PresetDevice",
    "PlanCall",
    # scan_variables
    "ScanVariables",
    "split_device_variable",
    "ScanVariable",
    "ScanVariableSpec",
    "PseudoScanVariable",
    "PseudoComponent",
    "CompositeMode",
    # trigger_profile
    "TriggerProfile",
    "TriggerState",
    "TriggerWrite",
    # experiment_defaults
    "ExperimentDefaults",
    "DefaultActions",
    # action_plan
    "ActionPlan",
    "ActionPlanLibrary",
    "ActionStep",
    "SetStep",
    "WaitStep",
    "CheckStep",
    "RunPlanStep",
    # analysis
    "AnalysisDiagnostic",
    "AnalysisGroup",
    "AnalysisRecipe",
    "AnalyzerRef",
    "AnalyzerSpec",
    "CameraConfig",
    "Line1DConfig",
    "RendererOptions",
    "ScanRuntime",
    # shot_offsets
    "ShotOffsets",
    "DeviceOffset",
    # log_entry
    "LogEntry",
    "Attachment",
    "AnalysisPayload",
    "ProblemPayload",
    "EntryPayload",
    "Book",
    "EntryKind",
    "EntryStatus",
    "SCHEMA_REGISTRY",
    "OptimizerConfig",
    "optimizer_required_devices",
    # sweep payload (nested in a plan call, not a standalone config document)
    "Sweep",
    "SweepAxis",
    "RelativeSweepAxis",
    "AxisSweep",
    "RangeAxis",
    "ListAxis",
    "LogAxis",
    "SpiralSweep",
    "FermatSpiralSweep",
    "SquareSpiralSweep",
    "X2XSweep",
]

# kind → top-level document model, for generic tooling (loaders, editors,
# docgen). Keys are the canonical config-kind identifiers.
from .optimizer_config import OptimizerConfig, optimizer_required_devices

SCHEMA_REGISTRY: dict[str, type[VersionedSchemaModel]] = {
    "optimizer_config": OptimizerConfig,
    "preset": Preset,
    "scan_request": ScanRequest,
    "scan_variables": ScanVariables,
    "trigger_profile": TriggerProfile,
    "action_plan": ActionPlan,
    "action_plan_library": ActionPlanLibrary,
    "experiment_defaults": ExperimentDefaults,
    "derived_channels": DerivedChannels,
    "shot_offsets": ShotOffsets,
    "analysis_diagnostic": AnalysisDiagnostic,
    "analysis_recipe": AnalysisRecipe,
    "analysis_group": AnalysisGroup,
    "log_entry": LogEntry,
}
