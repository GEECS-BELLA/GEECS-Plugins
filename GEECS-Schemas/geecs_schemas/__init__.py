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
    AnalysisGroup,
    AnalysisRecipe,
    AnalyzerRef,
    HimgJob,
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
from geecs_schemas.archive_policy import (
    ARCHIVE_POLICY_FILENAME,
    ARCHIVER_CONFIG_FOLDER,
    ArchivePolicy,
    SamplingOverride,
)
from geecs_schemas.derived_channels import (
    DERIVED_CHANNELS_FILENAME,
    GATEWAY_CONFIG_FOLDER,
    DerivedChannel,
    DerivedChannels,
    DerivedInput,
)
from geecs_schemas.epics import (
    DETECTOR_KINDS,
    NATIVE_DEVICES_FILENAME,
    AravisCamera,
    EpicsMotor,
    NativeDevice,
    NativeDevices,
)
from geecs_schemas.epics.family import (
    REGISTRY_KIND_FAMILIES,
    FamilyAction,
    FamilyFile,
    Role,
    load_families,
)
from geecs_schemas.epics.panel_devices import (
    PANEL_DEVICES_FILENAME,
    PanelDevice,
    PanelDevices,
)
from geecs_schemas.epics.operations import (
    OPERATOR_POLICY_FILENAME,
    JournalEntry,
    OperatorGrant,
    OperatorPolicy,
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
    # archive_policy
    "ArchivePolicy",
    "SamplingOverride",
    "ARCHIVER_CONFIG_FOLDER",
    "ARCHIVE_POLICY_FILENAME",
    # derived_channels
    "DerivedChannels",
    "GATEWAY_CONFIG_FOLDER",
    "DERIVED_CHANNELS_FILENAME",
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
    # epics (native EPICS devices)
    "NativeDevices",
    "NativeDevice",
    "AravisCamera",
    "EpicsMotor",
    "DETECTOR_KINDS",
    "NATIVE_DEVICES_FILENAME",
    # epics.family / epics.panel_devices (the web estate's device data)
    "FamilyFile",
    "FamilyAction",
    "Role",
    "load_families",
    "REGISTRY_KIND_FAMILIES",
    "PanelDevices",
    "PanelDevice",
    "PANEL_DEVICES_FILENAME",
    # epics.operations (GeecsCanopy's device module: who may write, what was tried)
    "OperatorPolicy",
    "OperatorGrant",
    "OPERATOR_POLICY_FILENAME",
    "JournalEntry",
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
    "AnalysisGroup",
    "AnalysisRecipe",
    "AnalyzerRef",
    "HimgJob",
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
    "archive_policy": ArchivePolicy,
    "shot_offsets": ShotOffsets,
    "native_devices": NativeDevices,
    "family_file": FamilyFile,
    "panel_devices": PanelDevices,
    "operator_policy": OperatorPolicy,
    "journal_entry": JournalEntry,
    "analysis_recipe": AnalysisRecipe,
    "analysis_group": AnalysisGroup,
    "himg_job": HimgJob,
    "log_entry": LogEntry,
}
