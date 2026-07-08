"""geecs_schemas — versioned Pydantic models for every GEECS scanner config.

Configs are schemas; YAML is just serialization.  This package is the single
home of the models (vision doc §4): scan requests, save sets, scan variables,
trigger profiles, and action plans — plus converters from every legacy YAML
dialect (``geecs_schemas.convert``) and a Markdown reference generator
(``geecs_schemas.docgen``).

It depends on Pydantic only, so anything — engine, GUI, scripts, docs
tooling — can import it without dragging in hardware or analysis stacks.
"""

from geecs_schemas._base import SchemaModel, VersionedSchemaModel
from geecs_schemas.action_plan import (
    ActionPlan,
    ActionPlanLibrary,
    ActionStep,
    CheckStep,
    RunPlanStep,
    SetStep,
    WaitStep,
)
from geecs_schemas.save_set import SaveRole, SaveSet, SaveSetEntry
from geecs_schemas.scan_request import (
    AcquisitionMode,
    ActionBindings,
    EvaluatorSpec,
    GeneratorSpec,
    OptimizationSpec,
    PositionList,
    PositionRange,
    Positions,
    ScanRequest,
    ScanRequestMode,
)
from geecs_schemas.scan_variables import (
    CompositeMode,
    PseudoScanVariable,
    PseudoTarget,
    ScanVariable,
    ScanVariables,
    ScanVariableSpec,
)
from geecs_schemas.trigger_profile import (
    TriggerProfile,
    TriggerState,
    TriggerVariant,
)

__all__ = [
    "SchemaModel",
    "VersionedSchemaModel",
    # scan_request
    "ScanRequest",
    "ScanRequestMode",
    "AcquisitionMode",
    "ActionBindings",
    "PositionRange",
    "PositionList",
    "Positions",
    "OptimizationSpec",
    "EvaluatorSpec",
    "GeneratorSpec",
    # save_set
    "SaveSet",
    "SaveSetEntry",
    "SaveRole",
    # scan_variables
    "ScanVariables",
    "ScanVariable",
    "ScanVariableSpec",
    "PseudoScanVariable",
    "PseudoTarget",
    "CompositeMode",
    # trigger_profile
    "TriggerProfile",
    "TriggerVariant",
    "TriggerState",
    # action_plan
    "ActionPlan",
    "ActionPlanLibrary",
    "ActionStep",
    "SetStep",
    "WaitStep",
    "CheckStep",
    "RunPlanStep",
    "SCHEMA_REGISTRY",
]

# kind → top-level document model, for generic tooling (loaders, editors,
# docgen). Keys are the canonical config-kind identifiers.
SCHEMA_REGISTRY: dict[str, type[VersionedSchemaModel]] = {
    "scan_request": ScanRequest,
    "save_set": SaveSet,
    "scan_variables": ScanVariables,
    "trigger_profile": TriggerProfile,
    "action_plan": ActionPlan,
    "action_plan_library": ActionPlanLibrary,
}
