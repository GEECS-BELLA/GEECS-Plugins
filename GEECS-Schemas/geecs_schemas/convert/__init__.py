"""Converters from every legacy scanner-config YAML dialect to the new schemas.

One module per legacy dialect; every converter accepts a parsed dict or a
YAML path and fails loudly (:class:`SchemaConversionError`) naming exactly
what could not be mapped.  Migration converters live next to the schemas
they migrate (vision doc §4) so a flag day is never needed.
"""

from geecs_schemas.convert._common import SchemaConversionError
from geecs_schemas.convert.actions import (
    convert_action_library,
    convert_assigned_actions,
)
from geecs_schemas.convert.optimizer_configs import (
    OptimizerConversion,
    convert_optimizer_config,
)
from geecs_schemas.convert.presets import (
    PresetConversion,
    compose_save_sets,
    convert_scan_preset,
)
from geecs_schemas.convert.save_elements import (
    SaveElementConversion,
    convert_save_element,
)
from geecs_schemas.convert.scan_variables import convert_scan_variables
from geecs_schemas.convert.trigger_profiles import (
    convert_shot_control,
    merge_trigger_variant,
)

__all__ = [
    "SchemaConversionError",
    "convert_action_library",
    "convert_assigned_actions",
    "convert_save_element",
    "SaveElementConversion",
    "convert_scan_variables",
    "convert_shot_control",
    "merge_trigger_variant",
    "convert_scan_preset",
    "compose_save_sets",
    "PresetConversion",
    "convert_optimizer_config",
    "OptimizerConversion",
]
