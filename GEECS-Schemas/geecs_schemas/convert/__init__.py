"""Converters from the remaining legacy scanner-config YAML dialects.

Scan variables have no converter: ``scan_variables.yaml`` is new-schema only
(GEECS-Plugins#779).  Analysis diagnostics have none either: the corpus was
rewritten to v2 once (0.19.0) and is authored v2-only since.  Presets have
none: the legacy save elements and scan presets were regenerated once as
``Preset`` documents (GEECS-Plugins#807, phase 1 PR 2) and the converters
went with them.

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
from geecs_schemas.convert.trigger_profiles import (
    convert_shot_control,
)

__all__ = [
    "SchemaConversionError",
    "convert_action_library",
    "convert_assigned_actions",
    "convert_shot_control",
    "convert_optimizer_config",
    "OptimizerConversion",
]
