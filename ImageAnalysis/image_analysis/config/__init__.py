"""Loading and instantiating analysis diagnostics.

The config *models* live in GEECS-Schemas (``geecs_schemas.analysis``): the
whole diagnostic document — ``analyzer:`` spec, ``image:`` processing
section, ``scan:`` runtime — validates with pydantic alone.  Import them
from there.  This package owns what needs the analysis stack:

* :mod:`loader` — YAML → typed model: :func:`load_diagnostic`,
  :func:`load_camera_config`, :func:`load_line_config`,
  :func:`list_diagnostics`, :func:`find_config_file`.
* :mod:`factory` — typed diagnostic → live analyzer:
  :func:`create_image_analyzer`.
* :mod:`registry` — analyzer ``kind`` → implementing class.
"""

from .factory import create_image_analyzer
from .loader import (
    find_config_file,
    list_diagnostics,
    load_camera_config,
    load_diagnostic,
    load_line_config,
)
from .registry import ANALYZER_CLASS_PATHS, analyzer_class

__all__ = [
    "ANALYZER_CLASS_PATHS",
    "analyzer_class",
    "create_image_analyzer",
    "find_config_file",
    "list_diagnostics",
    "load_camera_config",
    "load_diagnostic",
    "load_line_config",
]
