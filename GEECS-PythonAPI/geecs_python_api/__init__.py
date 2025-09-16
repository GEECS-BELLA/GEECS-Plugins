"""
Top-level initialization for the GEECS-Scanner package.

This module defines basic setup performed when the package is imported,
This module defines basic setup performed when the package is imported.

- Configuring a module-level logger with a `NullHandler` to prevent
  "No handler found" warnings. Applications using this package should
  configure logging explicitly.
- Exposing the `GEECS_Plugins_folder` path, which points two levels
  above the current file and is used for locating plugin resources.

Attributes
----------
logger : logging.Logger
    Package-level logger for the `geecs_scanner` namespace.
    By default, a `NullHandler` is attached.
GEECS_Plugins_folder : pathlib.Path
    Path to the root plugins directory, resolved relative to this file.
"""

import logging

from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

GEECS_Plugins_folder = Path(__file__).parents[2]
