"""Top-level initialization for the GEECS-Scanner package."""

import logging

from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

GEECS_Plugins_folder = Path(__file__).parents[2]
