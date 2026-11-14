"""The analysis config editor's mount (``--config-editor``), its previews the run page's own."""

from __future__ import annotations

import functools
import logging
from pathlib import Path

from fastapi import FastAPI

from geecs_portal.routes.analysis_configs import (
    SUMMARY_SHOTS_MAX,
    frame_preview,
    summary_preview,
)
from geecs_portal.state import PortalState

logger = logging.getLogger(__name__)


def mount_config_editor(
    app: FastAPI, portal: PortalState, *, read_only: bool = False
) -> None:
    """Mount ``scan_analysis.config_editor`` at ``/configs`` over the processing tree (``read_only``: every save a 405)."""
    try:
        from scan_analysis.config_editor import create_editor_router
        from scan_analysis.config_store import ConfigStore
    except ImportError as exc:  # the analysis extra without the editor extra
        logger.warning("config editor requested but not installed: %s", exc)
    else:
        app.include_router(
            create_editor_router(
                ConfigStore(Path(portal.processing_config_dir)),
                preview=functools.partial(frame_preview, portal),
                summary_preview=functools.partial(summary_preview, portal),
                summary_shots_max=SUMMARY_SHOTS_MAX,
                read_only=read_only,
                theme_url="/theme",
            ),
            prefix="/configs",
        )
        portal.config_editor_enabled = True
