"""The web config editor: a mountable FastAPI router over :class:`~scan_analysis.config_store.ConfigStore`.

Two hosts share it: the data portal mounts it at ``/configs`` (with a live
preview of the document under edit on the scan page's current shot), and
``scan-config-editor`` serves it standalone against a local configs
checkout.  Needs the ``editor`` extra (fastapi, jinja2, uvicorn).
"""

from scan_analysis.config_editor.app import (
    PreviewFn,
    create_editor_app,
    create_editor_router,
    main,
)

__all__ = ["PreviewFn", "create_editor_app", "create_editor_router", "main"]
