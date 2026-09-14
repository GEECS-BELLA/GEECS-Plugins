"""The web config editor: a mountable FastAPI router over :class:`~scan_analysis.config_store.ConfigStore`.

One host: the data portal mounts it at ``/configs`` (with a live preview
of the document under edit on the scan page's current shot) and its
**edit configs** link is the full-page form. The standalone
``scan-config-editor`` launcher was removed (owner ruling 2026-09-13):
the editor stays beside the analysis code it configures, and the portal
is its one host. Needs the ``editor`` extra (fastapi, jinja2).
"""

from scan_analysis.config_editor.app import PreviewFn, create_editor_router

__all__ = ["PreviewFn", "create_editor_router"]
