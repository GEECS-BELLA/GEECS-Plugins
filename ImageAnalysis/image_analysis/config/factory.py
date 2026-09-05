"""Factory: validated diagnostic → live ImageAnalyzer instance.

One public function, :func:`create_image_analyzer`.  The diagnostic's
``analyzer.kind`` picks the class through
:mod:`image_analysis.config.registry`; the analyzer receives its typed
spec (``spec=``), the validated ``image:`` section under the constructor
name its family expects (``camera_config`` / ``line_config``), and the
diagnostic's ``effective_output_name`` (``output_name=``).  Each kwarg is
passed only when the constructor declares it — analyzers with no
parameters take no ``spec``, no-image analyzers take no image config —
so a real signature mismatch still surfaces as the constructor's own
``TypeError``.

For Mode 2 (load from YAML) compose with :func:`load_diagnostic`:

>>> from image_analysis.config import load_diagnostic, create_image_analyzer
>>> diag = load_diagnostic("UC_Amp4_IR_Input")
>>> diag.image.roi.x_max = 200    # optional notebook tweak
>>> analyzer = create_image_analyzer(diag)

For Mode 1 (no YAML), build an :class:`AnalysisDiagnostic` in code and
hand it to the same factory.  ScanAnalysis' ``create_scan_analyzer``
builds on this: it makes the inner analyzer here, then wraps it in the
dimension-specific ``Array{1,2}DScanAnalyzer``.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Type

from geecs_schemas.analysis import AnalysisDiagnostic, CameraConfig, Line1DConfig

from .registry import analyzer_class

logger = logging.getLogger(__name__)

__all__ = ["create_image_analyzer"]


def create_image_analyzer(diag: AnalysisDiagnostic) -> Any:
    """Build a live ImageAnalyzer from a validated diagnostic.

    Parameters
    ----------
    diag : AnalysisDiagnostic
        Validated diagnostic (v1 documents are already lifted).

    Returns
    -------
    ImageAnalyzer
        Ready for ``analyze_image`` / ``analyze_image_file``.

    Raises
    ------
    ImportError, AttributeError
        If the kind's class cannot be imported on this host (vendor SDK).
    TypeError
        If the class rejects the kwargs — a signature/registry mismatch.
    """
    cls = analyzer_class(diag.analyzer.kind)
    kwargs: dict[str, Any] = {}
    if isinstance(diag.image, CameraConfig):
        kwargs["camera_config"] = diag.image
    elif isinstance(diag.image, Line1DConfig):
        kwargs["line_config"] = diag.image
    if _accepts_kwarg(cls, "spec"):
        kwargs["spec"] = diag.analyzer
    if _accepts_kwarg(cls, "output_name"):
        kwargs["output_name"] = diag.effective_output_name

    logger.info(
        "Instantiating %s (kind=%s) with kwargs %s",
        cls.__name__,
        diag.analyzer.kind,
        sorted(kwargs),
    )
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate {cls.__module__}.{cls.__name__} for analyzer "
            f"kind {diag.analyzer.kind!r} with kwargs {sorted(kwargs)}: {exc}"
        ) from exc


def _accepts_kwarg(cls: Type, name: str) -> bool:
    """Whether ``cls.__init__`` declares keyword ``name`` (or a ``**kwargs`` catch-all)."""
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.name == name or param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False
