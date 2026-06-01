"""Factory: typed diagnostic config → live ImageAnalyzer instance.

One public function:

* :func:`create_image_analyzer` — take a validated
  :class:`DiagnosticAnalysisConfig`, import the named class, instantiate
  it with the embedded image config. Returns a live ``ImageAnalyzer``.

For Mode 2 (load from YAML) compose with :func:`load_diagnostic` from
:mod:`image_analysis.config.loader`:

>>> from image_analysis.config import load_diagnostic, create_image_analyzer
>>> diag = load_diagnostic("UC_Amp4_IR_Input")
>>> diag.image.roi.x_max = 200    # optional notebook tweak
>>> analyzer = create_image_analyzer(diag)
>>> result = analyzer.analyze_image_file(some_path)

For Mode 1 (no YAML), construct a ``DiagnosticAnalysisConfig`` in code
and hand it to ``create_image_analyzer`` directly — same factory,
same result.

The ScanAnalysis-side ``create_scan_analyzer`` builds on this factory:
it calls ``create_image_analyzer`` for the inner analyzer, then wraps
it in the dimension-specific ``Array{1,2}DScanAnalyzer``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Type

from .array1d_processing import Line1DConfig
from .array2d_processing import CameraConfig
from .diagnostic import DiagnosticAnalysisConfig, ImageAnalyzerSpec

logger = logging.getLogger(__name__)

__all__ = ["create_image_analyzer"]


def create_image_analyzer(diag: DiagnosticAnalysisConfig) -> Any:
    """Build a live ImageAnalyzer instance from a validated diagnostic config.

    Imports the class named by ``diag.image_analyzer.class_path`` and
    instantiates it with the embedded image config plus any extra
    constructor kwargs. The embedded ``image:`` section has already
    been validated against the right Pydantic model
    (:class:`CameraConfig` or :class:`Line1DConfig`) at
    ``DiagnosticAnalysisConfig`` construction time — the discriminator
    on the ``image:`` field picked the variant.

    Parameters
    ----------
    diag : DiagnosticAnalysisConfig
        Validated diagnostic. The ``image_analyzer`` spec gives the
        class to import and any extra constructor kwargs; the type of
        ``diag.image`` decides which constructor kwarg name to use.

    Returns
    -------
    ImageAnalyzer
        Fully-configured analyzer instance. Caller can immediately
        call ``analyze_image`` / ``analyze_image_file`` on it.

    Raises
    ------
    ImportError, AttributeError
        If ``class_path`` does not resolve to an importable class.
    TypeError
        If the resolved class can't be instantiated with the inferred
        kwargs.
    """
    return _instantiate_image_analyzer(
        diag.image_analyzer,
        diag.image,
        output_name=diag.effective_output_name,
    )


def _instantiate_image_analyzer(
    spec: ImageAnalyzerSpec,
    image_config: Any,
    *,
    output_name: str | None = None,
) -> Any:
    """Import the class and instantiate with kwargs + image config.

    The kwarg name (``camera_config`` vs ``line_config``) is decided
    by the type of the validated image config. When ``image_config`` is
    ``None``, no image-config kwarg is injected — the analyzer takes
    only what ``spec.kwargs`` provides (HASO-style).

    ``output_name`` is forwarded to analyzers that accept it (Standard
    family + subclasses); analyzers that don't accept the kwarg get a
    fallback instantiation without it so HASO-style and other
    out-of-family analyzers stay compatible.
    """
    analyzer_class = _import_class(spec.class_path)
    kwargs = dict(spec.kwargs)

    if isinstance(image_config, CameraConfig):
        kwargs["camera_config"] = image_config
    elif isinstance(image_config, Line1DConfig):
        kwargs["line_config"] = image_config
    # image_config is None: kwargs is exactly what the YAML supplied.

    if output_name is not None:
        kwargs.setdefault("output_name", output_name)

    logger.info(
        "Instantiating %s with kwargs %s",
        spec.class_path,
        sorted(kwargs),
    )
    try:
        return analyzer_class(**kwargs)
    except TypeError as exc:
        # Fallback for analyzers that don't accept output_name
        # (out-of-family / legacy analyzers). Drop it and retry once.
        if "output_name" in kwargs:
            retry_kwargs = {k: v for k, v in kwargs.items() if k != "output_name"}
            try:
                return analyzer_class(**retry_kwargs)
            except TypeError:
                pass
        raise TypeError(
            f"Failed to instantiate {spec.class_path} with kwargs "
            f"{sorted(kwargs)}: {exc}"
        ) from exc


def _import_class(class_path: str) -> Type:
    """Resolve a dotted import path to a class object."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError as exc:
        raise ValueError(
            f"Invalid class path '{class_path}'. Must be fully qualified "
            f"(e.g. 'module.Class')."
        ) from exc

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import module '{module_path}' for class path '{class_path}': {exc}"
        ) from exc

    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_path}' has no class '{class_name}' "
            f"(from class path '{class_path}'): {exc}"
        ) from exc
