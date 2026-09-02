"""Ephemeral diagnostic runs: analyze in-memory frames, guaranteed write-free.

The scan pipeline runs analyzers *against the scan folder*: several
analyzers persist derived per-shot files next to their input (gated on
``auxiliary_data["file_path"]`` or constructor state — see the
filesystem-invariants section of ``CLAUDE.md``). Read-only viewers (the
data portal's processing selector, notebooks exploring a config) need
the *same* configured pipeline with a hard guarantee that nothing is
written anywhere.

:func:`run_diagnostic_ephemeral` is that seam. The write gate is
structural, not conventional:

* Input is in-memory frames only — the runner never receives, resolves,
  or forwards a file path, so ``analyze_image`` runs with no
  ``file_path`` in its auxiliary data and every path-gated writer stays
  dormant.
* ``auxiliary_data`` containing ``file_path`` is refused loudly
  (``ValueError``) rather than stripped — a caller passing it is
  confused about the contract, and silently dropping the key would turn
  that confusion into wrong-but-plausible output.
* Analyzers with unconditional side effects — writes or subprocess
  spawns not gated on ``file_path`` — are refused by class path via
  :data:`EPHEMERAL_DENYLIST` **before import**, so a denylisted
  analyzer's vendor SDK or DLL dependency is never even imported on
  hosts that lack it. The denylist shrinks as analyzers grow an
  explicit ephemeral mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import create_image_analyzer, load_diagnostic
from .types import Array2D, ImageAnalyzerResult

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from .base import ImageAnalyzer

__all__ = [
    "EPHEMERAL_DENYLIST",
    "render_diagnostic_ephemeral",
    "render_frame_figure",
    "render_result_figure",
    "run_diagnostic_ephemeral",
]

#: Analyzer class paths that cannot run ephemerally: their side effects
#: are not gated on ``auxiliary_data["file_path"]``, so no calling
#: convention makes them pure. Remove an entry only when the analyzer
#: gains an explicit no-write mode.
#:
#: * HASO writes five sidecars per shot from ``load_image`` (instance
#:   state), its module hard-imports the wavekit SDK, and its
#:   ``analyze_image`` without that ``load_image`` returns a
#:   meaningless pass-through — no ephemeral calling convention exists.
#: * Grenouille's ``analyze_image`` unconditionally writes transient
#:   temp files and spawns a ~seconds 32-bit DLL subprocess per frame —
#:   cleaned up afterwards, but a per-request viewer must not trigger
#:   either.
EPHEMERAL_DENYLIST = frozenset(
    {
        "image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor",
        "image_analysis.analyzers.grenouille_analyzer.GrenouilleAnalyzer",
    }
)


def run_diagnostic_ephemeral(
    name_or_path: Union[str, Path],
    frames: Sequence[Array2D],
    *,
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    auxiliary_data: Optional[Dict[str, Any]] = None,
) -> List[ImageAnalyzerResult]:
    """Run a configured diagnostic over in-memory frames without writing.

    Loads the diagnostic YAML, instantiates its analyzer once, and calls
    ``analyze_image`` per frame. See the module docstring for the
    write-free contract this enforces.

    Parameters
    ----------
    name_or_path : str or Path
        Diagnostic ID or YAML path, per :func:`~.config.load_diagnostic`.
    frames : sequence of ndarray
        In-memory frames to analyze (already loaded pixels — this seam
        never touches the filesystem). One result per frame.
    config_dir : Path, optional
        Configs-tree root forwarded to ``load_diagnostic``. Callers
        outside the scan pipeline (the portal) should pass their own
        explicitly rather than rely on the global default.
    overrides : dict, optional
        Deep-merged into the raw YAML, per ``load_diagnostic``.
    auxiliary_data : dict, optional
        Forwarded to every ``analyze_image`` call, top-level-copied per
        frame (an analyzer adding/removing keys cannot leak across
        frames; mutation *inside* a nested value still would — don't
        pass shared mutables you care about). Must not contain
        ``file_path``.

    Returns
    -------
    list of ImageAnalyzerResult
        One result per input frame, in order.

    Raises
    ------
    ValueError
        If ``auxiliary_data`` contains ``file_path``, or the diagnostic's
        analyzer class is on :data:`EPHEMERAL_DENYLIST`.
    """
    analyzer = _ephemeral_analyzer(
        name_or_path,
        config_dir=config_dir,
        overrides=overrides,
        auxiliary_data=auxiliary_data,
    )
    return _analyze_frames(analyzer, frames, auxiliary_data)


def _ephemeral_analyzer(
    name_or_path: Union[str, Path],
    *,
    config_dir: Optional[Path],
    overrides: Optional[Dict[str, Any]],
    auxiliary_data: Optional[Dict[str, Any]],
) -> "ImageAnalyzer":
    """The write-free gate + one analyzer instantiation (shared by both runners)."""
    if auxiliary_data is not None and "file_path" in auxiliary_data:
        raise ValueError(
            "auxiliary_data['file_path'] is forbidden in ephemeral runs: "
            "it is the gate several analyzers use to write derived files "
            "next to their input. Pass loaded frames only."
        )

    diag = load_diagnostic(name_or_path, config_dir=config_dir, overrides=overrides)

    class_path = diag.image_analyzer.class_path
    if class_path in EPHEMERAL_DENYLIST:
        raise ValueError(
            f"Analyzer {class_path} cannot run ephemerally: its side "
            f"effects (writes / subprocess spawns) are not gated on "
            f"auxiliary file paths. Use the scan pipeline for this "
            f"diagnostic, or give the analyzer a no-write mode and "
            f"remove it from EPHEMERAL_DENYLIST."
        )

    return create_image_analyzer(diag)


def _analyze_frames(
    analyzer: "ImageAnalyzer",
    frames: Sequence[Array2D],
    auxiliary_data: Optional[Dict[str, Any]],
) -> List[ImageAnalyzerResult]:
    return [
        analyzer.analyze_image(
            frame, dict(auxiliary_data) if auxiliary_data is not None else None
        )
        for frame in frames
    ]


def _new_figure(figsize: Tuple[float, float], dpi: int) -> Tuple["Figure", Any]:
    """An object-API figure + axes (no pyplot: no global registry, thread-safe)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=figsize, dpi=dpi, constrained_layout=True)
    return fig, fig.subplots()


def _window_limits(
    image: Optional[np.ndarray], window: Optional[Tuple[float, float]]
) -> Dict[str, float]:
    """``vmin``/``vmax`` from a percentile *window* over *image* (or nothing)."""
    if window is None or image is None:
        return {}
    lo, hi = np.nanpercentile(np.asarray(image, dtype=float), list(window))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return {}
    return {"vmin": float(lo), "vmax": float(hi)}


def render_result_figure(
    analyzer: "ImageAnalyzer",
    result: ImageAnalyzerResult,
    *,
    window: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (5.0, 4.2),
    dpi: int = 110,
) -> "Figure":
    """Draw *result* with *analyzer*'s own ``render_image`` into an object-API figure.

    The renderer is handed our axes, so its overlays (projections,
    markers, calibrated axes — whatever ``render_data`` carries) land on
    a ``matplotlib.figure.Figure`` that never touched pyplot. The base
    renderer skips its colorbar when given an axes, so one is added
    here whenever an image was drawn. 2D results take ``cmap`` and a
    percentile ``window`` (→ ``vmin``/``vmax`` over the processed
    image); 1D renderers take neither (their signature is
    ``ax`` + plot kwargs).

    Raises
    ------
    AttributeError
        If the analyzer has no ``render_image`` (every Standard-family
        analyzer does).
    ValueError
        Propagated from the renderer (e.g. a 1D result handed to a 2D
        renderer).
    """
    fig, ax = _new_figure(figsize, dpi)
    kwargs: Dict[str, Any] = {}
    if result.data_type == "2d":
        if cmap:
            kwargs["cmap"] = cmap
        kwargs.update(_window_limits(result.processed_image, window))
    analyzer.render_image(result, ax=ax, **kwargs)
    if ax.images:
        fig.colorbar(ax.images[0], ax=ax, shrink=0.65)
    return fig


def render_frame_figure(
    image: Array2D,
    *,
    window: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (5.0, 4.2),
    dpi: int = 110,
) -> "Figure":
    """Draw a bare 2D image with the base renderer (no analyzer overlays).

    For images that are not one analyzer result — an average of several
    processed frames, whose per-shot overlays do not average meaningfully.
    """
    from .tools.rendering import base_render_image

    fig, ax = _new_figure(figsize, dpi)
    result = ImageAnalyzerResult(data_type="2d", processed_image=np.asarray(image))
    kwargs: Dict[str, Any] = {"cmap": cmap} if cmap else {}
    kwargs.update(_window_limits(result.processed_image, window))
    base_render_image(result, ax=ax, **kwargs)
    if ax.images:
        fig.colorbar(ax.images[0], ax=ax, shrink=0.65)
    return fig


def render_diagnostic_ephemeral(
    name_or_path: Union[str, Path],
    frames: Sequence[Array2D],
    *,
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    auxiliary_data: Optional[Dict[str, Any]] = None,
    window: Optional[Tuple[float, float]] = None,
    cmap: Optional[str] = None,
    figsize: Tuple[float, float] = (5.0, 4.2),
    dpi: int = 110,
) -> List["Figure"]:
    """Run a diagnostic over in-memory frames and draw each result its own way.

    :func:`run_diagnostic_ephemeral` plus :func:`render_result_figure`
    per frame, with one analyzer instantiation for the batch and the
    same write-free contract (the ``file_path`` refusal and the
    denylist apply unchanged). Returns object-API figures — no pyplot
    state is created, so this is safe on a request threadpool.

    Parameters
    ----------
    name_or_path, frames, config_dir, overrides, auxiliary_data
        As for :func:`run_diagnostic_ephemeral`.
    window : (lo, hi) percentiles, optional
        Display window over each processed image → ``vmin``/``vmax``.
    cmap : str, optional
        Matplotlib colormap name for 2D results.
    figsize, dpi
        Figure geometry.

    Returns
    -------
    list of matplotlib.figure.Figure
        One figure per input frame, in order.
    """
    analyzer = _ephemeral_analyzer(
        name_or_path,
        config_dir=config_dir,
        overrides=overrides,
        auxiliary_data=auxiliary_data,
    )
    return [
        render_result_figure(
            analyzer, result, window=window, cmap=cmap, figsize=figsize, dpi=dpi
        )
        for result in _analyze_frames(analyzer, frames, auxiliary_data)
    ]
