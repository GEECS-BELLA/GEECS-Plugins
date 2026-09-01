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
* Analyzers that write unconditionally from instance state are refused
  by class path via :data:`EPHEMERAL_DENYLIST` **before import** — the
  one current entry (HASO) both writes five sidecars per shot and
  imports a vendor SDK, so the pre-import check doubles as keeping
  wavekit off non-Windows hosts. The denylist shrinks as analyzers grow
  an explicit ephemeral mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .config import create_image_analyzer, load_diagnostic
from .types import Array2D, ImageAnalyzerResult

__all__ = ["EPHEMERAL_DENYLIST", "run_diagnostic_ephemeral"]

#: Analyzer class paths that cannot run ephemerally: they write derived
#: files unconditionally (not gated on ``auxiliary_data["file_path"]``),
#: so no calling convention makes them pure. Remove an entry only when
#: the analyzer gains an explicit no-write mode.
EPHEMERAL_DENYLIST = frozenset(
    {
        "image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor",
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
        Forwarded to every ``analyze_image`` call (shallow-copied per
        frame so per-shot mutation by an analyzer cannot leak across
        frames). Must not contain ``file_path``.

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
            f"Analyzer {class_path} cannot run ephemerally: it writes "
            f"derived files unconditionally. Use the scan pipeline for "
            f"this diagnostic, or give the analyzer a no-write mode and "
            f"remove it from EPHEMERAL_DENYLIST."
        )

    analyzer = create_image_analyzer(diag)
    return [
        analyzer.analyze_image(
            frame, dict(auxiliary_data) if auxiliary_data is not None else None
        )
        for frame in frames
    ]
