"""Analyzer kind → implementing class: the one table ImageAnalysis keeps for the v2 schema.

The diagnostic document (``geecs_schemas.analysis.AnalysisDiagnostic``)
names its analyzer by ``kind``; the class path left the YAML in format v2
so a module refactor never breaks a config.  This module resolves a kind
to its class — lazily, so importing the registry never imports vendor
SDKs (HASO's WaveKit, the FROG DLL wrapper).

Adding an analyzer: one spec model in ``geecs_schemas.analysis.analyzers``
(joined into ``AnalyzerSpec``) and one line in :data:`ANALYZER_CLASS_PATHS`
here.  ``tests/test_config_registry.py`` pins that the two tables cover the
same kinds.
"""

from __future__ import annotations

import importlib
from typing import Type

from geecs_schemas.analysis import ANALYZER_SPECS

__all__ = ["ANALYZER_CLASS_PATHS", "analyzer_class", "import_class_path"]

#: kind → fully qualified class path of the ImageAnalyzer implementing it.
ANALYZER_CLASS_PATHS: dict[str, str] = {
    "standard": "image_analysis.analyzers.standard_analyzer.StandardAnalyzer",
    "line": "image_analysis.analyzers.line_analyzer.LineAnalyzer",
    "beam": "image_analysis.analyzers.beam_analyzer.BeamAnalyzer",
    "magspec": "image_analysis.analyzers.magspec_manual_calib_analyzer.MagSpecManualCalibAnalyzer",
    "frog_retrieval": "image_analysis.analyzers.grenouille_analyzer.GrenouilleAnalyzer",
    "frog_spectral_phase": "image_analysis.analyzers.frog_spectral_phase_analyzer.FrogSpectralPhaseAnalyzer",
    "ict": "image_analysis.analyzers.ict_1d_analyzer.ICT1DAnalyzer",
    "line_stitcher": "image_analysis.analyzers.line_stitcher.LineStitcher",
    "haso": "image_analysis.analyzers.HASO_himg_has_processor.HASOHimgHasProcessor",
    "downramp_phase": "image_analysis.analyzers.downramp_phase_analyzer.DownrampPhaseAnalyzer",
    "hi_res_mag_cam": "image_analysis.analyzers.Undulator.hi_res_mag_cam_analyzer.HiResMagCamAnalyzer",
    "bcave_magspec_stitcher": "image_analysis.analyzers.Undulator.BCaveMagSpecStitcher.BCaveMagSpecStitcherAnalyzer",
    "bcave_mag_opt": "image_analysis.analyzers.Undulator.BCaveMagSpecStitcherOpt.BCaveMagOpt",
    "phase_downramp": "image_analysis.analyzers.density_from_phase_analysis.PhaseDownrampProcessor",
}

_missing = set(ANALYZER_SPECS) ^ set(ANALYZER_CLASS_PATHS)
if _missing:  # pragma: no cover — a packaging error, caught at import
    raise RuntimeError(
        "ImageAnalysis analyzer registry and geecs_schemas.analysis.ANALYZER_SPECS "
        f"disagree on kinds: {sorted(_missing)}"
    )


def import_class_path(class_path: str) -> Type:
    """Resolve a dotted ``module.Class`` path to the class object.

    Raises
    ------
    ImportError
        If the module cannot be imported (a vendor SDK missing on this host,
        for example) — the original message is preserved.
    AttributeError
        If the module has no such class.
    """
    module_path, class_name = class_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import module '{module_path}' for analyzer class "
            f"'{class_path}': {exc}"
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_path}' has no class '{class_name}'"
        ) from exc


def analyzer_class(kind: str) -> Type:
    """Return the ImageAnalyzer class implementing analyzer ``kind`` (imported on demand)."""
    try:
        class_path = ANALYZER_CLASS_PATHS[kind]
    except KeyError as exc:
        raise KeyError(
            f"Unknown analyzer kind {kind!r}; known kinds: {sorted(ANALYZER_CLASS_PATHS)}"
        ) from exc
    return import_class_path(class_path)
