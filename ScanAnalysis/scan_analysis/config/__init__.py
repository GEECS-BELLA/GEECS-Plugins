"""Loading analysis groups and building scan analyzers from diagnostics.

The documents — :class:`~geecs_schemas.analysis.AnalysisDiagnostic` (one
YAML per diagnostic under ``scan_analysis_configs/analyzers/<namespace>/``,
with its ``analyzer:``, ``image:`` and typed ``scan:`` sections) and
:class:`~geecs_schemas.analysis.AnalysisGroup` (``groups/<namespace>/``,
what LiveWatch and the task queue dispatch) — live in GEECS-Schemas; import
them from ``geecs_schemas.analysis``.  This package owns what needs the
analysis stack:

* :mod:`analysis_group_loader` — discover and load groups, resolve each
  reference to a :class:`ResolvedDiagnosticConfig` (diagnostic + file-stem
  id + effective priority).
* :mod:`diagnostic_factory` — :func:`create_scan_analyzer`: a diagnostic →
  the wrapping :class:`~scan_analysis.base.ScanAnalyzer`.

Quick start::

    >>> from scan_analysis.config import load_analysis_group, create_scan_analyzer
    >>> group = load_analysis_group("baseline", config_dir=<scan_analysis_configs>)
    >>> analyzers = [
    ...     create_scan_analyzer(r.diagnostic, id=r.id, priority=r.priority)
    ...     for r in group.analyzers
    ... ]
    >>> for a in analyzers:
    ...     a.run_analysis(scan_tag)

Or one diagnostic directly::

    >>> from image_analysis.config import load_diagnostic
    >>> from scan_analysis.config import create_scan_analyzer
    >>> diag = load_diagnostic("UC_VisaEBeam1")
    >>> diag.image.roi.x_max = 200    # optional notebook tweak
    >>> analyzer = create_scan_analyzer(diag)

``SCAN_ANALYSIS_CONFIG_DIR`` (env var) or ``scan_analysis_configs_path``
in ``~/.config/geecs_python_api/config.ini`` sets the configs root;
ImageAnalysis derives its own search root as ``<root>/analyzers``.
"""

from .analysis_group_loader import (
    LoadedAnalysisGroup,
    ResolvedDiagnosticConfig,
    discover_analyzers,
    discover_groups,
    load_analysis_group,
    resolve_group,
)
from .diagnostic_factory import create_scan_analyzer

__all__ = [
    "LoadedAnalysisGroup",
    "ResolvedDiagnosticConfig",
    "create_scan_analyzer",
    "discover_analyzers",
    "discover_groups",
    "load_analysis_group",
    "resolve_group",
]
