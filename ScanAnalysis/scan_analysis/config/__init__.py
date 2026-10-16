"""Loading analysis documents and groups, and building scan analyzers from them.

The documents — :class:`~geecs_schemas.analysis.AnalysisRecipe` (one YAML
per diagnostic under ``scan_analysis_configs/analyzers/<namespace>/``), the
``.himg`` jobs' :class:`~geecs_schemas.analysis.HimgJob`, and
:class:`~geecs_schemas.analysis.AnalysisGroup` (``groups/<namespace>/``,
what explicit analysis runs dispatch) — live in GEECS-Schemas; import
them from ``geecs_schemas.analysis``.  This package owns what needs the
analysis stack:

* :mod:`loader` — :func:`load_diagnostic`: one document by stem or path,
  the client ``config.ini`` root as the default tree.
* :mod:`analysis_group_loader` — discover and load groups, resolve each
  reference to a :class:`ResolvedDiagnosticConfig` (document + file-stem
  id + effective priority).
* :mod:`diagnostic_factory` — :func:`create_scan_analyzer`: a document →
  the :class:`~scan_analysis.base.ScanAnalyzer` that runs it.

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

    >>> from scan_analysis.config import create_scan_analyzer, load_diagnostic
    >>> recipe = load_diagnostic("UC_VisaEBeam1")
    >>> analyzer = create_scan_analyzer(recipe)

``scan_analysis_configs_path`` in ``~/.config/geecs_python_api/config.ini``
sets the default configs root; every loader takes ``config_dir=`` instead.
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
from .loader import load_diagnostic

__all__ = [
    "LoadedAnalysisGroup",
    "ResolvedDiagnosticConfig",
    "create_scan_analyzer",
    "discover_analyzers",
    "discover_groups",
    "load_analysis_group",
    "load_diagnostic",
    "resolve_group",
]
