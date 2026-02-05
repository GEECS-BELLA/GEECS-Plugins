"""
Analysis Provenance - Python Reference Implementation.

This module implements the Analysis Provenance Standard v0.1 for tracking
the origin and history of derived data columns in analysis files.

See: standards/analysis-provenance/SPECIFICATION.md

Basic Usage:
    from scan_analysis.provenance import log_provenance

    log_provenance(
        data_file="path/to/s123.txt",
        columns_written=["centroid_x", "centroid_y"],
        software_name="my_analysis",
        software_version="1.0.0"
    )
"""

from scan_analysis.provenance.io import log_provenance, read_provenance
from scan_analysis.provenance.capture import (
    capture_code_version,
    capture_dependencies,
    extract_config_from_analyzer,
)
from scan_analysis.provenance.models import (
    ProvenanceFile,
    AnalysisEntry,
    Software,
    CodeVersion,
)

__all__ = [
    # Main API
    "log_provenance",
    "read_provenance",
    # Capture utilities
    "capture_code_version",
    "capture_dependencies",
    "extract_config_from_analyzer",
    # Models (for advanced use)
    "ProvenanceFile",
    "AnalysisEntry",
    "Software",
    "CodeVersion",
]
