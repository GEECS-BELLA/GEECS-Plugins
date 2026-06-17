"""Exploratory helpers that sit beside modeling packages without importing ``sklearn``.

Currently exposes correlation ranking; extend here as more scan/QC summaries arrive.
"""

from geecs_data_utils.analysis.correlation import CorrelationMethod, CorrelationReport

__all__ = ["CorrelationMethod", "CorrelationReport"]
