"""Exploratory tabular helpers over scan scalar frames.

Currently exposes correlation ranking; extend here as more scan/QC summaries arrive.
"""

from geecs_data_utils.analysis.correlation import CorrelationMethod, CorrelationReport

__all__ = ["CorrelationMethod", "CorrelationReport"]
