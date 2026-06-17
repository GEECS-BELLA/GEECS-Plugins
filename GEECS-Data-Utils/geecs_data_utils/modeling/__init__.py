"""High-level modeling namespace for GEECS workflows.

This package stays thin; concrete backends live in subpackages. The supported
backend is machine learning under :mod:`geecs_data_utils.modeling.ml`.

Notes
-----
Non-ML data loading and cleaning live in :mod:`geecs_data_utils.data`.
Non-ML analysis (e.g. correlation rankings) lives in
:mod:`geecs_data_utils.analysis`.

See Also
--------
geecs_data_utils.modeling.ml : Sklearn regression, artifacts, inference.
"""
