"""Root conftest.py — collection configuration for the full test suite.

Individual broken/optional test files that require unavailable dependencies
(online_analysis, pytestqt, etc.) are excluded here so they don't block
collection of the rest of the suite.
"""

collect_ignore = [
    # Requires online_analysis (unavailable)
    "ImageAnalysis/tests/analyzers/test_ICT_Analysis.py",
    "ImageAnalysis/tests/analyzers/test_U_FROG_Grenouille.py",
    # Requires GEECS-Plugins-Configs repo to be configured (integration dep)
    "ImageAnalysis/tests/analyzers/test_beamanalyzer.py",
]
