"""Shared experiment-name validation for the per-experiment config stores.

Every config store joins the experiment name onto the experiments root
(``scanner_configs/experiments/<Experiment>/...``) and creates parent
directories before writing.  The main window's experiment selector is
editable, so the raw text can be a traversal string like
``../OtherExperiment`` — joined unchecked, it escapes the intended
experiment folder and can create or corrupt another experiment's real
configs (issue #513).  :func:`check_experiment_name` is the one guard,
applied by every store **before** any path join, so load/save/list/delete
raise the store's own error before any directory is created.

The semantics mirror the original ``ScanVariableStore._check_experiment``
(the store that already had the guard): the empty string is *allowed* —
"no experiment selected" is a separate, friendlier error each store raises
itself — but any path separator or relative-path special name is rejected.
"""

from __future__ import annotations


def check_experiment_name(experiment: str, error_type: type[Exception]) -> None:
    r"""Reject an experiment name that would escape the experiments root.

    Parameters
    ----------
    experiment : str
        The candidate experiment folder name.  ``""`` passes — the stores
        handle "no experiment selected" separately.
    error_type : type of Exception
        The calling store's error class (each store surfaces its own
        status-bar-ready error type).

    Raises
    ------
    Exception
        An instance of *error_type* when the name contains a path separator
        (``/`` or ``\\``) or is a relative-path special name (``.`` or
        ``..``).
    """
    if any(sep in experiment for sep in ("/", "\\")) or experiment in (".", ".."):
        raise error_type(
            f"Experiment name {experiment!r} must be a plain folder name "
            "(no path separators)."
        )
