"""Match VOCS-style variable names to s-file column names (alias-tolerant).

GEECS save-elements rewrites VOCS-style names into the s-file's own format:

- The first colon (device/variable separator) becomes a space:
    VOCS:   ``U_ESP_JetXYZ:Position.Axis 1``
    s-file: ``U_ESP_JetXYZ Position.Axis 1``
- An optional alias is appended as ``" Alias:<alias text>"``:
    ``U_ESP_JetXYZ Position.Axis 1 Alias:Jet_X (mm)``

This module provides one function, :func:`match_vocs_to_sfile_column`, which
covers the common cases and raises a useful error otherwise.
"""

from __future__ import annotations

from typing import Iterable


def match_vocs_to_sfile_column(vocs_name: str, df_columns: Iterable[str]) -> str:
    """Resolve a VOCS-style variable name to its s-file column.

    Match order: exact name → colon-to-space bare form → bare form
    followed by ``" Alias:<anything>"`` → any column starting with either
    form.

    Raises
    ------
    KeyError
        On missing or ambiguous match. The message lists the candidates
        considered (and the first 30 columns when nothing matched).
    """
    cols = list(df_columns)
    if vocs_name in cols:
        return vocs_name

    bare = vocs_name.replace(":", " ", 1)
    if bare in cols:
        return bare

    aliased = [c for c in cols if c.startswith(bare + " Alias:")]
    if len(aliased) == 1:
        return aliased[0]
    if len(aliased) > 1:
        raise KeyError(
            f"Multiple alias matches for VOCS variable {vocs_name!r}: {aliased}"
        )

    starts = [c for c in cols if c.startswith(bare) or c.startswith(vocs_name)]
    if len(starts) == 1:
        return starts[0]
    if len(starts) > 1:
        raise KeyError(
            f"Ambiguous prefix match for VOCS variable {vocs_name!r}: {starts}"
        )

    raise KeyError(
        f"No s-file column matches VOCS variable {vocs_name!r}. "
        f"Tried exact, bare {bare!r}, and '{bare} Alias:*'. "
        f"First 30 columns: {cols[:30]}"
    )
