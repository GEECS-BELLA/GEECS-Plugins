"""The GEECS variable type, resolved from the DB the one way every consumer agrees on.

The canonical type of a GEECS variable is ``devicetype_variable.choice_id``
→ the ``choice`` table: ids 1–4 are the base types (``numeric``, ``string``,
``1darray``, ``image``; ``path`` also appears as a descriptor) and every row
from 5 up is an enum option list (``on,off``).  The ``variabletype`` column
on the same table is a secondary annotation, blank on roughly half the rows.
:func:`effective_vartype` folds the two into one answer, descriptor first.
Known DB defect for the type sweep (Sam, 2026-09-09): 18 Undulator rows
carry ``variabletype='numeric'`` with a filter-wheel style option list
(``1,2,3,4,5,6``) that names configurations, not numbers.  They *should* be
``choice`` — fix them in the DB (``variabletype``), not here: this rule is
what the gateway serves today, and every consumer's declared type must keep
matching the served PV (changing the rule alone would break the existing
scan path's ``float`` declarations for those variables).

This is the rule the CA gateway types its PVs with (``GeecsCAGateway``), the
PVA gateway picks image variables with, and GeecsBluesky declares its ophyd
signal types with — so a variable's declared client type always matches
the served PV.  It lived in ``geecs_ca_gateway.config`` until 2026-09-09
and moved here (GEECS-Core 0.5.0) so that no package has to import a
gateway's config module to know a variable's type.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Literal

logger = logging.getLogger(__name__)

#: The served-PV shape of a scalar variable, as the CA gateway names them.
DType = Literal["float", "int", "string", "path", "enum"]

#: GEECS effective type → served scalar dtype.  Types absent here (``image``,
#: ``1darray``) are not scalar CA data.  ``path`` is distinct from ``string``:
#: EPICS DBR_STRING caps at 40 characters, so path variables (file/save paths
#: routinely exceed that) are served as char-array PVs — the standard EPICS
#: long-string convention (areaDetector ``FilePath`` does the same).  A plain
#: ``string`` stays a native 40-char string PV.
VARTYPE_TO_DTYPE: dict[str, DType] = {
    "numeric": "float",
    "string": "string",
    "path": "path",
    "choice": "enum",
}

#: Effective types that are not scalar CA data (served over PVA, if at all).
SKIP_VARTYPES: frozenset[str] = frozenset({"image", "1darray"})

#: ``choices`` values that are bare type descriptors rather than option lists
#: (the ``choice`` table's low IDs double as type descriptors in the GEECS DB).
CHOICE_TYPE_DESCRIPTORS: frozenset[str] = SKIP_VARTYPES | {"numeric", "string", "path"}


def effective_vartype(variabletype: str | None, choices: str | None) -> str:
    """Resolve the effective GEECS variable type from DB metadata.

    The ``choice`` table's low IDs double as type descriptors, so when
    ``choices`` is a bare descriptor word (``image``, ``1darray``, ``numeric``,
    ``string``, ``path``) it is the AUTHORITATIVE type — even when
    ``variabletype`` says otherwise (e.g. ``variabletype='choice'`` with
    ``choices='image'`` is an image variable streaming raw bytes, not a
    one-option enum).  Otherwise trust ``variabletype``; if it is blank, a real
    option list is a ``choice``, else fall back to ``numeric``.

    Parameters
    ----------
    variabletype : str or None
        The DB ``variabletype`` column (may be blank/None).
    choices : str or None
        The DB ``choices`` column (an option list, a bare type descriptor,
        or blank/None).

    Returns
    -------
    str
        The effective type, lower-cased: one of the descriptor words above,
        a ``variabletype`` value, or ``"choice"`` / ``"numeric"`` fallbacks.
    """
    vartype = (variabletype or "").strip().lower()
    raw_choices = (choices or "").strip()
    descriptor = raw_choices.lower()
    if descriptor in CHOICE_TYPE_DESCRIPTORS:
        return descriptor
    if vartype:
        return vartype
    if "," in raw_choices:
        return "choice"
    return "numeric"


def is_scalar_vartype(effective: str) -> bool:
    """Whether an effective type is scalar CA data (not an image / array)."""
    return effective not in SKIP_VARTYPES


def image_variables(rows) -> list[str]:
    """Names of the image-typed variables among one device's DB metadata rows.

    The camera test shared by the PVA gateway (which serves them) and the
    worker's namespace (which makes such a device plugin-backed, #806):
    ``effective_vartype(variabletype, choices) == "image"``, sorted.
    """
    return sorted(
        str(row["name"])
        for row in rows
        if effective_vartype(row.get("variabletype"), row.get("choices")) == "image"
    )


#: The timestamp ladder the gateways subscribe for every device beside its
#: variables — ``acq_timestamp`` preferred, ``systimestamp`` fallback (the CA
#: gateway's PV_CONTRACT ladder; the PVA gateway's frame stamp).  One
#: constant so the PVA server's subscription and the per-frame scalar
#: filter below cannot drift apart.
TIMESTAMP_LADDER: tuple[str, ...] = ("acq_timestamp", "systimestamp")

#: Effective DB types the PVA file plugin writes as a ``DOUBLE`` per-frame
#: attribute.  Numbers only: an enum's wire value is its text label on both
#: gateways (the CA gateway keeps labels verbatim for ``enum_index``), so a
#: ``choice`` variable would be ``NaN`` in every row; ``string``/``path``/
#: ``image``/``1darray`` are not per-frame scalars.  A camera's enum and
#: text columns therefore ride only in its strict row, never in its stack.
SCALAR_ATTRIBUTE_VARTYPES: frozenset[str] = frozenset({"numeric"})


def scalar_attribute_variables(
    rows: Sequence[dict],
    subscribed: Sequence[str],
    *,
    normalize: Callable[[str], str] | None = None,
) -> list[str]:
    """The subscribed scalars the PVA file plugin writes per frame for one device.

    The subscribed (``get='yes'``) list in DB order — the list the worker's
    namespace makes a device's event columns from, so a gated row and a
    strict row carry the same numeric columns for that device
    (``Planning/native_bluesky/08_gated_batch.md`` §4.4) — restricted to
    :data:`SCALAR_ATTRIBUTE_VARTYPES` and minus :data:`TIMESTAMP_LADDER`
    (already the frame's stamp attributes).  A subscribed name with no
    metadata row has no type and is skipped.  With *normalize* (the PV
    naming contract's ``normalize_component``, which the plugin names its
    datasets with), a second variable normalizing to an earlier one's name
    is dropped with a warning rather than colliding in the file.

    The one home for this rule, beside :func:`image_variables`: the
    gateway builds its roster from it and the worker (phase 2c's s-file
    writer) recovers the row's columns through it.
    """
    types = {
        str(row["name"]): effective_vartype(row.get("variabletype"), row.get("choices"))
        for row in rows
    }
    out: list[str] = []
    seen: set[str] = set()
    for name in subscribed:
        if name in TIMESTAMP_LADDER or types.get(name) not in SCALAR_ATTRIBUTE_VARTYPES:
            continue
        key = normalize(name) if normalize is not None else name
        if key in seen:
            if name not in out:
                logger.warning(
                    "subscribed variable %r normalizes to %r, already taken by "
                    "another variable of the device; not written as a per-frame "
                    "attribute (curate the DB)",
                    name,
                    key,
                )
            continue
        seen.add(key)
        out.append(name)
    return out
