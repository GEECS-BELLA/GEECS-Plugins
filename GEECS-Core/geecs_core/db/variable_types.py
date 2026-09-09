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

from typing import Literal

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
