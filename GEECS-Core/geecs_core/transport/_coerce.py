"""Shared wire-value coercion for the GEECS transport layers."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any


def coerce_scalar(s: str) -> Any:
    """Best-effort numeric conversion; non-numeric text passes through as-is.

    Lossy for text that merely *looks* numeric (``'007'`` → ``7``) — string-typed
    variables must bypass this (the subscriber's ``text_variables`` parameter).
    Non-finite numerics (``inf``/``nan``) pass through as the raw string.
    """
    try:
        f = float(s)
    except ValueError:
        return s
    if not math.isfinite(f):
        return s
    return int(f) if f == int(f) and "." not in s else f


def format_float(value: float) -> str:
    """Render a float for the wire as its shortest round-trip decimal.

    The outbound pair of :func:`coerce_scalar`. Uses Python's shortest
    round-trip representation (``repr``), so the digits sent are exactly the
    digits the caller asked for — ``40854.24625`` transmits as
    ``40854.24625``, never a fixed-precision expansion such as
    ``40854.246249999997`` that exposes the binary representation and that
    LabVIEW's parser rejects as "not a number" (issue #819). Precision is
    never truncated: ``1e-05`` (a DB minimum) and ``0.001`` (a DB tolerance)
    survive intact, which a fixed-decimals format would silently zero.

    Exponent notation, which LabVIEW may not parse, is expanded to a plain
    decimal (``1e-07`` → ``0.0000001``, ``1e+16`` → ``10000000000000000.0``).
    Every finite value carries a decimal point, as the previous ``%.12f``
    format did. Non-finite values (``inf``/``nan``) pass through as their
    ``repr``, unchanged from before.
    """
    text = repr(float(value))
    if "e" not in text:  # also inf/nan: repr carries no exponent marker
        return text
    # repr chose exponent form (|value| < 1e-4 or >= 1e16): expand the same
    # shortest digits — Decimal(text) is exact — into plain positional form.
    expanded = format(Decimal(text), "f")
    return expanded if "." in expanded else expanded + ".0"
