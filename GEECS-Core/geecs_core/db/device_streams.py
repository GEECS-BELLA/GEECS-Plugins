"""Which of a device's non-scalar variables are capture streams — declared per devicetype.

The GEECS DB types a variable (``image``, ``1darray``) but says nothing about
whether the device actually *pushes* it per shot, or whether anyone wants it
recorded.  Both matter, and they have opposite costs:

- **Serving** a non-scalar variable over PVA is cheap and gated — a PV nobody
  subscribes to costs the device nothing — so serving stays DB-driven, minus
  the **exclusions** named here (a variable that pushes one number under an
  array type, a twin nobody needs, a name the device never publishes).
- **Capturing** a variable to the scan record is expensive and can break a
  scan: a plugin armed on a variable the device never pushes waits out its
  arm timeout on every prepare, and a right-but-unwanted stream costs real
  bytes per shot.  So capture is an explicit **allowlist**: a devicetype's
  ``capture`` tuple, in capture order (the first is the device's primary
  stream).  A devicetype with no entry here declares nothing, and the
  consumer keeps its historical default (the worker: the one ``image``
  variable, else the first image variable).

Every name is matched against the device's DB rows **case-insensitively** —
the GEECS DB spells one variable differently across tables, and the device
itself looks names up case-insensitively — and the **DB row's spelling** is
what comes back.  A declared name with no row for that devicetype is a typo
or a DB change; it is dropped with a WARNING rather than armed, and the
parity test in ``tests/test_device_streams.py`` pins every entry against
recorded DB rows so a misspelling fails offline, before it reaches a scan.

The entries record what was established live on the reference deployment
(2026-09-16..20): the FROG pushes only ``frogTrace`` (its ``SpatialImage``
and retrieved traces stay empty, so arming on them times out); the MagSpec
cameras push ``Image`` and ``ImageInterp`` plus the two lineouts, while
``EnergyAxis`` / ``AngleAxis`` each carry a single number; the stitcher's
``interpDiv`` arrives malformed (an axis that stops increasing after 190 of
8218 rows); the Picoscope's ``ScopeTraces`` / ``wfm`` / ``wfm info`` are
names the device never publishes and ``scopeTraceGUI.*`` is a downsampled
twin of ``scopeTrace.*`` for client rendering.  The Picoscope's capture set
is per *instance* (only the channels its ``Enable.Ch<X>`` reads ``on`` push
anything), which arrives with array support — hence its empty ``capture``.

This module lives beside :mod:`geecs_core.db.variable_types` because the
worker (which arms the plugins), the PVA gateway (which serves the PVs) and
the CA gateway all need one answer and none may import another.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceTypeStreams:
    """One devicetype's non-scalar stream declaration.

    Attributes
    ----------
    capture :
        The variables the file plugin records per shot, in capture order;
        the first is the device's primary stream.  Empty means "nothing is
        captured" — distinct from a devicetype with no entry at all.
    exclude :
        Variables never served over PVA and never captured, whatever the DB
        types them.
    """

    capture: tuple[str, ...] = ()
    exclude: frozenset[str] = frozenset()


def _key(devicetype: str) -> str:
    return devicetype.strip().lower()


#: The declaration, keyed by lower-cased devicetype (:func:`streams_for`
#: normalizes the lookup).  Adding an entry means recording that
#: devicetype's rows into ``tests/fixtures/devicetype_variables.json`` too:
#: the parity test refuses a table entry with no fixture behind it.
DEVICE_TYPE_STREAMS: Mapping[str, DeviceTypeStreams] = {
    "point grey camera": DeviceTypeStreams(
        capture=("image",),
        exclude=frozenset({"bakground image", "processed image"}),
    ),
    "magspeccamera": DeviceTypeStreams(
        capture=("Image", "ImageInterp", "interpSpec", "interpDiv"),
        exclude=frozenset({"EnergyAxis", "AngleAxis"}),
    ),
    "magspecstitcher": DeviceTypeStreams(
        capture=("Image", "interpSpec"),
        exclude=frozenset({"interpDiv"}),
    ),
    "frog": DeviceTypeStreams(
        capture=("frogTrace",),
        exclude=frozenset(
            {"SpatialImage", "retrieved FrogTrace", "retrievedFrogTrace"}
        ),
    ),
    "picoscopev2": DeviceTypeStreams(
        capture=(),
        exclude=frozenset(
            {
                "ScopeTraces",
                "wfm",
                "wfm info",
                "scopeTraceGUI.Channel0",
                "scopeTraceGUI.Channel1",
                "scopeTraceGUI.Channel2",
                "scopeTraceGUI.Channel3",
            }
        ),
    ),
}


def streams_for(devicetype: str) -> DeviceTypeStreams | None:
    """The declaration for *devicetype* (matched case- and whitespace-insensitively), or ``None``."""
    return DEVICE_TYPE_STREAMS.get(_key(devicetype))


def _resolve(
    names: Iterable[str],
    rows: Sequence[Mapping[str, object]],
    *,
    devicetype: str,
    which: str,
) -> list[str]:
    """Declared *names* → the DB rows' spellings, dropping (and warning on) unknown ones."""
    by_lower: dict[str, str] = {}
    for row in rows:
        name = str(row["name"])
        by_lower.setdefault(name.lower(), name)
    out: list[str] = []
    for declared in names:
        found = by_lower.get(declared.lower())
        if found is None:
            logger.warning(
                "device streams: devicetype %r declares %s variable %r, which the "
                "DB does not list for it (a typo, or the DB changed) — ignored",
                devicetype,
                which,
                declared,
            )
            continue
        out.append(found)
    return out


def capture_variables(
    devicetype: str, rows: Sequence[Mapping[str, object]]
) -> list[str] | None:
    """The declared capture streams of one device, spelled as its DB rows spell them.

    Parameters
    ----------
    devicetype :
        The device's devicetype (``GeecsDb.get_experiment_device_types``).
    rows :
        The device's variable metadata rows (``name`` at least).

    Returns
    -------
    list of str or None
        The declared names present in *rows*, in capture order; ``[]`` when
        the devicetype declares that nothing is captured; ``None`` when the
        devicetype has no entry, so the caller applies its own default.
    """
    entry = streams_for(devicetype)
    if entry is None:
        return None
    return _resolve(entry.capture, rows, devicetype=devicetype, which="capture")


def excluded_variables(
    devicetype: str, rows: Sequence[Mapping[str, object]]
) -> list[str]:
    """The declared exclusions of one device present in *rows*, sorted, DB-spelled.

    Empty for a devicetype with no entry.
    """
    entry = streams_for(devicetype)
    if entry is None:
        return []
    return _resolve(sorted(entry.exclude), rows, devicetype=devicetype, which="exclude")
