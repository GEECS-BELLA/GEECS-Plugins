"""Which of a device's non-scalar variables are capture streams — declared per devicetype.

The GEECS DB types a variable (``image``, ``1darray``) but says nothing about
whether the device actually *pushes* it per shot, or whether anyone wants it
recorded.  Capturing a variable to the scan record is expensive and can
break a scan: a file plugin armed on a variable the device never pushes
waits out its arm timeout on every prepare, and a right-but-unwanted stream
costs real bytes per shot.  So capture is an explicit **allowlist**: a
devicetype's ``capture`` tuple, in capture order (the first is the device's
primary stream).  A devicetype with no entry here declares nothing, and the
consumer keeps its historical default (the worker: the one ``image``
variable, else the first image variable).

Serving over PVA is the opposite case — cheap and gated, a PV nobody
subscribes to costs the device nothing — so it stays DB-driven.  The
exclusion list that will trim it (names a device never publishes, a
downsampled twin, an axis variable that pushes one number) arrives with the
gateway's array support, beside its consumer, not here ahead of it.

Every name is matched against the device's DB rows **case-insensitively** —
the GEECS DB spells one variable differently across tables, and the device
itself looks names up case-insensitively — and the **DB row's spelling** is
what comes back.  A declared name with no row for that devicetype is a typo
or a DB change; it is dropped with a WARNING rather than armed, and the
parity test in ``tests/test_device_streams.py`` pins every entry against
recorded DB rows so a misspelling fails offline, before it reaches a scan.

The entries record what was established live on the reference deployment
(2026-09-16..20): the FROG pushes only ``frogTrace`` (its ``SpatialImage``
is an alignment view and the retrieved traces stay empty, so arming on any
of them times out); the MagSpec cameras push ``Image`` and ``ImageInterp``
plus the two lineouts; the stitcher pushes ``Image`` and ``interpSpec``
(its ``interpDiv`` arrives malformed — an axis that stops increasing after
190 of 8218 rows — and is left out on purpose).  The Picoscope's capture
set is per *instance* (only the channels its ``Enable.Ch<X>`` reads ``on``
push anything) and arrives with array support — hence its empty
``capture``, which is a decision ("nothing yet"), not an absence.

Two neighbours to know about.  The worker (which arms the plugins) reads
this today and the PVA gateway (which serves the PVs) follows with array
support; neither may import the other, which is the ``scalar_policy``
precedent for a DB-derived rule living in GEECS-Core (the CA gateway serves
scalars only and never reads it).  And ``geecs_bluesky.assets.registry``
carries a per-devicetype table that looks like this one and is not: it
describes the files LabVIEW writes natively (a spatial *and* a temporal
FROG image, the stitcher's ``interpDiv`` TSV), not what the device pushes,
and it retires with PNG retirement (#738).  Extend this table, not that one.

Adding an entry: record the devicetype's rows first —
``poetry run python scripts/record_devicetype_variables.py "<devicetype>"``
from ``GEECS-Core/`` on the lab network — because the parity test refuses a
table entry with no recorded fixture behind it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from geecs_core.db.variable_types import rows_by_lower

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceTypeStreams:
    """One devicetype's capture declaration.

    Attributes
    ----------
    capture :
        The variables the file plugin records per shot, in capture order;
        the first is the device's primary stream.  Empty means "nothing is
        captured" — distinct from a devicetype with no entry at all.
    """

    capture: tuple[str, ...] = ()


def _key(devicetype: str) -> str:
    return devicetype.strip().lower()


#: The declaration, keyed by lower-cased devicetype (:func:`streams_for`
#: normalizes the lookup).  Every entry has recorded rows in
#: ``tests/fixtures/devicetype_variables.json`` (see the module docstring).
DEVICE_TYPE_STREAMS: Mapping[str, DeviceTypeStreams] = {
    "point grey camera": DeviceTypeStreams(capture=("image",)),
    "magspeccamera": DeviceTypeStreams(
        capture=("Image", "ImageInterp", "interpSpec", "interpDiv")
    ),
    "magspecstitcher": DeviceTypeStreams(capture=("Image", "interpSpec")),
    "frog": DeviceTypeStreams(capture=("frogTrace",)),
    # Nothing yet: its channels are armed per instance (``Enable.Ch<X>``)
    # once arrays are capturable.
    "picoscopev2": DeviceTypeStreams(capture=()),
}


def streams_for(devicetype: str) -> DeviceTypeStreams | None:
    """The declaration for *devicetype* (matched case- and whitespace-insensitively), or ``None``."""
    return DEVICE_TYPE_STREAMS.get(_key(devicetype))


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
    by_lower = rows_by_lower(rows)
    out: list[str] = []
    for declared in entry.capture:
        row = by_lower.get(declared.lower())
        if row is None:
            logger.warning(
                "device streams: devicetype %r declares capture variable %r, "
                "which the DB does not list for it (a typo, or the DB changed) "
                "— ignored",
                devicetype,
                declared,
            )
            continue
        out.append(str(row["name"]))
    return out
