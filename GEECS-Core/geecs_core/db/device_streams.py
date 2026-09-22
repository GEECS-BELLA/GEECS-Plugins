"""Which of a device's non-scalar variables are capture streams — declared per devicetype.

The GEECS DB types a variable (``image``, ``1darray``) but says nothing about
whether the device actually *pushes* it per shot, or whether anyone wants it
recorded.  Two declarations per devicetype answer that, with opposite
polarities because their costs are opposite:

- **Capturing** a variable to the scan record is expensive and can break a
  scan: a file plugin armed on a variable the device never pushes waits out
  its arm timeout on every prepare, and a right-but-unwanted stream costs
  real bytes per shot.  So capture is an explicit **allowlist** — the
  ``capture`` tuple, in capture order (the first is the device's primary
  stream).  A devicetype with no entry here declares nothing, and the
  consumer keeps its historical default (the worker: the one ``image``
  variable, else the first image variable).
- **Serving** a variable over PVA is cheap and gated — a PV nobody subscribes
  to costs the device nothing — so serving stays DB-driven and only the
  **exclusions** are named: ``1darray``-typed variables the gateway must not
  serve because the device never publishes them (the FROG's spectra, the
  Picoscope's dead ``ScopeTraces``/``wfm`` rows), because they are a
  downsampled twin for GUIs (``scopeTraceGUI.*``), because they arrive
  malformed (the stitcher's ``interpDiv``, an axis that stops increasing
  after 190 of 8218 rows), or because they repeat a captured stream's own
  axis (the MagSpec ``EnergyAxis``/``AngleAxis`` are ``interpSpec``/
  ``interpDiv`` column 0).  **Exclusions name array variables only** —
  no image PV is ever removed here; a Point Grey's ``processed image``
  stays served, gated, for whoever watches it.

A third, numeric fact rides with them: ``array_ceiling``, the row count the
gateway pads a variable-length array to (NaN fill) so the PV shape, the
Bluesky descriptor and the HDF5 stack stay constant across shots — the
MagSpec lineouts' length is the energy span over a fixed ``dE`` and moves
with the magnet current (a camera at ~285 rows, the stitcher at ~8218; the
1 x 2 magnet-off default is an ordinary frame).  Longer than the ceiling is
dropped and counted, never truncated.  ``None`` means "serve at native
length" (a scope trace's length is its configured record).

Every name is matched against the device's DB rows **case-insensitively** —
the GEECS DB spells one variable differently across tables, and the device
itself looks names up case-insensitively — and the **DB row's spelling** is
what comes back.  A declared name with no row for that devicetype is a typo
or a DB change; it is dropped with a WARNING rather than acted on, and the
parity test in ``tests/test_device_streams.py`` pins every entry against
recorded DB rows so a misspelling fails offline, before it reaches a scan.

The entries record what was established live on the reference deployment
(2026-09-16..21): the FROG pushes only ``frogTrace`` (its ``SpatialImage``
is an alignment view and the retrieved traces and spectra stay empty); the
MagSpec cameras push ``Image`` and ``ImageInterp`` plus the two lineouts
and their two axes; the stitcher pushes ``Image``, ``interpSpec`` and a
malformed ``interpDiv``; the Picoscope pushes ``scopeTrace.Channel<N>`` for
each enabled channel (and the GUI twins), its capture set being per
*instance* — hence the ``gate`` column: the worker arms a channel's plugin
only when that instance's ``Enable.Ch<X>`` reads ``on``.

Two neighbours to know about.  The worker (which arms the plugins) and the
PVA gateway (which serves the PVs) both read this; neither may import the
other, which is the ``scalar_policy`` precedent for a DB-derived rule living
in GEECS-Core (the CA gateway serves scalars only and never reads it).  And
``geecs_bluesky.assets.registry`` carries a per-devicetype table that looks
like this one and is not: it describes the files LabVIEW writes natively (a
spatial *and* a temporal FROG image, the stitcher's ``interpDiv`` TSV), not
what the device pushes, and it retires with PNG retirement (#738).  Extend
this table, not that one.

Adding an entry: record the devicetype's rows first —
``poetry run python scripts/record_devicetype_variables.py "<devicetype>"``
from ``GEECS-Core/`` on the lab network — because the parity test refuses a
table entry with no recorded fixture behind it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from geecs_core.db.variable_types import array_variables, rows_by_lower

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceTypeStreams:
    """One devicetype's stream declaration.

    Attributes
    ----------
    capture :
        The variables the file plugin records per shot, in capture order;
        the first is the device's primary stream.  Empty means "nothing is
        captured" — distinct from a devicetype with no entry at all.
    exclude :
        ``1darray``-typed variables the PVA gateway never serves (and so
        nobody captures).  Never an image variable.
    array_ceiling :
        Rows the gateway pads this devicetype's variable-length arrays to
        (NaN fill; longer is dropped and counted).  ``None`` = native length.
    gate :
        Capture variable → the device's on/off variable that says whether
        *this instance* pushes it (the Picoscope's ``Enable.Ch<X>`` per
        channel).  The worker reads the gate on every prepare and arms the
        plugin only when it reads ``on``; a capture variable with no gate is
        armed unconditionally.  The gate variable must be subscribed
        (``get='yes'``) so the CA gateway serves it — the worker refuses to
        arm a gated stream whose gate it cannot read.
    """

    capture: tuple[str, ...] = ()
    exclude: frozenset[str] = frozenset()
    array_ceiling: int | None = None
    gate: Mapping[str, str] = field(default_factory=dict)


def _key(devicetype: str) -> str:
    return devicetype.strip().lower()


#: The declaration, keyed by lower-cased devicetype (:func:`streams_for`
#: normalizes the lookup).  Every entry has recorded rows in
#: ``tests/fixtures/devicetype_variables.json`` (see the module docstring).
DEVICE_TYPE_STREAMS: Mapping[str, DeviceTypeStreams] = {
    "point grey camera": DeviceTypeStreams(
        capture=("image",),
        # Empty on the wire on every camera probed; out of scope (Sam, 2026-09-16).
        exclude=frozenset({"HorizontalLineout", "VerticalLineout", "lineouts"}),
    ),
    "magspeccamera": DeviceTypeStreams(
        capture=("Image", "ImageInterp", "interpSpec", "interpDiv"),
        exclude=frozenset({"EnergyAxis", "AngleAxis"}),  # = the lineouts' column 0
        array_ceiling=2048,
    ),
    "magspecstitcher": DeviceTypeStreams(
        capture=("Image", "interpSpec"),
        exclude=frozenset({"interpDiv"}),  # malformed on this devicetype
        array_ceiling=16384,
    ),
    "frog": DeviceTypeStreams(
        capture=("frogTrace",),
        # The six spectra never carry a value on the wire.
        exclude=frozenset(
            {
                "spectrum x",
                "spectrum y",
                "spectrum phase y",
                "temporal x",
                "temporal intensity y",
                "temporal phase y",
            }
        ),
    ),
    # Four channels, each armed per instance when its ``Enable.Ch<X>`` reads
    # ``on`` (a two-channel unit, or a four-channel one with two wired,
    # pushes nothing on the others — probed live 2026-09-20: A/B on, C/D
    # empty).  Served at the configured record length (no ceiling).
    "picoscopev2": DeviceTypeStreams(
        capture=(
            "scopeTrace.Channel0",
            "scopeTrace.Channel1",
            "scopeTrace.Channel2",
            "scopeTrace.Channel3",
        ),
        gate={
            "scopeTrace.Channel0": "Enable.ChA",
            "scopeTrace.Channel1": "Enable.ChB",
            "scopeTrace.Channel2": "Enable.ChC",
            "scopeTrace.Channel3": "Enable.ChD",
        },
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
    by_lower = rows_by_lower(rows)
    out: list[str] = []
    for declared in names:
        row = by_lower.get(declared.lower())
        if row is None:
            logger.warning(
                "device streams: devicetype %r declares %s variable %r, which the "
                "DB does not list for it (a typo, or the DB changed) — ignored",
                devicetype,
                which,
                declared,
            )
            continue
        out.append(str(row["name"]))
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


def served_array_variables(
    devicetype: str, rows: Sequence[Mapping[str, object]]
) -> list[str]:
    """The ``1darray`` variables the PVA gateway serves for one device: typed minus excluded.

    The array counterpart of :func:`geecs_core.db.variable_types.image_variables`
    for the served set, shared by the gateway (its roster) and the worker
    (which may only capture what is served).  Sorted, DB-spelled.
    """
    excluded = {name.lower() for name in excluded_variables(devicetype, rows)}
    return [name for name in array_variables(rows) if name.lower() not in excluded]


def capture_gates(
    devicetype: str, rows: Sequence[Mapping[str, object]]
) -> dict[str, str]:
    """``{capture variable: gate variable}`` for one device, both DB-spelled.

    Only pairs whose two names the DB lists for the device; an unknown name on
    either side is dropped with a WARNING (the same rule as every other
    declared name).  Empty for a devicetype with no entry or no gates.
    """
    entry = streams_for(devicetype)
    if entry is None or not entry.gate:
        return {}
    by_lower = rows_by_lower(rows)
    out: dict[str, str] = {}
    for captured, gate in entry.gate.items():
        row = by_lower.get(captured.lower())
        if row is None:
            continue  # the capture list's own resolution already warned about it
        resolved = _resolve([gate], rows, devicetype=devicetype, which="gate")
        if resolved:
            out[str(row["name"])] = resolved[0]
    return out


def array_ceiling(devicetype: str) -> int | None:
    """The padding ceiling declared for *devicetype*'s arrays, or ``None`` (native length)."""
    entry = streams_for(devicetype)
    return None if entry is None else entry.array_ceiling
