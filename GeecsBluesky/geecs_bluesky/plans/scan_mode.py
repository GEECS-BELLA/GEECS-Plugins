"""Scan mode in the bound plans: the bracket, the run's device list, the preflight, the release plan.

The device is :class:`~geecs_bluesky.devices.scan_mode.ScanModeControl`;
``plans/registry.py`` and ``plans/optimize.py`` drive it through here:

- :func:`scan_mode_bracket` enters scan mode inside the trigger bracket,
  right after ARMED and before the stage and the claim (a refused entry
  claims no scan number, the liveness gate's posture), and ends it in a
  finalizer before STANDBY, on every exit path but ``RE.halt()``.
- :func:`run_device_names` is the ``ScanStart`` gate's list: the run's
  detectors (their owners), its non-essential devices and the profile's
  box.  Movers and the background telemetry are not required: an unlocked
  mover still takes sets, and a background readback is read from the
  gateway's cache either way.
- :func:`scan_mode_preflight` refuses, before the box is driven, what scan
  mode cannot carry: a gated run, a worker without the control or the
  experiment's endpoint, a run device the roster lacks, a profile write
  that disagrees with the device's own DB boundary row (the device applies
  its ``startvalue`` on ``ScanStart``, after the bracket's ARMED), an
  announced address that does not resolve to this host, and a worker on
  another host than the CA gateway (``EPICS_CA_ADDR_LIST``).  The DB's
  ``RootPath`` is cross-checked against the device-side root (a WARNING).
- :func:`release_scan_mode_plan` is the registered ``release_scan_mode``
  plan: end scan mode on the devices of a run that never ended it.
"""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Mapping, Sequence
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from geecs_schemas.trigger_profile import TriggerState

from geecs_bluesky.data_paths import local_data_root, pva_addr_tokens
from geecs_bluesky.devices.ca._view import owner_of
from geecs_bluesky.devices.scan_mode import EXIT, EnterScanMode, ScanModeControl
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_core.configs_repo import read_config_entry

logger = logging.getLogger(__name__)

#: The profile state the device's DB ``startvalue`` must agree with (the
#: bracket opens ARMED, then ``ScanStart`` applies the row), and the one its
#: ``endvalue`` must agree with (``ScanEnd`` applies the row, then STANDBY).
_BOUNDARY_PAIRS: tuple[tuple[TriggerState, str], ...] = (
    (TriggerState.ARMED, "startvalue"),
    (TriggerState.STANDBY, "endvalue"),
)


def run_device_names(
    detectors: Sequence[Any],
    non_essential: Sequence[Any] = (),
    shot_control: Any | None = None,
) -> tuple[str, ...]:
    """The GEECS devices the entry waits for, in plan order, each once.

    The detectors' owners (a ``.scalars`` view names its owner), the
    non-essential devices' owners, then the trigger profile's device(s);
    a plan-side object with no GEECS device behind it (the bin counter,
    the background snapshot) is skipped.
    """
    names: dict[str, None] = {}
    for obj in [*detectors, *non_essential]:
        device = getattr(owner_of(obj), "_geecs_device_name", None)
        if device:
            names.setdefault(str(device))
    for device in getattr(shot_control, "liveness_signals", None) or {}:
        names.setdefault(str(device))
    return tuple(names)


def scan_parameter_for(plan_name: str) -> str:
    """The second ``ScanStart`` argument: Master Control sends ``Shot #`` for a NoScan."""
    return "Shot #" if plan_name == "count" else plan_name


def scan_mode_bracket(plan: Any, control: ScanModeControl, entry: EnterScanMode):
    """Enter scan mode before *plan*; end it in a finalizer (every exit path but a halt)."""
    yield from bps.mv(control, entry)

    def leave():
        yield from bps.mv(control, EXIT)

    return (yield from bpp.finalize_wrapper(plan, leave()))


def values_agree(a: str | None, b: str | None) -> bool:
    """Whether two wire strings name the same value (``'4'`` and ``'4.0'`` do).

    A ``None`` on either side (a blank DB column) agrees with anything:
    the device writes nothing for it.
    """
    if a is None or b is None:
        return True
    x, y = a.strip(), b.strip()
    if x == y:
        return True
    try:
        return float(x) == float(y)
    except ValueError:
        return x.lower() == y.lower()


def boundary_conflicts(
    shot_control: Any, boundary_writes: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    """Profile writes that disagree with the device's own DB boundary row, described."""
    writes = getattr(shot_control, "writes", None)
    if writes is None:
        return []
    conflicts: list[str] = []
    for state, column in _BOUNDARY_PAIRS:
        for device, variable, value in writes.writes_for_state(state):
            for row in boundary_writes.get(device, ()):
                if row.get("variable") != variable:
                    continue
                db_value = row.get(column)
                if not values_agree(value, db_value):
                    conflicts.append(
                        f"{device}:{variable} — the profile's {state.value} "
                        f"writes {value!r}, the device's DB row applies "
                        f"{db_value!r} as its {column}"
                    )
    return conflicts


def gateway_hosts() -> tuple[str, ...]:
    """The CA gateway's address list: the worker's ``EPICS_CA_ADDR_LIST``, else config.ini's."""
    raw = os.environ.get("EPICS_CA_ADDR_LIST") or read_config_entry(
        "epics", "ca_addr_list"
    )
    return tuple(pva_addr_tokens(raw))


def _resolves_to(host: str) -> set[str]:
    try:
        infos = socket.getaddrinfo(host.rsplit(":", 1)[0], None, socket.AF_INET)
    except OSError:
        return set()
    return {info[4][0] for info in infos}


def check_gateway_host(local_ip: str, gateways: Sequence[str]) -> None:
    """Refuse a worker that would announce an address the CA gateway does not have.

    A device in scan mode takes commands only from the announced address,
    and the gateway's puts (the fire, every set) come from the gateway's
    host — so the two must be one host.  An empty *gateways* (no
    ``EPICS_CA_ADDR_LIST``) cannot be checked and only warns.
    """
    if not gateways:
        logger.warning(
            "scan mode: EPICS_CA_ADDR_LIST is not set — cannot check that this "
            "worker shares the CA gateway's host (a device in scan mode takes "
            "commands only from the announced address %s)",
            local_ip,
        )
        return
    known: set[str] = set()
    for gateway in gateways:
        known |= _resolves_to(gateway)
    if known and local_ip not in known:
        raise GeecsConfigurationError(
            f"scan mode: this worker would announce {local_ip} but the CA "
            f"gateway is at {', '.join(gateways)} — a device in scan mode "
            "takes commands only from the announced address, so the worker "
            "and the gateway must run on one host"
        )


def root_path_check(control: ScanModeControl) -> None:
    """WARN when the DB's ``RootPath`` and the device-side root the paths use disagree."""
    db_root = control.db_root_path
    if not db_root:
        return
    local = local_data_root()
    if not local:
        return
    translated = control.device_path(str(local)).rstrip("\\/")
    if translated and not db_root.rstrip("\\/").lower().startswith(translated.lower()):
        logger.warning(
            "scan mode: the DB's RootPath for %r is %r but the shot paths are "
            "built on %r (config.ini [Paths] geecs_device_server_data_base_path) "
            "— the devices may not find the folder",
            control.experiment,
            db_root,
            translated,
        )


def scan_mode_preflight(
    shot_control: Any,
    control: ScanModeControl | None,
    *,
    acquisition: str,
    run_devices: Sequence[str],
) -> None:
    """Refuse, before the box is driven, a run scan mode cannot carry.

    The rules are the module docstring's; every refusal is a
    :exc:`~geecs_bluesky.exceptions.GeecsConfigurationError` naming the
    cause.  Runs only for a profile with ``use_scan_mode``.
    """
    name = getattr(shot_control, "profile_name", "?")
    if acquisition != "strict":
        raise GeecsConfigurationError(
            f"trigger profile {name!r} runs in scan mode, which is strict "
            "only: a gated batch has no per-shot announce (use "
            "acquisition='strict', or a profile without use_scan_mode)"
        )
    if control is None:
        raise GeecsConfigurationError(
            f"trigger profile {name!r} asks for scan mode but this worker has "
            "no scan-mode control (a hermetic worker, or the experiment's "
            "scan-mode facts could not be read at environment open)"
        )
    if not control.enabled:
        raise GeecsConfigurationError(
            f"trigger profile {name!r} asks for scan mode but experiment "
            f"{control.experiment!r} has no multicast endpoint in the GEECS DB "
            "(expt.MultiCastIP / MultiCastRemotePort)"
        )
    if not run_devices:
        raise GeecsConfigurationError(
            "scan mode: the run has no GEECS device to announce to"
        )
    endpoints = control.device_endpoints
    unknown = [d for d in run_devices if d not in endpoints]
    if unknown:
        raise GeecsConfigurationError(
            f"scan mode: {', '.join(unknown)} not in the experiment's device "
            "roster — a run device must be an enabled device of "
            f"{control.experiment!r}"
        )
    conflicts = boundary_conflicts(shot_control, control.boundary_writes)
    if conflicts:
        raise GeecsConfigurationError(
            f"trigger profile {name!r} conflicts with the devices' own "
            "scan-boundary rows (applied by the device on ScanStart/ScanEnd): "
            + "; ".join(conflicts)
            + " — make the profile and the DB rows agree, or drop the row"
        )
    local_ip = control.resolve_local_ip(endpoints[run_devices[0]][0])
    control.check_announce_address(control.announce_address(local_ip), local_ip)
    check_gateway_host(local_ip, gateway_hosts())
    root_path_check(control)


def release_scan_mode_plan(control: ScanModeControl | None):
    """Build the registered ``release_scan_mode`` plan over *control*."""

    def release_scan_mode():
        """End GEECS scan mode on the devices of the last run, if it never ended.

        A run's exit runs in its finalizer on every path but ``RE.halt()``;
        after a halt the devices stay in scan mode, paused, until this
        item (or the next scan-mode run, which ends the old one first).
        Nothing happens when no run is in scan mode.  Opens no run.
        """
        if control is None:
            raise GeecsConfigurationError(
                "this worker has no scan-mode control (a hermetic worker, or "
                "the experiment's scan-mode facts could not be read at "
                "environment open) — nothing to release"
            )
        yield from bps.mv(control, EXIT)

    return release_scan_mode


__all__ = [
    "boundary_conflicts",
    "check_gateway_host",
    "gateway_hosts",
    "release_scan_mode_plan",
    "root_path_check",
    "run_device_names",
    "scan_mode_bracket",
    "scan_mode_preflight",
    "scan_parameter_for",
    "values_agree",
]
