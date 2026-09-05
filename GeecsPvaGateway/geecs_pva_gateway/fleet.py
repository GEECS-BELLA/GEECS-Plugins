"""Fleet roster: which hosts *should* run a gateway, and which are deployed.

Two facts, one home each — nothing hand-curated in code:

- **Roster** — the GEECS DB: every endpoint IP that hosts an enabled device
  with an image-typed variable for the experiment (the same derivation
  :meth:`~geecs_pva_gateway.config.PvaGatewayConfig.from_geecs_experiment`
  applies per host at serve time, here across the whole experiment).
- **Deployed** — the client ``~/.config/geecs_python_api/config.ini``::

      [pva]
      addr_list = 192.168.6.100 192.168.7.161

  the hosts running an instance, space-separated (commas tolerated; a
  ``host:port`` entry counts by its host) — the same list a cross-subnet
  client puts in ``EPICS_PVA_ADDR_LIST`` (DEPLOYMENT.md "Client access"),
  mirroring ``[epics] ca_addr_list``. Only the fleet tooling reads the key;
  ``scripts/fleet_status.sh`` exports it for its own probe. A roster host
  absent from it is **not deployed**: the box hosts cameras only nominally
  and no instance was ever installed there — a failed probe is not an
  outage. When the key is absent every roster host counts as deployed (the
  pre-``[pva]`` behaviour).

Consumers: ``deploy/gen_fleet_status.py`` (the Phoebus fleet screen) and
``scripts/fleet_status.sh`` (the observed-fleet probe).
"""

from __future__ import annotations

import configparser
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from geecs_pva_gateway.config import image_variables, instance_pv_prefix

logger = logging.getLogger(__name__)

USER_CONFIG_PATH = Path("~/.config/geecs_python_api/config.ini")


class FleetHost(BaseModel):
    """One camera server as the roster sees it."""

    ip: str
    cameras: list[str] = Field(default_factory=list)
    deployed: bool = True

    def instance_pv(self, experiment: str, name: str) -> str:
        """Full name of one instance PV (``version``, ``heartbeat``, ``restart``)."""
        return f"{instance_pv_prefix(experiment, self.ip)}:{name}"


def _ip_key(host: str) -> tuple:
    """Sort key: dotted quads numerically, anything else (a hostname) after them."""
    parts = host.split(".")
    if all(p.isdigit() for p in parts):
        return (0, tuple(int(p) for p in parts))
    return (1, (host,))


def camera_endpoints(
    experiment: str, *, enabled_only: bool = True
) -> dict[str, list[str]]:
    """Return ``{endpoint_ip: [camera device, ...]}`` for *experiment* from the DB.

    Two batched queries; a device counts as a camera when it exposes at
    least one image-typed variable.
    """
    from geecs_core.db.geecs_db import GeecsDb

    endpoints = GeecsDb.get_experiment_devices(experiment, enabled_only=enabled_only)
    var_map = GeecsDb.get_experiment_device_variables(
        experiment, enabled_only=enabled_only
    )
    by_ip: dict[str, list[str]] = {}
    for device, (ip, _port) in endpoints.items():
        if image_variables(var_map.get(device, [])):
            by_ip.setdefault(ip, []).append(device)
    return {
        ip: sorted(devs)
        for ip, devs in sorted(by_ip.items(), key=lambda kv: _ip_key(kv[0]))
    }


def read_config(config_path: Path | None = None) -> configparser.ConfigParser:
    """Parse the client config.ini (missing file → empty parser, never raises)."""
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    path = (config_path or USER_CONFIG_PATH).expanduser()
    if path.exists():
        parser.read(path)
    return parser


def default_experiment(config_path: Path | None = None) -> str:
    """The experiment named in config.ini ``[Experiment]`` (``expt``/``exp_name``), or ``""``."""
    parser = read_config(config_path)
    for key in ("expt", "exp_name"):
        value = parser.get("Experiment", key, fallback="").strip()
        if value:
            return value
    return ""


def deployed_addr_list(config_path: Path | None = None) -> list[str] | None:
    """Hosts listed in config.ini ``[pva] addr_list``; ``None`` when the key is absent.

    Entries are space- or comma-separated; a ``host:port`` entry (EPICS
    address-list syntax) counts by its host.
    """
    raw = read_config(config_path).get("pva", "addr_list", fallback=None)
    if raw is None:
        return None
    return [entry.split(":", 1)[0] for entry in raw.replace(",", " ").split()]


def fleet_roster(
    experiment: str,
    *,
    config_path: Path | None = None,
    enabled_only: bool = True,
) -> list[FleetHost]:
    """The experiment's camera servers, each marked deployed or not.

    Roster hosts come from the DB (:func:`camera_endpoints`); ``deployed``
    is membership in ``[pva] addr_list`` (all deployed when the key is
    absent). A listed address with no cameras in the DB is kept as a
    deployed host with an empty camera list — a stale entry worth seeing —
    and logged as a warning.
    """
    by_ip = camera_endpoints(experiment, enabled_only=enabled_only)
    listed = deployed_addr_list(config_path)
    deployed = set(by_ip) if listed is None else set(listed)
    hosts = [
        FleetHost(ip=ip, cameras=cams, deployed=ip in deployed)
        for ip, cams in by_ip.items()
    ]
    for ip in sorted(deployed - set(by_ip), key=_ip_key):
        logger.warning(
            "[pva] addr_list names %s but the DB has no enabled camera on it", ip
        )
        hosts.append(FleetHost(ip=ip, cameras=[], deployed=True))
    return sorted(hosts, key=lambda h: _ip_key(h.ip))
