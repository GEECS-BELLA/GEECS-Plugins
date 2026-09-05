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
  mirroring ``[epics] ca_addr_list``. Only the fleet tooling reads the key
  (``geecs-pva-gateway fleet`` searches exactly those hosts). A roster host
  absent from it is **not deployed**: the box hosts cameras only nominally
  and no instance was ever installed there — a failed probe is not an
  outage. When the key is absent every roster host counts as deployed (the
  pre-``[pva]`` behaviour).

Consumers: ``deploy/gen_fleet_status.py`` (the Phoebus fleet screen) and
``geecs-pva-gateway fleet`` (:func:`probe_fleet` / :func:`fleet_main`), the
read-only liveness probe ``scripts/fleet_status.sh`` calls — so the roster,
the instance PV names, and the probe live in one package.
"""

from __future__ import annotations

import argparse
import configparser
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
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


# --- the liveness probe (`geecs-pva-gateway fleet`) ----------------------------

FLEET_ROLE = "PVA image gateways"


class HostProbe(BaseModel):
    """One deployed host's answer to the version/heartbeat gets."""

    host: FleetHost
    version: str | None = None
    heartbeat: int | None = None
    error: str | None = None

    @property
    def up(self) -> bool:
        """True when both instance PVs answered."""
        return self.version is not None


class FleetProbe(BaseModel):
    """The whole fleet's probe: per-host results plus the roster's not-deployed hosts."""

    experiment: str
    probes: list[HostProbe] = Field(default_factory=list)
    not_deployed: list[FleetHost] = Field(default_factory=list)

    @property
    def versions(self) -> dict[str, list[str]]:
        """``{version: [ip, ...]}`` over the hosts that answered."""
        out: dict[str, list[str]] = {}
        for p in self.probes:
            if p.up:
                out.setdefault(str(p.version), []).append(p.host.ip)
        return out

    @property
    def down(self) -> list[HostProbe]:
        """Deployed hosts that did not answer."""
        return [p for p in self.probes if not p.up]

    def lines(self) -> list[str]:
        """Human lines in the fleet_status.sh style (``[ OK ]``/``[DOWN]``/``[ -- ]``/``[WARN]``)."""
        out = []
        for p in self.probes:
            n = len(p.host.cameras)
            if p.up:
                out.append(
                    f"  [ OK ] PVA gateway  {p.host.ip:<15}  geecs-pva-gateway {p.version}"
                    f"  heartbeat={p.heartbeat}  {n} cameras"
                )
            else:
                out.append(
                    f"  [DOWN] PVA gateway  {p.host.ip:<15}  ({p.error}; {n} cameras)"
                )
        for h in self.not_deployed:
            out.append(
                f"  [ -- ] PVA gateway  {h.ip:<15}  not deployed ({len(h.cameras)} cameras in the DB: "
                f"{', '.join(h.cameras)}) — add to config.ini [pva] addr_list once installed"
            )
        if len(self.versions) > 1:
            out.append(
                "  [WARN] PVA fleet runs mixed versions — a rollout is incomplete or a box missed its pull-on-restart:"
            )
            for ver, hs in sorted(self.versions.items()):
                out.append(f"         {ver}: {', '.join(hs)}")
        return out

    def record(self) -> str:
        """One tab-separated ``key=value`` record for the fleet table.

        ``info=`` carries facts (hosts up, hosts not deployed); ``note=``
        carries findings (unreachable hosts, mixed versions) — the table
        marks a row ``!`` on notes only.
        """
        versions = self.versions
        n_ok = sum(len(hs) for hs in versions.values())
        ver_txt = (
            ", ".join(f"{v} ×{len(hs)}" for v, hs in sorted(versions.items())) or "?"
        )
        fields = [
            f"role={FLEET_ROLE}",
            f"state={'ok' if n_ok else 'down'}",
            "runs=NSSM",
            "checkout=share clone",
            f"version={ver_txt}",
            f"info={n_ok} of {len(self.probes)} deployed up",
        ]
        if self.not_deployed:
            fields.append(f"info={len(self.not_deployed)} not deployed")
        if self.down:
            fields.append(
                f"note={len(self.down)} unreachable: {' '.join(p.host.ip for p in self.down)}"
            )
        if len(versions) > 1:
            fields.append("note=MIXED versions")
        return "\t".join(fields)


@contextmanager
def _p4p_context(hosts: list[str], timeout: float) -> Iterator[Callable[[str], object]]:
    """Yield a ``get(pv) -> value`` over a p4p context searching *hosts* by unicast.

    UDP broadcast does not cross a VPN, so the address list is the deployed
    hosts themselves (``EPICS_PVA_AUTO_ADDR_LIST=NO``) — passed as the
    context's own ``conf``, never written to the process environment; the
    context is closed on exit so a library caller leaks nothing.
    """
    from p4p.client.thread import Context

    conf = {"EPICS_PVA_ADDR_LIST": " ".join(hosts), "EPICS_PVA_AUTO_ADDR_LIST": "NO"}
    # unwrap=False: raw Values (str(NT wrapper) would carry a timestamp).
    with Context("pva", conf=conf, useenv=True, unwrap=False) as ctx:
        yield lambda pv: ctx.get(pv, timeout=timeout)["value"]


def probe_fleet(
    experiment: str,
    hosts: list[FleetHost],
    *,
    timeout: float = 2.0,
    getter: Callable[[str], object] | None = None,
) -> FleetProbe:
    """Read every deployed host's ``version`` + ``heartbeat`` instance PVs (read-only).

    *getter* is injectable (tests); the default opens a p4p context.
    """
    deployed = [h for h in hosts if h.deployed]
    result = FleetProbe(
        experiment=experiment, not_deployed=[h for h in hosts if not h.deployed]
    )
    if not deployed:
        return result
    source = (
        nullcontext(getter)
        if getter
        else _p4p_context([h.ip for h in deployed], timeout)
    )
    with source as get:
        for host in deployed:
            try:
                version = str(get(host.instance_pv(experiment, "version")))
                beats = int(get(host.instance_pv(experiment, "heartbeat")))
            except Exception as exc:  # noqa: BLE001 — a failed probe is a finding
                result.probes.append(HostProbe(host=host, error=type(exc).__name__))
                continue
            result.probes.append(HostProbe(host=host, version=version, heartbeat=beats))
    return result


def fleet_main(argv: list[str] | None = None) -> int:
    """``geecs-pva-gateway fleet``: print the probe's lines and its record; exit 0 when any host is up."""
    parser = argparse.ArgumentParser(
        prog="geecs-pva-gateway fleet",
        description="Probe every deployed PVA image gateway's version/heartbeat PVs (read-only).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="GEECS experiment (default: config.ini [Experiment])",
    )
    parser.add_argument(
        "--timeout", type=float, default=2.0, help="seconds per PVA get (default 2)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="client config.ini (default: the user's)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="  [WARN] PVA fleet: %(message)s")

    experiment = args.experiment or default_experiment(args.config)
    if not experiment:
        print(
            "no experiment: pass --experiment or set config.ini [Experiment]",
            file=sys.stderr,
        )
        print(f"role={FLEET_ROLE}\tstate=down\tnote=no experiment name")
        return 2
    try:
        hosts = fleet_roster(experiment, config_path=args.config)
    except Exception as exc:  # noqa: BLE001 — an unreadable roster is a finding
        print(
            f"  [DOWN] PVA fleet    roster unreadable from the DB ({type(exc).__name__}: {exc})"
        )
        print(f"role={FLEET_ROLE}\tstate=down\tnote=DB roster unreadable")
        return 1
    result = probe_fleet(experiment, hosts, timeout=args.timeout)
    for line in result.lines():
        print(line)
    print(result.record())
    return 0 if result.versions else 1
