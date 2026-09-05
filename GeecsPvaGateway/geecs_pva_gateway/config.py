"""DB-driven configuration: which cameras this host serves, and their PV names.

The served set is self-scoping: enumerate the experiment's enabled devices and
keep those whose GEECS endpoint IP belongs to this machine and that expose at
least one image-typed variable. The GEECS DB is the source of truth — there is
no per-host config file.
"""

from __future__ import annotations

import logging
import socket

from pydantic import BaseModel, Field

from geecs_ca_gateway.config import effective_vartype
from geecs_core.pv_naming import normalize_component, pv_name
from geecs_core.transport.udp_client import detect_local_ip

logger = logging.getLogger(__name__)


def local_ip_addresses(probe_target: str | None = None) -> set[str]:
    """Return this machine's IPv4 addresses (hostname lookup + route probe)."""
    addresses: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    if probe_target:
        # The address the OS would source from toward the lab (no packet sent).
        addresses.add(detect_local_ip(probe_target))
        addresses.discard("")
    return addresses


def instance_pv_prefix(experiment: str, host: str) -> str:
    """Prefix of one instance's identity PVs (``version``/``heartbeat``/``restart``).

    The identity component is the served host (an IP, dots normalised to
    underscores — PV_CONTRACT). The one composition shared by the server
    and the fleet tooling, so the probe and the screen can never drift from
    what an instance actually serves.
    """
    return pv_name(experiment, "pvagateway", normalize_component(host))


def image_variables(metadata: list[dict]) -> list[str]:
    """Names of the image-typed variables among one device's DB metadata rows.

    The camera test everywhere in this package (a device with none is not a
    camera); shared with :mod:`geecs_pva_gateway.fleet`.
    """
    return sorted(
        meta["name"]
        for meta in metadata
        if effective_vartype(meta.get("variabletype"), meta.get("choices")) == "image"
    )


class CameraSpec(BaseModel):
    """One GEECS camera device: endpoint, image variables, PV names."""

    device: str
    host: str
    port: int
    experiment: str
    image_variables: list[str] = Field(default_factory=lambda: ["image"])

    def pv_name_for(self, variable: str) -> str:
        """Full PV name for one image variable, minted by the shared contract."""
        return pv_name(self.experiment, self.device, variable)


class PvaGatewayConfig(BaseModel):
    """The set of cameras one gateway instance serves."""

    experiment: str
    cameras: list[CameraSpec] = Field(default_factory=list)

    @classmethod
    def from_geecs_experiment(
        cls,
        experiment: str,
        *,
        host: str | None = None,
        devices: list[str] | None = None,
        enabled_only: bool = True,
    ) -> "PvaGatewayConfig":
        """Build the served set from the GEECS database (two batched queries).

        Parameters
        ----------
        experiment : str
            GEECS experiment name; also the PV namespace prefix.
        host : str, optional
            Endpoint IP to scope to. Default: this machine's own addresses.
        devices : list of str, optional
            Restrict to these device names (after host scoping).
        enabled_only : bool
            Skip devices not enabled in the experiment (default true).
        """
        from geecs_core.db.geecs_db import GeecsDb

        endpoints = GeecsDb.get_experiment_devices(
            experiment, enabled_only=enabled_only
        )
        var_map = GeecsDb.get_experiment_device_variables(
            experiment, enabled_only=enabled_only
        )
        if host:
            hosts = {host}
        else:
            # Probe toward any device endpoint so the lab-facing interface's
            # address is included even when hostname lookup misses it.
            any_ip = next(iter(endpoints.values()), ("", 0))[0]
            hosts = local_ip_addresses(probe_target=any_ip or None)

        cameras: list[CameraSpec] = []
        for device, (ip, port) in sorted(endpoints.items()):
            if ip not in hosts:
                continue
            if devices is not None and device not in devices:
                continue
            image_vars = image_variables(var_map.get(device, []))
            if not image_vars:
                continue  # not a camera (e.g. a timing box on the same host)
            cameras.append(
                CameraSpec(
                    device=device,
                    host=ip,
                    port=port,
                    experiment=experiment,
                    image_variables=image_vars,
                )
            )
        if devices:
            for name in set(devices) - {c.device for c in cameras}:
                logger.warning(
                    "requested device %s not served: not on host(s) %s, not "
                    "enabled, or has no image variables",
                    name,
                    sorted(hosts),
                )
        return cls(experiment=experiment, cameras=cameras)
