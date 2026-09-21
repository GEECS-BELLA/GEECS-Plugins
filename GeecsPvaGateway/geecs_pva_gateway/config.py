"""DB-driven configuration: which devices this host serves, and their PV names.

The served set is self-scoping: enumerate the experiment's enabled devices and
keep those whose GEECS endpoint IP belongs to this machine and that expose at
least one **stream** variable — an image-typed one, or a ``1darray``-typed one
the devicetype does not exclude (:func:`geecs_core.db.device_streams.
served_array_variables`).  The GEECS DB is the source of truth — there is no
per-host config file; the per-devicetype exclusions and padding ceilings are
the one curated overlay, and they live in GEECS-Core beside the type rule.
"""

from __future__ import annotations

import logging
import socket

from pydantic import BaseModel, Field

from geecs_core.db.device_streams import array_ceiling, served_array_variables
from geecs_core.db.scalar_policy import GeecsDbScalarPolicy
from geecs_core.db.variable_types import (  # noqa: F401 - image_variables re-exported
    image_variables,
    scalar_attribute_variables,
)
from geecs_core.pv_naming import connected_pv, normalize_component, pv_name
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


class DeviceSpec(BaseModel):
    """One served GEECS device: endpoint, its stream variables, PV names.

    A stream variable is served as one ``NTNDArray`` PV with one file plugin:
    an ``image_variables`` entry decodes as an IMAQ image, an
    ``array_variables`` entry as one of the three array wire shapes
    (:mod:`geecs_pva_gateway.streams`), padded to ``array_ceiling`` rows
    when the devicetype declares one.
    """

    device: str
    host: str
    port: int
    experiment: str
    devicetype: str = ""
    image_variables: list[str] = Field(default_factory=list)
    #: The ``1darray`` variables the devicetype does not exclude
    #: (:func:`geecs_core.db.device_streams.served_array_variables`).
    array_variables: list[str] = Field(default_factory=list)
    #: Rows the arrays are padded to (NaN) — ``None`` = native length.
    array_ceiling: int | None = None
    #: The subscribed numeric scalars the file plugin writes beside every
    #: frame (:func:`geecs_core.db.variable_types.scalar_attribute_variables`,
    #: the one home for the rule); they join the stream variable's one TCP
    #: subscription.  Empty = frames and stamps only.
    scalar_variables: list[str] = Field(default_factory=list)

    @property
    def stream_variables(self) -> list[str]:
        """Every served variable of the device, images first."""
        return [*self.image_variables, *self.array_variables]

    def is_array(self, variable: str) -> bool:
        """Whether *variable* is served as an array (else as an image)."""
        return variable in self.array_variables

    def pv_name_for(self, variable: str) -> str:
        """Full PV name for one stream variable, minted by the shared contract."""
        return pv_name(self.experiment, self.device, variable)

    def connected_pv_for(self, variable: str) -> str:
        """The variable's subscription-state PV (``geecs_core.pv_naming.connected_pv``).

        ``Idle`` (gated off: nobody watching, nothing known) / ``Disconnected``
        (watched and unreachable — the boot-order gap of GEECS-Plugins#854,
        visible here instead of at the scan's first arm; MAJOR alarm) /
        ``Connected``.
        """
        return connected_pv(self.experiment, self.device, variable)


class PvaGatewayConfig(BaseModel):
    """The set of devices one gateway instance serves, and the host it serves them from."""

    experiment: str
    #: The served host's address — what the instance's identity PVs
    #: (``version`` / ``heartbeat`` / ``restart``) are named after, so the
    #: fleet probe and the Phoebus screen (which ask by ``[pva] addr_list``
    #: IP) find the instance even while it has no device to serve.  The
    #: ``--host`` argument, else the lab-facing local address.
    host: str | None = None
    devices: list[DeviceSpec] = Field(default_factory=list)

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
        # The per-devicetype exclusions and ceilings key on the devicetype.
        types = GeecsDb.get_experiment_device_types(
            experiment, enabled_only=enabled_only
        )
        # The per-frame scalar attributes: the same rule the worker builds a
        # device's row from, from its one home.  Degrades to empty with a
        # warning on a DB blip (the roster queries above already succeeded).
        subscribed = GeecsDbScalarPolicy(
            experiment, enabled_only=enabled_only, db=GeecsDb
        ).subscribed_by_device()
        if host:
            hosts = {host}
            served_host: str | None = host
        else:
            # Probe toward any device endpoint so the lab-facing interface's
            # address is included even when hostname lookup misses it.
            any_ip = next(iter(endpoints.values()), ("", 0))[0]
            hosts = local_ip_addresses(probe_target=any_ip or None)
            served_host = detect_local_ip(any_ip) or None if any_ip else None

        served: list[DeviceSpec] = []
        for device, (ip, port) in sorted(endpoints.items()):
            if ip not in hosts:
                continue
            if devices is not None and device not in devices:
                continue
            rows = var_map.get(device, [])
            devicetype = types.get(device, "")
            image_vars = image_variables(rows)
            array_vars = served_array_variables(devicetype, rows)
            if not image_vars and not array_vars:
                continue  # no stream variable (e.g. a timing box on the same host)
            served.append(
                DeviceSpec(
                    device=device,
                    host=ip,
                    port=port,
                    experiment=experiment,
                    devicetype=devicetype,
                    image_variables=image_vars,
                    array_variables=array_vars,
                    array_ceiling=array_ceiling(devicetype),
                    scalar_variables=scalar_attribute_variables(
                        var_map.get(device, []),
                        subscribed.get(device, []),
                        normalize=normalize_component,
                    ),
                )
            )
        if devices:
            for name in set(devices) - {c.device for c in served}:
                logger.warning(
                    "requested device %s not served: not on host(s) %s, not "
                    "enabled, or has no stream (image / served array) variables",
                    name,
                    sorted(hosts),
                )
        return cls(experiment=experiment, host=served_host, devices=served)
