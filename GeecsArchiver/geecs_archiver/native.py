"""The registry pass: the archive set of the devices native EPICS IOCs serve.

A migrated device is disabled in the GEECS DB, so the DB pass
(:func:`geecs_archiver.archive_set.derive_candidates`) never sees it.  This
pass reads the configs repository instead — ``native_devices.yaml`` (the
worker's registry, :class:`~geecs_schemas.NativeDevices`) and
``panel_devices.yaml`` (:class:`~geecs_schemas.PanelDevices`, the panel data)
— and the installed family files (:func:`~geecs_schemas.load_families`).
It never needs the DB: given the DB roster it also refuses doubled
ownership, without it it says it did not check.

Per device (``NATIVE_DEVICES.md`` §10, ``Planning/epics_estate.md`` §2.5):

* the device's **family**: the one ``panel_devices.yaml`` names, else the
  one family file of the registry kind;
* its **archived roles**: the family's ``archive: true`` roles, or — for a
  family that marks none — the kind's own rule (:func:`default_archive_roles`);
  then the device's ``archive`` opt-ins added and ``no_archive`` /
  ``omit`` removed (``PanelDevice.resolve``);
* scalars only: never an array, an image, a command or a char-array text
  role, whatever a file says;
* a role whose record is its own write target (a motor's ``.VAL``, a
  supply's ``current:SP``) is a **setpoint**, under ``include_setpoints``;
  every other role is a readback (an ``_RBV`` of a setting, a bit, a
  switch);
* under ``include_status``: the kind's liveness record
  (:data:`LIVENESS_SUFFIXES`) and, once per IOC a device names (``ioc``),
  devIocStats' heartbeat (:func:`geecs_core.pv_naming.ioc_heartbeat_pv`) —
  the analogue of the gateway's ``cagateway:heartbeat``.

PV name = the device's prefix (:func:`geecs_core.pv_naming.native_device_prefix`,
or a bench device's own ``prefix``) + the role's suffix, per axis for a
per-axis family (``<prefix><axis><suffix>``).

Who is archived: every registry device, and every ``panel_devices.yaml``
device the DB does not own (a bench device whose DB twin is enabled is the
gateway's, and the DB pass has it).  A registry device the DB also owns is
a **conflict**: reported, and left to the DB pass — never archived twice.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field

from geecs_core.pv_naming import (
    ioc_heartbeat_pv,
    native_device_prefix,
    normalize_component,
)
from geecs_schemas import (
    ArchivePolicy,
    EpicsMotor,
    FamilyFile,
    NativeDevice,
    NativeDevices,
    PanelDevice,
    PanelDevices,
    Role,
)

from geecs_archiver.archive_set import ArchiveCandidate, Kind

#: Role types that are never CA scalars the appliance can keep.
UNARCHIVABLE_ROLE_TYPES: frozenset[str] = frozenset({"array", "image", "command"})

#: Per family kind: the records that say whether the IOC reaches the
#: hardware, archived under ``include_status`` like the gateway's
#: ``<device>:connected``.  The camera's is the asyn port's connection
#: (``NATIVE_DEVICES.md`` §3 "Liveness"), which its family file has no
#: role for; the scope contract's ``Connected_RBV`` is a role, archived by
#: its family.
LIVENESS_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "aravis_camera": ("cam1:AsynIO.CNCT",),
}

#: The motor record's fields §10's rule archives when its family marks none.
_MOTOR_DEFAULT_SUFFIXES = frozenset({".RBV", ".VAL", ".DMOV", ".HLS", ".LLS"})
#: The run lock's record (geecs-epics ``run_lock.db``).
_RUN_LOCK_SUFFIX = "RUN:Lock"


def default_archive_roles(family: FamilyFile) -> set[str]:
    """The roles §10's per-kind rule archives, for a family whose file marks none.

    A motor: ``.RBV``, ``.VAL``, ``.DMOV`` and the limit switches.  A
    setpoint device: its readbacks, setpoints and bits.  A detector (camera
    or scope): its settings' readbacks (the ``_RBV`` beside a written
    record) and the run lock.
    """
    if family.kind == "epics_motor":
        return {r.name for r in family.roles if r.suffix in _MOTOR_DEFAULT_SUFFIXES}
    if family.kind == "setpoint_device":
        return {
            r.name for r in family.roles if r.type in ("readback", "setpoint", "bit")
        }
    return {
        r.name
        for r in family.roles
        if r.write is not None or r.suffix == _RUN_LOCK_SUFFIX
    }


def archived_roles(family: FamilyFile, entry: PanelDevice | None) -> list[Role]:
    """The device's roles the archiver takes: the family's (or the kind's) defaults, then the device's opt-ins and opt-outs."""
    if not any(role.archive for role in family.roles):
        defaults = default_archive_roles(family)
        family = family.model_copy(
            update={
                "roles": [
                    r.model_copy(update={"archive": r.name in defaults})
                    for r in family.roles
                ]
            }
        )
    roles = entry.resolve(family) if entry is not None else list(family.roles)
    return [r for r in roles if r.archive and is_archivable(r)]


def is_archivable(role: Role) -> bool:
    """Whether the appliance can keep the role: a scalar, never a char array."""
    return role.type not in UNARCHIVABLE_ROLE_TYPES and role.text != "char_array"


def role_kind(role: Role) -> Kind:
    """``setpoint`` for a role whose record is its own write target, else ``readback``."""
    return "setpoint" if role.write_suffix == role.suffix else "readback"


def role_dtype(role: Role) -> str:
    """The candidate's dtype, from the role's type."""
    if role.text is not None:
        return "string"
    if role.type == "enum":
        return "enum"
    if role.type in ("bit", "counter"):
        return "int"
    return "float"


@dataclass
class RegistryPass:
    """What the registry pass derived, and what it left out and why."""

    candidates: list[ArchiveCandidate] = field(default_factory=list)
    #: Registry devices the DB also owns (enabled): reported, left to the DB pass.
    conflicts: list[str] = field(default_factory=list)
    #: Devices not archived here, with the reason (DB-owned bench device, no family …).
    skipped: list[str] = field(default_factory=list)
    #: Whether the DB roster was consulted (``False``: ``--no-db``).
    db_checked: bool = True


def family_for_kind(kind: str, families: Mapping[str, FamilyFile]) -> FamilyFile | None:
    """The one family file serving a registry *kind*, or ``None`` when none or several do."""
    matches = [f for f in families.values() if f.kind == kind]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class _Device:
    """One device the pass archives: its family as it has it, where its records are."""

    name: str
    entry: PanelDevice | None
    family: FamilyFile
    prefix: str
    axes: tuple[str, ...]


def _device(
    name: str,
    entry: PanelDevice | None,
    family: FamilyFile,
    prefix: str,
    axes: tuple[str, ...],
) -> _Device | str:
    """The device, or why it is skipped (a per-axis family with no axes)."""
    if family.per_axis and not axes:
        return f"{name}: a {family.template} device names no axes"
    return _Device(name, entry, family, prefix, axes)


def _registry_device(
    experiment: str,
    name: str,
    native: NativeDevice,
    entry: PanelDevice | None,
    families: Mapping[str, FamilyFile],
) -> _Device | str:
    """A ``native_devices.yaml`` device: the panel data's family, else its kind's; the derived prefix."""
    family = (
        families.get(entry.family)
        if entry is not None
        else family_for_kind(native.kind, families)
    )
    if family is None or family.kind != native.kind:
        named = f" (panel data names {entry.family!r})" if entry is not None else ""
        return f"{name}: no family of kind {native.kind!r}{named}"
    if isinstance(native, EpicsMotor):
        axes = tuple(native.axes)
    else:
        axes = tuple((entry.axes if entry is not None else None) or ())
    return _device(name, entry, family, native_device_prefix(experiment, name), axes)


def _panel_device(
    experiment: str, name: str, entry: PanelDevice, families: Mapping[str, FamilyFile]
) -> _Device | str:
    """A ``panel_devices.yaml`` device the registry does not list: its own prefix, else the derived one."""
    family = families.get(entry.family)
    if family is None:
        return f"{name}: no family {entry.family!r}"
    prefix = entry.prefix or native_device_prefix(experiment, name)
    return _device(name, entry, family, prefix, tuple(entry.axes or ()))


def _devices(
    experiment: str,
    registry: NativeDevices,
    panels: PanelDevices,
    owned: Mapping[str, str] | None,
    families: Mapping[str, FamilyFile],
    out: RegistryPass,
) -> list[_Device]:
    """Every device the pass archives; conflicts and skips recorded on *out*."""
    devices: list[_Device | str] = []
    for name, native in registry.devices.items():
        twin = None if owned is None else owned.get(normalize_component(name))
        if twin is not None:
            out.conflicts.append(
                f"{name}: in native_devices.yaml and enabled in the GEECS DB as "
                f"{twin!r} — one owner per device; archived from the DB only"
            )
            continue
        entry = panels.devices.get(name)
        devices.append(_registry_device(experiment, name, native, entry, families))
    for name, entry in panels.devices.items():
        if name in registry.devices:
            continue
        twin = None if owned is None else owned.get(normalize_component(name))
        if twin is not None:
            devices.append(f"{name}: a bench device the GEECS DB owns ({twin!r})")
            continue
        devices.append(_panel_device(experiment, name, entry, families))
    out.skipped.extend(d for d in devices if isinstance(d, str))
    return [d for d in devices if isinstance(d, _Device)]


def _device_candidates(
    device: _Device, policy: ArchivePolicy
) -> list[ArchiveCandidate]:
    """One device's candidates: its archived roles (per axis), then its kind's liveness record."""
    family = device.family
    axes = device.axes if family.per_axis else ("",)
    out: list[ArchiveCandidate] = []
    for role in archived_roles(family, device.entry):
        kind = role_kind(role)
        if kind == "setpoint" and not policy.include_setpoints:
            continue
        out.extend(
            ArchiveCandidate(
                device.prefix + axis + role.suffix,
                device.name,
                f"{axis}:{role.name}" if axis else role.name,
                kind,
                role_dtype(role),
                "registry",
            )
            for axis in axes
        )
    if policy.include_status:
        out.extend(
            ArchiveCandidate(
                device.prefix + suffix,
                device.name,
                suffix,
                "status",
                "enum",
                "registry",
            )
            for suffix in LIVENESS_SUFFIXES.get(family.kind, ())
        )
    return out


def derive_native_candidates(
    experiment: str,
    registry: NativeDevices,
    panels: PanelDevices,
    *,
    policy: ArchivePolicy,
    db_devices: Collection[str] | None,
    families: Mapping[str, FamilyFile],
) -> RegistryPass:
    """Apply the per-kind rule to the registry and the panel data (no DB, no network).

    Parameters
    ----------
    experiment : str
        GEECS experiment name; the PV namespace prefix.
    registry : NativeDevices
        The experiment's ``native_devices.yaml`` (empty when absent).
    panels : PanelDevices
        The experiment's ``panel_devices.yaml`` (empty when absent).
    policy : ArchivePolicy
        The overlay: ``include_setpoints`` and ``include_status`` apply here;
        the ``exclude`` globs apply where both passes meet
        (:func:`~geecs_archiver.archive_set.derive_candidates`).
    db_devices : collection of str, optional
        The DB's enabled devices; ``None`` when the DB was not read.
    families : mapping
        The family files by template (:func:`~geecs_schemas.load_families`).

    Returns
    -------
    RegistryPass
        Candidates tagged ``source="registry"``, in device order, before the
        policy's ``exclude`` globs.
    """
    owned = (
        None
        if db_devices is None
        else {normalize_component(name): name for name in db_devices}
    )
    out = RegistryPass(db_checked=owned is not None)
    devices = _devices(experiment, registry, panels, owned, families, out)
    for device in devices:
        out.candidates.extend(_device_candidates(device, policy))
    if policy.include_status:
        # One heartbeat per IOC, however many devices it serves.
        iocs = {
            d.entry.ioc.upper(): d.entry.ioc for d in devices if d.entry and d.entry.ioc
        }
        out.candidates.extend(
            ArchiveCandidate(
                ioc_heartbeat_pv(ioc), ioc, "HEARTBEAT", "status", "int", "registry"
            )
            for ioc in iocs.values()
        )
    return out
