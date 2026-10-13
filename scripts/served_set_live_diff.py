"""Read-only live diff: each consumer's former served-set rule vs ``geecs_core.db.served_set``.

Each consumer's former copy of the served-set rule is frozen here, verbatim in
effect, and compared against the consumer's current code on the real GEECS DB.
Every difference is printed by device and variable; no output under a part's
header means no change. The DB client only reads.

Run each part from its own package env (the three do not share one)::

    (cd GeecsCAGateway && poetry run python ../scripts/served_set_live_diff.py ca Undulator)
    (cd GeecsArchiver && poetry run python ../scripts/served_set_live_diff.py archiver Undulator)
    (cd GeecsBluesky && poetry run python ../scripts/served_set_live_diff.py bluesky Undulator)

Run ``scripts/lab_status.sh`` first: an unreachable DB blocks ~75 s.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from geecs_core.db.geecs_db import GeecsDb
from geecs_core.pv_naming import pv_name, setpoint_pv


def _fetch(experiment: str):
    """Endpoints, variable rows and get-list — the three read-only queries."""
    return (
        GeecsDb.get_experiment_devices(experiment),
        GeecsDb.get_experiment_device_variables(experiment),
        GeecsDb.get_subscribed_variables(experiment),
    )


def _report(label: str, old: set, new: set) -> int:
    """Print every difference; return how many."""
    for item in sorted(old - new):
        print(f"  {label} only-old: {item}")
    for item in sorted(new - old):
        print(f"  {label} only-new: {item}")
    print(f"  {label}: old={len(old)} new={len(new)} differences={len(old ^ new)}")
    return len(old ^ new)


def _facts(endpoints, var_map, sub_map) -> None:
    """The DB facts each drift case turns on, so a zero diff is explained."""
    print(f"  devices: endpoints={len(endpoints)} with-rows={len(var_map)}")
    print(f"  rows without an endpoint: {sorted(set(var_map) - set(endpoints))}")
    for device, rows in sorted(var_map.items()):
        names = [r["name"] for r in rows]
        dups = sorted(n for n, k in Counter(names).items() if k > 1)
        if dups:
            print(f"  duplicate rows: {device}: {dups}")
        folded = Counter(n.lower() for n in set(names))
        twins = sorted(n for n, k in folded.items() if k > 1)
        if twins:
            print(f"  case-only twins: {device}: {twins}")
        exact = set(names)
        lowered = {n.lower() for n in names}
        for v in sub_map.get(device, ()):
            if v not in exact and v.lower() in lowered:
                print(f"  get-list case mismatch: {device}: {v!r}")


def part_ca(experiment: str) -> int:
    """The CA PV list: the old get-list include path vs ``from_geecs_experiment``."""
    from geecs_ca_gateway.config import DeviceSpec, GatewayConfig

    endpoints, var_map, sub_map = _fetch(experiment)
    _facts(endpoints, var_map, sub_map)

    def pvs(specs) -> set[str]:
        names: Counter[str] = Counter()
        for spec in specs:
            for var in spec.variables:
                readback = pv_name(experiment, spec.name, var.geecs_var)
                names[f"{readback} [{var.dtype}]"] += 1
                if var.settable:
                    names[setpoint_pv(readback)] += 1
        # A PV twice is a startup collision: keep it visible, not collapsed.
        return {pv if n == 1 else f"{pv} x{n} COLLISION" for pv, n in names.items()}

    old_specs = []
    for name, (host, port) in endpoints.items():  # the pre-change loop
        try:
            spec = DeviceSpec.from_db_metadata(
                name,
                host,
                port,
                var_map.get(name, []),
                include=sub_map.get(name, []),
                include_settable=True,
                experiment=experiment,
            )
        except Exception:  # noqa: BLE001
            continue
        if spec.variables:
            old_specs.append(spec)
    new_specs = GatewayConfig.from_geecs_experiment(experiment).devices
    n = _report("CA devices", {s.name for s in old_specs}, {s.name for s in new_specs})
    return n + _report("CA PVs", pvs(old_specs), pvs(new_specs))


def part_archiver(experiment: str) -> int:
    """The archive set: the old loop vs ``build_archive_set`` (default policy)."""
    from geecs_archiver.archive_set import (
        UNARCHIVABLE_DTYPES,
        build_archive_set,
        classify,
        is_timestamp_variable,
    )
    from geecs_core.pv_naming import device_status_pv
    from geecs_schemas import ArchivePolicy

    endpoints, var_map, sub_map = _fetch(experiment)
    old: set[str] = set()
    for device in sorted(endpoints):  # the pre-change loop, default policy
        monitored = set(sub_map.get(device, ()))
        seen: set[str] = set()
        serves_anything = False
        for row in var_map.get(device, ()):
            name = str(row["name"])
            if name in seen:
                continue
            seen.add(name)
            settable = bool(row.get("settable", False))
            if not (name in monitored or settable):
                continue
            dtype = classify(row)
            if dtype is None:
                continue
            serves_anything = True
            if is_timestamp_variable(name) or dtype in UNARCHIVABLE_DTYPES:
                continue
            readback = pv_name(experiment, device, name)
            if name in monitored:
                old.add(readback)
            if settable:
                old.add(setpoint_pv(readback))
        if serves_anything:
            old.add(device_status_pv(experiment, device))
    new = {c.pv for c in build_archive_set(experiment, policy=ArchivePolicy())}
    return _report("archive set", old, new)


def part_bluesky(experiment: str) -> int:
    """The namespace children/columns: the old provider + loop vs the roster."""
    from geecs_bluesky.namespace import DeviceRoster, python_type
    from geecs_core.db.served_set import served_variables

    _endpoints, var_map, sub_map = _fetch(experiment)
    old_served: dict[str, set[str]] = {d: set(v) for d, v in sub_map.items()}
    for device, rows in var_map.items():  # the pre-change provider
        settable = {r["name"] for r in rows if bool(r.get("settable", False))}
        if settable:
            old_served.setdefault(device, set()).update(settable)
    old: set[tuple] = set()
    for device, rows in var_map.items():  # the pre-change _build typed loop
        served = {v.lower() for v in old_served.get(device, ())}
        typed = {}
        for row in rows:
            if row["name"].lower() in served and python_type(row) is not None:
                typed[row["name"]] = bool(row.get("settable"))
        old |= {(device, n, s) for n, s in typed.items()}

    roster = DeviceRoster.from_geecs_db(experiment)
    new: set[tuple] = set()
    for device, rows in roster.variables.items():  # the current _build typed loop
        served = roster.served_for(device)
        sub = {device: roster.subscribed.get(device, ())}
        cand = served_variables({device: rows}, sub)
        for sv in cand.get(device, ()):
            if sv.name.lower() in served and python_type(sv.row) is not None:
                new.add((device, sv.name, sv.settable))
    return _report("namespace children/columns (device, variable, settable)", old, new)


def main() -> int:
    """Run one part; exit 1 on any difference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["ca", "archiver", "bluesky"])
    parser.add_argument("experiment")
    args = parser.parse_args()
    print(f"== {args.part} ({args.experiment})")
    differences = globals()[f"part_{args.part}"](args.experiment)
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
