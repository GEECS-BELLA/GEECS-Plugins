"""The catalogue: every registered kind, where it is kept, its tier, and who reads it.

Storage and tier come from GEECS-Schemas (``KIND_STORAGE``, each model's
``tier``), never restated here. What this module adds is the service's own
column: each file kind's **consumers** (who reads it, when, and the apply
verb that makes a committed change take effect) and its **dry run** (the
consumer's own check, run on an unsaved draft). Applies and dry runs run in
later slices; the table says now what they will be. Pinned to the file
kinds by ``tests/test_store.py``.
"""

from __future__ import annotations

from geecs_schemas import KIND_STORAGE, SCHEMA_REGISTRY, FileStorage, NotAFile
from pydantic import BaseModel

#: The worker reopens its environment to see these (the queue idle and empty).
_REOPEN = "environment reopen (when the queue is idle and empty)"
_PER_REQUEST = "per request"


class Consumer(BaseModel):
    """One reader of a kind: who, when it reads, and what applies a committed change."""

    name: str
    reads: str
    apply: str


def _c(name: str, reads: str, apply: str = "nothing to do") -> Consumer:
    return Consumer(name=name, reads=reads, apply=apply)


#: The device registry's other readers (native_devices.yaml and panel_devices.yaml are read together).
_CANOPY_DIRECTORY = _c("Canopy's directory adapter", "both registry files on every directory call", "not yet surveyed (Raccoon caches the directory's answers)")  # fmt: skip
_ARCHIVER_REGISTRY = _c("archiver (its registry pass)", "when onboard runs", "geecs-archiver onboard")  # fmt: skip
_CONTROL_ROOM_VIA_CANOPY = _c("Control Room (device pages, home, screens)", "through Canopy's GEECS module, per request", "as Canopy's directory")  # fmt: skip

#: Every file kind's readers (the brief's pickup survey and apply table).
CONSUMERS: dict[str, list[Consumer]] = {
    "trigger_profile": [
        _c("worker", "at environment open", _REOPEN),
        _c("scanner preflight", "per submit, by name"),
    ],
    "scan_variables": [
        _c("worker", "at environment open", _REOPEN),
        _c("scanner", "on file change"),
    ],
    "native_devices": [
        _c("worker", "at environment open", _REOPEN),
        _CANOPY_DIRECTORY,
        _ARCHIVER_REGISTRY,
        _CONTROL_ROOM_VIA_CANOPY,
    ],
    "shot_offsets": [_c("worker", "at environment open", _REOPEN)],
    "preset": [_c("scanner", "per submit")],
    "optimizer_config": [_c("worker", _PER_REQUEST)],
    "experiment_defaults": [_c("worker", "per plan")],
    "action_plan_library": [
        _c("Canopy's GEECS module", "per request, short cache"),
        _c("worker", "per call"),
    ],
    "operator_policy": [_c("Canopy's GEECS module", "per request, short cache")],
    "screen": [_c("Control Room", _PER_REQUEST)],
    "home_layout": [_c("Control Room", _PER_REQUEST)],
    "panel_devices": [_CANOPY_DIRECTORY, _ARCHIVER_REGISTRY, _CONTROL_ROOM_VIA_CANOPY],
    "derived_channels": [
        _c(
            "CA gateway",
            "at startup",
            "gateway restart: pending, the maintainer's, never during a scan",
        )  # fmt: skip
    ],
    "archive_policy": [_c("archiver", "when onboard runs", "geecs-archiver onboard")],
    "setup": [_c("worker and Canopy's GEECS module", "per apply")],
    "analysis_recipe": [_c("ScanAnalysis (the portal's runs)", "per run")],
    "himg_job": [_c("ScanAnalysis (the portal's runs)", "per run")],
    "analysis_group": [_c("ScanAnalysis (the portal's runs)", "per run")],
}

#: The consumer's own check of an unsaved draft, and the slice that wires it.
DRY_RUNS: dict[str, str] = {
    "trigger_profile": "the scanner's preflight with the draft inline (slice 3)",
    "archive_policy": "geecs-archiver onboard --dry-run --policy <draft> (slice 3)",
    "setup": "the scanner's settables check the targets (slice 4)",
    "analysis_recipe": "the portal's validate, bound to the analysis core (slice 6)",
    "analysis_group": "the portal's validate (slice 6)",
}


class KindInfo(BaseModel):
    """One registered kind as the API lists it."""

    kind: str
    title: str
    storage: str  # "file", "not-a-file", or "code" (package data, read-only)
    tier: str | None
    path: str | None
    reason: str | None
    consumers: list[Consumer]
    dry_run: str


def kinds() -> list[KindInfo]:
    """Every registered kind, file kinds first, so none is silently missing."""
    out: list[KindInfo] = []
    for kind, storage in KIND_STORAGE.items():
        model = SCHEMA_REGISTRY[kind]
        code = kind == "family_file"
        out.append(
            KindInfo(
                kind=kind,
                title=(model.__doc__ or kind).strip().splitlines()[0],
                storage="code" if code else storage.storage,
                tier="code" if code else model.tier,
                path=storage.path_pattern()
                if isinstance(storage, FileStorage)
                else None,
                reason=storage.reason if isinstance(storage, NotAFile) else None,
                consumers=CONSUMERS.get(kind, []),
                dry_run=DRY_RUNS.get(kind, "none"),
            )
        )
    order = {"file": 0, "code": 1, "not-a-file": 2}
    return sorted(out, key=lambda k: order[k.storage])
