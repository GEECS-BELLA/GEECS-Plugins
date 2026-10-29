"""Every Canopy-internal name the package uses, and the image they were verified at (condition (d)).

Canopy declares none of them as a library surface. ``self_test`` runs before
the wrapper imports Canopy and refuses to start, naming the rollback, unless
``pin.py`` names ``VERIFIED_IMAGE``, the installed ``canopy`` is that commit
and every name resolves. A pin bump re-verifies the list and moves
``VERIFIED_IMAGE`` (DEPLOYMENT.md §5).
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Mapping, Sequence

from geecs_canopy import pin

VERIFIED_IMAGE = "ghcr.io/slaclab/canopy@sha256:219da36a98713bf524cd623d1364ead9eef95016fc474e15ab27fcc62d4d8abd"

#: ``module:attribute[.attribute]``: the wrapper's imports, the broker methods it calls, the adapter's base.
NAMES: tuple[str, ...] = tuple(
    """
    canopy.main:app canopy.core.modules:tag_module canopy.core.security:require_read_access
    canopy.core.errors:CanopyError canopy.core.errors:register_canopy_error_handler
    canopy.hare.gateway:get_value canopy.hare.gateway:get_broker
    canopy.hare.services.broker:PVBroker.put canopy.hare.services.broker:PVBroker.get_describe
    canopy.hare.services.broker:PVBroker.add_client canopy.hare.services.broker:PVBroker.subscribe
    canopy.raccoon.adapters.base:AdapterHealth canopy.raccoon.adapters.base:RaccoonError
    canopy.raccoon.adapters.base:DirectoryUnavailable canopy.raccoon.adapters.base:InvalidQuery
    canopy.raccoon.adapters.base:AdapterMisconfigured
    """.split()
)

#: The wire (alias) names read off Hare's frames and describes.
FIELDS: Mapping[str, tuple[str, ...]] = {
    "canopy.contracts.pv:PVValue": ("value", "severity", "alarmStatus", "connected"),
    "canopy.contracts.pv:PVMeta": ("enumChoices", "controlLow", "controlHigh"),
}

ROLLBACK = (
    "Roll back: the unwrapped api command `uvicorn canopy.main:app` with the "
    "previous pin (GeecsCanopy/DEPLOYMENT.md section 5)."
)


class SurfaceDrift(RuntimeError):
    """The Canopy in this process is not the one the module was verified against."""


def _resolve(name: str) -> object:
    module, _, path = name.partition(":")
    found: object = importlib.import_module(module)
    for part in path.split("."):
        found = getattr(found, part)
    return found


def self_test(
    names: Sequence[str] = NAMES,
    fields: Mapping[str, tuple[str, ...]] = FIELDS,
    *,
    image: str = pin.CANOPY_IMAGE,
    installed: str | None = None,
) -> None:
    """Raise ``SurfaceDrift`` naming every mismatch and the rollback; else return."""
    problems = []
    if image != VERIFIED_IMAGE:
        problems.append(f"pin.py names {image}, verified at {VERIFIED_IMAGE}")
    try:
        installed = installed or importlib.metadata.version("canopy")
    except importlib.metadata.PackageNotFoundError:
        installed = "nothing"
    if f"g{pin.CANOPY_COMMIT}" not in installed:
        problems.append(f"canopy {installed} is not commit {pin.CANOPY_COMMIT}")
    for name in [*names, *fields]:
        try:
            found = _resolve(name)
        except (ImportError, AttributeError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        have = {f.alias or n for n, f in getattr(found, "model_fields", {}).items()}
        problems += [
            f"{name} lacks {a!r}" for a in fields.get(name, ()) if a not in have
        ]
    if problems:
        raise SurfaceDrift("; ".join(problems) + f". {ROLLBACK}")
