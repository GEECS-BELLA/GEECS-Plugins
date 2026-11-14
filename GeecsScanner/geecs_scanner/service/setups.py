"""Setups through the scanner: the preview, the hazard confirmations, the diff and the apply's questions.

A setup (``<experiment>/setups/<name>.yaml``) is applied by the worker's
``apply_setup`` as one idle-only queue item; this module is the client side
of it, over GEECS-Schemas alone (never the worker's plans):

- **The walk** is :func:`geecs_schemas.flatten_setup`, the one the worker
  runs, so the preview promises the order the worker executes.
- **The hazard confirmations** are :func:`geecs_schemas.setup_hazards`
  over the same sources the device module's gate reads: the family files
  (as ``panel_devices.yaml`` amends them), the operator policy's
  ``hazards``, a native device's family and a DB device's device type.
  An unreadable source refuses the apply rather than asking nothing.
- **The questions** follow the preflight's pattern: each has a ``check``,
  the client relays it and sends it back acknowledged; the server never
  acknowledges one itself. Every check carries the setup's digest, so an
  acknowledgement given to one version of a setup never covers another.
- **The queue path checks no grants**, as a move and a scan check none:
  the operator policy's grants are the device module's gate (screen
  buttons), never this route's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from geecs_scanner.service.errors import ScannerError
from geecs_scanner.service.models import (
    PreflightQuestionOut,
    SetupHazardOut,
    SetupStepOut,
)

logger = logging.getLogger(__name__)

#: How many characters of the digest a question's ``check`` carries.
CHECK_DIGEST_CHARS = 16


@dataclass
class Flattened:
    """One setup resolved for the scanner: its steps in order, digest and file."""

    name: str
    description: str
    steps: list[tuple[Any, Optional[str]]]
    digest: str
    path: str


def flatten(name: str, setup: Any, setups: Mapping[str, Any], path: str) -> Flattened:
    """Every concrete step of *setup*, nested ``apply`` steps inlined, with its digest.

    Raises
    ------
    ScannerError
        ``invalid_request`` for an ``apply`` naming an unknown setup, or a loop.
    """
    from geecs_schemas import (
        SetupCycleError,
        SetupNotFoundError,
        flatten_setup,
        setup_digest,
    )

    try:
        steps = flatten_setup(setup, setups=setups)
    except (SetupNotFoundError, SetupCycleError) as exc:
        raise ScannerError("invalid_request", f"setup {name!r}: {exc}") from exc
    return Flattened(name, setup.description, steps, setup_digest(steps), path)


def _fmt(value: Any) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def describe(index: int, step: Any, origin: Optional[str]) -> SetupStepOut:
    """One flattened step as the page lists it."""
    do = str(step.do)
    out = SetupStepOut(index=index, do=do, from_setup=origin, text=do)
    if do == "set":
        out.device, out.variable, out.value = step.device, step.variable, step.value
        out.text = f"set {step.device}:{step.variable} = {_fmt(step.value)}"
    elif do == "check":
        out.device, out.variable = step.device, step.variable
        out.expected, out.tolerance = step.expected, step.tolerance
        slack = f" ± {step.tolerance:g}" if step.tolerance else ""
        out.text = (
            f"check {step.device}:{step.variable} == {_fmt(step.expected)}{slack}"
        )
    elif do == "wait":
        out.seconds = float(step.seconds)
        out.text = f"wait {step.seconds:g} s"
    return out


def hazards(
    flat: Flattened,
    *,
    resolver: Any,
    db_families: Optional[Mapping[str, str]],
) -> list[SetupHazardOut]:
    """The setup's hazard confirms, from the configs the device module reads.

    Raises
    ------
    ScannerError
        ``invalid_request`` when the operator policy, the native-device
        registry or the panel devices cannot be read: the confirmations
        cannot be asked, so the setup is not applied.
    """
    from geecs_schemas import load_families, setup_hazards

    try:
        policy = resolver.resolve_operator_policy()
        native = resolver.resolve_native_devices()
        panels = resolver.resolve_panel_devices()
    except Exception as exc:
        raise ScannerError(
            "invalid_request",
            f"the hazard confirmations of setup {flat.name!r} cannot be asked: {exc}",
        ) from exc
    found = setup_hazards(
        flat.steps,
        policy=policy,
        families=load_families(),
        native=native,
        panels=panels,
        db_families=db_families,
    )
    return [SetupHazardOut(**h.model_dump()) for h in found]


def questions(
    flat: Flattened, found: list[SetupHazardOut]
) -> list[PreflightQuestionOut]:
    """The apply's questions: the setup's writes, then one hazard confirm per hazardous step."""
    tag = flat.digest[:CHECK_DIGEST_CHARS]
    writes = [describe(i, s, o) for i, (s, o) in enumerate(flat.steps, 1) if s.do == "set"]  # fmt: skip
    others = len(flat.steps) - len(writes)
    lines = [f"{w.index}. {w.text}" for w in writes] or ["(no writes)"]
    first = PreflightQuestionOut(
        check=f"setup:{tag}",
        title=f"Apply setup {flat.name}",
        message=(
            f"Setup {flat.name!r} writes {len(writes)} value(s), in this order"
            + (f", with {others} wait/check step(s) between" if others else "")
            + ":\n"
            + "\n".join(lines)
            + "\nA refused write or a failed check stops it there; the writes "
            "before it stay."
        ),
        continue_label="Apply",
        abort_label="Cancel",
    )
    return [first] + [
        PreflightQuestionOut(
            check=f"hazard:{tag}:{h.index}",
            title=f"Hazard: {h.device} {h.variable}",
            message=(
                f"Step {h.index} writes {_fmt(h.value)} to {h.device}:{h.variable}, "
                "which needs the hazard confirm: " + "; ".join(h.reasons) + "."
            ),
            continue_label="Confirm and write",
            abort_label="Cancel",
        )
        for h in found
    ]


def provenance(
    flat: Flattened,
    *,
    operator: str,
    commit: Optional[str],
    answered: list[PreflightQuestionOut],
    client: str,
) -> dict[str, Any]:
    """The item's ``md``: the setup's name, path, commit and digest, the operator, the record.

    The worker reads ``digest`` and ``name``; the rest is provenance
    (``EVENT_SCHEMA.md`` § Queue items that open no run).
    """
    from geecs_bluesky.qs_client import build_submission_record

    record = build_submission_record(
        [(q.check, "continued", q.title) for q in answered], client=client
    )
    return {
        "geecs": {
            "operator": operator,
            "setup": {
                "name": flat.name,
                "path": flat.path,
                "commit": commit,
                "digest": flat.digest,
            },
            "submission": record.model_dump(mode="json"),
        }
    }
