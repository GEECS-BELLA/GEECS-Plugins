"""One line for a queue item, and the shot count a plan call implies.

The manager stores a stock plan item as ``name`` / ``args`` / ``kwargs``:
``args[0]`` is the detectors list the preset expansion built, the rest
are the verb's own positionals, and the GEECS keywords ride in ``kwargs``
(``shots_per_step``, ``acquisition``, ``trigger_profile``, ``md``).  The
summary reads those back into what an operator recognises — the axis, the
range, the step count, the shots, the mode — and never guesses at a device
it cannot see.  Tolerant of every shape the queue may hold: a history item
outlives the client that submitted it.

Ported from GEECS-Console's ``queue_panel.summarize_item`` and rewritten
for the stock verbs (the console summarized ``ScanRequest`` documents,
which the queue no longer carries).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Manager ``exit_status`` → (the word a chip shows, the kit state it wears).
EXIT_WORDS: dict[str, tuple[str, str]] = {
    "completed": ("ok", "ok"),
    "success": ("ok", "ok"),
    "failed": ("failed", "failed"),
    "fail": ("failed", "failed"),
    "stopped": ("stopped", "failed"),
    "aborted": ("stopped", "failed"),
    "abort": ("stopped", "failed"),
    "halted": ("halted", "failed"),
    "unknown": ("unknown", "unknown"),
}

#: Verbs whose positionals come in (motor, start, stop) triplets with one
#: trailing ``num``.
_TRIPLET_VERBS = {"scan", "rel_scan", "log_scan", "rel_log_scan"}
#: Verbs whose positionals come in (motor, start, stop, num) quadruplets.
_QUAD_VERBS = {"grid_scan", "rel_grid_scan"}
#: Verbs whose positionals come in (motor, points) pairs.
_LIST_VERBS = {"list_scan", "rel_list_scan", "list_grid_scan", "rel_list_grid_scan"}


@dataclass
class ItemSummary:
    """What one queue item says about itself, read back from its arguments."""

    plan: str
    text: str
    steps: Optional[int] = None
    shots_per_step: Optional[int] = None
    planned_shots: Optional[int] = None
    acquisition: Optional[str] = None
    trigger_profile: Optional[str] = None
    #: The run-level LabVIEW-files switch as the item carries it (``None`` =
    #: the experiment default, which only the worker knows).
    native_image_save: Optional[bool] = None
    preset: Optional[str] = None
    description: str = ""
    background: bool = False
    detectors: list[str] = field(default_factory=list)


def exit_word(exit_status: Any) -> tuple[str, str]:
    """``(word, kit state)`` for a manager or document exit status."""
    return EXIT_WORDS.get(str(exit_status or "unknown").lower(), ("unknown", "unknown"))


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return str(value)


def _int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _axis(motor: Any, start: Any, stop: Any) -> str:
    return f"{motor} {_fmt(start)} → {_fmt(stop)}"


def summarize_item(item: dict) -> ItemSummary:
    """Read one manager queue/history item back into an :class:`ItemSummary`.

    Parameters
    ----------
    item : dict
        The manager's item shape (``name`` / ``args`` / ``kwargs`` /
        ``user`` / ``item_uid`` [/ ``result``]).

    Returns
    -------
    ItemSummary
        Never raises: an unrecognised shape summarizes as its plan name
        and argument count.
    """
    name = str(item.get("name") or "?")
    args = list(item.get("args") or [])
    kwargs = dict(item.get("kwargs") or {})
    md = kwargs.get("md") if isinstance(kwargs.get("md"), dict) else {}
    geecs = md.get("geecs") if isinstance(md.get("geecs"), dict) else {}

    detectors = [str(d) for d in args[0]] if args and isinstance(args[0], list) else []
    rest = args[1:] if detectors or (args and isinstance(args[0], list)) else args

    summary = ItemSummary(
        plan=name,
        text=name,
        acquisition=kwargs.get("acquisition"),
        trigger_profile=kwargs.get("trigger_profile"),
        native_image_save=(
            None
            if kwargs.get("native_image_save") is None
            else bool(kwargs.get("native_image_save"))
        ),
        preset=geecs.get("preset"),
        description=str(md.get("description") or "").strip(),
        background=bool(md.get("background")),
        detectors=detectors,
    )
    sps = _int(kwargs.get("shots_per_step"))
    axes: list[str] = []
    steps: Optional[int] = None

    try:
        if name == "count":
            num = _int(kwargs.get("num", rest[0] if rest else 1)) or 1
            steps, sps = 1, num
            summary.text = f"count · {num} shots"
        elif name == "optimize":
            steps = _int(kwargs.get("max_iterations"))
            summary.acquisition = "strict"
            summary.text = f"optimize · {kwargs.get('optimizer_config', '?')} · ≤ {steps or '?'} iterations"
        elif name == "sweep":
            from geecs_schemas import Sweep

            payload = Sweep.model_validate(kwargs.get("sweep"))
            steps = payload.n_steps()
            for a in payload.axis_references():
                label = a.axis + (" (relative)" if a.relative else "")
                kind = getattr(a, "kind", None)
                if kind == "range":
                    label = _axis(label, a.start, a.stop)
                elif kind == "list":
                    label += f" [{len(a.positions)} pts]"
                elif kind == "log":
                    label += f" 10^{_fmt(a.start_exp)} → 10^{_fmt(a.stop_exp)}"
                axes.append(label)
            sps = sps or 1
        elif name in _TRIPLET_VERBS:
            num = _int(kwargs.get("num")) if "num" in kwargs else None
            body = rest
            if num is None and len(rest) % 3 == 1:
                num, body = _int(rest[-1]), rest[:-1]
            for i in range(0, len(body) - 2, 3):
                axes.append(_axis(body[i], body[i + 1], body[i + 2]))
            steps = num
        elif name in _QUAD_VERBS:
            counts: list[int] = []
            for i in range(0, len(rest) - 3, 4):
                axes.append(_axis(rest[i], rest[i + 1], rest[i + 2]))
                n = _int(rest[i + 3])
                if n:
                    counts.append(n)
            steps = 1
            for n in counts:
                steps *= n
            if not counts:
                steps = None
        elif name in _LIST_VERBS:
            lengths: list[int] = []
            for i in range(0, len(rest) - 1, 2):
                pts = rest[i + 1] if isinstance(rest[i + 1], list) else []
                axes.append(f"{rest[i]} [{len(pts)} pts]")
                lengths.append(len(pts))
            if lengths:
                steps = 1
                for n in lengths if "grid" in name else lengths[:1]:
                    steps *= n
        elif name == "mv":
            pairs = [
                f"{rest[i]} → {_fmt(rest[i + 1])}" for i in range(0, len(rest) - 1, 2)
            ]
            summary.text = "move " + ", ".join(pairs)
        elif name == "run_action":
            summary.text = f"action {rest[0]}" if rest else "action"
        elif name in ("measure_shot_offsets", "check_shot_sync"):
            summary.text = name.replace("_", " ")
    except (IndexError, TypeError, ValueError):  # a shape we do not know
        axes = []

    if axes:
        parts = [f"{name} " + " × ".join(axes)]
        if steps:
            parts.append(f"{steps} steps")
        if sps:
            parts.append(f"{sps} shots/step")
        summary.text = " · ".join(parts)
    if summary.acquisition and name not in ("mv", "run_action"):
        summary.text += f" · {summary.acquisition}"
    if summary.native_image_save is False and name not in ("mv", "run_action"):
        summary.text += " · no LabVIEW files"
    if summary.background:
        summary.text = "background · " + summary.text
    if summary.description:
        summary.text += f' — "{summary.description}"'

    summary.steps = steps
    summary.shots_per_step = sps
    if steps is not None and sps is not None:
        summary.planned_shots = steps * sps
    elif name == "count" and sps is not None:
        summary.planned_shots = sps
    return summary
