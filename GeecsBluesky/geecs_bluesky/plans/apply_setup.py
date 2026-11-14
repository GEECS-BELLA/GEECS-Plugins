"""``apply_setup``: a named setup as one queue item over the device namespace.

A setup (``geecs_schemas.Setup``, ``<experiment>/setups/<name>.yaml``) is a
named device state: ``set`` / ``wait`` / ``check`` steps, with ``apply``
steps inlining other setups. The worker runs one as plain plan stubs, so
it inherits the RunEngine's pause, stop and abort. It opens no run,
claims no scan number and writes no file.

The rules (pinned by ``tests/test_apply_setup.py``): the setup is read
and flattened when the item runs; a digest in ``md["geecs"]["setup"]``
(what the operator confirmed) that differs from the setup's now refuses
the item before the first write; every target is resolved and read
before the first write; a ``set`` is the device's own completion
(``abs_set(wait=True)``, no second confirmation layer) and a ``check``
matches as the confirm settable does; a failure stops the setup and
names the step and the writes already made
(:class:`~geecs_bluesky.exceptions.SetupStepFailedError`), while a
RunEngine pause, stop or abort passes through. ``md`` is otherwise
provenance, kept in the queue item and the log (``EVENT_SCHEMA.md``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Optional, Protocol, runtime_checkable

import bluesky.plan_stubs as bps
from bluesky.protocols import Movable, Readable
from bluesky.utils import MsgGenerator, RunEngineControlException
from geecs_schemas import (
    SetupCheckStep,
    SetupCycleError,
    SetupNotFoundError,
    SetupSetStep,
    flatten_setup,
    setup_digest,
    setup_value_matches,
)

from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    SetupCheckFailedError,
    SetupStepFailedError,
    failure_cause_text,
)
from geecs_bluesky.utils import resolve_annotations

logger = logging.getLogger(__name__)

__all__ = ["SettableFactory", "apply_setup_plan", "describe_setup_step"]


@runtime_checkable
class SettableFactory(Protocol):
    """Hands out signals for GEECS ``(device, variable)`` pairs.

    The plan never builds a PV string: the production factory is the
    device namespace (setpoint puts riding GEECS blocking sets, or a native
    device's own records), while tests return in-memory mock signals.
    Factories raise their own errors for unknown devices or variables.
    """

    def get_settable(self, device: str, variable: str) -> Movable:
        """Return a settable signal (accepted by ``bps.abs_set``) for a ``set`` step."""
        ...

    def get_readable(self, device: str, variable: str) -> Readable:
        """Return a readable signal (accepted by ``bps.rd``) for a ``check`` step."""
        ...


def describe_setup_step(step: Any) -> str:
    """One flattened step in the words a log line or a refusal uses."""
    if isinstance(step, SetupSetStep):
        return f"set {step.device}:{step.variable} = {step.value!r}"
    if isinstance(step, SetupCheckStep):
        slack = f" ± {step.tolerance:g}" if step.tolerance else ""
        return f"check {step.device}:{step.variable} == {step.expected!r}{slack}"
    return f"wait {step.seconds:g} s"


def _confirmed(md: Optional[Mapping[str, Any]]) -> tuple[dict[str, Any], str]:
    """``(md["geecs"]["setup"], operator)`` from the item's metadata; empty when absent."""
    geecs = md.get("geecs") if isinstance(md, Mapping) else None
    if not isinstance(geecs, Mapping):
        return {}, ""
    record = geecs.get("setup")
    operator = str(geecs.get("operator") or "")
    return (dict(record) if isinstance(record, Mapping) else {}), operator


def _flatten_checked(
    resolver: Any, name: str, record: Mapping[str, Any]
) -> tuple[list[tuple[Any, Optional[str]]], str]:
    """The setup's steps and digest, refused when they are not what *record* says was confirmed."""
    setup, setups = resolver.resolve_setup(name)
    try:
        steps = flatten_setup(setup, setups=setups)
    except (SetupNotFoundError, SetupCycleError) as exc:
        raise GeecsConfigurationError(f"setup {name!r}: {exc}") from exc
    digest = setup_digest(steps)
    if record.get("name") not in (None, name):
        raise GeecsConfigurationError(
            f"apply_setup {name!r}: the item's md names the setup "
            f"{record.get('name')!r}; nothing was written"
        )
    confirmed = record.get("digest")
    if confirmed is not None and confirmed != digest:
        raise GeecsConfigurationError(
            f"setup {name!r} changed since it was confirmed (the file, or a "
            "setup it applies, was edited): nothing was written; look at it "
            "again and apply it anew"
        )
    return steps, digest


def _signals(
    settables: SettableFactory, name: str, steps: list[tuple[Any, Optional[str]]]
) -> list[Any]:
    """Each step's signal (``None`` for a wait), every one resolved before anything runs."""
    signals: list[Any] = []
    for index, (step, _origin) in enumerate(steps, 1):
        try:
            if isinstance(step, SetupSetStep):
                signals.append(settables.get_settable(step.device, step.variable))
            elif isinstance(step, SetupCheckStep):
                signals.append(settables.get_readable(step.device, step.variable))
            else:
                signals.append(None)
        except Exception as exc:
            what = describe_setup_step(step)
            cause = failure_cause_text(exc)
            raise SetupStepFailedError(
                name, index, len(steps), what, (), cause
            ) from exc
    return signals


def _connect(name: str, total: int, signals: list[Any]) -> MsgGenerator:
    """Read each distinct target once, the read that connects it: a dead device fails before any write."""
    distinct = {id(s): s for s in signals if s is not None}
    for target in distinct.values():
        try:
            yield from bps.rd(target)
        except RunEngineControlException:
            raise
        except Exception as exc:
            what = f"reading {getattr(target, 'name', target)!r} before the first write"
            cause = failure_cause_text(exc)
            raise SetupStepFailedError(name, 0, total, what, (), cause) from exc


def _run_step(step: Any, signal: Any) -> MsgGenerator:
    """One step: a set to the device's own completion, a check by the confirm settable's match, a wait."""
    if isinstance(step, SetupSetStep):
        yield from bps.abs_set(signal, step.value, wait=True)
    elif isinstance(step, SetupCheckStep):
        actual = yield from bps.rd(signal)
        if not setup_value_matches(step.expected, actual, step.tolerance):
            raise SetupCheckFailedError(
                step.device, step.variable, step.expected, actual, step.tolerance
            )
    else:
        yield from bps.sleep(step.seconds)


def apply_setup_plan(
    resolver: Any | None, settables: SettableFactory | None
) -> Callable[..., Any]:
    """The ``apply_setup`` queue plan over *resolver* and *settables* (the namespace).

    Parameters
    ----------
    resolver :
        The experiment's config resolver (``resolve_setup``,
        ``setup_repo_path``).
    settables :
        Hands out the signals the steps touch: the device namespace.

    Either ``None`` registers a plan that refuses to run (the hermetic
    worker, ``QS_DEVICE_NAMESPACE=off``), so the manager's plan list is the
    same in every mode.
    """

    def apply_setup(name: str, *, md: Optional[dict] = None) -> MsgGenerator:
        """Apply the setup *name* from the experiment's ``setups/``: its steps, in order.

        Opens no run. Every target is resolved and read before the first
        write; a ``check`` mismatch or a refused ``set`` stops the setup
        there, and the writes before it stay.

        Parameters
        ----------
        name : str
            The setup's name (its file stem).
        md : dict, optional
            Provenance from the client: ``md["geecs"]["setup"]`` (``name``,
            ``path``, ``commit``, ``digest``) and ``md["geecs"]["operator"]``.
            A ``digest`` that differs from the setup's now refuses the item.
        """
        if resolver is None or settables is None:
            raise GeecsConfigurationError(
                "apply_setup: this worker has no device namespace "
                "(QS_DEVICE_NAMESPACE=off), so setups cannot be applied"
            )
        record, operator = _confirmed(md)
        steps, digest = _flatten_checked(resolver, name, record)
        signals = _signals(settables, name, steps)
        total = len(steps)
        logger.info(
            "apply_setup %s for %s: %d step(s), path %s, commit %s, digest %s%s",
            name,
            operator or "an unnamed operator",
            total,
            record.get("path") or resolver.setup_repo_path(name),
            record.get("commit") or "unknown",
            digest[:12],
            "" if record.get("digest") else " (no confirmed digest: not checked)",
        )
        yield from _connect(name, total, signals)
        written: list[int] = []
        for index, ((step, _origin), signal) in enumerate(zip(steps, signals), 1):
            try:
                yield from _run_step(step, signal)
            except RunEngineControlException:
                raise  # a pause, stop or abort is the RunEngine's, never a step's failure
            except Exception as exc:
                what, cause = describe_setup_step(step), failure_cause_text(exc)
                raise SetupStepFailedError(
                    name, index, total, what, written, cause
                ) from exc
            if isinstance(step, SetupSetStep):
                written.append(index)
            logger.info("apply_setup %s: step %d/%d done: %s", name, index, total, describe_setup_step(step))  # fmt: skip
        logger.info("apply_setup %s: all %d step(s) done", name, total)

    return resolve_annotations(apply_setup, {"name": str, "md": Optional[dict]})
