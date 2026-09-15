"""CaPseudoPositioner — a scan-variable catalog pseudo as a pseudo positioner.

The runtime for :class:`~geecs_schemas.scan_variables.PseudoScanVariable`:
one scanned number, several GEECS components, a bidirectional relation
between them.  ``set(value)`` computes every component's setting and moves
them together **through each component's own** ``set()`` (a
:class:`~geecs_bluesky.devices.ca.motor.CaMotor` waits for the device's
reply under its stall rule, a
:class:`~geecs_bluesky.devices.ca.settable.CaSettable` rides the native
blocking set) — GEECS-Plugins#910: no put budget of this class's own.  The
readback is **derived from the components' live readbacks** through the
relation's inverse, so it is defined before any set, after a restart, after
a hand move, and ``locate()`` is real (the case GEECS-Plugins#855 was filed
about).

The relation is an ophyd-async :class:`~ophyd_async.core.Transform`:
``derived_to_raw`` is the catalog's ``forward`` formulas, ``raw_to_derived``
the inverse — derived by the software for an affine ``forward``
(:func:`~geecs_bluesky.forward_expr.affine_coefficients`: the identity
component where one exists, else the first), supplied by the physicist as
the catalog's ``inverse`` otherwise (``R56_at_100MeV``).  A
:class:`~ophyd_async.core.DerivedSignalFactory` over the component readback
signals produces the readback child, and the parameters of the transform
are the components' **user offsets** (:attr:`CaSettable.offset`).

Two kinds of entry, one class (the rulings are in ``GeecsBluesky/CLAUDE.md``):

- a **plain pseudo positioner** (the catalog's ``mode: absolute``) reads its
  components in the *dial* frame — the offsets are wired as zeros.  Its
  value has absolute meaning (an R56 in mm, a compressor position);
  components off the formula before a scan is normal (someone moved the
  mode imager by hand): a WARNING at ``locate``, and the first step snaps
  every component onto the formula.
- ``mode: relative`` reads its components in the *user* frame, zeroed at
  every ``stage()`` (:meth:`CaSettable.set_current_position`), so the value
  is a **deviation from today's alignment** — the steering bumps.  The
  readback is 0 by construction before the first step, ``set(0)`` puts the
  components back (every relative ``forward`` is pinned ``f(0) = 0`` at
  build), and ``unstage()`` restores the captured baselines — end of scan,
  abort and halt alike (the RunEngine unstages every leftover staged
  object on every exit path; on a ``halt`` it does not wait for the
  status, so a restore that fails there surfaces only in the journal).
  A restore that failed leaves the pseudo **owing** its components their
  baselines: the next ``stage()`` refuses
  (:class:`~geecs_bluesky.exceptions.PseudoRestorePendingError`) rather
  than zero with the leftover bump baked in, and ``mv <pseudo> 0`` — the
  offsets still hold the true baselines — puts them back and clears it.

**The disagreement check** carries the weight of the over-determined
readback: ``forward(inverse(readbacks))`` is compared with what the
components actually read, per component, within its tolerance **plus what
the inverse propagates**: the components the inverse reads sit within
their own tolerances too, and that error reaches every other component's
prediction scaled by the relation (×2 on the S4H of an angle bump), so the
allowance is each component's tolerance plus how far its prediction moves
when the value shifts by the inverse's own uncertainty.  When they agree
every inverse choice gives the same value.  When they disagree
*after this pseudo has moved them* the scan **fails**
(:class:`~geecs_bluesky.exceptions.PseudoComponentsDisagreeError`) — a
component moved under the scan, and moving the others onto the formula is
the paired-magnet incident nobody wants.  Before the first move a plain
pseudo warns and snaps; a relative one cannot disagree (its deviations were
just zeroed), so it fails there too.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from bluesky.protocols import Location
from ophyd_async.core import (
    DEFAULT_TIMEOUT,
    AsyncStatus,
    DerivedSignalFactory,
    StandardReadable,
    Transform,
)
from pydantic import create_model

from geecs_bluesky.devices.ca.motor import within_tolerance
from geecs_bluesky.devices.ca.settable import CaSettable
from geecs_bluesky.exceptions import (
    GeecsConfigurationError,
    PseudoComponentsDisagreeError,
    PseudoRestorePendingError,
)
from geecs_bluesky.forward_expr import (
    CompiledForward,
    CompiledInverse,
    affine_coefficients,
    compile_forward,
    compile_inverse,
)
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

#: Agreement tolerance for a component whose DB tolerance is unset or 0
#: (the steering magnets, GEECS-Plugins#780) — in the component's own
#: units.  Well above the ~0.5 mA readback scatter of a magnet supply,
#: well below any step a scan takes; a DB tolerance replaces it.
DEFAULT_AGREEMENT_TOLERANCE = 0.01


class PseudoTransform(Transform):
    """Base of the per-entry transform classes (:func:`transform_class`).

    Fields (one ``offset_<key>: float`` per component) are the components'
    user offsets, fed live from their :attr:`CaSettable.offset` signals for
    a relative entry and wired as ``0.0`` for a plain one.  The relation
    itself is class state: ``forward`` maps the value to each component's
    *user* position, ``inverse`` recovers the value from the components'
    user positions.  ``user = dial + offset`` (the EPICS motor record's
    convention); the raw signals are the dial.
    """

    keys: ClassVar[tuple[str, ...]] = ()
    forward: ClassVar[tuple[Callable[[float], float], ...]] = ()
    inverse: ClassVar[Callable[[Mapping[str, float]], float]] = lambda user: math.nan

    def _offset(self, key: str) -> float:
        return float(getattr(self, f"offset_{key}"))

    def raw_to_derived(self, **kwargs: float) -> dict[str, float]:
        """The value from the components' dial readbacks (``kwargs``: key → dial)."""
        user = {k: float(kwargs[k]) + self._offset(k) for k in self.keys}
        return {"value": float(type(self).inverse(user))}

    def derived_to_raw(self, *, value: float) -> dict[str, float]:
        """Each component's dial setting for *value* (key → dial)."""
        return {
            k: float(f(value)) - self._offset(k)
            for k, f in zip(self.keys, type(self).forward)
        }


def transform_class(
    name: str,
    keys: Sequence[str],
    forward: Sequence[Callable[[float], float]],
    inverse: Callable[[Mapping[str, float]], float],
) -> type[PseudoTransform]:
    """A :class:`PseudoTransform` subclass with one ``offset_<key>`` field per component."""
    cls = create_model(
        f"{name}Transform",
        __base__=PseudoTransform,
        **{f"offset_{k}": (float, ...) for k in keys},
    )
    cls.keys = tuple(keys)
    cls.forward = tuple(forward)
    cls.inverse = staticmethod(inverse)  # type: ignore[assignment]
    return cls


def affine_inverse(
    keys: Sequence[str], coefficients: Sequence[tuple[float, float]]
) -> Callable[[Mapping[str, float]], float]:
    """The inverse of an affine relation: read the identity component, else the first.

    11 of the 13 corpus pseudos carry a component whose ``forward`` is the
    value itself (``x``, ``x * 1``): that component *is* the value.  The
    choice only affects the number in a disagreement message — when the
    components agree every choice gives the same value.
    """
    index = next(
        (i for i, (a, b) in enumerate(coefficients) if a == 1.0 and b == 0.0),
        next((i for i, (a, _) in enumerate(coefficients) if a != 0.0), None),
    )
    if index is None:
        raise GeecsConfigurationError(
            "no component depends on the scanned value (every forward is a "
            "constant) — nothing can define the inverse"
        )
    key, (a, b) = keys[index], coefficients[index]

    def inverse(user: Mapping[str, float]) -> float:
        return (float(user[key]) - b) / a

    return inverse


class CaPseudoPositioner(StandardReadable):
    """A pseudo scan variable as a Bluesky ``Movable`` + ``Locatable`` over its components.

    Parameters
    ----------
    components : sequence of (target, CaSettable)
        Each ``"Device:Variable"`` target with the namespace's Movable for
        it (a ``CaMotor`` or ``CaSettable``).  Component moves go through
        these objects' own ``set()``.
    forward : sequence of callables
        Per component, the value → user-position formula (the catalog's
        ``forward``).
    inverse : callable
        Components' user positions (by key, see :func:`component_key`) →
        the value.
    relative : bool
        ``True`` for the catalog's ``mode: relative`` (components zeroed at
        stage, restored at unstage); ``False`` for a plain pseudo positioner.
    tolerances : sequence of float
        Per component, the agreement tolerance (its DB tolerance, else
        :data:`DEFAULT_AGREEMENT_TOLERANCE`).
    variable_name : str
        The catalog's friendly name (the exporter column header).
    name : str
        ophyd-async device name (namespaces the event keys).
    """

    def __init__(
        self,
        components: Sequence[tuple[str, CaSettable]],
        forward: Sequence[Callable[[float], float]],
        inverse: Callable[[Mapping[str, float]], float],
        *,
        relative: bool,
        tolerances: Sequence[float],
        variable_name: str,
        name: str = "pseudo",
    ) -> None:
        if len(components) != len(forward) or len(components) != len(tolerances):
            raise ValueError("one forward formula and one tolerance per component")
        keys = [component_key(target) for target, _ in components]
        if len(set(keys)) != len(keys):
            raise GeecsConfigurationError(
                f"pseudo {variable_name!r}: a component is listed twice ({keys})"
            )
        self._targets: list[str] = [target for target, _ in components]
        self._components: list[CaSettable] = [comp for _, comp in components]
        self._keys = keys
        self._relative = relative
        self._tolerances = [float(t) for t in tolerances]
        self._variable_name = variable_name
        raw = {
            key: getattr(comp, comp._readback_attr_name)
            for key, comp in zip(keys, self._components)
        }
        params: dict[str, Any] = {
            f"offset_{key}": (comp.offset if relative else 0.0)
            for key, comp in zip(keys, self._components)
        }
        self._factory = DerivedSignalFactory(
            transform_class(safe_name(name), keys, forward, inverse), **raw, **params
        )
        with self.add_children_as_readables():
            # The recorded column: the value, from the components' readbacks.
            self.readback = self._factory.derived_signal_r(float, "value")
        super().__init__(name=name)
        self._zeroed = False  # relative: offsets captured by this pseudo
        self._moved = False  # this pseudo has moved its components since stage
        self._restore_pending = False  # staged, and the baselines not yet put back
        self._staged = False  # between stage() and unstage(): a scan is driving us
        #: ``{"Device:Variable": dial setting}`` of the last completed set
        #: (``None`` until one succeeds) — operator feedback for manual moves.
        self.last_commanded: dict[str, float] | None = None

    @property
    def _column_headers(self) -> dict[str, str]:
        """Readback event key → the catalog's friendly name (the s-file header)."""
        return {self.readback.name: self._variable_name}

    @property
    def relative(self) -> bool:
        """Whether the components are read in their user frame, zeroed at stage."""
        return self._relative

    async def connect(
        self,
        mock: Any = False,
        timeout: float = DEFAULT_TIMEOUT,
        force_reconnect: bool = False,
    ) -> None:
        """Connect the readback and the components it derives from.

        The components are another device's children (never re-parented
        here) and a derived signal's own connect assumes its inputs are
        connected — so a pseudo touched on demand (``connect_on_demand``)
        connects them itself.  A real connect is cached by ophyd-async; a
        mock one is not, so a component already carrying a mock keeps it
        (re-mocking would drop the callbacks a test registered on it).
        """
        await asyncio.gather(
            super().connect(
                mock=mock, timeout=timeout, force_reconnect=force_reconnect
            ),
            *(
                comp.connect(
                    mock=mock, timeout=timeout, force_reconnect=force_reconnect
                )
                for comp in self._components
                if not (mock and comp._mock is not None and not force_reconnect)
            ),
        )

    # --------------------------------------------------------------- frame
    async def _zero_components(self) -> None:
        """Zero every component's user offset here: today's alignment is the baseline."""
        await asyncio.gather(
            *(comp.set_current_position(0.0) for comp in self._components)
        )
        self._zeroed = True
        self._moved = False
        offsets = await asyncio.gather(
            *(comp.offset.get_value() for comp in self._components)
        )
        logger.info(
            "%s: baselines captured (components zeroed): %s",
            self.name,
            {t: -float(o) for t, o in zip(self._targets, offsets)},
        )

    @AsyncStatus.wrap
    async def stage(self) -> None:
        """Stage the readback; a relative pseudo zeroes its components first.

        Refused while a previous scan's restore is still owed (it failed):
        zeroing now would make the leftover bump the new baseline.
        """
        if self._relative:
            if self._restore_pending:
                offsets = await asyncio.gather(
                    *(comp.offset.get_value() for comp in self._components)
                )
                held = {t: -float(o) for t, o in zip(self._targets, offsets)}
                raise PseudoRestorePendingError(
                    f"{self._variable_name}: the previous scan's restore did not "
                    f"complete — the baselines still owed are {held}. Move the "
                    "pseudo to 0 (mv …, 0) to put the components back, then scan"
                )
            await self._zero_components()
            self._restore_pending = True
        self._moved = False
        self._staged = True
        await super().stage().task

    @AsyncStatus.wrap
    async def unstage(self) -> None:
        """Unstage; a relative pseudo puts its components back at their baselines.

        Runs at the end of a scan, on abort and on halt (the stock plans'
        ``stage_wrapper`` unstages as a finalize, and the RunEngine sweeps
        every leftover staged object on exit — without awaiting it on a
        ``halt``).  The restore is formula-independent — each
        component goes to the dial position it had when zeroed — and rides
        the components' own ``set()``, so a failed restore fails the plan
        visibly (each component ERROR-logs its own refused put).
        """
        try:
            if self._relative and self._zeroed:
                await self._restore_baselines()
                # Only a completed restore releases the frame: after a failed
                # one the offsets still hold the baselines and set(0) restores.
                self._zeroed = False
                self._restore_pending = False
        finally:
            self._moved = False
            self._staged = False
            await super().unstage().task

    async def _restore_baselines(self) -> None:
        offsets = await asyncio.gather(
            *(comp.offset.get_value() for comp in self._components)
        )
        baselines = [-float(o) for o in offsets]  # user 0 = dial −offset
        logger.info(
            "%s: restoring baselines: %s",
            self.name,
            dict(zip(self._targets, baselines)),
        )
        await _move_all([(comp, b) for comp, b in zip(self._components, baselines)])
        self.last_commanded = dict(zip(self._targets, baselines))

    # ------------------------------------------------------------ position
    async def _read_dials(self) -> dict[str, float]:
        values = await asyncio.gather(
            *(
                getattr(comp, comp._readback_attr_name).get_value()
                for comp in self._components
            )
        )
        return {k: float(v) for k, v in zip(self._keys, values)}

    async def _check_agreement(self, *, fail: bool) -> float:
        """The value from the components, after checking they sit on the formula.

        Returns the value.  Components off the formula by more than their
        tolerance raise :class:`PseudoComponentsDisagreeError` when *fail*,
        else log a WARNING naming each one and its discrepancy.
        """
        transform = await self._factory.transform()
        dials = await self._read_dials()
        value = transform.raw_to_derived(**dials)["value"]
        predicted = transform.derived_to_raw(value=value)
        allowance = self._allowance(transform, dials, value, predicted)
        off = {
            target: (dials[key], predicted[key])
            for target, key in zip(self._targets, self._keys)
            if not within_tolerance(dials[key], predicted[key], allowance[key])
        }
        if not off:
            return value
        detail = ", ".join(
            f"{t} reads {read:.6g}, formula says {pred:.6g} at {value:.6g}"
            for t, (read, pred) in off.items()
        )
        if fail:
            raise PseudoComponentsDisagreeError(
                f"{self._variable_name}: components disagree with the formula — "
                f"{detail}. A component moved under the scan; nothing was moved."
            )
        logger.warning(
            "%s: components off the formula (normal before the first step; the "
            "first move snaps them onto it): %s",
            self.name,
            detail,
        )
        return value

    def _allowance(
        self,
        transform: PseudoTransform,
        dials: Mapping[str, float],
        value: float,
        predicted: Mapping[str, float],
    ) -> dict[str, float]:
        """Per component: its tolerance plus what the inverse's own uncertainty adds.

        The components the inverse reads sit within *their* tolerances, so
        the value is uncertain by however much shifting each of them by its
        tolerance moves the inverse; every prediction then moves by that
        value shift through its own ``forward``.  For an angle bump read
        through S3H this is S4H's tolerance plus twice S3H's.  Evaluated
        numerically so it holds for any transform, not only the affine.
        """
        spread = 0.0
        for key, tol in zip(self._keys, self._tolerances):
            for sign in (1.0, -1.0):
                shifted = dict(dials)
                shifted[key] = dials[key] + sign * tol
                try:
                    v = transform.raw_to_derived(**shifted)["value"]
                except GeecsConfigurationError:
                    continue
                if math.isfinite(v):
                    spread = max(spread, abs(v - value))
        allowance: dict[str, float] = {}
        for key, tol in zip(self._keys, self._tolerances):
            moved = 0.0
            for sign in (1.0, -1.0):
                try:
                    shifted_prediction = transform.derived_to_raw(
                        value=value + sign * spread
                    )[key]
                except GeecsConfigurationError:
                    continue
                if math.isfinite(shifted_prediction):
                    moved = max(moved, abs(shifted_prediction - predicted[key]))
            allowance[key] = tol + moved
        return allowance

    async def locate(self) -> Location:
        """Where the pseudo is — the inverse over the components' readbacks, both fields.

        Implements :class:`bluesky.protocols.Locatable` — what the ``rel_*``
        plans stash before the first move and restore afterwards.  A
        relative pseudo reads 0 here by construction (its components were
        zeroed at stage, or are zeroed now for an unstaged caller); a plain
        one reads its real value and warns about components off the
        formula.
        """
        if self._relative and not self._zeroed:
            await self._zero_components()
        value = await self._check_agreement(fail=self._relative or self._moved)
        return Location(setpoint=value, readback=value)

    # ----------------------------------------------------------------- move
    def set(self, value: float) -> AsyncStatus:
        """Move every component to its setting for *value*; completes when all have.

        Implements :class:`bluesky.protocols.Movable`.
        """
        return AsyncStatus(self._set_and_wait(float(value)))

    async def _set_and_wait(self, value: float) -> None:
        if not math.isfinite(value):
            raise ValueError(f"{self._variable_name}: cannot move to {value}")
        if self._relative and not self._zeroed:
            # Unstaged caller (a manual mv): today's positions are the baseline.
            await self._zero_components()
        restoring = (
            self._relative
            and self._restore_pending
            and not self._staged  # a scan point at 0 is an ordinary step
            and value == 0.0
        )
        if restoring:
            # The recovery gesture (an unstaged ``mv <pseudo> 0``) after a
            # partial restore: the components disagree by construction (one
            # is back, one is not) and the move sends each to its own
            # captured baseline — safe without the agreement check, which
            # would otherwise refuse the cure.
            logger.info("%s: restoring the owed baselines", self.name)
        else:
            await self._check_agreement(fail=self._relative or self._moved)
        transform = await self._factory.transform()
        try:
            dials = transform.derived_to_raw(value=value)
        except GeecsConfigurationError as exc:
            # A forward outside its domain (sqrt of a negative R56): the
            # request is wrong, not the config — the status fails naming it.
            raise ValueError(
                f"{self._variable_name}: no setting for {value}: {exc}"
            ) from exc
        bad = [
            self._targets[i]
            for i, k in enumerate(self._keys)
            if not math.isfinite(dials[k])
        ]
        if bad:
            raise ValueError(
                f"{self._variable_name}: the formula gives no finite setting for "
                f"{', '.join(bad)} at {value} (outside the relation's domain)"
            )
        commanded = {t: dials[k] for t, k in zip(self._targets, self._keys)}
        logger.info("%s: %s → %s", self.name, value, commanded)
        self._moved = True  # commanded, whether or not every component arrives
        await _move_all(
            [(comp, dials[k]) for comp, k in zip(self._components, self._keys)]
        )
        self.last_commanded = commanded
        if restoring:
            # Back at the baselines: nothing is owed any more.  Only the
            # unstaged recovery move clears this — a staged scan point at 0
            # still owes its restore at unstage (review of #918).
            self._restore_pending = False


async def _move_all(moves: Sequence[tuple[CaSettable, float]]) -> None:
    """Set every component concurrently; wait for **all**, then raise the first failure.

    A bare ``gather`` raises on the first failed component while the
    others are still moving, and a restore issued on top of a move in
    progress is a second GEECS blocking set on a busy device — the
    incident class the restore exists to prevent.
    """
    results = await asyncio.gather(
        *(comp.set(target) for comp, target in moves), return_exceptions=True
    )
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise failures[0]


# ------------------------------------------------------------------ build


def component_key(target: str) -> str:
    """The transform's keyword for a ``"Device:Variable"`` target (its :func:`safe_name`)."""
    return safe_name(target)


def inverse_symbols(targets: Sequence[str]) -> dict[str, str]:
    """The names a catalog ``inverse`` may use for each target → its transform key.

    Two spellings per component: the device name alone when it is unique
    among the entry's targets (``U_ChicaneInner``), and always the full
    target with every non-identifier character replaced by ``_``
    (``U_ESP302_02_Position_Axis_3``).
    """
    symbols: dict[str, str] = {}
    devices = [t.partition(":")[0] for t in targets]
    for target, device in zip(targets, devices):
        key = component_key(target)
        symbols["".join(c if c.isalnum() or c == "_" else "_" for c in target)] = key
        if devices.count(device) == 1:
            symbols[device] = key
    return symbols


def build_pseudo(
    variable_name: str,
    spec: Any,
    resolve: Callable[[str], CaSettable],
    *,
    tolerance: Callable[[str], float | None] | None = None,
    name: str | None = None,
) -> CaPseudoPositioner:
    """A :class:`CaPseudoPositioner` from a catalog entry — every formula checked here.

    Parameters
    ----------
    variable_name :
        The catalog key (the friendly name; also the column header).
    spec :
        The :class:`~geecs_schemas.scan_variables.PseudoScanVariable`.
    resolve :
        ``"Device:Variable"`` → the namespace's Movable for it.
    tolerance :
        ``"Device:Variable"`` → its DB tolerance (``None``/``0`` → the
        module default).
    name :
        ophyd device name; defaults to ``safe_name(variable_name)``.

    Raises
    ------
    GeecsConfigurationError
        A formula that does not compile; a relative entry whose ``forward``
        is not ``0`` at ``0`` (``set(0)`` would not restore); an entry that
        is not affine and gives no ``inverse``; an ``inverse`` that
        compiles but does not invert the ``forward`` formulas at a probe
        value; a target the namespace has no Movable for.
    """
    targets = [str(c.target) for c in spec.targets]
    relative = str(getattr(spec.mode, "value", spec.mode)) == "relative"
    forwards: list[CompiledForward] = [
        compile_forward(str(c.forward)) for c in spec.targets
    ]
    coefficients = [affine_coefficients(f.source) for f in forwards]
    if relative:
        not_zero = [t for t, f in zip(targets, forwards) if f(0.0) != 0.0]
        if not_zero:
            raise GeecsConfigurationError(
                f"pseudo {variable_name!r} is relative but its forward for "
                f"{', '.join(not_zero)} is not 0 at 0 — set(0) would not restore "
                "the baselines; a relative formula is a*x"
            )
    keys = [component_key(t) for t in targets]
    inverse: Callable[[Mapping[str, float]], float]
    if spec.inverse is not None:
        symbols = inverse_symbols(targets)
        compiled: CompiledInverse = compile_inverse(str(spec.inverse), set(symbols))

        def inverse(user: Mapping[str, float]) -> float:
            return compiled({sym: user[key] for sym, key in symbols.items()})

        _check_inverse(variable_name, keys, forwards, inverse)
    elif all(c is not None for c in coefficients):
        inverse = affine_inverse(keys, coefficients)  # type: ignore[arg-type]
    else:
        non_affine = [t for t, c in zip(targets, coefficients) if c is None]
        raise GeecsConfigurationError(
            f"pseudo {variable_name!r}: the forward for {', '.join(non_affine)} is "
            "not affine in the scanned value and the entry gives no 'inverse' — "
            "add one (the scanned value as an expression of the components, e.g. "
            "'560968.636 * U_ChicaneInner**2 / 100**2')"
        )
    components: list[tuple[str, CaSettable]] = []
    for target in targets:
        comp = resolve(target)
        if not isinstance(comp, CaSettable):
            raise GeecsConfigurationError(
                f"pseudo {variable_name!r}: {target} is not a settable numeric "
                f"variable (got {type(comp).__name__})"
            )
        components.append((target, comp))
    tolerances = []
    for target in targets:
        db = tolerance(target) if tolerance is not None else None
        tolerances.append(
            float(db)
            if db is not None and float(db) > 0
            else DEFAULT_AGREEMENT_TOLERANCE
        )
    return CaPseudoPositioner(
        components,
        forwards,
        inverse,
        relative=relative,
        tolerances=tolerances,
        variable_name=variable_name,
        name=name or safe_name(variable_name),
    )


def _check_inverse(
    variable_name: str,
    keys: Sequence[str],
    forwards: Sequence[Callable[[float], float]],
    inverse: Callable[[Mapping[str, float]], float],
) -> None:
    """A supplied ``inverse`` must undo ``forward`` at a probe value, or the entry is refused."""
    checked = 0
    for probe in (1.0, 2.5, -1.0):
        try:
            user = {k: f(probe) for k, f in zip(keys, forwards)}
            back = inverse(user)
        except GeecsConfigurationError:
            continue  # outside the relation's domain at this probe
        checked += 1
        if not math.isclose(back, probe, rel_tol=1e-9, abs_tol=1e-9):
            raise GeecsConfigurationError(
                f"pseudo {variable_name!r}: 'inverse' does not undo 'forward' — at "
                f"{probe} the forwards give {user} and the inverse returns {back}"
            )
    if not checked:
        logger.warning(
            "pseudo %r: 'inverse' could not be checked against 'forward' (the "
            "relation's domain excludes every probe value); it is taken on trust",
            variable_name,
        )


__all__ = [
    "DEFAULT_AGREEMENT_TOLERANCE",
    "CaPseudoPositioner",
    "PseudoTransform",
    "affine_inverse",
    "build_pseudo",
    "component_key",
    "inverse_symbols",
    "transform_class",
]
