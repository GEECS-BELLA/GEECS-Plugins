"""GeecsSession — headless GEECS scans on the standard access layer.

The run *discipline* of a GUI scan — scan numbering, ScanInfo, save-path
layout, event schema v1, Tiled writes, s-file export, shot-control bracketing —
without the GUI: a session owns a RunEngine in the calling thread and composes
the CA device family (gateway PVs), the existing plans, and
:func:`~geecs_bluesky.plans.run_wrapper.geecs_run_wrapper`.

The session is **CA-only by design**: the gateway is the standard access layer
(see ``Planning/geecs_session/00_overview.md``).  Requires the ``ca`` extra and
a running GeecsCAGateway; set ``EPICS_CA_ADDR_LIST`` before constructing.

Example::

    from geecs_bluesky.session import GeecsSession

    s = GeecsSession("Undulator")
    cam = s.detector("UC_Amp2_IR_input", ["centroidx"], save_images=True)
    jet = s.motor("U_ESP_JetXYZ", "Position.Axis 1")
    s.shot_control("HTU-LaserOFF")

    s.scan(detectors=[cam], motor=jet, start=4.0, end=5.0, step=0.5,
           shots_per_step=3)
    s.noscan(detectors=[cam], shots=10, mode="strict")
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from bluesky import RunEngine

from geecs_bluesky.data_paths import asset_resource_root_paths, device_server_save_path
from geecs_bluesky.exceptions import GeecsConfigurationError
from geecs_bluesky.devices.ca import (
    CaActionSignalFactory,
    CaConfirmSettable,
    CaGenericDetector,
    CaMotor,
    CaSettable,
    CaSnapshotReadable,
    CaTelemetryReadable,
    CaTimestampedReadable,
)
from geecs_bluesky.models.shot_control import ShotControlConfig, ShotControlWrites
from geecs_bluesky.optimize import BinData, Suggester
from geecs_bluesky.plans.optimize import geecs_adaptive_scan
from geecs_bluesky.plans.orchestration import build_step_scan_plan
from geecs_bluesky.plans.run_wrapper import claim_scan_number, geecs_run_wrapper
from geecs_bluesky.plans.step_scan import normalize_motors
from geecs_bluesky.scan_log import scan_log
from geecs_bluesky.scanner_configs import load_shot_control_config
from geecs_bluesky.shot_controller import ShotController
from geecs_bluesky.tiled_integration import subscribe_tiled
from geecs_bluesky.utils import safe_name

logger = logging.getLogger(__name__)

_FREE_RUN = "free_run"
_STRICT = "strict"


def _positions(start: float, end: float, step: float) -> list[float]:
    """Convert start/end/step to an explicit list of positions."""
    if step == 0 or start == end:
        return [start]
    n = max(2, round(abs(end - start) / abs(step)) + 1)
    return list(np.linspace(start, end, n))


def _json_safe(value: Any) -> Any:  # Any: mirrors json.dumps's input surface
    """Recursively replace non-finite floats (NaN/inf) with ``None``.

    ``json.dumps`` would otherwise emit bare ``NaN`` tokens — invalid JSON
    that strict parsers (``jq``, ``json.loads`` consumers) reject.  Failed
    objective evaluations are recorded as NaN, so optimization histories
    must pass through here before serialization.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class GeecsSession:
    """Headless scan session for one experiment, over the CA gateway.

    Parameters
    ----------
    experiment : str
        GEECS experiment name — both the PV namespace prefix and the data-tree
        experiment folder (e.g. ``"Undulator"``).
    rep_rate_hz : float
        Machine repetition rate; sizes shot-id derivation and quiescence waits.
    tiled : bool
        Subscribe a TiledWriter (default true); the catalog location comes
        from ``tiled_uri``/``tiled_api_key`` or, when omitted, the standard
        config file.  When the server is unreachable (bounded TCP pre-check,
        e.g. off the lab network), construction proceeds promptly with Tiled
        persistence disabled for the session (warning logged).
    mock : bool
        Connect devices with ophyd-async mock backends (hermetic tests only).
    """

    def __init__(
        self,
        experiment: str,
        *,
        rep_rate_hz: float = 1.0,
        tiled: bool = True,
        tiled_uri: str | None = None,
        tiled_api_key: str | None = None,
        mock: bool = False,
    ) -> None:
        self.experiment = experiment
        self.rep_rate_hz = rep_rate_hz
        self._mock = mock
        # context_managers=[] so sessions also work off the main thread.
        self.RE = RunEngine(context_managers=[])
        self._last_run_uid: str | None = None
        self.RE.subscribe(self._remember_uid, name="start")
        if tiled:
            subscribe_tiled(self.RE, tiled_uri, tiled_api_key)
        self._shot_controller: ShotController | None = None

    def _remember_uid(self, _name: str, doc: dict) -> None:
        self._last_run_uid = doc.get("uid")

    # ------------------------------------------------------------------
    # Device factories (constructed connected; explicit names)
    # ------------------------------------------------------------------

    def _connect(self, device: Any) -> Any:
        """Connect a device in the RunEngine's persistent event loop."""
        import asyncio

        future = asyncio.run_coroutine_threadsafe(
            device.connect(mock=self._mock), self.RE._loop
        )
        future.result(timeout=20.0)
        return device

    def _read_movable(self, movable: Any) -> float:
        """Current readback value of a session settable/motor."""
        import asyncio

        signal = getattr(movable, getattr(movable, "_readback_attr_name", "readback"))
        return float(
            asyncio.run_coroutine_threadsafe(signal.get_value(), self.RE._loop).result(
                timeout=10.0
            )
        )

    def _move_movables(self, variables: dict[str, Any], targets: dict) -> None:
        """Drive movables to *targets* outside a plan (best-effort).

        Goes through each movable's own ``set()`` (the Bluesky ``Movable``
        protocol), not the raw ``:SP`` signal directly — a bypass here would
        skip ``CaMotor``'s readback poll and, worse, ``CaConfirmSettable``'s
        confirming-variable poll: the exact "the limit register converged but
        nothing physically moved" failure this device exists to catch would
        go silently unconfirmed on every optimize on_finish move.
        """
        import asyncio

        for name, value in targets.items():
            movable = variables.get(name)
            if movable is None:
                continue

            async def _move(m=movable, v=float(value)) -> None:
                await m.set(v)

            try:
                asyncio.run_coroutine_threadsafe(_move(), self.RE._loop).result(
                    timeout=30.0
                )
            except Exception:
                logger.warning("could not move %s to %s", name, value, exc_info=True)

    def disconnect(self, *devices: Any) -> None:
        """Disconnect devices in the RE's event loop (best-effort)."""
        import asyncio

        for device in devices:
            try:
                asyncio.run_coroutine_threadsafe(
                    device.disconnect(), self.RE._loop
                ).result(timeout=10.0)
            except Exception:
                logger.debug("disconnect raised for %s", device, exc_info=True)

    def detector(
        self,
        device: str,
        variables: list[str],
        *,
        save_images: bool = False,
        name: str | None = None,
    ) -> CaGenericDetector:
        """Triggered detector (free-run reference / strict triggered role)."""
        det = CaGenericDetector(
            device,
            variables,
            experiment=self.experiment,
            name=name or safe_name(device),
            save_nonscalar_data=save_images,
        )
        return self._connect(det)

    def contributor(
        self,
        device: str,
        variables: list[str],
        *,
        save_images: bool = False,
        name: str | None = None,
    ) -> CaTimestampedReadable:
        """Free-run contributor (non-blocking, reference-relative labeling)."""
        det = CaTimestampedReadable(
            device,
            variables,
            experiment=self.experiment,
            name=name or safe_name(device),
            save_nonscalar_data=save_images,
        )
        return self._connect(det)

    def snapshot(
        self,
        device: str,
        variables: list[str],
        *,
        name: str | None = None,
    ) -> CaSnapshotReadable:
        """Asynchronous readback sampled once per event row."""
        return self._connect(
            CaSnapshotReadable(
                device,
                variables,
                experiment=self.experiment,
                name=name or safe_name(device),
            )
        )

    def telemetry(
        self,
        device: str,
        variables: list[str],
        *,
        name: str | None = None,
    ) -> CaTelemetryReadable | None:
        """Soft Tier-2 telemetry readable; ``None`` if the device is unreachable.

        Builds a :class:`~geecs_bluesky.devices.ca.telemetry.CaTelemetryReadable`
        (read-only, fault-tolerant read) and connects it.  A connect failure
        means the device is dead/unreachable at scan start — the background
        telemetry contract says such a device is **dropped with a log line,
        never an error** — so this returns ``None`` after warning instead of
        propagating (unlike the strict device factories, whose failures fail
        loudly).

        The telemetry readable infers each variable's dtype from its PV
        (numeric stays float, enum/string is captured as a label), so a device
        is dropped here only when it is genuinely unreachable — never for a
        variable *type* mismatch.  A single non-numeric ``get='yes'`` variable
        no longer takes the device's other columns down with it.
        """
        det = CaTelemetryReadable(
            device,
            variables,
            experiment=self.experiment,
            name=name,
            shot_rep_rate_hz=self.rep_rate_hz,
        )
        try:
            return self._connect(det)
        except Exception:
            logger.warning(
                "Dropping background-telemetry device %s: unreachable at scan "
                "start (soft tier — never aborts the scan)",
                device,
                exc_info=True,
            )
            return None

    def telemetry_batch(
        self,
        selected: dict[str, list[str]],
    ) -> list[CaTelemetryReadable]:
        """Build and connect all Tier-2 telemetry readables **concurrently**.

        The per-device :meth:`telemetry` factory connects sequentially — at
        ~87 telemetry devices that alone cost ~9 s of the operator's
        start-to-execution latency (each connect is a network round trip;
        measured live 2026-07-13).  This batch variant constructs every
        readable first, then awaits all connects in one ``asyncio.gather``
        on the RunEngine loop, so wall time is the slowest single device
        rather than the sum.

        Per-device semantics are identical to :meth:`telemetry`: a device
        unreachable at scan start is dropped with a warning, never an error
        (the soft-tier contract).

        Parameters
        ----------
        selected:
            ``{device: [variables]}`` — the telemetry selection
            (:func:`~geecs_bluesky.db_runtime.select_telemetry_variables`).

        Returns
        -------
        list of CaTelemetryReadable
            The connected readables, in *selected* order (dropped devices
            omitted).
        """
        import asyncio

        devices = [
            CaTelemetryReadable(
                device,
                variables,
                experiment=self.experiment,
                shot_rep_rate_hz=self.rep_rate_hz,
            )
            for device, variables in selected.items()
        ]

        async def _connect_all() -> list:
            return await asyncio.gather(
                *(d.connect(mock=self._mock) for d in devices),
                return_exceptions=True,
            )

        results = asyncio.run_coroutine_threadsafe(
            _connect_all(), self.RE._loop
        ).result(timeout=30.0)
        connected: list[CaTelemetryReadable] = []
        for device_obj, result in zip(devices, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Dropping background-telemetry device %s: unreachable at "
                    "scan start (soft tier — never aborts the scan)",
                    device_obj._geecs_device_name,
                    exc_info=result,
                )
            else:
                connected.append(device_obj)
        return connected

    def motor(
        self,
        device: str,
        variable: str,
        *,
        tolerance: float = 0.005,
        name: str | None = None,
    ) -> CaMotor:
        """Position-feedback motor (blocking :SP put + readback poll)."""
        return self._connect(
            CaMotor(
                device,
                variable,
                experiment=self.experiment,
                name=name or safe_name(f"{device}_{variable}"),
                tolerance=tolerance,
            )
        )

    def settable(
        self,
        device: str,
        variable: str,
        *,
        name: str | None = None,
    ) -> CaSettable:
        """Plain settable (blocking :SP put, streamed readback)."""
        return self._connect(
            CaSettable(
                device,
                variable,
                experiment=self.experiment,
                name=name or safe_name(f"{device}_{variable}"),
            )
        )

    def confirm_settable(
        self,
        device: str,
        variable: str,
        *,
        confirm_device: str,
        confirm_variable: str,
        tolerance: float = 0.05,
        timeout: float = 10.0,
        name: str | None = None,
    ) -> CaConfirmSettable:
        """Settable that confirms completion on a *different* variable.

        The topology-C case (``ScanVariable.confirm``): ``variable`` is
        written, but ``confirm_variable`` (possibly on a different device) is
        polled for the actual physical result. See
        :class:`~geecs_bluesky.devices.ca.confirm.CaConfirmSettable`.
        """
        return self._connect(
            CaConfirmSettable(
                device,
                variable,
                confirm_device=confirm_device,
                confirm_variable=confirm_variable,
                experiment=self.experiment,
                name=name or safe_name(f"{device}_{variable}"),
                tolerance=tolerance,
                timeout=timeout,
            )
        )

    def action_signal_factory(self) -> CaActionSignalFactory:
        """Build the SettableFactory for compiled action plans (CA-backed).

        Returns a fresh per-scan factory: ``get_settable`` puts wire strings
        to the variable's gateway ``:SP`` (put-completion rides the GEECS
        blocking set — the ``wait_for_execution`` semantics), ``get_readable``
        reads the streamed readback natively typed.  Signals are cached per
        ``(device, variable)`` and connected in this session's RE loop at
        creation; pre-connect everything a plan will touch **before** running
        it (see ``scan_request_runner.prefetch_action_signals``) and pass the
        factory to :meth:`disconnect` with the scan's other devices.
        """
        return CaActionSignalFactory(self.experiment, self._connect, mock=self._mock)

    # ------------------------------------------------------------------
    # Shot control
    # ------------------------------------------------------------------

    def shot_control(
        self, config: str | dict | ShotControlConfig | ShotControlWrites | None
    ) -> ShotController | None:
        """Attach shot control from a configs-repo name, dict, config, or writes.

        A configs-repo *name* (e.g. ``"HTU-LaserOFF"``) is loaded and
        validated; ``None`` or an empty/blank config (e.g. ``{}``) detaches.
        A :class:`~geecs_bluesky.models.shot_control.ShotControlWrites`
        (generalized per-state ordered write lists, possibly multi-device —
        the TriggerProfile shape) builds the ordered controller; anything
        else takes the legacy single-device
        :class:`~geecs_bluesky.models.shot_control.ShotControlConfig` path
        unchanged.  Either way the controller drives the gateway's ``:SP``
        PVs; their reachability is verified here (short CA connect) so a
        typo'd device name fails now, not ~10 s per caput mid-plan.
        """
        import asyncio

        controller: ShotController | None = None
        attached_to = ""
        if isinstance(config, ShotControlWrites):
            if config.states:
                controller = ShotController.from_writes(
                    config, experiment=self.experiment, rep_rate_hz=self.rep_rate_hz
                )
                attached_to = controller.describe_target
        else:
            if isinstance(config, str):
                config = load_shot_control_config(config, self.experiment)
            else:
                config = ShotControlConfig.from_information(config)
            if config is not None:
                controller = ShotController.over_ca(
                    config, experiment=self.experiment, rep_rate_hz=self.rep_rate_hz
                )
                attached_to = config.device
        if controller is None:
            self._shot_controller = None
            logger.info("Shot control detached")
            return None
        if not self._mock:
            asyncio.run_coroutine_threadsafe(
                controller.connect_setters(), self.RE._loop
            ).result(timeout=20.0)
        self._shot_controller = controller
        logger.info("Shot control attached: %s", attached_to)
        return self._shot_controller

    # ------------------------------------------------------------------
    # Scans
    # ------------------------------------------------------------------

    def noscan(
        self,
        *,
        detectors: Sequence[Any],
        shots: int,
        mode: str = _FREE_RUN,
        description: str = "",
        save_data: bool = True,
        md: dict | None = None,
    ) -> str | None:
        """Statistics collection: *shots* rows at fixed settings."""
        return self.scan(
            detectors=detectors,
            motor=None,
            positions=[None],
            shots_per_step=shots,
            mode=mode,
            description=description,
            save_data=save_data,
            md=md,
        )

    def scan(
        self,
        *,
        detectors: Sequence[Any],
        motor: Any | Sequence[Any] | None = None,
        start: float | None = None,
        end: float | None = None,
        step: float | None = None,
        positions: Sequence[Any] | None = None,
        shots_per_step: int = 1,
        mode: str = _FREE_RUN,
        description: str = "",
        save_data: bool = True,
        md: dict | None = None,
        scan_number: int | None = None,
        scan_folder: str | None = None,
        scan_info: dict | None = None,
        setup: Any | None = None,
        per_step: Any | None = None,
        closeout: Any | None = None,
    ) -> str | None:
        """Run one scan with the full GEECS run discipline; return the run uid.

        The first entry of *detectors* is the free-run reference (pacemaker);
        contributors made with :meth:`contributor` are anchored to it
        automatically.  ``mode="strict"`` requires attached shot control with
        an ``ARMED`` state.  ``save_data=False`` skips scan-number claiming,
        ScanInfo, native saving, and the s-file export (ad-hoc acquisition).
        A pre-claimed ``scan_number``/``scan_folder`` pair (e.g. from a caller
        that opened a per-scan log first) skips the claim; ``scan_info``
        overrides individual ScanInfo ini fields (``scan_parameter``,
        ``start``, ``end``, ``step``, ``shots``, ``background``,
        ``scan_mode``) for legacy-format fidelity.

        *motor* may be a **sequence** of movables for a multi-axis grid scan
        (outermost axis first); *positions* is then the explicit list of
        grid points as tuples aligned with the motors (one bin per grid
        point; only changed axes are re-moved).  All motors' readbacks are
        recorded in every event row, exactly like the single-motor case.

        ``setup`` / ``per_step`` / ``closeout`` are optional plan-stub
        callables (typically compiled ActionPlans): setup runs in the plan
        preamble before the first step, per_step at every step boundary
        (after the move, before the shots), closeout in a finalize wrapper
        that runs even on abort, after the trigger disarm.  See
        :func:`~geecs_bluesky.plans.orchestration.build_step_scan_plan`.
        """
        if mode not in (_FREE_RUN, _STRICT):
            raise ValueError(f"mode={mode!r} invalid; use 'free_run' or 'strict'")
        detectors = list(detectors)
        if not detectors:
            raise ValueError("scan() needs at least one detector")
        motors = normalize_motors(motor)
        if positions is None:
            if len(motors) > 1:
                raise ValueError(
                    "multi-axis scans need explicit positions (a list of "
                    "grid-point tuples, outermost axis first)"
                )
            if motors:
                if None in (start, end, step):
                    raise ValueError("motor scans need start/end/step or positions")
                positions = _positions(float(start), float(end), float(step))
            else:
                positions = [None]
        positions = list(positions)

        controller = self._shot_controller
        if mode == _STRICT:
            if controller is None:
                raise ValueError("mode='strict' requires shot_control(...) first")
            controller.require_strict_single_shot()
        if save_data and scan_number is not None and scan_folder is None:
            raise GeecsConfigurationError(
                "scan_number was given without scan_folder; pass both "
                "(pre-claimed) or neither to claim a new scan"
            )

        claimed_here = False
        if save_data:
            if scan_number is None:
                scan_number, scan_folder = claim_scan_number(self.experiment)
                claimed_here = True
            if scan_number is not None:
                self._write_scan_info(
                    scan_number,
                    scan_folder,
                    motor=motor,
                    positions=positions,
                    shots_per_step=shots_per_step,
                    description=description,
                    overrides=scan_info,
                )
        else:
            scan_number = scan_folder = None

        # Role wiring: schema-v1 shot ids + contributor anchoring.
        reference = detectors[0]
        for det in detectors:
            if hasattr(det, "configure_shot_id"):
                det.configure_shot_id(self.rep_rate_hz)
            if hasattr(det, "set_reference") and det is not reference:
                det.set_reference(reference)

        saving_detectors = self._configure_saving(detectors, scan_number, scan_folder)

        plan = build_step_scan_plan(
            strict=mode == _STRICT,
            motor=motor,
            positions=positions,
            reference=reference,
            detectors=detectors,
            shots_per_step=shots_per_step,
            controller=controller,
            experiment=self.experiment,
            scan_number=scan_number,
            scan_folder=scan_folder,
            saving_detectors=saving_detectors,
            extra_md={"description": description, **(md or {})},
            setup=setup,
            per_step=per_step,
            closeout=closeout,
        )

        self._last_run_uid = None
        # Headless scans get the same per-scan scan.log as bridge scans
        # (Gate-2 finding).  Attach only when this call claimed the number:
        # a pre-claimed number means the caller (e.g. the GUI bridge) owns
        # the scan.log handler, and a second one would duplicate every line.
        with scan_log(scan_number, scan_folder) if claimed_here else nullcontext():
            self.RE(plan)

            if save_data and self._last_run_uid and scan_number is not None:
                self._export_scalar_files(scan_number)
        return self._last_run_uid

    def optimize(
        self,
        *,
        variables: dict[str, Any],
        detectors: Sequence[Any],
        objective: "Callable[[BinData], float]",
        suggester: Suggester,
        shots_per_iteration: int = 5,
        max_iterations: int = 20,
        mode: str = _FREE_RUN,
        description: str = "",
        save_data: bool = True,
        md: dict | None = None,
        on_finish: str = "hold",
        scan_number: int | None = None,
        scan_folder: str | None = None,
    ) -> tuple[str | None, list[dict]]:
        """Run an optimization **as a scan** (iteration = bin); return (uid, history).

        One scan number, one Tiled run, the same schema/data tree as any scan:
        each suggester iteration acquires ``shots_per_iteration`` shot-matched
        rows as one bin, then *objective* is evaluated on that bin's
        :class:`~geecs_bluesky.optimize.BinData` scalar rows and fed back
        through ``suggester.observe`` before the next ``suggester.suggest``.
        Image/diagnostic-based objectives run ScanAnalysis analyzers against
        the natively saved files (by scan tag), exactly as in any scan — see
        the GUI optimization bridge.  A failed objective evaluates to NaN
        rather than aborting the run.  The per-iteration record is returned
        and, when saving, written to ``optimization.json`` in the scan folder.

        Parameters
        ----------
        variables:
            ``{name: Movable}`` — session settables/motors; readbacks are
            recorded in every event row.  Suggester inputs use these names.
        detectors:
            As in :meth:`scan` (first entry is the free-run reference).
        objective:
            ``objective(bin_data) -> float`` (higher is better by suggester
            convention; flip the sign for minimization).
        suggester:
            Anything implementing :class:`~geecs_bluesky.optimize.Suggester`
            (``RandomSuggester``, ``XoptSuggester``, or your own).
        on_finish:
            Where the variables end up: ``"hold"`` (last visited point — the
            scan convention), ``"initial"`` (restore pre-optimization values,
            also applied on abort/failure), or ``"best"`` (move to the
            highest-objective iteration; falls back to ``initial``-style
            restore if nothing finite was observed).
        scan_number, scan_folder:
            Pre-claimed scan number/folder (as in :meth:`scan`).  When omitted
            and *save_data* is true, they are claimed here.
        """
        if mode not in (_FREE_RUN, _STRICT):
            raise ValueError(f"mode={mode!r} invalid; use 'free_run' or 'strict'")
        if on_finish not in ("hold", "initial", "best"):
            raise ValueError(
                f"on_finish={on_finish!r} invalid; use 'hold', 'initial', or 'best'"
            )
        detectors = list(detectors)
        if not detectors:
            raise ValueError("optimize() needs at least one detector")
        controller = self._shot_controller
        if mode == _STRICT:
            if controller is None:
                raise ValueError("mode='strict' requires shot_control(...) first")
            controller.require_strict_single_shot()
        if save_data and scan_number is not None and scan_folder is None:
            raise GeecsConfigurationError(
                "scan_number was given without scan_folder; pass both "
                "(pre-claimed) or neither to claim a new scan"
            )

        initial_values = {
            name: self._read_movable(movable) for name, movable in variables.items()
        }

        claimed_here = False
        if save_data:
            if scan_number is None:
                scan_number, scan_folder = claim_scan_number(self.experiment)
                claimed_here = True
            if scan_number is not None:
                self._write_scan_info(
                    scan_number,
                    scan_folder,
                    motor=None,
                    positions=[None],
                    shots_per_step=shots_per_iteration,
                    description=description,
                    overrides={
                        "scan_parameter": ",".join(variables),
                        "scan_mode": "optimization",
                        "shots": shots_per_iteration,
                    },
                )

        reference = detectors[0]
        for det in detectors:
            if hasattr(det, "configure_shot_id"):
                det.configure_shot_id(self.rep_rate_hz)
        saving_detectors = self._configure_saving(detectors, scan_number, scan_folder)

        # Collect primary-stream rows in-process so the objective can be
        # evaluated between bins without a Tiled round trip.
        descriptors: dict[str, str] = {}
        rows: list[dict] = []

        def _collect(name: str, doc: dict) -> None:
            if name == "descriptor":
                descriptors[doc["uid"]] = doc.get("name", "")
            elif name == "event" and descriptors.get(doc["descriptor"]) == "primary":
                rows.append(dict(doc["data"]))

        history: list[dict] = []
        pending: dict[str, float] | None = None

        def _observe_previous(iteration: int) -> None:
            nonlocal pending
            if pending is None:
                return
            bin_rows = [r for r in rows if r.get("bin_number") == iteration]
            bin_data = BinData(iteration, bin_rows)
            try:
                value = float(objective(bin_data))
            except Exception:
                logger.warning(
                    "objective failed on iteration %d; recording NaN",
                    iteration,
                    exc_info=True,
                )
                value = float("nan")
            suggester.observe(pending, value, bin_data)
            history.append(
                {
                    "iteration": iteration,
                    "inputs": pending,
                    "objective": value,
                    "n_rows": len(bin_rows),
                }
            )
            pending = None

        def _propose(iteration: int) -> dict[str, float] | None:
            nonlocal pending
            _observe_previous(iteration - 1)
            inputs = suggester.suggest()
            if inputs is None:
                logger.info("suggester stopped at iteration %d", iteration)
                return None
            unknown = set(inputs) - set(variables)
            if unknown:
                raise KeyError(f"suggester proposed unknown variables: {unknown}")
            pending = inputs
            logger.info("iteration %d: %s", iteration, inputs)
            return inputs

        plan = geecs_adaptive_scan(
            movables=dict(variables),
            propose=_propose,
            detectors=detectors[1:] if mode == _FREE_RUN else detectors,
            reference=reference if mode == _FREE_RUN else None,
            shots_per_iteration=shots_per_iteration,
            max_iterations=max_iterations,
            fire_shot=controller.fire_shot if mode == _STRICT and controller else None,
            setup_trigger=(
                (lambda: controller.arm_single_shot(detectors))
                if mode == _STRICT and controller
                else None
            ),
            arm_trigger=controller.arm if mode == _FREE_RUN and controller else None,
            disarm_trigger=(
                controller.disarm if mode == _FREE_RUN and controller else None
            ),
            quiesce_trigger=(
                controller.quiesce if mode == _FREE_RUN and controller else None
            ),
            md={"description": description, **(md or {})},
        )
        run_plan = geecs_run_wrapper(
            plan,
            experiment=self.experiment,
            scan_number=scan_number,
            scan_folder=scan_folder,
            saving_detectors=saving_detectors,
            devices=detectors + list(variables.values()),
            extra_md={"description": description, **(md or {})},
        )
        if controller is not None:
            import bluesky.preprocessors as bpp

            run_plan = bpp.finalize_wrapper(run_plan, controller.disarm())

        # Headless optimizations get the same per-scan scan.log as bridge
        # runs (see scan()): attach only when this call claimed the number.
        with scan_log(scan_number, scan_folder) if claimed_here else nullcontext():
            token = self.RE.subscribe(_collect)
            self._last_run_uid = None
            try:
                self.RE(run_plan)
            except BaseException:
                if on_finish in ("initial", "best"):
                    self._move_movables(variables, initial_values)
                raise
            finally:
                self.RE.unsubscribe(token)
            _observe_previous(len(history) + 1)  # final bin

            if on_finish in ("initial", "best"):
                target = initial_values
                if on_finish == "best":
                    finite = [h for h in history if np.isfinite(h["objective"])]
                    if finite:
                        target = max(finite, key=lambda h: h["objective"])["inputs"]
                    else:
                        logger.warning(
                            "on_finish='best' but no finite objectives; restoring initial"
                        )
                self._move_movables(variables, target)
                logger.info("optimize on_finish=%s -> %s", on_finish, target)

            if save_data and scan_folder is not None and history:
                import json

                try:
                    path = Path(scan_folder) / "optimization.json"
                    # Failed objectives are NaN in-process; serialize them as
                    # null (allow_nan=False makes any regression fail loudly
                    # instead of writing invalid JSON).
                    path.write_text(
                        json.dumps(_json_safe(history), indent=2, allow_nan=False)
                    )
                    logger.info("Optimization history written to %s", path)
                except Exception:
                    logger.warning(
                        "Could not write optimization history", exc_info=True
                    )
            if save_data and self._last_run_uid and scan_number is not None:
                self._export_scalar_files(scan_number)
            return self._last_run_uid, history

    def run(
        self,
        request: Any,  # geecs_schemas.ScanRequest (imported lazily below)
        resolver: Any | None = None,  # scan_request_runner.ConfigResolver
        *,
        objective: Any | None = None,
        suggester: Any | None = None,
    ) -> str | None:
        """Run one :class:`~geecs_schemas.scan_request.ScanRequest`; return the uid.

        The schema submission front door: names resolve through *resolver*
        (default: :class:`~geecs_bluesky.config_resolver.ConfigsRepoResolver`
        over this session's experiment) and the request is mapped onto
        :meth:`scan` / :meth:`optimize` by
        :func:`~geecs_bluesky.scan_request_runner.run_scan_request` (see its
        docstring for the documented v1 gaps).

        Parameters
        ----------
        request :
            The scan request to run.
        resolver :
            Optional name resolver (defaults to the configs-repo resolver).
        objective, suggester :
            Ready-made optimization callables, required for ``optimize``
            mode (the evaluator/generator specs are instantiated by the
            caller's stack, not here).

        Returns
        -------
        str or None
            The Bluesky run uid (``None`` when nothing was persisted).
        """
        from geecs_bluesky.scan_request_runner import (
            ConfigsRepoResolver,
            run_scan_request,
        )

        if resolver is None:
            resolver = ConfigsRepoResolver(self.experiment)
        return run_scan_request(
            self, request, resolver, objective=objective, suggester=suggester
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _configure_saving(
        self,
        detectors: Sequence[Any],
        scan_number: int | None,
        scan_folder: str | None,
    ) -> list[tuple]:
        """Wire save paths + external asset docs for image-saving detectors."""
        saving: list[tuple] = []
        if scan_folder is None:
            return saving
        for det in detectors:
            if not getattr(det, "_save_nonscalar_data", False):
                continue
            device_name = getattr(det, "_geecs_device_name", det.name)
            save_path = os.path.join(scan_folder, device_name)
            det.configure_nonscalar_file_logging(save_path)
            if scan_number is not None:
                self._configure_assets(det, device_name, scan_number, scan_folder)
            saving.append((det, save_path, device_server_save_path(save_path)))
        return saving

    def _configure_assets(
        self, det: Any, device_name: str, scan_number: int, scan_folder: str
    ) -> None:
        """Attach external asset definitions (best-effort; needs the DB)."""
        try:
            from geecs_bluesky.assets import get_asset_definitions
            from geecs_ca_gateway.db.geecs_db import GeecsDb

            device_type = GeecsDb.get_device_type(device_name)
            definitions = get_asset_definitions(device_type)
            if not definitions:
                return
            asset_root, asset_local_root = asset_resource_root_paths()
            det.configure_external_asset_logging(
                scan_number=scan_number,
                asset_definitions=definitions,
                root_path=asset_root or scan_folder,
                local_root_path=asset_local_root or scan_folder,
            )
        except Exception:
            logger.warning(
                "Could not configure external assets for %s",
                device_name,
                exc_info=True,
            )

    def _write_scan_info(
        self,
        scan_number: int,
        scan_folder: str,
        *,
        motor: Any | None,
        positions: Sequence[float | None],
        shots_per_step: int,
        description: str,
        overrides: dict | None = None,
    ) -> None:
        """Write ``ScanInfoScanNNN.ini`` (legacy [Scan Info] format).

        Writes only into the already-claimed ``scans/ScanNNN/`` folder — it
        never creates the scan folder (cross-package invariant).  *overrides*
        replaces individual derived fields (legacy-format fidelity for the
        GUI bridge).  A ``Scanner = "bluesky"`` key is stamped so future
        tooling can tell Bluesky-produced scans from legacy MC scans
        (metadata only — nothing may depend on it for correctness).
        """
        folder = Path(scan_folder)
        if not folder.is_dir():
            logger.warning(
                "Scan folder %s does not exist; skipping ScanInfo write", folder
            )
            return
        real = [p for p in positions if p is not None]
        if real and isinstance(real[0], (list, tuple)):
            # Multi-axis grid: the legacy 1-D ini fields (Start/End/Step)
            # describe the outermost (slowest) axis; the full grid lives in
            # the run metadata (scan_axes / grid_shape).
            real = [p[0] for p in real]
        o = overrides or {}
        motors = normalize_motors(motor)
        default_var = (
            ",".join(getattr(m, "name", str(m)) for m in motors) if motors else None
        )
        scan_var = o.get("scan_parameter") or default_var or "Shotnumber"
        start = o.get("start", real[0] if real else 0)
        end = o.get("end", real[-1] if real else 0)
        step = o.get("step", (real[1] - real[0]) if len(real) > 1 else 0)
        shots = o.get("shots", shots_per_step)
        background = str(bool(o.get("background", False))).lower()
        scan_mode = o.get("scan_mode", "standard" if real else "noscan")
        info = f"Bluesky scan. scanning {scan_var}. {description}".strip()
        lines = [
            "[Scan Info]\n",
            f"Scan No = {scan_number}\n",
            f'ScanStartInfo = "{info}"\n',
            f'Scan Parameter = "{scan_var}"\n',
            f"Start = {start}\n",
            f"End = {end}\n",
            f"Step size = {step}\n",
            f"Shots per step = {shots}\n",
            'ScanEndInfo = ""\n',
            f"Background = {background}\n",
            f'ScanMode = "{scan_mode}"\n',
            'Scanner = "bluesky"\n',
        ]
        path = folder / f"ScanInfo{folder.name}.ini"
        try:
            with path.open("w") as f:
                f.writelines(lines)
            logger.info("Scan info written to %s", path)
        except Exception:
            logger.warning("Could not write %s", path, exc_info=True)

    def _export_scalar_files(self, scan_number: int) -> None:
        """Best-effort legacy s-file export from Tiled (mirrors the scanner)."""
        try:
            from geecs_data_utils import write_scalar_files_from_tiled

            result = write_scalar_files_from_tiled(self._last_run_uid)
            logger.info("Wrote legacy scalar files: %s", result)
        except Exception:
            logger.warning(
                "Could not export legacy scalar files for scan %s (uid=%s)",
                scan_number,
                self._last_run_uid,
                exc_info=True,
            )
