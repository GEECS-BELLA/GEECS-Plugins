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
import os
from pathlib import Path
from typing import Any, Sequence

import bluesky.preprocessors as bpp
import numpy as np
from bluesky import RunEngine

from geecs_bluesky.data_paths import asset_resource_root_paths, device_server_save_path
from geecs_bluesky.devices.ca import (
    CaGenericDetector,
    CaMotor,
    CaSettable,
    CaSnapshotReadable,
    CaTimestampedReadable,
)
from geecs_bluesky.models.shot_control import ShotControlConfig
from geecs_bluesky.plans.free_run_step_scan import geecs_free_run_step_scan
from geecs_bluesky.plans.run_wrapper import claim_scan_number, geecs_run_wrapper
from geecs_bluesky.plans.step_scan import geecs_step_scan
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
        Subscribe a TiledWriter from the standard config (default true).
    mock : bool
        Connect devices with ophyd-async mock backends (hermetic tests only).
    """

    def __init__(
        self,
        experiment: str,
        *,
        rep_rate_hz: float = 1.0,
        tiled: bool = True,
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
            subscribe_tiled(self.RE)
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

    # ------------------------------------------------------------------
    # Shot control
    # ------------------------------------------------------------------

    def shot_control(
        self, config: str | dict | ShotControlConfig | None
    ) -> ShotController | None:
        """Attach shot control from a configs-repo name, dict, or config.

        A configs-repo *name* (e.g. ``"HTU-LaserOFF"``) is loaded and
        validated; ``None`` detaches.  The controller drives the device
        through the gateway's ``:SP`` PVs.
        """
        if config is None:
            self._shot_controller = None
            return None
        if isinstance(config, str):
            config = load_shot_control_config(config, self.experiment)
        else:
            config = ShotControlConfig.from_information(config)
        self._shot_controller = ShotController.over_ca(
            config, experiment=self.experiment, rep_rate_hz=self.rep_rate_hz
        )
        logger.info("Shot control attached: %s", config.device)
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
        motor: Any | None = None,
        start: float | None = None,
        end: float | None = None,
        step: float | None = None,
        positions: Sequence[float | None] | None = None,
        shots_per_step: int = 1,
        mode: str = _FREE_RUN,
        description: str = "",
        save_data: bool = True,
        md: dict | None = None,
    ) -> str | None:
        """Run one scan with the full GEECS run discipline; return the run uid.

        The first entry of *detectors* is the free-run reference (pacemaker);
        contributors made with :meth:`contributor` are anchored to it
        automatically.  ``mode="strict"`` requires attached shot control with
        an ``ARMED`` state.  ``save_data=False`` skips scan-number claiming,
        ScanInfo, native saving, and the s-file export (ad-hoc acquisition).
        """
        if mode not in (_FREE_RUN, _STRICT):
            raise ValueError(f"mode={mode!r} invalid; use 'free_run' or 'strict'")
        detectors = list(detectors)
        if not detectors:
            raise ValueError("scan() needs at least one detector")
        if positions is None:
            if motor is not None:
                if None in (start, end, step):
                    raise ValueError("motor scans need start/end/step or positions")
                positions = _positions(float(start), float(end), float(step))
            else:
                positions = [None]
        positions = list(positions)

        scan_number: int | None = None
        scan_folder: str | None = None
        if save_data:
            scan_number, scan_folder = claim_scan_number(self.experiment)
            if scan_number is not None:
                self._write_scan_info(
                    scan_number,
                    scan_folder,
                    motor=motor,
                    positions=positions,
                    shots_per_step=shots_per_step,
                    description=description,
                )

        # Role wiring: schema-v1 shot ids + contributor anchoring.
        reference = detectors[0]
        for det in detectors:
            if hasattr(det, "configure_shot_id"):
                det.configure_shot_id(self.rep_rate_hz)
            if hasattr(det, "set_reference") and det is not reference:
                det.set_reference(reference)

        saving_detectors = self._configure_saving(detectors, scan_number, scan_folder)

        controller = self._shot_controller
        if mode == _FREE_RUN:
            inner = geecs_free_run_step_scan(
                motor=motor,
                positions=positions,
                reference=reference,
                detectors=detectors[1:],
                shots_per_step=shots_per_step,
                arm_trigger=controller.arm if controller else None,
                disarm_trigger=controller.disarm if controller else None,
                quiesce_trigger=controller.quiesce if controller else None,
            )
        else:
            if controller is None:
                raise ValueError("mode='strict' requires shot_control(...) first")
            controller.require_strict_single_shot()
            inner = geecs_step_scan(
                motor=motor,
                positions=positions,
                detectors=detectors,
                shots_per_step=shots_per_step,
                setup_trigger=lambda: controller.arm_single_shot(detectors),
                fire_shot=controller.fire_shot,
            )

        scalar_devices = detectors + ([motor] if motor is not None else [])
        plan = geecs_run_wrapper(
            inner,
            experiment=self.experiment,
            scan_number=scan_number,
            scan_folder=scan_folder,
            saving_detectors=saving_detectors,
            devices=scalar_devices,
            extra_md={"description": description, **(md or {})},
        )
        if controller is not None:
            plan = bpp.finalize_wrapper(plan, controller.disarm())

        self._last_run_uid = None
        self.RE(plan)

        if save_data and self._last_run_uid and scan_number is not None:
            self._export_scalar_files(scan_number)
        return self._last_run_uid

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
            from geecs_bluesky.db.geecs_db import GeecsDb

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
    ) -> None:
        """Write ``ScanInfoScanNNN.ini`` (legacy [Scan Info] format).

        Writes only into the already-claimed ``scans/ScanNNN/`` folder — it
        never creates the scan folder (cross-package invariant).
        """
        folder = Path(scan_folder)
        if not folder.is_dir():
            logger.warning(
                "Scan folder %s does not exist; skipping ScanInfo write", folder
            )
            return
        scan_var = getattr(motor, "name", None) or "Shotnumber"
        real = [p for p in positions if p is not None]
        start = real[0] if real else 0
        end = real[-1] if real else 0
        step = (real[1] - real[0]) if len(real) > 1 else 0
        info = f"Bluesky session scan. scanning {scan_var}. {description}".strip()
        lines = [
            "[Scan Info]\n",
            f"Scan No = {scan_number}\n",
            f'ScanStartInfo = "{info}"\n',
            f'Scan Parameter = "{scan_var}"\n',
            f"Start = {start}\n",
            f"End = {end}\n",
            f"Step size = {step}\n",
            f"Shots per step = {shots_per_step}\n",
            'ScanEndInfo = ""\n',
            "Background = false\n",
            'ScanMode = "standard"\n' if real else 'ScanMode = "noscan"\n',
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
