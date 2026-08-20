"""Headless GeecsSession scan.log attachment tests (mock CA backends).

The shared-helper tests live in ``test_scan_log.py`` (CI-safe, no CA);
these need the ``ca`` extra's mock backends to run a real session scan.
Pins the Gate-2 fix: a session-*claimed* scan writes ``scan.log`` into its
folder; a pre-claimed scan does not self-attach (the claiming caller —
e.g. the optimize-mode runner, which claims pre-bind — owns the handler).
"""

from __future__ import annotations

import pytest

pytest.importorskip("aioca")

from tests.ca_mock_helpers import start_pacer  # noqa: E402


def _mock_session():
    from geecs_bluesky.session import GeecsSession

    return GeecsSession("TestExp", tiled=False, mock=True)


def _run_noscan(session, monkeypatch, scan_folder, *, preclaimed: bool) -> None:
    """Run one 2-shot free-run noscan through session.scan into *scan_folder*."""
    import geecs_bluesky.session as session_module

    if not preclaimed:
        monkeypatch.setattr(
            session_module,
            "claim_scan_number",
            lambda experiment: (7, str(scan_folder)),
        )
    det = session.detector("U_Cam", ["Sig"])
    det_kwargs = {}
    if preclaimed:
        det_kwargs = {"scan_number": 7, "scan_folder": str(scan_folder)}
    from ophyd_async.core import set_mock_value

    set_mock_value(det.acq_timestamp, 1000.0)
    pacer = start_pacer(session.RE, [(det, 1000.0)], initial_delay=1.0, interval=0.15)
    try:
        session.scan(
            detectors=[det],
            motor=None,
            positions=[None],
            shots_per_step=2,
            mode="free_run",
            **det_kwargs,
        )
    finally:
        pacer.cancel()
        session.disconnect(det)


def test_headless_session_scan_writes_scan_log(tmp_path, monkeypatch) -> None:
    """A session-claimed scan gets a scan.log with the run's log lines."""
    scan_folder = tmp_path / "Scan007"
    scan_folder.mkdir()
    session = _mock_session()
    _run_noscan(session, monkeypatch, scan_folder, preclaimed=False)

    log_file = scan_folder / "scan.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "scan Scan007: starting" in content
    assert "scan Scan007: finished" in content
    assert "scan=Scan007" in content
    # The run's own package log lines are captured (ScanInfo write happens
    # pre-attach; the export warning is emitted inside the run block).
    assert "geecs_bluesky" in content


def test_session_does_not_attach_for_preclaimed_scans(tmp_path, monkeypatch) -> None:
    """Pre-claimed number/folder → the caller owns scan.log (no duplicate)."""
    scan_folder = tmp_path / "Scan007"
    scan_folder.mkdir()
    session = _mock_session()
    _run_noscan(session, monkeypatch, scan_folder, preclaimed=True)
    assert not (scan_folder / "scan.log").exists()
