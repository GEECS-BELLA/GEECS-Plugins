"""The claim preprocessor, the run path provider and the headers preprocessor (PR 2, #807)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("aioca")

import bluesky.plan_stubs as bps  # noqa: E402
import bluesky.plans as bp  # noqa: E402
import bluesky.preprocessors as bpp  # noqa: E402
from bluesky import RunEngine  # noqa: E402
from functools import partial  # noqa: E402

from geecs_bluesky.devices.ca import CaSnapshotReadable  # noqa: E402
from geecs_bluesky.exceptions import GeecsConfigurationError  # noqa: E402
from geecs_bluesky.plans.claim_scan import (  # noqa: E402
    GeecsScanPathProvider,
    claim_scan_preprocessor,
)
from geecs_bluesky.preprocessors import scalar_headers  # noqa: E402
from tests.ca_mock_helpers import DocCollector, connect_mock  # noqa: E402


class FakeClaim:
    """Counts claims and hands out ScanNNN folders under a tmp day."""

    def __init__(self, day: Path, fail: bool = False) -> None:
        self.day = day
        self.fail = fail
        self.claimed: list[int] = []

    def __call__(self, experiment: str):
        if self.fail:
            return None, None
        number = len(self.claimed) + 1
        self.claimed.append(number)
        folder = self.day / "scans" / f"Scan{number:03d}"
        folder.mkdir(parents=True)
        tag = SimpleNamespace(year=2026, month=9, day=10, number=number)
        return tag, str(folder)


@pytest.fixture
def RE() -> RunEngine:
    return RunEngine()


def _device(RE: RunEngine, name: str = "U_Gauge") -> CaSnapshotReadable:
    dev = CaSnapshotReadable(
        name, ["Pressure"], experiment="TestExp", name=name.lower()
    )
    connect_mock(RE, dev)
    return dev


# ------------------------------------------------------------ path provider
def test_provider_refuses_outside_a_run_and_names_the_device_dir(tmp_path):
    provider = GeecsScanPathProvider()
    with pytest.raises(GeecsConfigurationError, match="no scan claimed"):
        provider("UC_Cam")
    provider.point_at(tmp_path / "Scan001")
    info = provider("UC_Cam")
    assert Path(info.directory_path) == tmp_path / "Scan001" / "UC_Cam"
    assert info.filename == "UC_Cam"
    with pytest.raises(ValueError):
        provider("")
    provider.point_at(None)
    assert provider.folder is None


# -------------------------------------------------------------- the claim
def test_every_run_claims_and_the_provider_follows(RE, tmp_path):
    claim = FakeClaim(tmp_path)
    provider = GeecsScanPathProvider()
    RE.preprocessors.append(
        partial(
            claim_scan_preprocessor,
            experiment="TestExp",
            path_provider=provider,
            claim=claim,
        )
    )
    dev = _device(RE)
    col = DocCollector()
    RE.subscribe(col)
    seen: list[Path | None] = []

    def two_runs():
        yield from bp.count([dev], 1)
        seen.append(provider.folder)
        yield from bp.count([dev], 1)

    def peek():
        seen.append(provider.folder)
        return
        yield

    RE(bpp.finalize_wrapper(two_runs(), peek()))
    assert claim.claimed == [1, 2]
    starts = col.docs["start"]
    assert [s["scan_number"] for s in starts] == [1, 2]
    assert [s["scan_id"] for s in starts] == [1, 2]
    assert starts[0]["scan_folder"] == str(tmp_path / "scans" / "Scan001")
    assert starts[0]["experiment"] == "TestExp"
    assert starts[1]["scan_tag"] == {
        "year": 2026,
        "month": 9,
        "day": 10,
        "number": 2,
        "experiment": "TestExp",
    }
    # released at close_run, both after a run and at the very end
    assert seen == [None, None]


def test_provider_points_at_the_run_while_it_is_open(RE, tmp_path):
    claim = FakeClaim(tmp_path)
    provider = GeecsScanPathProvider()
    RE.preprocessors.append(
        partial(
            claim_scan_preprocessor,
            experiment="TestExp",
            path_provider=provider,
            claim=claim,
        )
    )
    inside: list[Path | None] = []

    @bpp.run_decorator()
    def look():
        inside.append(provider.folder)
        yield from bps.null()

    RE(look())
    assert inside == [tmp_path / "scans" / "Scan001"]
    assert provider.folder is None


def test_failed_claim_refuses_the_run_before_it_opens(RE, tmp_path):
    claim = FakeClaim(tmp_path, fail=True)
    RE.preprocessors.append(
        partial(claim_scan_preprocessor, experiment="TestExp", claim=claim)
    )
    dev = _device(RE)
    col = DocCollector()
    RE.subscribe(col)
    with pytest.raises(GeecsConfigurationError, match="could not claim"):
        RE(bp.count([dev], 1))
    assert col.docs["start"] == []


def test_plan_metadata_survives_the_claim(RE, tmp_path):
    claim = FakeClaim(tmp_path)
    RE.preprocessors.append(
        partial(claim_scan_preprocessor, experiment="TestExp", claim=claim)
    )
    dev = _device(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([dev], 1, md={"description": "gauge check", "geecs": {"preset": "p"}}))
    start = col.docs["start"][0]
    assert start["description"] == "gauge check"
    assert start["geecs"] == {"preset": "p"}
    assert start["plan_name"] == "count" and start["scan_number"] == 1


# ------------------------------------------------------------ the headers
def test_staged_devices_headers_land_in_the_start_document(RE):
    RE.preprocessors.append(scalar_headers)
    gauge = _device(RE, "U_Gauge")
    other = _device(RE, "U_Other")
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([gauge], 1))
    RE(bp.count([other], 1))
    first, second = col.docs["start"]
    assert first["geecs_scalar_headers"] == {"u_gauge-pressure": "U_Gauge Pressure"}
    assert second["geecs_scalar_headers"] == {"u_other-pressure": "U_Other Pressure"}


def test_a_header_map_the_plan_carries_is_kept(RE):
    RE.preprocessors.append(scalar_headers)
    gauge = _device(RE)
    col = DocCollector()
    RE.subscribe(col)
    RE(bp.count([gauge], 1, md={"geecs_scalar_headers": {"x": "X"}}))
    assert col.docs["start"][0]["geecs_scalar_headers"] == {"x": "X"}
