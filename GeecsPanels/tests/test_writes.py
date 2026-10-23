"""The panel's writes: the allow-list, the coordinate conversion, the routes."""

from __future__ import annotations

import pytest

from geecs_panels.camera import WRITABLE, NotWritable, check_write, monitored_suffixes
from geecs_panels.camera import counter_suffix
from geecs_panels.coords import Axis, to_image, to_roi
from geecs_panels.feed import WriteFailed
from tests.test_app import CAM, FakeCameras, _client


# ---- the allow-list -----------------------------------------------------------


@pytest.mark.parametrize(
    "suffix",
    [
        "cam1:AcquireTime",
        "cam1:Acquire",
        "HDF1:Capture",
        "OVER1:3:CenterX",
        "ROI1:BinX",
        "TRANS1:Type",
    ],
)
def test_nothing_outside_the_allow_list_is_writable(suffix: str) -> None:
    """Above all, nothing that changes what a scan saves (the driver, HDF1)."""
    with pytest.raises(NotWritable, match="not writable"):
        check_write(suffix, 1)


def test_no_writable_pv_touches_scan_data() -> None:
    assert all(s.split(":")[0] in {"OVER1", "ROI1"} for s in WRITABLE)


@pytest.mark.parametrize(
    ("suffix", "value", "expected"),
    [
        ("OVER1:1:CenterX", 640, 640),
        ("OVER1:1:CenterX", 640.0, 640),
        ("OVER1:2:Use", True, 1),
        ("OVER1:2:Use", 0, 0),
        ("OVER1:1:Shape", "Cross", "Cross"),
        ("OVER1:1:Name", "ref1", "ref1"),
        ("ROI1:SizeX", 512, 512),
    ],
)
def test_allowed_values_are_coerced_to_the_pv_type(suffix, value, expected) -> None:
    assert check_write(suffix, value) == expected


@pytest.mark.parametrize(
    ("suffix", "value"),
    [
        ("OVER1:1:CenterX", 640.5),
        ("OVER1:1:CenterX", True),
        ("OVER1:1:CenterX", "640"),
        ("OVER1:1:Use", 2),
        ("OVER1:1:Shape", "Ellipse"),
        ("OVER1:1:Name", "x" * 40),
        ("ROI1:SizeX", 0),  # a cleared field must not collapse the ROI
        ("ROI1:MinX", -1),
        ("OVER1:1:CenterX", -5),
        ("OVER1:1:Red", 256),
        ("OVER1:1:CenterX", float("nan")),
    ],
)
def test_wrong_values_are_refused(suffix, value) -> None:
    with pytest.raises(NotWritable):
        check_write(suffix, value)


def test_the_crosshairs_and_roi_are_monitored() -> None:
    suffixes = monitored_suffixes()
    assert "OVER1:2:CenterX_RBV" in suffixes and "ROI1:ReverseX_RBV" in suffixes


# ---- coordinates --------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "roi", "image"),
    [
        (Axis(0, 1280), 640.0, 640.0),
        (Axis(100, 200), 0.0, 100.0),
        (Axis(100, 200, bin=2), 0.0, 100.5),  # bin 2: output 0 covers 100..101
        (
            Axis(100, 200, reverse=True),
            0.0,
            299.0,
        ),  # reversed: output 0 is the far edge
        (Axis(100, 200, bin=4, reverse=True), 1.0, 293.5),  # output 1 covers 295..292
        # A size that is not whole bins: NDROI drops the remainder, so the
        # reversed far edge is min + 100 - 1, not min + 101 - 1.
        (Axis(100, 101, bin=2, reverse=True), 0.0, 198.5),
    ],
)
def test_roi_positions_map_to_image_pixels(axis, roi, image) -> None:
    assert to_image(axis, roi) == pytest.approx(image)
    assert to_roi(axis, image) == pytest.approx(roi)


# ---- routes -------------------------------------------------------------------


class RecordingCameras(FakeCameras):
    def __init__(self, fail: bool = False) -> None:
        super().__init__()
        self.puts: list[tuple[str, object]] = []
        self.fail = fail

    async def put(self, device, suffix, value) -> None:
        if self.fail:
            raise WriteFailed(f"{device.prefix}{suffix}: put timed out")
        self.puts.append((suffix, value))


def test_an_allowed_write_is_put_under_the_cameras_prefix() -> None:
    cameras = RecordingCameras()
    r = _client(cameras=cameras).post(
        "/camera/ADTEST_CAM1/put", json={"pv": "OVER1:1:CenterX", "value": 700}
    )
    assert r.status_code == 200
    assert cameras.puts == [("OVER1:1:CenterX", 700)]


def test_a_refused_write_is_a_422_and_puts_nothing() -> None:
    cameras = RecordingCameras()
    r = _client(cameras=cameras).post(
        "/camera/ADTEST_CAM1/put", json={"pv": "cam1:AcquireTime", "value": 1}
    )
    assert r.status_code == 422 and "not writable" in r.json()["detail"]
    assert cameras.puts == []


def test_a_read_only_panel_refuses_every_write() -> None:
    cameras = RecordingCameras()
    client = _client(cameras=cameras, writable=False)
    r = client.post(
        "/camera/ADTEST_CAM1/put", json={"pv": "OVER1:1:CenterX", "value": 1}
    )
    assert r.status_code == 403
    assert client.post("/camera/ADTEST_CAM1/crosshair/1/to-centroid").status_code == 403
    assert cameras.puts == []
    assert 'data-writable="false"' in client.get("/camera/ADTEST_CAM1").text


def test_an_ioc_that_refuses_is_a_502_saying_why() -> None:
    r = _client(cameras=RecordingCameras(fail=True)).post(
        "/camera/ADTEST_CAM1/put", json={"pv": "OVER1:1:CenterX", "value": 1}
    )
    assert r.status_code == 502 and "timed out" in r.json()["detail"]


def _with_centroid(cameras: RecordingCameras, *, fresh: bool = True) -> None:
    feed = cameras.feed(CAM)
    for suffix, value in {
        "STAT:ComputeCentroid": 1,
        "STAT:CentroidX_RBV": 10.0,
        "STAT:CentroidY_RBV": 20.0,
        "ROI1:MinX_RBV": 100,
        "ROI1:MinY_RBV": 200,
        "ROI1:SizeX_RBV": 400,
        "ROI1:SizeY_RBV": 300,
        "ROI1:BinX_RBV": 2,
        "ROI1:BinY_RBV": 1,
        "ROI1:ReverseX_RBV": 0,
        "ROI1:ReverseY_RBV": 0,
    }.items():
        feed.update(suffix, value)
    feed.update(counter_suffix("STAT"), 1)
    if fresh:
        feed.update(counter_suffix("STAT"), 2)


def test_to_centroid_puts_the_centroid_in_image_pixels() -> None:
    """ROI (100, 200), bin x2: centroid (10, 20) is image (120.5 → 121, half up; 220)."""
    cameras = RecordingCameras()
    _with_centroid(cameras)
    r = _client(cameras=cameras).post("/camera/ADTEST_CAM1/crosshair/2/to-centroid")
    assert r.status_code == 200, r.text
    assert cameras.puts == [("OVER1:2:CenterX", 121), ("OVER1:2:CenterY", 220)]


def test_to_centroid_refuses_a_stale_centroid() -> None:
    cameras = RecordingCameras()
    _with_centroid(cameras, fresh=False)
    r = _client(cameras=cameras).post("/camera/ADTEST_CAM1/crosshair/1/to-centroid")
    assert r.status_code == 409 and cameras.puts == []


def test_there_are_two_crosshairs() -> None:
    assert (
        _client().post("/camera/ADTEST_CAM1/crosshair/3/to-centroid").status_code == 404
    )


def test_a_colour_is_written_as_its_three_channels() -> None:
    cameras = RecordingCameras()
    r = _client(cameras=cameras).post(
        "/camera/ADTEST_CAM1/crosshair/2/colour", json={"r": 255, "g": 128, "b": 0}
    )
    assert r.status_code == 200
    assert cameras.puts == [
        ("OVER1:2:Red", 255),
        ("OVER1:2:Green", 128),
        ("OVER1:2:Blue", 0),
    ]


def test_an_out_of_range_colour_writes_nothing() -> None:
    cameras = RecordingCameras()
    r = _client(cameras=cameras).post(
        "/camera/ADTEST_CAM1/crosshair/1/colour", json={"r": 300, "g": 0, "b": 0}
    )
    assert r.status_code == 422 and cameras.puts == []
