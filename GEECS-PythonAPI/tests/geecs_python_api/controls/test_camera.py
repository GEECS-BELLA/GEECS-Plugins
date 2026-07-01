"""Unit tests for ``camera.latest_image`` (no hardware, no database).

Covers the guardrail behaviour — non-camera devices and not-yet-arrived frames
return ``None`` rather than raising — and the happy path of decoding a frame
present in ``device.state``.
"""

from __future__ import annotations

import struct

import numpy as np

from geecs_data_utils.io.images import _IMAQ_CLASS_MARKER
from geecs_python_api.controls.devices.camera import latest_image, on_image
from geecs_python_api.controls.devices.geecs_device import GeecsDevice

_NAME = b"UC_TestCam"


class _FakeDevice:
    """Minimal stand-in exposing only the ``state`` mapping ``latest_image`` reads."""

    def __init__(self, state: dict):
        self.state = state


class _FakeStreamDevice:
    """Stand-in for ``on_image``: records listeners and replays the real parser.

    Mirrors the bits of ``GeecsDevice`` that ``on_image`` touches —
    ``register_update_listener`` / ``unregister_update_listener`` and the
    ``_subscription_parser`` static method — so the push path can be exercised
    by feeding synthetic raw messages through :meth:`emit`.
    """

    _subscription_parser = staticmethod(GeecsDevice._subscription_parser)

    def __init__(self):
        self._listeners: dict = {}

    def register_update_listener(self, name, fn):
        self._listeners[name] = fn

    def unregister_update_listener(self, name):
        self._listeners.pop(name, None)

    def emit(self, message: str):
        """Simulate one TCP frame arriving (pre-parse, as the real callback fires)."""
        for fn in list(self._listeners.values()):
            fn(message)


def _uncompressed_flatten(
    width: int, height: int, border: int
) -> tuple[str, np.ndarray]:
    """Build an 8-bit uncompressed flattened-image ``str`` and its expected image."""
    rows = height + 2 * border
    cols = width + 2 * border
    img = np.arange(1, height * width + 1, dtype="<u1").reshape(height, width)
    alloc = np.zeros((rows, cols), dtype="<u1")
    alloc[border : border + height, border : border + width] = img

    sb = bytearray(64)
    struct.pack_into("<i", sb, 40, width)
    struct.pack_into("<i", sb, 48, height)
    struct.pack_into("<i", sb, 56, border)

    prefixed_name = struct.pack(">I", len(_NAME)) + _NAME
    wrapper = b"nivissvc.*" + _IMAQ_CLASS_MARKER + b"\x00" * 8 + bytes(sb)
    blob = prefixed_name + wrapper + prefixed_name + alloc.tobytes()
    # device.state holds the latin-1-decoded str, not raw bytes
    return blob.decode("latin-1"), img


def test_non_camera_device_returns_none():
    """A device with no ``image`` key (i.e. not a camera) yields None, not an error."""
    assert (
        latest_image(_FakeDevice({"shot number": 3, "Device Status": "Initialized"}))
        is None
    )


def test_no_frame_yet_returns_none():
    """Empty / absent frame values are treated as 'no image yet'."""
    assert latest_image(_FakeDevice({"image": ""})) is None
    assert latest_image(_FakeDevice({"image": None})) is None


def test_missing_state_attribute_returns_none():
    """An object without a ``state`` attribute is handled gracefully."""

    class _Bare:
        pass

    assert latest_image(_Bare()) is None


def test_decodes_frame_present_in_state():
    """A device carrying a valid flattened image decodes to the expected array."""
    blob, expected = _uncompressed_flatten(width=5, height=4, border=1)
    out = latest_image(_FakeDevice({"image": blob}))

    assert out is not None
    assert out.shape == (4, 5)
    np.testing.assert_array_equal(out, expected)


def _message_with_image(blob: str, dev: str = "UC_TestCam", shot: int = 0) -> str:
    """Wrap an image payload in a full subscription message (with a scalar too)."""
    return (
        f"{dev}>>{shot}>>Device Status nval,Initialized nvar,\r\nimage nval,{blob} nvar"
    )


def test_on_image_pushes_decoded_frames():
    """``on_image`` decodes each frame from the raw message and calls back."""
    dev = _FakeStreamDevice()
    received: list = []
    name = on_image(dev, received.append)
    assert name == "image"

    blob, expected = _uncompressed_flatten(width=5, height=4, border=1)
    dev.emit(_message_with_image(blob))

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], expected)


def test_on_image_skips_frames_without_image():
    """Frames carrying no image variable never reach the callback (guardrail)."""
    dev = _FakeStreamDevice()
    received: list = []
    on_image(dev, received.append)

    dev.emit("UC_TestCam>>0>>Device Status nval,Initialized nvar")

    assert received == []


def test_on_image_decode_false_passes_raw_payload():
    """With ``decode=False`` the callback receives the undecoded payload str."""
    dev = _FakeStreamDevice()
    received: list = []
    on_image(dev, received.append, decode=False)

    blob, _ = _uncompressed_flatten(width=5, height=4, border=1)
    dev.emit(_message_with_image(blob))

    assert received == [blob]


def test_on_image_unregister_stops_callbacks():
    """Unregistering by the returned name stops further callbacks."""
    dev = _FakeStreamDevice()
    received: list = []
    name = on_image(dev, received.append)
    dev.unregister_update_listener(name)

    blob, _ = _uncompressed_flatten(width=5, height=4, border=1)
    dev.emit(_message_with_image(blob))

    assert received == []
