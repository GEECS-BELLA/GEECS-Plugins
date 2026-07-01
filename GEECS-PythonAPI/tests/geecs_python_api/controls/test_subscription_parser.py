"""Unit tests for ``GeecsDevice._subscription_parser``.

These are pure-logic tests (no hardware, no database) that pin the subscription
message format and, in particular, the requirement that *binary* variable values
— such as IMAQ flattened camera images — survive parsing intact even when they
contain comma (``0x2C``) and ``>>`` bytes.
"""

from __future__ import annotations

from geecs_python_api.controls.devices.geecs_device import GeecsDevice


def make_msg(dev: str, shot: int, pairs: list[tuple[str, str]]) -> str:
    """Build a subscription message in the GEECS wire format."""
    body = ",\r\n".join(f"{name} nval,{value} nvar" for name, value in pairs)
    return f"{dev}>>{shot}>>{body}"


def test_parses_plain_scalars() -> None:
    msg = make_msg(
        "DevA",
        7,
        [("Device Status", "Initialized"), ("x", "1.5"), ("y", "-3")],
    )
    dev, shot, vals = GeecsDevice._subscription_parser(msg)
    assert dev == "DevA"
    assert shot == 7
    assert vals == {"Device Status": "Initialized", "x": "1.5", "y": "-3"}


def test_empty_value_preserved() -> None:
    _, _, vals = GeecsDevice._subscription_parser(
        make_msg("D", 0, [("device error", "")])
    )
    assert vals["device error"] == ""


def test_binary_value_with_commas_and_arrows_survives() -> None:
    # A value carrying the very bytes the old comma/>>-splitting parser choked on:
    # commas, a ">>" sequence, and JPEG SOI/EOI markers.
    blob = b"\x00\x00\x00\x03ABC\xff\xd8\xff,>>,\x2c\xff\xd9".decode("latin-1")
    msg = make_msg("Cam", 0, [("a", "1"), ("image", blob), ("b", "2")])
    dev, shot, vals = GeecsDevice._subscription_parser(msg)

    assert dev == "Cam"
    assert shot == 0
    # scalar variables on either side of the image are unaffected
    assert vals["a"] == "1"
    assert vals["b"] == "2"
    # the binary image round-trips byte-for-byte
    assert vals["image"].encode("latin-1") == (
        b"\x00\x00\x00\x03ABC\xff\xd8\xff,>>,\x2c\xff\xd9"
    )


def test_malformed_message_returns_empty() -> None:
    assert GeecsDevice._subscription_parser("no-delimiters-here") == ("", 0, {})
