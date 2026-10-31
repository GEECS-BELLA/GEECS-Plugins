"""camera_rules.js under node: each row one rule of the camera page (skipped without node)."""

from __future__ import annotations

from typing import Any

from tests._node import call_all, needs_node

VIEW = {"ok": True, "state": "live", "roles": {}}
W = {"role": "exposure", "writable": True, "locked": True}
AX = {"min": 10, "size": 20, "bin": 2, "reverse": False}
RAX = {"min": 10, "size": 20, "bin": 2, "reverse": True}
V0 = {"scale": 2, "ox": 10, "oy": 20, "moved": True}

ROWS: list[tuple[str, str, list[Any], Any]] = [
    # levels and the colour map
    (
        "levels: white kept above black",
        "levels",
        ["200", "100"],
        {"black": 200, "white": 201},
    ),
    ("levels: junk is the floor", "levels", ["x", "255"], {"black": 0, "white": 255}),
    (
        "auto levels: the 0.1 % and 99.9 % points",
        "autoLevels",
        [[0] * 10 + [500] + [0] * 89 + [500] + [0] * 155],
        {"black": 10, "white": 100},
    ),  # fmt: skip
    (
        "auto levels: one grey level opens one step",
        "autoLevels",
        [[0] * 50 + [9] + [0] * 205],
        {"black": 50, "white": 51},
    ),  # fmt: skip
    # the JPEG stretched over MJPG1's size when the shapes agree
    ("size: unread is the JPEG's", "imageSize", [640, 480, None, None], [640, 480]),
    (
        "size: a scaled JPEG takes the IOC's size",
        "imageSize",
        [1600, 130, 2448, 200],
        [2448, 200],
    ),
    (
        "size: a disagreeing shape keeps the JPEG's",
        "imageSize",
        [960, 1200, 1280, 1024],
        [960, 1200],
    ),
    # the view
    ("view: image to view", "toView", [V0, 3, 4], [16, 28]),
    ("view: view to image", "toImage", [V0, 16, 28], [3, 4]),
    (
        "fit: centred with a margin",
        "fit",
        [532, 432, 100, 100],
        {"scale": 4, "ox": 66, "oy": 16, "moved": False},
    ),  # fmt: skip
    (
        "zoom: the point under the pointer stays",
        "zoomAt",
        [{"scale": 1, "ox": 0, "oy": 0}, 50, 50, 2, 1],
        {"scale": 2, "ox": -50, "oy": -50, "moved": True},
    ),  # fmt: skip
    (
        "zoom: never below half the fit",
        "zoomAt",
        [{"scale": 1, "ox": 0, "oy": 0}, 0, 0, 0.01, 1],
        {"scale": 0.5, "ox": 0, "oy": 0, "moved": True},
    ),  # fmt: skip
    # the analysis ROI's pixels
    ("roi: output pixel to image (binned)", "roiToImage", [AX, 0], 10.5),
    ("roi: reversed counts from the far edge", "roiToImage", [RAX, 0], 28.5),
    ("roi: image to output pixel is the inverse", "imageToRoi", [AX, 14.5], 2),
    ("roi: reversed inverse", "imageToRoi", [RAX, 28.5], 0),
    (
        "drag: the east edge follows, inside the frame",
        "dragRoi",
        [{"minx": 10, "miny": 10, "sizex": 20, "sizey": 20}, "e", 999, 0, 100, 100],
        {"minx": 10, "miny": 10, "sizex": 90, "sizey": 20},
    ),  # fmt: skip
    (
        "drag: the north-west corner keeps 1 px",
        "dragRoi",
        [{"minx": 10, "miny": 10, "sizex": 20, "sizey": 20}, "nw", 40, 40, 100, 100],
        {"minx": 29, "miny": 29, "sizex": 1, "sizey": 1},
    ),  # fmt: skip
    (
        "handle: a corner",
        "roiHandle",
        [{"l": 0, "t": 0, "r": 100, "b": 100}, 2, 98, 9, 6],
        "sw",
    ),
    (
        "handle: inside is none",
        "roiHandle",
        [{"l": 0, "t": 0, "r": 100, "b": 100}, 50, 50, 9, 6],
        "",
    ),
    # values
    ("fmt: nothing is a dash", "fmt", [None, {}], "—"),
    (
        "fmt: an enum's label",
        "fmt",
        [{"value": 1}, {"enumChoices": ["Off", "On"]}],
        "On",
    ),
    ("fmt: four significant digits", "fmt", [{"value": 147.80138}, {}], "147.8"),
    ("fmt: large is an exponent", "fmt", [{"value": 8587609600.5}, {}], "8.588e+9"),
    ("fmt: an integer as is", "fmt", [{"value": 65400}, {}], "65400"),
    # which control is enabled: the module's word only
    (
        "control: the module silent",
        "control",
        [{"ok": False}, W, False],
        {"enabled": False, "why": "the module cannot say"},
    ),  # fmt: skip
    (
        "control: no such role",
        "control",
        [VIEW, None, False],
        {"enabled": False, "why": "the module names no such role for this camera"},
    ),  # fmt: skip
    (
        "control: read only",
        "control",
        [VIEW, {"writable": False}, False],
        {"enabled": False, "why": "read only (the module's policy)"},
    ),  # fmt: skip
    (
        "control: run-locked stays enabled, says why",
        "control",
        [VIEW, W, False],
        {"enabled": True, "why": "refused while a run holds the camera (the run lock)"},
    ),  # fmt: skip
    (
        "control: a write in flight",
        "control",
        [VIEW, W, True],
        {"enabled": False, "why": "a write is in flight"},
    ),  # fmt: skip
    # words
    (
        "image chip: stale",
        "imageChip",
        ["stale"],
        {"text": "image stale", "word": "stale"},
    ),
    (
        "image chip: silent is unknown",
        "imageChip",
        [None],
        {"text": "image unknown", "word": "unknown"},
    ),  # fmt: skip
    (
        "alarms: the module's error verdicts by label",
        "alarms",
        [
            {
                "ok": True,
                "roles": {
                    "a": {"role": "state", "label": "detector", "verdict": "error"},
                    "b": {"role": "rate", "verdict": "live"},
                },
            }
        ],
        ["detector"],
    ),  # fmt: skip
    ("alarms: none while the module is silent", "alarms", [{"ok": False}], []),
    # the module's hazard marks, asked once for a whole write list
    (
        "hazards: the roles the module marks, in order",
        "hazards",
        [
            [["min_x", 0], ["size_x", 64], ["size_y", 64]],
            {
                "min_x": {"writable": True},
                "size_x": {"confirm": "hazard"},
                "size_y": {"confirm": "hazard"},
            },
        ],
        ["size_x", "size_y"],
    ),  # fmt: skip
    ("hazards: an unknown role is none", "hazards", [[["gain", 2]], {}], []),
    ("valid: the waveform's length", "valid", [[1, 2, 3, 4], 2], [1, 2]),
    ("valid: unread length keeps all", "valid", [[1, 2], None], [1, 2]),
    ("whole: a typed number, rounded", "whole", [" 12.6 "], 13),
    ("whole: empty writes nothing", "whole", [""], None),
    ("number: not a number writes nothing", "number", ["abc"], None),
]


@needs_node
def test_every_row() -> None:
    got = call_all("camera_rules.js", [(fn, args) for _, fn, args, _ in ROWS])
    wrong = [(name, want, have) for (name, _, _, want), have in zip(ROWS, got) if have != want]  # fmt: skip
    assert not wrong, wrong


@needs_node
def test_the_lut_maps_through_the_levels() -> None:
    grey, viridis = call_all("camera_rules.js", [("lut", ["grey", "linear", 100, 200]), ("lut", ["viridis", "linear", 0, 255])])  # fmt: skip
    assert len(grey) == 768 and grey[:3] == [0, 0, 0] and grey[150 * 3] == 128 and grey[-3:] == [255, 255, 255]  # fmt: skip
    assert viridis[:3] == [68, 1, 84] and viridis[-3:] == [253, 231, 37]
