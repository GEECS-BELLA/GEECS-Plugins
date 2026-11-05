"""camera_rules.js under node: each row one rule of the camera page (skipped without node)."""

from __future__ import annotations

from typing import Any

from tests._node import call_all, needs_node, run

VIEW = {"ok": True, "state": "live", "roles": {}}
W = {"role": "exposure", "writable": True, "locked": True}
AX = {"min": 10, "size": 20, "bin": 2, "reverse": False}
RAX = {"min": 10, "size": 20, "bin": 2, "reverse": True}
V0 = {"scale": 2, "ox": 10, "oy": 20, "moved": True}
RAW_HINT = (
    "The camera's 16-bit frame, levels applied in this browser; nothing is written to the camera."
    " Raw is the driver's frame: no background subtracted, rotate / flip applied here."
)

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
    # levels at the raw frame's full scale (X-Frame-Full-Scale)
    (
        "levels: counts of a 12-bit full scale",
        "levels",
        ["56", "4000", 4095],
        {"black": 56, "white": 4000},
    ),
    (
        "levels: past the full scale is clamped",
        "levels",
        [5000, 9999, 4095],
        {"black": 4094, "white": 4095},
    ),
    (
        "histogram: full + 1 bins, above the scale in the top bin",
        "histogram",
        [[0, 1, 1, 3, 9], 3],
        [1, 2, 0, 2],
    ),
    (
        "auto levels at 4095: a lone hot pixel is past the 99.9 % point",
        "autoLevels",
        [[0] * 53 + [1428] * 7 + [0] * 3940 + [1] + [0] * 95, 4095],
        {"black": 53, "white": 59},
    ),
    (
        "auto levels at 4095: the two ends",
        "autoLevels",
        [[0] * 56 + [500] + [0] * 3943 + [500] + [0] * 95, 4095],
        {"black": 56, "white": 4000},
    ),
    # levels per mode, stored as fractions of the mode's full scale
    (
        "store: fractions of the mode's scale",
        "storeLevels",
        [{}, "raw", 56, 4095, 4095],
        {"raw": {"black": 56 / 4095, "white": 1}},
    ),
    (
        "store: the other mode's kept",
        "storeLevels",
        [{"levels": {"mjpeg": {"black": 0, "white": 1}}}, "raw", 0, 4095, 4095],
        {"mjpeg": {"black": 0, "white": 1}, "raw": {"black": 0, "white": 1}},
    ),
    (
        "stored: an 8-bit white of 255 is the raw full scale, not 255",
        "storedLevels",
        [{"levels": {"raw": {"black": 0, "white": 1}}}, "raw", 4095],
        {"black": 0, "white": 4095},
    ),
    (
        "stored: a 12-bit black of 56 read at 255",
        "storedLevels",
        [
            {"levels": {"mjpeg": {"black": 56 / 4095, "white": 4000 / 4095}}},
            "mjpeg",
            255,
        ],
        {"black": 3, "white": 249},
    ),
    (
        "stored: nothing for the mode is 0 to full scale",
        "storedLevels",
        [{"levels": {"mjpeg": {"black": 0.5, "white": 1}}}, "raw", 4095],
        {"black": 0, "white": 4095},
    ),
    (
        "stored: the JPEG's levels from before modes",
        "storedLevels",
        [{"black": "10", "white": "200"}, "mjpeg", 255],
        {"black": 10, "white": 200},
    ),
    (
        "stored: the JPEG's old levels never reach raw",
        "storedLevels",
        [{"black": "10", "white": "200"}, "raw", 4095],
        {"black": 0, "white": 4095},
    ),
    # the raw poll's answers: a 304 is nothing new, never an error
    ("raw answer: 200 is a frame", "rawAnswer", [200], "frame"),
    ("raw answer: 304 is nothing new", "rawAnswer", [304], "same"),
    ("raw answer: 503 is the error", "rawAnswer", [503], "error"),
    ("raw answer: 404 is the error", "rawAnswer", [404], "error"),
    # the transform role's value → TRANS1's mapping, on a 3×2 frame [[0, 1, 2], [3, 4, 5]]
    *[
        (
            f"transform: {name}",
            "transformFrame",
            [name, [0, 1, 2, 3, 4, 5], 3, 2],
            {"data": data, "width": w, "height": h},
        )
        for name, data, w, h in [
            ("None", [0, 1, 2, 3, 4, 5], 3, 2),
            ("Rot90", [3, 0, 4, 1, 5, 2], 2, 3),
            ("Rot180", [5, 4, 3, 2, 1, 0], 3, 2),
            ("Rot270", [2, 5, 1, 4, 0, 3], 2, 3),
            ("Mirror", [2, 1, 0, 5, 4, 3], 3, 2),
            ("Rot90Mirror", [0, 3, 1, 4, 2, 5], 2, 3),
            ("Rot180Mirror", [3, 4, 5, 0, 1, 2], 3, 2),
            ("Rot270Mirror", [5, 2, 4, 1, 3, 0], 2, 3),
        ]
    ],
    (
        "transform: an unknown name is none",
        "transformFrame",
        ["Rot45", [0, 1], 2, 1],
        None,
    ),
    (
        "transform: a prototype name is none",
        "transformFrame",
        ["toString", [0, 1], 2, 1],
        None,
    ),
    (
        "transform name: the enum's label",
        "transformName",
        [{"value": 1}, {"enumChoices": ["None", "Rot90"]}],
        "Rot90",
    ),
    ("transform name: unread is null", "transformName", [None, {}], None),
    (
        "transform name: an index without labels is null",
        "transformName",
        [{"value": 1}, {}],
        None,
    ),
    # the toggle's words
    (
        "mode words: MJPEG",
        "modeWords",
        [False],
        {
            "note": "",
            "hint": "The 8-bit stream, stretched in this browser: nothing is written to the camera.",
            "value": "level",
            "counterChip": True,
            "overlays": True,
        },
    ),
    (
        "mode words: raw, untransformed, full scale read",
        "modeWords",
        [True, {"transform": "None", "full": 4095, "assumed": False}],
        {
            "note": "full scale 4095",
            "hint": RAW_HINT,
            "value": "counts",
            "counterChip": False,
            "overlays": True,
        },
    ),
    (
        "mode words: raw, full scale assumed",
        "modeWords",
        [True, {"transform": "Rot90", "full": 4095, "assumed": True}],
        {
            "note": "Rot90 applied · 4095 · full scale assumed",
            "hint": RAW_HINT,
            "value": "counts",
            "counterChip": False,
            "overlays": True,
        },
    ),
    (
        "mode words: raw, transform unknown hides the overlays",
        "modeWords",
        [True, {"transform": None, "full": 4095, "assumed": False}],
        {
            "note": "rotate / flip unknown: the driver's frame as is, overlays hidden · full scale 4095",
            "hint": RAW_HINT,
            "value": "counts",
            "counterChip": False,
            "overlays": False,
        },
    ),
    (
        "size: a raw frame is its served size",
        "imageSize",
        [1600, 1200, 2592, 2048, True],
        [1600, 1200],
    ),
    # the JPEG stretched over MJPG1's size when the shapes agree
    ("size: unread is the JPEG's", "imageSize", [640, 480, None, None], [640, 480]),
    ("gesture: a completed pointer commits", "gestureEnds", ["pointerup"], "commit"),
    (
        "gesture: a cancelled pointer discards",
        "gestureEnds",
        ["pointercancel"],
        "discard",
    ),
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
    # a computation switched off: STAT holds its last array, so nothing is drawn
    ("not computed: off", "notComputed", [0, "profiles"], "profiles not computed"),
    ("not computed: on", "notComputed", [1, "histogram"], None),
    ("not computed: unread is not off", "notComputed", [None, "histogram"], None),
    # the histogram plot: Raw's own frame histogram, else STAT's (bin i centred on lo + (i + 0.5) * step)
    (
        "histogram plot: raw is the frame's, 0..full, STAT's off word ignored",
        "histogramPlot",
        [{"values": [1, 2, 0, 2], "full": 3}, [9, 9], 0, 10, 20],
        {"values": [1, 2, 0, 2], "lo": -0.5, "step": 1, "empty": False},
    ),  # fmt: skip
    (
        "histogram plot: no raw is STAT's over hist_min..hist_max",
        "histogramPlot",
        [None, [5, 6, 7, 8], 1, 100, 300],
        {"values": [5, 6, 7, 8], "lo": 100, "step": 50, "empty": False},
    ),  # fmt: skip
    (
        "histogram plot: STAT's not computed is drawn as nothing, and said",
        "histogramPlot",
        [None, [5, 6], 0, 0, 2],
        {"values": [], "lo": 0, "step": 1, "empty": "histogram not computed"},
    ),  # fmt: skip
    (
        "histogram plot: STAT's range unread is its own length from 0",
        "histogramPlot",
        [None, [5, 6, 7], None, None, None],
        {"values": [5, 6, 7], "lo": 0, "step": 1, "empty": False},
    ),  # fmt: skip
    (
        "histogram plot: nothing read is empty",
        "histogramPlot",
        [None, None, None, None, None],
        {"values": [], "lo": 0, "step": 0, "empty": False},
    ),  # fmt: skip
    # the raw loop draws a new representation: the counter, or the full scale landing under the same counter
    (
        "raw is new: the same ETag is not",
        "rawIsNew",
        ['"7;4095"', '"7;4095"', '"7"'],
        False,
    ),  # fmt: skip
    (
        "raw is new: the scale landing is",
        "rawIsNew",
        ['"7;4095"', '"7"', '"7"'],
        True,
    ),  # fmt: skip
    (
        "raw is new: the counter moving is",
        "rawIsNew",
        ['"8;4095"', '"7;4095"', '"8"'],
        True,
    ),  # fmt: skip
    ("raw is new: no tag is always", "rawIsNew", ['"7"', '"7"', None], True),
    ("raw is new: no ETag is always", "rawIsNew", [None, None, '"7"'], True),
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


_SINGLE = """
const R = require(process.argv[1]);
(async () => {
  const sent = [], gate = {};
  const held = new Promise((r) => { gate.open = r; });
  const write = R.single(async (name) => { sent.push(name); await held; return name; });
  const first = write("drag roi");
  const busy = write.busy();
  const second = await write("rail exposure"); // refused while the first is in flight
  gate.open();
  const done = await first;
  const third = await write("after");
  console.log(JSON.stringify({ busy, second, done, third, sent, idle: !write.busy() }));
})();
"""


@needs_node
def test_one_write_list_at_a_time_for_the_page() -> None:
    got = run("camera_rules.js", _SINGLE)
    assert got == {"busy": True, "second": None, "done": "drag roi", "third": "after", "sent": ["drag roi", "after"], "idle": True}  # fmt: skip


_WITHIN = """
const R = require(process.argv[1]);
(async () => {
  let aborted = false;
  const stalled = R.within(40, (signal) => new Promise(() => { signal.addEventListener("abort", () => { aborted = true; }); }));
  const slow = await stalled.then(() => "answered", (e) => e.message);
  const quick = await R.within(1000, async () => "answered");
  // the deadline covers the body: a response whose body never arrives still times out
  const body = await R.within(40, async () => { await new Promise(() => {}); return "body"; }).then(() => "read", (e) => e.message);
  console.log(JSON.stringify({ slow, aborted, quick, body }));
})();
"""


@needs_node
def test_a_stalled_request_times_out_and_is_aborted() -> None:
    got = run("camera_rules.js", _WITHIN)
    assert got == {"slow": "timed out", "aborted": True, "quick": "answered", "body": "timed out"}  # fmt: skip


@needs_node
def test_the_lut_at_a_12_bit_full_scale() -> None:
    grey, log = call_all("camera_rules.js", [("lut", ["grey", "linear", 0, 4095, 4095]), ("lut", ["inferno", "log", 56, 4000, 4095])])  # fmt: skip
    assert len(grey) == 4096 * 3 and grey[:3] == [0, 0, 0] and grey[2048 * 3] == 128 and grey[-3:] == [255, 255, 255]  # fmt: skip
    assert len(log) == 4096 * 3 and log[56 * 3 : 57 * 3] == [0, 0, 4] and log[4000 * 3 : 4001 * 3] == [252, 255, 164]  # fmt: skip
    # the log scale lifts the floor: 3 counts above black is already past 7 % of the map
    assert log[59 * 3] > log[56 * 3] + 10


_DECODE = """
const R = require(process.argv[1]);
const px = new Uint16Array([0, 1, 40000, 4095, 56, 65535]);
const ok = R.decodeRaw(px.buffer, { width: "3", height: "2", fullScale: null, tag: '"7"' });
const short = R.decodeRaw(px.buffer.slice(2), { width: "3", height: "2", fullScale: "4095" });
const bare = R.decodeRaw(px.buffer, { width: null, height: "2" });
const read = R.decodeRaw(px.buffer, { width: "2", height: "3", fullScale: "65535" });
console.log(JSON.stringify({
  ok: [ok.ok, ok.width, ok.height, ok.full, ok.assumed, ok.tag, ok.data.constructor.name, Array.from(ok.data)],
  read: [read.full, read.assumed, read.data[2 * 2 + 1]],
  short, bare,
}));
"""


@needs_node
def test_the_decode_of_a_frame_and_its_refusals() -> None:
    got = run("camera_rules.js", _DECODE)
    assert got["ok"] == [True, 3, 2, 4095, True, '"7"', "Uint16Array", [0, 1, 40000, 4095, 56, 65535]]  # fmt: skip
    assert got["read"] == [65535, False, 65535]
    assert got["short"] == {"ok": False, "why": "the frame is 10 bytes for 3×2"}
    assert got["bare"] == {"ok": False, "why": "the frame came without its geometry"}


_ROUND_TRIP = """
const R = require(process.argv[1]);
// stored in one mode at its scale, read back in each mode at each scale
const s = { levels: R.storeLevels({ levels: R.storeLevels({}, "mjpeg", 3, 255, 255) }, "raw", 56, 4000, 4095) };
console.log(JSON.stringify({
  mjpeg: R.storedLevels(s, "mjpeg", 255),
  raw: R.storedLevels(s, "raw", 4095),
  raw16: R.storedLevels(s, "raw", 65535),
}));
"""


@needs_node
def test_levels_round_trip_as_fractions_between_the_scales() -> None:
    got = run("camera_rules.js", _ROUND_TRIP)
    assert got == {"mjpeg": {"black": 3, "white": 255}, "raw": {"black": 56, "white": 4000}, "raw16": {"black": 896, "white": 64015}}  # fmt: skip


_TRANSFORM_TYPED = """
const R = require(process.argv[1]);
// a typed frame stays typed, and a value above 32767 survives the move
const t = R.transformFrame("Rot90", new Uint16Array([0, 40000, 2, 3, 4, 65535]), 3, 2);
console.log(JSON.stringify([t.data.constructor.name, Array.from(t.data), t.width, t.height]));
"""


@needs_node
def test_a_typed_frame_transforms_as_typed() -> None:
    assert run("camera_rules.js", _TRANSFORM_TYPED) == ["Uint16Array", [3, 0, 4, 40000, 65535, 2], 2, 3]  # fmt: skip
