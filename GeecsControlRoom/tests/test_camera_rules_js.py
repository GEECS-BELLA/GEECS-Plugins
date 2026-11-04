"""camera_rules.js under node: each row one rule of the camera page (skipped without node)."""

from __future__ import annotations

from typing import Any

from tests._node import call_all, needs_node, run

VIEW = {"ok": True, "state": "live", "roles": {}}
W = {"role": "exposure", "writable": True, "locked": True}
AX = {"min": 10, "size": 20, "bin": 2, "reverse": False}
RAX = {"min": 10, "size": 20, "bin": 2, "reverse": True}
V0 = {"scale": 2, "ox": 10, "oy": 20, "moved": True}
TYPES = [
    "None",
    "Rot90",
    "Rot180",
    "Rot270",
    "Mirror",
    "Rot90Mirror",
    "Rot180Mirror",
    "Rot270Mirror",
]  # TRANS1:Type

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
    # the raw frame's scale: 16-bit values, full scale + 1 bins and LUT entries
    (
        "levels: inside the raw full scale",
        "levels",
        [5000, 9000, 4095],
        {"black": 4094, "white": 4095},
    ),  # fmt: skip
    (
        "histogram: full scale + 1 bins, above it in the top",
        "histogram",
        [[0, 3, 3, 9, 2], 3],
        [1, 0, 1, 3],
    ),  # fmt: skip
    (
        "auto levels at 4095: the floor and the hot pixel's tail",
        "autoLevels",
        [[0] * 56 + [500] + [0] * 3943 + [500] + [0] * 95, 4095],
        {"black": 56, "white": 4000},
    ),  # fmt: skip
    (
        "auto levels at 4095: a flat floor opens one count",
        "autoLevels",
        [[0] * 56 + [999] + [0] * 4039, 4095],
        {"black": 56, "white": 57},
    ),  # fmt: skip
    # levels per mode, as fractions of the mode's full scale
    (
        "stored: an 8-bit white of 255 is the raw view's full scale",
        "levelsFromStore",
        [{"black": 0, "white": 1}, 4095],
        {"black": 0, "white": 4095},
    ),  # fmt: skip
    (
        "stored: nothing kept is 0 … full scale",
        "levelsFromStore",
        [None, 4095],
        {"black": 0, "white": 4095},
    ),  # fmt: skip
    (
        "stored: a pre-mode store is the JPEG's 8-bit counts",
        "storedLevels",
        [{"black": "51", "white": "204"}, "jpeg"],
        {"black": 0.2, "white": 0.8},
    ),  # fmt: skip
    (
        "stored: a pre-mode store is not the raw view's",
        "storedLevels",
        [{"black": 51, "white": 204}, "raw"],
        None,
    ),  # fmt: skip
    (
        "stored: the raw view's black never reaches the JPEG",
        "storedLevels",
        [{"levels": {"raw": {"black": 0.5, "white": 0.9}}}, "jpeg"],
        None,
    ),  # fmt: skip
    # the transform role's value, as TRANS1:Type names it
    (
        "transform: an enum index through its choices",
        "transformOf",
        [{"value": 1}, {"enumChoices": TYPES}],
        {"name": "Rot90", "known": True, "word": ""},
    ),  # fmt: skip
    (
        "transform: an index past the choices is unknown",
        "transformOf",
        [{"value": 9}, {"enumChoices": TYPES}],
        {
            "name": None,
            "known": False,
            "word": "transform 9 unknown: the driver's frame as read, overlays hidden",
        },
    ),  # fmt: skip
    (
        "transform: a name the IOC does not have",
        "transformOf",
        [{"value": "Flip"}, {}],
        {
            "name": None,
            "known": False,
            "word": 'transform "Flip" unknown: the driver\'s frame as read, overlays hidden',
        },
    ),  # fmt: skip
    (
        "transform: unread",
        "transformOf",
        [None, {}],
        {
            "name": None,
            "known": False,
            "word": "transform unread: the driver's frame as read, overlays hidden",
        },
    ),  # fmt: skip
    (
        "transform: a name not of the eight is not drawn",
        "transformFrame",
        [[0, 1, 2, 3, 4, 5], 3, 2, "Flip"],
        None,
    ),  # fmt: skip
    # the toggle's words
    (
        "raw toggle: the module names the raw frame",
        "rawToggle",
        [{"role": "raw_frame"}],
        {"enabled": True, "why": "the camera's 16-bit frame, before the transform"},
    ),  # fmt: skip
    (
        "raw toggle: no raw frame role",
        "rawToggle",
        [None],
        {"enabled": False, "why": "the module names no raw frame for this camera"},
    ),  # fmt: skip
    (
        "mode words: the JPEG",
        "modeWords",
        [False, 255, False],
        {
            "hint": "The 8-bit stream, stretched in this browser: nothing is written to the camera.",
            "scale": "",
            "chip": True,
            "full": 255,
        },
    ),  # fmt: skip
    (
        "mode words: raw, full scale read",
        "modeWords",
        [True, 4095, False],
        {
            "hint": "The camera's 16-bit frame, levels applied in this browser; nothing is written to the camera.",
            "scale": "counts of 4095",
            "chip": False,
            "full": 4095,
        },
    ),  # fmt: skip
    (
        "mode words: raw, full scale assumed",
        "modeWords",
        [True, 4095, True],
        {
            "hint": "The camera's 16-bit frame, levels applied in this browser; nothing is written to the camera.",
            "scale": "counts of 4095, full scale assumed",
            "chip": False,
            "full": 4095,
        },
    ),  # fmt: skip
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
def test_the_lut_at_the_raw_full_scale() -> None:
    (grey,) = call_all("camera_rules.js", [("lut", ["grey", "log", 50, 4000, 4095])])
    assert len(grey) == 3 * 4096 and grey[:3] == [0, 0, 0] and grey[50 * 3] == 0 and grey[4000 * 3 :] == [255] * 3 * 96  # fmt: skip
    assert (
        0 < grey[60 * 3] < grey[400 * 3] < 255
    )  # the log scale lifts the floor's first counts


# A 3×2 frame, row-major (row 0: 0 1 2; row 1: 3 4 5), through each TRANS1:Type, pixel for pixel as
# ADCore's NDPluginTransform.cpp maps a mono frame: Rot90 out[(h-1-y) + x·h], Rot270 out[y + (w-1-x)·h],
# Rot90Mirror out[y + x·h], Rot270Mirror out[(h-1-y) + (w-1-x)·h], Mirror and Rot180Mirror flip one axis.
TRANSFORMED = {
    "None": ([0, 1, 2, 3, 4, 5], 3, 2),
    "Rot90": ([3, 0, 4, 1, 5, 2], 2, 3),
    "Rot180": ([5, 4, 3, 2, 1, 0], 3, 2),
    "Rot270": ([2, 5, 1, 4, 0, 3], 2, 3),
    "Mirror": ([2, 1, 0, 5, 4, 3], 3, 2),
    "Rot90Mirror": ([0, 3, 1, 4, 2, 5], 2, 3),
    "Rot180Mirror": ([3, 4, 5, 0, 1, 2], 3, 2),
    "Rot270Mirror": ([5, 2, 4, 1, 3, 0], 2, 3),
}


@needs_node
def test_each_transform_maps_as_trans1() -> None:
    got = call_all("camera_rules.js", [("transformFrame", [[0, 1, 2, 3, 4, 5], 3, 2, name]) for name in TYPES])  # fmt: skip
    assert {n: (g["data"], g["width"], g["height"]) for n, g in zip(TYPES, got)} == TRANSFORMED  # fmt: skip


_FRAMES = """
const R = require(process.argv[1]);
(async () => {
  const H = (h) => (k) => (k in h ? h[k] : null);
  const head = { "X-Frame-Width": "3", "X-Frame-Height": "2", "X-Frame-Full-Scale": "4095", "X-Frame-Tag": '"7"', ETag: '"7"' };
  const body = Uint16Array.from([56, 40000, 65535, 0, 4000, 57]).buffer;
  const f = R.decodeRaw(body, H(head));
  const decoded = { width: f.width, height: f.height, full: f.full, assumed: f.assumed, tag: f.tag, etag: f.etag, typed: f.data instanceof Uint16Array, pixel: f.data[1 * 3 + 1], above: f.data[1], top: f.data[2] };
  const fails = (b, h) => { try { R.decodeRaw(b, H(h)); return null; } catch (e) { return e.message; } };
  const short = fails(Uint16Array.from([1, 2, 3, 4, 5]).buffer, head);
  const bare = fails(body, {});
  const { "X-Frame-Full-Scale": _, ...noScale } = head;
  const assumed = R.decodeRaw(body, H(noScale));
  // the poll: a fake fetch answering 304, 503 and 200 in turn, its requests kept
  const asked = [], answers = [
    { status: 304, ok: false, headers: new Map() },
    { status: 503, ok: false, json: async () => ({ detail: "the frame counter (bench:cam:ARR:ArrayCounter_RBV) could not be read" }) },
    { status: 200, ok: true, headers: { get: (k) => H(head)(k) }, arrayBuffer: async () => body },
  ];
  const fetchFn = async (url, opts) => { asked.push([url, opts.cache, opts.headers]); return answers.shift(); };
  const same = await R.fetchRaw(fetchFn, "/raw", '"7"');
  const err = await R.fetchRaw(fetchFn, "/raw", '"7"').then(() => null, (e) => e.message);
  const fresh = await R.fetchRaw(fetchFn, "/raw", null);
  console.log(JSON.stringify({ decoded, short, bare, assumed: [assumed.full, assumed.assumed], same, err, fresh: [fresh.fresh, fresh.frame.etag], asked }));
})();
"""


@needs_node
def test_a_raw_frame_decodes_and_a_304_is_nothing_new() -> None:
    got = run("camera_rules.js", _FRAMES)
    assert got["decoded"] == {"width": 3, "height": 2, "full": 4095, "assumed": False, "tag": '"7"', "etag": '"7"', "typed": True, "pixel": 4000, "above": 40000, "top": 65535}  # fmt: skip
    assert got["short"] == "geometry disagrees: 5 values for 3×2"
    assert got["bare"] == "the frame came without its geometry"
    assert got["assumed"] == [
        4095,
        True,
    ]  # no X-Frame-Full-Scale: 4095, and the words say so
    assert got["same"] == {"fresh": False}  # a 304 is no new frame, never an error
    assert (
        got["err"]
        == "the frame counter (bench:cam:ARR:ArrayCounter_RBV) could not be read"
    )
    assert got["fresh"] == [True, '"7"']
    assert got["asked"] == [["/raw", "no-store", {"If-None-Match": '"7"'}]] * 2 + [
        ["/raw", "no-store", {}]
    ]


_ROUND_TRIPS = """
const R = require(process.argv[1]);
const trip = (b, w, from, to) => R.levelsFromStore(R.levelsToStore(b, w, from), to);
console.log(JSON.stringify({
  raw: trip(56, 400, 4095, 4095),           // the raw view's own levels come back exact
  jpeg: trip(10, 250, 255, 255),            // and the JPEG's
  white: trip(0, 255, 255, 4095),           // an 8-bit white is the raw full scale, never 255 counts
  black: trip(2048, 4095, 4095, 255),       // a 12-bit black is the JPEG's half, never its white
  wider: trip(56, 400, 4095, 65535),        // a full scale that changes keeps the fractions
}));
"""


@needs_node
def test_levels_round_trip_between_the_scales() -> None:
    got = run("camera_rules.js", _ROUND_TRIPS)
    assert got == {
        "raw": {"black": 56, "white": 400},
        "jpeg": {"black": 10, "white": 250},
        "white": {"black": 0, "white": 4095},
        "black": {"black": 128, "white": 255},
        "wider": {"black": 896, "white": 6401},
    }
