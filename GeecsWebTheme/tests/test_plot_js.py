"""plot.js — the one Plotly template every GEECS line plot is drawn through.

The real ``plot.js`` runs under node. The template is built through a fake
token reader that answers each name with a sentinel, so a colour that did
not come from a token (a literal, a default) shows as itself and fails. The
repaint is driven through a fake window, document and ``Plotly``.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from geecs_web_theme import static_dir
from geecs_web_theme.testing import defined_tokens, js_colour_literals, node_available

_PLOT = static_dir() / "plot.js"

_HARNESS = r"""
const P = require(process.argv[1]);
const read = (palette) => (name) => palette[name] ?? "";
const asked = new Set();
const sentinels = new Proxy({}, { get: (_, name) => (/^--trace-([5-9]|1\d)$/.test(name) ? undefined : (asked.add(name), `tok(${name})`)) });
const t = P.template(read(sentinels));
// the repaint: a fake page whose theme flips, a watched host still in the page and one removed
const calls = [], listeners = {};
let ink = "first";
global.window = { addEventListener: (k, fn) => (listeners[k] = fn), Plotly: { relayout: (h, keys) => calls.push([h.id, keys]) } };
global.document = { documentElement: {} };
global.getComputedStyle = () => ({ getPropertyValue: (name) => (/^--trace-[1-4]$/.test(name) || !name.startsWith("--trace") ? `${ink}(${name})` : "") });
const live = { id: "live", isConnected: true, _fullLayout: {} }, gone = { id: "gone", isConnected: false, _fullLayout: {} };
P.watch(live); P.watch(gone); P.watch(live);
ink = "second";
listeners["geecs:theme"]();
console.log(JSON.stringify({
  layout: t.layout, config: t.config, asked: [...asked].sort(),
  tokenOf: [0, 1, 3, 4, 9].map((i) => P.traceToken(i, 4)),
  count: [P.traceCount(read({})), P.traceCount(read({ "--trace-1": "a", "--trace-2": "b" }))],
  merged: P.merge({ a: { b: 1, c: 2 }, l: [1, 2] }, { a: { c: 3 }, l: [9] }),
  keys: P.themeKeys(t.layout), calls,
  formats: ["s", "V", "A", "Hz", "m", "W", "nm", "counts", "%", "px", "", null, " s "].map((u) => [u, P.axisFormat(u).exponentformat]),
  hover: [P.axisFormat("s").hoverformat, P.axisFormat("nm").hoverformat],
}));
"""


@pytest.fixture(scope="module")
def out() -> dict:
    """The harness's answers, from one node run."""
    if not node_available():  # pragma: no cover - CI and dev machines have it
        pytest.skip("node not available")
    done = subprocess.run(
        ["node", "-e", _HARNESS, str(_PLOT)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _leaves(obj: object, path: str = ""):
    """Every (path, value) leaf of a JSON object."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, obj


def test_the_script_carries_no_literal_colour() -> None:
    assert js_colour_literals(_PLOT.read_text()) == []


def test_every_colour_and_face_in_the_template_is_a_token(out: dict) -> None:
    themed = [
        (p, v)
        for p, v in _leaves(out["layout"])
        if p.endswith("color") or "colorway" in p or p.endswith("family")
    ]
    assert len(themed) > 20  # the axes, legend, hover label, modebar, fonts
    assert [p for p, v in themed if not str(v).startswith("tok(--")] == []


def test_each_trace_index_maps_to_a_trace_token(out: dict) -> None:
    # the palette defines --trace-1..4: the colorway is those four, in order, and trace i cycles them
    assert out["layout"]["colorway"] == [f"tok(--trace-{k})" for k in range(1, 5)]
    assert out["tokenOf"] == ["--trace-1", "--trace-2", "--trace-4", "--trace-1", "--trace-2"]  # fmt: skip
    assert out["count"] == [1, 2]  # no trace token: one colour, never zero


def test_every_token_the_template_reads_is_defined_in_every_palette(out: dict) -> None:
    blocks = defined_tokens((static_dir() / "theme.css").read_text())
    root = blocks[":root"]
    missing = [n for n in out["asked"] if n not in root]
    assert missing == []
    palettes = [b for s, b in blocks.items() if s != ":root"]
    colours = [
        n for n in out["asked"] if not n.startswith("--ff-")
    ]  # faces are root-only by design
    assert [n for n in colours for b in palettes if n not in b] == []


def test_the_config_is_the_rule(out: dict) -> None:
    c = out["config"]
    assert (c["displaylogo"], c["doubleClick"], c["responsive"]) == (
        False,
        "reset",
        True,
    )
    # the selection tools removed; zoom, pan, autoscale, reset and PNG export stay (Plotly's defaults)
    assert {"select2d", "lasso2d"} <= set(c["modeBarButtonsToRemove"])
    assert not {"zoom2d", "pan2d", "autoScale2d", "resetScale2d", "toImage"} & set(
        c["modeBarButtonsToRemove"]
    )
    # never the buttons by name: Plotly 3.1.1 hangs comparing that config on the next react (a live plot's 2 Hz)
    assert "modeBarButtons" not in c
    assert c["toImageButtonOptions"]["format"] == "png"


def test_a_page_merges_its_keys_over_the_template(out: dict) -> None:
    assert out["merged"] == {"a": {"b": 1, "c": 3}, "l": [9]}


def test_the_repaint_touches_colours_and_faces_only(out: dict) -> None:
    keys = out["keys"]
    assert (
        "xaxis.tickfont.color" in keys and "colorway" in keys and "font.family" in keys
    )
    # never a range, a margin or a mode: a repaint leaves the viewer's zoom alone
    assert [k for k in keys if not (k.endswith("color") or k.endswith("family") or k == "colorway")] == []  # fmt: skip


def test_a_theme_change_repaints_each_watched_host_once_with_the_new_tokens(
    out: dict,
) -> None:
    # watched twice, repainted once; a host no longer in the page is dropped, never relayout
    assert [h for h, _ in out["calls"]] == ["live"]
    keys = out["calls"][0][1]
    assert keys["xaxis.gridcolor"] == "second(--rule-soft)"
    assert keys["colorway"] == [f"second(--trace-{k})" for k in range(1, 5)]


def test_every_page_names_the_bundle_the_theme_vendors() -> None:
    """A template that loads Plotly names the one vendored file, which exists: an upgrade renames it everywhere."""
    repo = Path(__file__).resolve().parents[2]
    vendored = sorted(p.name for p in static_dir().glob("plotly-*.min.js"))
    assert len(vendored) == 1, vendored  # one bundle, one version
    pages = list(repo.glob("*/*/templates/*.html")) + [static_dir() / "kit.html"]
    named = {(p.relative_to(repo).as_posix(), m) for p in pages for m in re.findall(r"plotly-[a-z0-9]+-[0-9.]+\.min\.js", p.read_text())}  # fmt: skip
    assert {"GEECS-DataPortal/geecs_portal/templates/run.html", "GeecsControlRoom/geecs_control_room/templates/device_trace.html"} <= {p for p, _ in named}  # fmt: skip
    assert [(p, m) for p, m in named if m != vendored[0]] == []


def test_si_prefixes_only_on_base_si_units(out: dict) -> None:
    # a wavelength axis in nm read "knm" when every unit was prefixed: only s, V, A, Hz, m, W take SI prefixes
    si = {u for u, f in out["formats"] if f == "SI"}
    assert si == {"s", "V", "A", "Hz", "m", "W", " s "}
    assert {u: f for u, f in out["formats"] if u in ("nm", "counts", "%", "px", "", None)} == dict.fromkeys(("nm", "counts", "%", "px", "", None), "none")  # fmt: skip
    assert out["hover"] == [".4~s", ".5~g"]
