"""Execute the scan page's script: the Images tab's bins and the shot trace."""

from pathlib import Path
import shutil
import subprocess

import pytest

TEMPLATES = Path(__file__).parents[1] / "geecs_portal" / "templates"


def run_js(tmp_path, body):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for browser-control regressions")
    script = tmp_path / "scan-page.cjs"
    script.write_text(body)
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


HARNESS = r"""
const assert = require('node:assert/strict');
// The Images tab, binned, opened from a 2-D map cell (imagebin) on the Plot page.
const S = {tab:'images',view:'bin',imagebin:'1',filters:'',processing:'',display:'',bincfg:JSON.stringify({bin_col:'bin_number',min_count:1})};
const elements = new Map();
const document = {
 getElementById(id) {
  if(!elements.has(id)) elements.set(id,{value:'',innerHTML:'',textContent:'',hidden:false});
  return elements.get(id);
 },
};
let IMG_KEY=null;
const ROOT='',UID='uid-002',DAY='',VERSION='test',SEL_DEVICE='cam';
// A camera device: its shots are pixels, so the per-bin view applies.
const IS_TRACE=false;
const esc=String,escAttr=String,settleDraft=()=>{};
const paneState=(kind,text)=>`<div class="state" data-state="${kind}">${text}</div>`;
const writeState=()=>{};
const setView=value=>{S.view=value;};
let imageBins=[{bin:1,count:3},{bin:2,count:3}];
const api=async path=>{
 assert.equal(path,'bin-images');
 return {bins:imageBins,bin_col:'bin_number'};
};
"""


def test_a_map_cell_shows_its_bin_until_the_bins_change(tmp_path):
    """The Plot page's "View this cell's images" opens one bin here; new bin
    settings replace that selection (they define other bins)."""
    page = (TEMPLATES / "run.html").read_text()
    images = page[
        page.index("function refreshImages()") : page.index("function dispCfg(")
    ]
    bins = page[
        page.index("function applyBinset()") : page.index(
            "// ---------------- utilities + boot"
        )
    ]
    body = (
        HARNESS
        + images
        + bins
        + r"""
const figures = () => (document.getElementById('imggrid').innerHTML.match(/<figure/g)||[]).length;
(async()=>{
 refreshImages();await new Promise(setImmediate);
 assert.equal(figures(),1);
 assert.ok(document.getElementById('image-bin-selection').innerHTML.includes('bin 1'));
 document.getElementById('bs-col').value='signal';
 document.getElementById('bs-width').value='2';
 applyBinset();await new Promise(setImmediate);
 assert.equal(S.imagebin,'');
 assert.equal(figures(),2);
 assert.equal(document.getElementById('image-bin-selection').innerHTML,'');
})().catch(err=>{console.error(err);process.exit(1);});
"""
    )
    run_js(tmp_path, body)


#: The trace view needs far less than HARNESS (and must not inherit its
#: bin-shaped `api`), so it carries its own stubs: a DOM, the page's
#: `S`, and counters for what `loadTrace` did.
TRACE_HARNESS = r"""
const assert = require('node:assert/strict');
const S = {tab:'plot'};
const elements = new Map();
const document = {getElementById(id) {return elements.get(id);}};
const esc = String;
const paneState = (kind, text) => text;
let fetches = 0, drawn = 0, cleared = 0;
const Plotly = {react(){drawn++;}, purge(){}};
const PLOT_CONFIG = {};
const resolveThemeTokens = v => v;
const api = async (path, params) => {
  assert.equal(path, "trace");
  fetches++;
  return {figure:{data:[],layout:{}}, params};
};
const listeners = {};
const window = {addEventListener(name, fn){listeners[name] = fn;}};
const SEL_DEVICE = "U_ICT", SHOT = 3;
// A live graph: no .state child, so traceHost must NOT wipe it.
const liveHost = {innerHTML:"live", querySelector(){return null;}};
"""


def _trace_script(body):
    """The template's trace block plus *body*, over TRACE_HARNESS."""
    page = (TEMPLATES / "run.html").read_text()
    block = page[
        page.index("// ---------------- the shot trace") : page.index(
            "function imgFailed(img)"
        )
    ]
    return TRACE_HARNESS + block + body


def test_a_trace_is_drawn_only_while_its_pane_is_visible(tmp_path):
    """``loadTrace`` must not lay a figure out into a hidden pane.

    Plotly sizes a figure against its container and the vendored build
    carries no ResizeObserver, so a trace drawn while ``pane-images`` is
    ``display:none`` stays zero-sized until the window is resized. A
    shared link whose tab is ``plot`` opens exactly that way.
    """
    run_js(
        tmp_path,
        _trace_script(
            r"""
(async()=>{
 elements.set("shottrace", liveHost);
 loadTrace(); await new Promise(setImmediate);
 assert.equal(fetches, 0, "a hidden pane must not be drawn into");
 S.tab = "images";
 loadTrace(); await new Promise(setImmediate);
 assert.equal(fetches, 1); assert.equal(drawn, 1);
 // Re-entering the tab redraws the cached figure (it may have been
 // hidden when the theme last changed) but never refetches: the shot
 // form navigates, so one fetch per page load is right.
 loadTrace(); await new Promise(setImmediate);
 assert.equal(fetches, 1, "one fetch per page load");
 assert.equal(drawn, 2, "re-entry redraws from the cache");
 // A theme change redraws the last figure and must leave the live
 // graph's own DOM alone (react updates an SVG that must still be
 // in the document).
 listeners["geecs:theme"]();
 assert.equal(drawn, 3); assert.equal(fetches, 1);
 assert.equal(liveHost.innerHTML, "live", "a live graph must not be wiped");
})();
"""
        ),
    )


def test_a_theme_change_on_another_tab_does_not_strand_the_trace(tmp_path):
    """The theme handler is the other door into a hidden pane, and the guard shuts it.

    Re-theming while the Images pane is hidden must not lay the figure
    out (same zero-size container as drawing at boot), and returning to
    the tab must re-draw it in the new palette — ``TRACE_DRAWN`` stops
    the refetch, so without a redraw on re-entry the trace would keep
    the old colours for the life of the page.
    """
    run_js(
        tmp_path,
        _trace_script(
            r"""
(async()=>{
 elements.set("shottrace", liveHost);
 S.tab = "images";
 loadTrace(); await new Promise(setImmediate);
 assert.equal(drawn, 1); assert.equal(fetches, 1);
 // Away from the Images tab: the pane is display:none.
 S.tab = "overview";
 listeners["geecs:theme"]();
 assert.equal(drawn, 1, "a hidden pane must not be re-laid-out");
 // Back: the figure must be redrawn (new palette) but never refetched.
 S.tab = "images";
 loadTrace(); await new Promise(setImmediate);
 assert.equal(drawn, 2, "re-entry must redraw the cached figure");
 assert.equal(fetches, 1, "re-entry must not refetch");
})();
"""
        ),
    )


def test_a_camera_device_never_calls_the_trace_endpoint(tmp_path):
    """No ``#shottrace`` host (an image device) means no trace fetch at all."""
    run_js(
        tmp_path,
        _trace_script(
            r"""
(async()=>{
 S.tab = "images";   // the template rendered an <img>, so no host exists
 loadTrace(); await new Promise(setImmediate);
 assert.equal(fetches, 0);
})();
"""
        ),
    )


def test_the_logbook_caption_names_the_trace(tmp_path):
    """A trace sent to the logbook must say what it shows, not bare "plot"
    (the caption a host the scan page does not recognise falls back to)."""
    page = (TEMPLATES / "run.html").read_text()
    caption = page[
        page.index("function plotCaption(gd)") : page.index(
            "function openSendToLogbook"
        )
    ]
    body = (
        r"""
const assert = require('node:assert/strict');
const S = {view: "shot"};
const GRID_DATA = null;
const prettyName = String;
const SEL_DEVICE = "U_BCaveICT", SHOT = 7;
const LAST_TRACE = {figure: {layout: {yaxis: {title: {text: "scopetrace_channel0"}}}}};
"""
        + caption
        + r"""
const trace = plotCaption({id: "shottrace"});
assert.ok(trace.includes("U_BCaveICT"), trace);
assert.ok(trace.includes("scopetrace_channel0"), trace);
assert.ok(trace.includes("7"), trace);
assert.equal(plotCaption({id: "elsewhere"}), "plot");
"""
    )
    run_js(tmp_path, body)
