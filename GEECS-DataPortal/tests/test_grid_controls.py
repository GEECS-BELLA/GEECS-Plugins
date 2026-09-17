"""Execute Grid controls to pin cross-tab state and invalid-edit behavior."""

from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess

import pytest

TEMPLATES = Path(__file__).parents[1] / "geecs_portal" / "templates"


def run_js(tmp_path, body):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for browser-control regressions")
    script = tmp_path / "grid-controls.cjs"
    script.write_text(body)
    result = subprocess.run(
        [node, str(script)], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stdout + result.stderr


HARNESS = r"""
const assert = require('node:assert/strict');
const S = {tab:'grid',view:'bin',gridcfg:JSON.stringify({value:'signal',min_count:3,lower:.25,upper:.75,visit:1}),gridbin:'',imagebin:'',bincfg:JSON.stringify({bin_col:'signal',bin_width:2})};
const elements = new Map();
const document = {
 getElementById(id) {
  if(!elements.has(id)) elements.set(id,{value:'',innerHTML:'',textContent:'',hidden:false,parentElement:{},replaceChildren(){}});
  return elements.get(id);
 },querySelector(){return {};}
};
const ResizeObserver = class {observe(){}};
const Option = class {};
let BOOTED=true, writes=0, requests=0, IMG_KEY=null;
const ROOT='',UID='uid-002',DAY='',VERSION='test',SEL_DEVICE='cam';
const esc=String,escAttr=String,flashNote=()=>{},closeModals=()=>{},refresh=()=>{};
const writeState=()=>{writes++;};
const setPassCount=()=>{};
const setTab=value=>{S.tab=value;};
const setView=value=>{S.view=value;};
let imageBins=[{bin:3.25,count:3},{bin:5.25,count:3}];
const api=async path=>{
 requests++;
 if(path==='bin-images')return {bins:imageBins,bin_col:'signal'};
 return {config:JSON.parse(S.gridcfg),cells:[{bin:1,x:0,y:0,visit:1}],visits:1,kind:'grid',pretty:{},error_label:'IQR',notes:[],pass:3,total:3,bin_column:'bin_number'};
};
"""


def test_grid_visit_does_not_restrict_image_bins(tmp_path):
    page = (TEMPLATES / "run.html").read_text()
    images = page[
        page.index("function refreshImages()") : page.index(
            "// ---------------- column picker"
        )
    ]
    bins = page[
        page.index("function applyBinset()") : page.index(
            "// ---------------- display settings popup"
        )
    ]
    body = (
        HARNESS
        + (TEMPLATES / "grid_script.html").read_text()
        + images
        + bins
        + r"""
drawGrid=async()=>{};
(async()=>{
 await refreshGrid();
 assert.equal(S.gridbin,'1');
 assert.equal(S.imagebin,'');
 refreshImages(); // merely switching tabs must keep non-acquisition bin labels
 S.tab='images';refreshImages();await new Promise(setImmediate);
 assert.equal((document.getElementById('imggrid').innerHTML.match(/<figure/g)||[]).length,2);
 assert.equal(document.getElementById('grid-image-selection').innerHTML,'');
 gridOpenImages();
 assert.equal(S.imagebin,'1');assert.equal(JSON.parse(S.bincfg).bin_col,'bin_number');
 imageBins=[{bin:1,count:3},{bin:2,count:3}];
 refreshImages();await new Promise(setImmediate);
 assert.equal((document.getElementById('imggrid').innerHTML.match(/<figure/g)||[]).length,1);
 S.gridbin='2'; // another map selection is independent of the Images filter
 assert.equal(S.imagebin,'1');
 document.getElementById('bs-col').value='signal';
 document.getElementById('bs-width').value='2';
 applyBinset();await new Promise(setImmediate);
 assert.equal(S.imagebin,'');assert.equal(S.gridbin,'2');
})().catch(err=>{console.error(err);process.exit(1);});
"""
    )
    run_js(tmp_path, body)


@pytest.mark.parametrize(
    "name,value,expected",
    [
        ("min_count", "", None),
        ("lower", "", None),
        ("upper", "", None),
        ("min_count", "0", None),
        ("min_count", "1.5", None),
        ("min_count", "-3", None),
        ("upper", "20", None),
        ("lower", "80", None),
        ("upper", "101", None),
        ("min_count", "5", 5),
        ("lower", "0", 0),
        ("upper", "100", 1),
    ],
)
def test_numeric_edits_preserve_last_valid_state(tmp_path, name, value, expected):
    body = (
        HARNESS
        + (TEMPLATES / "grid_script.html").read_text()
        + "\n"
        + f"""
BOOTED=false;
const name={json.dumps(name)}, input={{value:{json.dumps(value)}}};
const before=S.gridcfg;
changeGridNumber(name,input);
const expected={json.dumps(expected)};
if(expected===null){{
 assert.equal(S.gridcfg,before);assert.equal(writes,0);assert.equal(requests,0);
 assert.equal(Number(input.value),JSON.parse(before)[name]*(name==='min_count'?1:100));
}}else{{
 assert.equal(JSON.parse(S.gridcfg)[name],expected);assert.equal(writes,1);
}}
"""
    )
    run_js(tmp_path, body)


class GridNumberInputs(HTMLParser):
    """Collect the ``onchange`` handler of every ``<input>`` carrying an id."""

    def __init__(self) -> None:
        super().__init__()
        self.handlers: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record ``onchange`` for an identified input element."""
        attributes = dict(attrs)
        if tag == "input" and attributes.get("id"):
            self.handlers[attributes["id"]] = attributes.get("onchange") or ""


@pytest.mark.parametrize("field", ["grid-lower", "grid-upper", "grid-min_count"])
def test_grid_number_inputs_call_the_validator(field):
    """Pin the wiring: a validator the inputs bypass guards nothing.

    ``test_numeric_edits_preserve_last_valid_state`` calls
    ``changeGridNumber`` directly, so it stays green even if an input is
    rewired to ``changeGrid(..., Number(this.value))`` -- the shape that
    let an empty field write ``0`` into the config and the shared URL.
    """
    parser = GridNumberInputs()
    parser.feed((TEMPLATES / "run.html").read_text())
    handler = parser.handlers[field]
    assert handler.startswith("changeGridNumber("), (
        f"{field} must route through the validator, got: {handler}"
    )
    assert "Number(this" not in handler, (
        f"{field} coerces its own value instead of letting the validator "
        f"reject empty and out-of-range input, got: {handler}"
    )
