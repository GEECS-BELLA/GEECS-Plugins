"""Exercise editor state transitions and deferred rendering using the real scripts."""

import json
import subprocess
from pathlib import Path

import pytest
from geecs_web_theme.testing import node_available

STATIC = Path(__file__).resolve().parents[1] / "geecs_scanner/static"

# Only DOM operations used by these components; no imitation of their logic.
DOM = r"""
class Element {
  constructor(tag) { this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {}; this.dataset = {}; this.value = ''; this.open = false; }
  setAttribute(k,v) {this.attrs[k] = String(v); if (k.startsWith('data-')) this.dataset[k.slice(5).replace(/-([a-z])/g,(_,c)=>c.toUpperCase())] = String(v);}
  appendChild(n) { n.parent = this; this.children.push(n); return n; }
  replaceChildren() {this.children = []; this.textContent = '';}
  addEventListener() {}
  querySelector(s) {
    for (const c of this.children) { if ((s[0] === '[' && Object.hasOwn(c.attrs,s.slice(1,-1))) || c.tagName === s.toUpperCase()) return c; const found = c.querySelector(s); if (found) return found; }
    return null;
  }
  closest(s) {return s === 'details' ? disclosure : null;}
}
const document = {createElement: t=>new Element(t), createElementNS: (_,t)=>new Element(t), createTextNode: t=>{const n=new Element('text'); n.textContent=t; return n;}};
const disclosure = new Element('details'), elements = {};
const root = {querySelector: s=>elements[s] || (elements[s]=new Element('div')), querySelectorAll: ()=>[], addEventListener() {}};
const window = {};
let timers = new Map(), nextTimer = 0;
function setTimeout(fn) { timers.set(++nextTimer,fn); return nextTimer; }
function clearTimeout(id) {timers.delete(id);}
async function tick() {const first = timers.entries().next().value; if(first) {timers.delete(first[0]); first[1]();} await new Promise(setImmediate);}
const sample = {total_steps:2, sampled:false, indices:[0,1], axes:[{axis:'A',relative:false,positions:[1,2]}]};
const good = {trajectory:{kind:'axes',axes:[{kind:'list',axis:'A',positions:[1,2]}]}};
"""


def run_js(body):
    if not node_available():
        pytest.skip("Node is not installed")
    script = (
        DOM
        + (STATIC / "trajectory-view.js").read_text()
        + (STATIC / "sweep-composer.js").read_text()
    )
    result = subprocess.run(
        ["node", "-"],
        input=script
        + "\n(async()=>{\n"
        + body
        + "\n})().catch(e=>{console.error(e);process.exitCode=1});",
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_bad_load_discards_prior_result_and_cancels_pending_preview():
    result = run_js("""
let pending = [], signals = [];
const c = window.GEECS_SWEEP.create(root, (p,s)=>{signals.push(s); return new Promise(resolve=>pending.push(resolve));}, ()=>{});
c.load(good); await tick(); pending.shift()(sample); await new Promise(setImmediate);
const hadResult = !!c.result();
let failures=[];
for (const bad of [undefined,{trajectory:{kind:'unknown'}},{trajectory:{kind:'axes',axes:[{kind:'list',axis:'A',positions:42}]}}]) {
 c.load(good); await tick(); const old = pending.shift();
 try {c.load(bad);} catch(e) { failures.push(e.message); }
 old(sample); await new Promise(setImmediate);
 if (c.result() !== null) throw new Error('stale result survived');
 try {c.value(); throw new Error('old trajectory survived');} catch(e) {if(e.message==='old trajectory survived') throw e;}
}
console.log(JSON.stringify({hadResult, failures, aborted: signals.every(s=>s.aborted), timers: timers.size}));
""")
    assert result["hadResult"] and len(result["failures"]) == 3
    assert result["aborted"] and result["timers"] == 0


def test_busy_preview_retries_twice_then_stops_with_visible_message():
    result = run_js("""
let calls=0;
const c=window.GEECS_SWEEP.create(root,()=>{calls++;const e=new Error('busy');e.status=409;return Promise.reject(e);},()=>{});
c.load(good); await tick(); await tick(); await tick();
console.log(JSON.stringify({calls,timers:timers.size,message:elements['#sweep-message'].textContent,value:c.value()}));
""")
    assert result["calls"] == 3 and result["timers"] == 0
    assert "Preview queue busy" in result["message"]
    assert result["value"]["trajectory"]["axes"][0]["positions"] == [1, 2]


def test_point_table_builds_only_when_open_and_uses_latest_result():
    result = run_js("""
const plots=new Element('div'), table=new Element('div');
window.GEECS_TRAJECTORY.render(plots,table,sample); const collapsed=table.children.length;
const latest={...sample,indices:[10,11]}; window.GEECS_TRAJECTORY.render(plots,table,latest);
disclosure.open=true; disclosure.ontoggle();
const count=table.children.length, first=table.children[0].querySelector('td').textContent;
disclosure.ontoggle();
console.log(JSON.stringify({collapsed,count,first,after:table.children.length}));
""")
    assert result == {"collapsed": 0, "count": 1, "first": "11", "after": 1}
