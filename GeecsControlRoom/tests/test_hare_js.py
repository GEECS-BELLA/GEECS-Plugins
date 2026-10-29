"""hare.js under node with a fake WebSocket: the key, the subscribe, the resubscribe, the silent socket dropped."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from geecs_web_theme.testing import node_available

_SCRIPT = Path(__file__).resolve().parents[1] / "geecs_control_room/static/hare.js"

_HARNESS = r"""
const sockets = [];
class FakeSocket {
  constructor(url) { this.url = url; this.sent = []; this.readyState = 0; sockets.push(this); }
  send(m) { this.sent.push(JSON.parse(m)); }
  close() { this.readyState = 3; this.onclose && this.onclose(); }
  up() { this.readyState = 1; this.onopen(); }
  frame(f) { this.onmessage({ data: JSON.stringify(f) }); }
}
globalThis.window = { location: { href: "http://room.test/screens/x" } };
require(process.argv[2]);
const seen = [], states = [];
const hare = window.GeecsHare.open({
  url: "http://canopy.test/hare/v1/ws/pvs", key: "k 1", WebSocketImpl: FakeSocket, silentMs: 300,
});
hare.onStatus((s) => states.push(s));
hare.watch(["A:x"], (pv, wire) => seen.push([pv, wire && wire.value]));
sockets[0].up();
hare.watch(["A:y"], () => {});
sockets[0].frame({ type: "initial", data: { "A:x": { value: 1 } }, meta: { "A:x": { units: "mm" } } });
sockets[0].close();  // the drop
setTimeout(() => {
  sockets[1].up();
  sockets[1].frame({ type: "diff", data: { "A:x": { value: 2 } } });
  sockets[1].close = () => {};  // half-open: close() never calls back
  setTimeout(() => {           // ... yet the silence alone drops it
    console.log(JSON.stringify({
      url: sockets[0].url, first: sockets[0].sent, again: sockets[1].sent,
      seen, states, units: hare.meta("A:x").units, redialled: sockets.length,
    }));
    hare.close();
  }, 900);
}, 700);
"""


@pytest.mark.skipif(not node_available(), reason="node is not installed")
def test_a_new_socket_resubscribes_and_a_silent_one_is_dropped(tmp_path: Path) -> None:
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS)
    out = subprocess.run(
        ["node", str(harness), str(_SCRIPT)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout
    got = json.loads(out)
    assert got["url"] == "ws://canopy.test/hare/v1/ws/pvs?api_key=k+1"
    assert got["first"] == [
        {"type": "subscribe", "pvNames": ["A:x"]},
        {"type": "subscribe", "pvNames": ["A:y"]},
    ]
    assert got["again"] == [{"type": "subscribe", "pvNames": ["A:x", "A:y"]}]
    # a drop, or two heartbeats of silence, tells every watcher nothing is known
    assert got["seen"] == [["A:x", 1], ["A:x", None], ["A:x", 2], ["A:x", None]]
    assert got["states"] == [
        "open",
        "closed",
        "connecting",
        "open",
        "closed",
        "connecting",
    ]
    assert got["redialled"] == 3
    assert got["units"] == "mm"
