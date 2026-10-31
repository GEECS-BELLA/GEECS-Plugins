"""poller.js under node with a fake fetch whose answers the harness releases (skipped without node)."""

from __future__ import annotations

from typing import Any

import pytest

from tests._node import needs_node, run

_HARNESS = r"""
const { makePoller } = require(process.argv[1]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const flush = () => sleep(0);

// a fetch that answers only when told: calls[i].answer(v) / .fail(e)
function rig(opts = {}) {
  const clock = [0], calls = [], views = [];
  let live = 0, peak = 0;
  const poller = makePoller({
    pollMs: 100, timeoutMs: 50, now: () => clock[0], ...opts,
    onView: (t, v) => views.push([t, v.ok ? v.value : v.error, v.slow]),
    fetch: (t, signal) => new Promise((ok, no) => {
      live++; peak = Math.max(peak, live);
      const done = (f) => (x) => { live--; f(x); };
      calls.push({ t, signal, answer: done(ok), fail: done(no) });
    }),
  });
  return { poller, clock, calls, views, peak: () => peak };
}

(async () => {
  const out = {};

  // in flight: a second tick while the first request waits sends nothing
  let r = rig();
  r.poller.tick(["a"]); await flush();
  r.poller.tick(["a"]); await flush();
  out.inFlight = r.calls.length;

  // late: a timed-out answer that arrives after a newer one is dropped
  r = rig();
  r.poller.tick(["a"]); await flush();
  await sleep(70);                                   // the first times out
  r.clock[0] = 1000;                                 // past its back-off
  r.poller.tick(["a"]); await flush();
  r.calls[1].answer("new"); await flush();
  r.calls[0].answer("old"); await flush();           // the abandoned one answers late
  out.late = { views: r.views, aborted: r.calls[0].signal.aborted };

  // back-off: period × 2 per failure, capped at 10 s, reset by a success
  r = rig();
  const gaps = [];
  for (let n = 0; n < 10; n++) {
    const sent = r.calls.length, from = r.clock[0];
    while (r.calls.length === sent) { r.clock[0] += 50; r.poller.tick(["a"]); await flush(); }
    gaps.push(r.clock[0] - from);
    r.calls[r.calls.length - 1][n < 8 ? "fail" : "answer"]("x"); await flush();
  }
  out.gaps = gaps;
  out.slow = r.views.map((v) => v[2]);

  // concurrency: 31 targets, at most 8 at once, every one served
  r = rig({ timeoutMs: 10000 });
  const many = Array.from({ length: 31 }, (_, i) => `d${i}`);
  r.poller.tick(many); await flush();
  const first = r.calls.length;
  for (let i = 0; i < r.calls.length; i++) { r.calls[i].answer(i); await flush(); }
  out.concurrency = { first, peak: r.peak(), served: new Set(r.views.map((v) => v[0])).size };

  console.log(JSON.stringify(out));
  process.exit(0);
})();
"""


@pytest.fixture(scope="module")
def got() -> Any:
    return run("poller.js", _HARNESS)


@needs_node
def test_a_target_still_waiting_is_skipped(got: Any) -> None:
    assert got["inFlight"] == 1


@needs_node
def test_a_late_answer_never_overwrites_a_newer_one(got: Any) -> None:
    assert got["late"]["views"] == [["a", "timed out", True], ["a", "new", False]]
    assert got["late"]["aborted"] is True


@needs_node
def test_failures_back_off_doubling_to_ten_seconds_and_a_success_resets(
    got: Any,
) -> None:
    # the first request at the first tick; then 200, 400 … capped at 10 s; after a success, the next tick
    assert got["gaps"] == [50, 200, 400, 800, 1600, 3200, 6400, 10000, 10000, 50]
    assert got["slow"] == [True] * 8 + [False, False]


@needs_node
def test_at_most_eight_requests_at_once_and_every_target_served(got: Any) -> None:
    assert got["concurrency"] == {"first": 8, "peak": 8, "served": 31}
