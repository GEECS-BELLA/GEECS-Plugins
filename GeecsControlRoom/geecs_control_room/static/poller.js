// One poll loop for a page's targets (a screen's devices, the home's reads).
//
// makePoller({fetch(target, signal), pollMs, onView(target, view)}) -> {tick(targets)}.
// view is {ok: true, value} or {ok: false, error}, with `slow` while the target backs off.
// - One request in flight per target: a tick skips a target still waiting.
// - A sequence number per target: an answer older than the one shown is dropped.
// - A timeout per request (1.5 × the period): the request is aborted and reads as a failure.
// - Back-off on consecutive failures: period × 2 per failure, at most 10 s; a success resets it.
// - At most 8 requests at once across the page; the rest queue.
// window.GeecsPoller in the page, module.exports under node.
(function (exports) {
  "use strict";
  const MAX_BACKOFF_MS = 10000, CONCURRENT = 8;

  function makePoller({ fetch, pollMs, onView, timeoutMs = 1.5 * pollMs, concurrent = CONCURRENT, now = Date.now }) {
    const slots = new Map(); // target -> {waiting, issued, shown, failures, nextAt}
    const queue = [];
    let active = 0;

    function slot(t) {
      if (!slots.has(t)) slots.set(t, { waiting: false, issued: 0, shown: 0, failures: 0, nextAt: 0 });
      return slots.get(t);
    }

    function settle(t, seq, view) {
      const s = slot(t);
      if (seq <= s.shown) return; // late: a newer answer, or this one's timeout, is already shown
      s.shown = seq;
      s.failures = view.ok ? 0 : s.failures + 1;
      s.nextAt = view.ok ? 0 : now() + Math.min(pollMs * 2 ** s.failures, MAX_BACKOFF_MS);
      onView(t, { ...view, slow: s.failures > 0 });
    }

    function start(t) {
      const s = slot(t), seq = ++s.issued, ctl = new AbortController();
      let open = true;
      active++;
      const finish = (view) => {
        if (open) { open = false; clearTimeout(timer); s.waiting = false; active--; }
        settle(t, seq, view);
        pump();
      };
      const timer = setTimeout(() => { ctl.abort(); finish({ ok: false, error: "timed out" }); }, timeoutMs);
      // the abandoned request may still answer later: settle drops it by its sequence number
      Promise.resolve()
        .then(() => fetch(t, ctl.signal))
        .then((value) => finish({ ok: true, value }), (err) => finish({ ok: false, error: String((err && err.message) || err) }));
    }

    function pump() {
      while (active < concurrent && queue.length) start(queue.shift());
    }

    function tick(targets) {
      const t0 = now();
      for (const t of targets) {
        const s = slot(t);
        if (s.waiting || t0 < s.nextAt) continue;
        s.waiting = true;
        queue.push(t);
      }
      pump();
    }

    return { tick };
  }

  exports.makePoller = makePoller;
})(typeof module === "object" ? module.exports : (window.GeecsPoller = {}));
