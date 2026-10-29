// Hare's live PV stream (Canopy /hare/v1/ws/pvs), one socket per page.
//
// The protocol as given: `initial` (the only frame with `meta`) answers a
// subscribe, `diff` carries changed values at most every 100 ms and may
// precede `initial`, `heartbeat` every 5 s. There is no resume: after any
// drop a new socket subscribes again to everything wanted. The read key
// travels as `?api_key=` (a browser socket sends no headers).
//
// GeecsHare.fromPage() (or .open({url, key})) -> hare; hare.watch(pvs, fn(pv, wire, meta));
// hare.onStatus(fn("connecting" | "open" | "closed")). While the socket is
// down nothing is known: `wire` is null for every watched PV.
(function () {
  "use strict";
  const SILENT_MS = 12000; // two heartbeats missed: the socket is dead
  const BACKOFF_MS = [500, 1000, 2000, 5000];

  function open({ url, key, WebSocketImpl, silentMs = SILENT_MS }) {
    const WS = WebSocketImpl || window.WebSocket;
    const target = new URL(url, window.location.href);
    target.protocol = target.protocol === "https:" ? "wss:" : "ws:";
    target.searchParams.set("api_key", key);
    const watchers = new Map(); // pv -> Set(fn)
    const meta = {}, latest = {};
    const statusFns = new Set();
    let ws = null, tries = 0, timer = null, watchdog = null;

    const tell = (pv) => {
      for (const fn of watchers.get(pv) || []) fn(pv, latest[pv] || null, meta[pv] || {});
    };
    const status = (s) => { for (const fn of statusFns) fn(s); };
    const send = (msg) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); };

    function heard() {
      clearTimeout(watchdog);
      watchdog = setTimeout(drop, silentMs);
    }

    // The socket is gone, or silent for two heartbeats: forget every value at
    // once (a half-open socket's close event can take a minute) and redial.
    function drop() {
      clearTimeout(watchdog);
      if (ws) {
        ws.onopen = ws.onmessage = ws.onclose = null;
        try { ws.close(); } catch (_) { /* already closed */ }
      }
      ws = null;
      for (const pv of Object.keys(latest)) delete latest[pv];
      for (const pv of watchers.keys()) tell(pv);
      status("closed");
      timer = setTimeout(connect, BACKOFF_MS[Math.min(tries++, BACKOFF_MS.length - 1)]);
    }

    function connect() {
      status("connecting");
      ws = new WS(target.toString());
      heard(); // a dial into nothing is dropped like a silent socket
      ws.onopen = () => {
        tries = 0;
        status("open");
        heard();
        if (watchers.size) send({ type: "subscribe", pvNames: [...watchers.keys()] });
      };
      ws.onmessage = (event) => {
        heard();
        let frame;
        try { frame = JSON.parse(event.data); } catch (_) { return; }
        if (frame.meta) Object.assign(meta, frame.meta);
        if (frame.type === "initial" || frame.type === "diff") {
          for (const [pv, wire] of Object.entries(frame.data || {})) {
            latest[pv] = wire;
            tell(pv);
          }
        }
      };
      ws.onclose = drop;
    }

    const hare = {
      watch(pvs, fn) {
        const fresh = pvs.filter((pv) => !watchers.has(pv));
        for (const pv of pvs) {
          if (!watchers.has(pv)) watchers.set(pv, new Set());
          watchers.get(pv).add(fn);
          if (latest[pv]) fn(pv, latest[pv], meta[pv] || {});
        }
        if (fresh.length) send({ type: "subscribe", pvNames: fresh });
      },
      onStatus(fn) { statusFns.add(fn); },
      meta(pv) { return meta[pv] || {}; },
      close() { clearTimeout(timer); clearTimeout(watchdog); if (ws) { ws.onclose = null; ws.close(); } },
    };
    connect();
    return hare;
  }

  // The page's own socket, from what base.html put on <body>.
  function fromPage() {
    const d = document.body.dataset;
    return open({ url: d.hareUrl, key: d.hareKey });
  }

  window.GeecsHare = { open, fromPage };
})();
