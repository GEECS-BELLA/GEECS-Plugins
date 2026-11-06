// The trace page's live half, DOM only: the rail's controls and the readings
// (the family's panel; values from Hare, words and PVs the module's, through
// window.GeecsDevicePage), and the plot: the trace pulled from
// /api/devices/<name>/trace at 2 Hz while the tab is visible, one request in
// flight, If-None-Match the last ETag (the counter: a 304 is nothing new), drawn
// by Plotly through the theme's template (window.GeecsPlot: one trace per
// channel in --trace-1..n, the modebar, the hover readout, the repaint on a theme
// change). This viewer's scale, window (Plotly's x range) and average are its
// own, written nowhere. Writes go through device_writes.js (the module's put
// route, sent once). The rules are trace_rules.js's.
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsTraceRules, box = document.getElementById("trace");
  if (!P || !box) return;
  const $ = (id) => document.getElementById(id);
  const KIT = JSON.parse($("kit-state").textContent);
  const POLL_MS = 500, TIMEOUT_MS = 5000;
  const value = (role) => { const f = P.frame(role); return f ? f.value : undefined; };
  const focused = (el) => document.activeElement === el;
  const known = () => { const v = P.view(); return v && v.ok ? Object.values(v.roles).map((e) => e.role) : []; };

  // ---- the rail and the readings: filled unless focused, enabled on the module's word, a channel's group hidden while absent
  const writes = window.GeecsDeviceWrites.make(P, () => fill());
  const controls = [...box.querySelectorAll(".rail [data-role]")];
  function fill() {
    for (const el of controls) {
      const role = el.dataset.role, c = R.control(P.view(), P.entry(role), writes.busy()), v = value(role);
      el.disabled = !c.enabled;
      el.title = c.why;
      if (el.tagName === "BUTTON") continue;
      if (el.type === "checkbox") { el.checked = !!v; continue; }
      if (el.tagName === "SELECT") {
        const choices = P.meta(role).enumChoices || []; // the PV's own choices, never a list of the page's
        if (el.options.length !== choices.length) el.replaceChildren(...choices.map((t, i) => new Option(t, String(i))));
        if (!focused(el) && typeof v === "number") el.value = String(v);
        continue;
      }
      if (!focused(el) && typeof v === "number") el.value = Number.isInteger(v) ? v : Number(v.toPrecision(6));
    }
    for (const el of box.querySelectorAll("[data-rb]")) el.textContent = `now ${R.fmt(P.frame(el.dataset.rb), P.meta(el.dataset.rb))}`;
    for (const el of box.querySelectorAll("[data-ro]")) el.textContent = R.fmt(P.frame(el.dataset.ro), P.meta(el.dataset.ro));
    const names = known();
    for (const g of box.querySelectorAll(".rail .group")) {
      const present = R.presentRole(g.dataset.roles.split(" "), names);
      g.hidden = !!present && R.hidden(P.frame(present));
    }
    for (const el of box.querySelectorAll(".readings .live")) {
      const role = el.dataset.role, present = R.presentRole([role], names), word = P.word(role);
      el.hidden = !!present && R.hidden(P.frame(present));
      el.querySelector(".v").textContent = R.fmt(P.frame(role), P.meta(role));
      el.querySelector(".dot").dataset.state = KIT[word] || "unknown";
      el.querySelector(".word").textContent = word;
    }
    const v = P.view();
    $("alarms").textContent = v && v.ok ? Object.values(v.roles).filter((e) => e.verdict === "error").map((e) => e.label || e.role).join(" · ") : "";
    const counter = box.dataset.counter, word = counter ? P.word(counter) : "unknown";
    const chip = $("trace-state");
    if (chip) { chip.dataset.state = KIT[word] || "unknown"; chip.textContent = `trace ${word}`; }
  }
  for (const el of controls) {
    if (el.tagName === "BUTTON") { el.addEventListener("click", () => writes.write([[el.dataset.role, Number(el.dataset.value)]])); continue; }
    el.addEventListener("change", () => {
      const v = el.type === "checkbox" ? (el.checked ? 1 : 0) : el.tagName === "SELECT" ? Number(el.value) : el.step === "1" ? R.whole(el.value) : R.number(el.value);
      if (v !== null) writes.write([[el.dataset.role, v]]).then(fill);
    });
  }
  P.on(fill);
  fill();

  // ---- the plot
  const host = $("trace-plot"), G = window.GeecsPlot;
  if (!host || !G || !window.Plotly) return;
  const yscale = $("yscale"), avgIn = $("average"), note = $("plot-note");
  let held = [], shown = null, etag = null, win = null, seen = [], lastTag = null, shape = null, watching = false;
  const ds = box.dataset, xFmt = G.axisFormat(ds.xUnits), yFmt = G.axisFormat(ds.yUnits);

  // Plotly.react with the template's layout under the page's axes. `uirevision` keeps the viewer's zoom, legend
  // toggles and hover across captures; x's revision is the capture's shape (a new shape is the whole trace again),
  // y's is the scale (a log/linear change rescales y).
  function draw() {
    const log = yscale.value === "log";
    let drawn = 0;
    if (shown) {
      const now = `${shown.channels}×${shown.samples}`;
      if (shape !== null && now !== shape) win = null;
      shape = now;
      const [x0, x1] = R.xRange(shown.axis, win), cols = Math.max(200, Math.round(host.clientWidth || 800));
      const d = R.plotData(shown, x0, x1, cols, log, ds.yLabel);
      drawn = d.drawn;
      window.Plotly.react(host, d.traces, G.layout({
        uirevision: "trace", showlegend: shown.channels > 1,
        xaxis: { title: { text: R.axisTitle(ds.xLabel, ds.xUnits) }, ...xFmt, uirevision: shape },
        yaxis: { title: { text: R.axisTitle(ds.yLabel, ds.yUnits) }, type: log ? "log" : "linear", ...yFmt, uirevision: yscale.value },
      }), G.config());
      if (!watching) {
        watching = true;
        // the window is Plotly's x range: a box zoom or a pan narrows it and the trace is decimated inside it again
        // (every sample there shown); autoscale, reset or a double-click is the whole trace
        host.on("plotly_relayout", (ev) => { const w = R.windowFromRelayout(ev); if (w !== undefined) { win = w; draw(); } });
        G.watch(host);
      }
    }
    const why = R.emptyWhy(shown, drawn, log);
    note.textContent = why;
    note.hidden = !why;
  }
  yscale.addEventListener("change", draw);
  avgIn.addEventListener("change", () => { avgIn.value = R.averageOf(avgIn.value); held = held.slice(-R.averageOf(avgIn.value)); if (held.length) shown = R.averaged(held); draw(); });

  function say(why) {
    $("trace-error").hidden = !why;
    $("trace-error-text").textContent = why || "";
  }

  async function poll() {
    if (document.hidden) return;
    let fresh = false;
    try {
      await R.within(TIMEOUT_MS, async (signal) => {
        const r = await fetch(box.dataset.traceUrl, { cache: "no-store", headers: R.ifNoneMatch(etag), signal });
        const what = R.rawAnswer(r.status);
        if (what === "same") return;
        if (what === "error") { etag = null; say((await r.json().catch(() => ({}))).detail || `answered ${r.status}`); return; }
        const h = { channels: r.headers.get("X-Trace-Channels"), samples: r.headers.get("X-Trace-Samples"), tag: r.headers.get("X-Trace-Tag") };
        const t = R.decodeTrace(await r.arrayBuffer(), h);
        if (!t.ok) { etag = null; say(t.why); return; }
        say(null);
        etag = r.headers.get("ETag");
        if (t.tag !== lastTag) { seen.push(performance.now()); lastTag = t.tag; }
        held = R.keep(held, t, R.averageOf(avgIn.value));
        shown = R.averaged(held);
        fresh = true;
      });
    } catch (e) { etag = null; say(e && e.message === "timed out" ? "the Control Room did not answer in time: the trace held" : "the Control Room did not answer"); }
    const rt = R.rate(seen, performance.now());
    seen = rt.seen;
    $("rate").textContent = rt.text;
    if (fresh) draw(); // outside the request's try: a drawing fault is never reported as the server not answering
  }
  async function loop() { await poll(); setTimeout(loop, POLL_MS); }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) etag = null; });
  draw();
  loop();
})();
