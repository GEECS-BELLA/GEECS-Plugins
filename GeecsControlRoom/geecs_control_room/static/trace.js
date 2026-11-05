// The trace page's live half, DOM only: the rail's controls and the readings
// (the family's panel; values from Hare, words and PVs the module's, through
// window.GeecsDevicePage), and the plot: the trace pulled from
// /api/devices/<name>/trace at 2 Hz while the tab is visible, one request in
// flight, If-None-Match the last ETag (the counter: a 304 is nothing new). One
// series per channel in the theme's --trace-1..4; this viewer's scale, window
// and average are its own, written nowhere. Writes go through device_writes.js
// (the module's put route, sent once). The rules are trace_rules.js's.
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsTraceRules, box = document.getElementById("trace");
  if (!P || !box) return;
  const $ = (id) => document.getElementById(id);
  const KIT = JSON.parse($("kit-state").textContent);
  const POLL_MS = 500, TIMEOUT_MS = 5000, COLOURS = ["--trace-1", "--trace-2", "--trace-3", "--trace-4"];
  const token = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
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
  const canvas = $("trace-plot");
  if (!canvas) return;
  const yscale = $("yscale"), avgIn = $("average"), fullBtn = $("window-full");
  let held = [], shown = null, etag = null, win = null, seen = [], lastTag = null, error = null, hover = null, drag = null, plot = null;
  const PAD = { l: 64, r: 12, t: 10, b: 34 };

  function draw() {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
    c.font = "11px " + (token("--ff-mono") || "monospace");
    const muted = token("--muted"), log = yscale.value === "log";
    if (!shown) { c.fillStyle = muted; c.fillText(error || "waiting for the first trace", PAD.l, h / 2); plot = null; return; }
    const [x0, x1] = R.xRange(shown.axis, win), yr = R.yRange(shown.axis, shown.series, x0, x1, log);
    const pw = w - PAD.l - PAD.r, ph = h - PAD.t - PAD.b;
    plot = { x0, x1, pw };
    c.strokeStyle = token("--rule-soft"); c.lineWidth = 1;
    c.strokeRect(PAD.l + 0.5, PAD.t + 0.5, pw - 1, ph - 1);
    const px = (x) => PAD.l + ((x - x0) / (x1 - x0)) * pw;
    const xs = R.si(Math.max(Math.abs(x0), Math.abs(x1))), xt = R.ticks(x0, x1, Math.max(2, Math.floor(pw / 90)));
    c.fillStyle = muted; c.textAlign = "center";
    for (const t of xt) { const X = px(t); c.beginPath(); c.moveTo(X, PAD.t + ph); c.lineTo(X, PAD.t + ph + 4); c.stroke(); c.fillText(R.tickLabel(t, xs.factor), X, PAD.t + ph + 15); }
    c.fillText(R.axisTitle(box.dataset.xLabel, box.dataset.xUnits, xs.prefix), PAD.l + pw / 2, h - 4);
    if (!yr) { c.textAlign = "left"; c.fillText(log ? "nothing positive to draw on a log scale" : "every channel is off or empty", PAD.l + 8, PAD.t + 16); return; }
    const ty = (v) => (log ? Math.log10(v) : v), [ya, yb] = [ty(yr[0]), ty(yr[1])];
    const py = (v) => PAD.t + ph - ((ty(v) - ya) / (yb - ya || 1)) * ph;
    const ys = R.si(Math.max(Math.abs(yr[0]), Math.abs(yr[1])));
    c.textAlign = "right";
    for (const t of log ? R.logTicks(yr[0], yr[1]) : R.ticks(yr[0], yr[1], Math.max(2, Math.floor(ph / 50)))) {
      const Y = py(t); c.strokeStyle = token("--rule-soft"); c.beginPath(); c.moveTo(PAD.l, Y); c.lineTo(PAD.l + pw, Y); c.stroke();
      c.fillText(log ? t.toExponential(0) : R.tickLabel(t, ys.factor), PAD.l - 6, Y + 4);
    }
    c.save(); c.translate(12, PAD.t + ph / 2); c.rotate(-Math.PI / 2); c.textAlign = "center";
    c.fillText(R.axisTitle(box.dataset.yLabel, box.dataset.yUnits, log ? "" : ys.prefix), 0, 0); c.restore();
    c.save(); c.beginPath(); c.rect(PAD.l, PAD.t, pw, ph); c.clip();
    shown.series.forEach((s, i) => {
      if (s.off) return;
      c.strokeStyle = token(COLOURS[i % COLOURS.length]); c.lineWidth = 1.25; c.beginPath();
      let pen = false;
      for (const p of R.decimate(shown.axis, s.values, x0, x1, Math.round(pw))) {
        if (p === null || (log && p[1] <= 0)) { pen = false; continue; }
        if (pen) c.lineTo(px(p[0]), py(p[1])); else c.moveTo(px(p[0]), py(p[1]));
        pen = true;
      }
      c.stroke();
    });
    if (drag) { c.fillStyle = token("--accent"); c.globalAlpha = 0.15; c.fillRect(Math.min(drag.a, drag.b), PAD.t, Math.abs(drag.b - drag.a), ph); c.globalAlpha = 1; }
    if (hover !== null) { c.strokeStyle = muted; c.setLineDash([3, 3]); c.beginPath(); c.moveTo(hover, PAD.t); c.lineTo(hover, PAD.t + ph); c.stroke(); }
    c.restore();
  }

  function legend() {
    const el = $("legend");
    if (!shown) { el.replaceChildren(); return; }
    el.replaceChildren(...shown.series.map((s, i) => {
      const k = document.createElement("span");
      k.className = "key"; k.dataset.n = String(i % COLOURS.length + 1);
      k.textContent = R.seriesName(i, shown.channels, box.dataset.yLabel) + (s.off ? " off" : "");
      return k;
    }));
  }

  function readout(clientX) {
    const r = canvas.getBoundingClientRect(), X = clientX - r.left;
    if (!shown || !plot || X < PAD.l || X > PAD.l + plot.pw) { hover = null; $("cursor").textContent = ""; return; }
    hover = X;
    const x = plot.x0 + ((X - PAD.l) / plot.pw) * (plot.x1 - plot.x0), i = R.nearest(shown.axis, x), xs = R.si(Math.abs(shown.axis[i]));
    const yu = box.dataset.yUnits ? ` ${box.dataset.yUnits}` : "";
    const ys = shown.series.map((s, n) => (s.off ? null : `${R.seriesName(n, shown.channels, box.dataset.yLabel)} ${R.fmt({ value: s.values[i] }, {})}${yu}`)).filter(Boolean);
    $("cursor").textContent = `${R.tickLabel(shown.axis[i], xs.factor)} ${xs.prefix}${box.dataset.xUnits} · ${ys.join(" · ")}`;
  }

  canvas.addEventListener("pointerdown", (e) => { const X = e.clientX - canvas.getBoundingClientRect().left; drag = { a: X, b: X }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => { if (drag) drag.b = e.clientX - canvas.getBoundingClientRect().left; readout(e.clientX); draw(); });
  canvas.addEventListener("pointerup", () => {
    if (drag && plot) { const next = R.windowFromDrag(drag.a - PAD.l, drag.b - PAD.l, plot.pw, plot.x0, plot.x1); if (next) { win = next; fullBtn.disabled = false; } }
    drag = null; draw();
  });
  canvas.addEventListener("pointercancel", () => { drag = null; draw(); });
  canvas.addEventListener("pointerleave", () => { hover = null; $("cursor").textContent = ""; draw(); });
  fullBtn.addEventListener("click", () => { win = null; fullBtn.disabled = true; draw(); });
  yscale.addEventListener("change", draw);
  avgIn.addEventListener("change", () => { avgIn.value = R.averageOf(avgIn.value); held = held.slice(-R.averageOf(avgIn.value)); if (held.length) shown = R.averaged(held); draw(); });
  window.addEventListener("resize", draw);

  function say(why) {
    error = why;
    $("trace-error").hidden = !why;
    $("trace-error-text").textContent = why || "";
  }

  async function poll() {
    if (document.hidden) return;
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
        legend();
      });
    } catch (e) { etag = null; say(e && e.message === "timed out" ? "the Control Room did not answer in time: the trace held" : "the Control Room did not answer"); }
    const rt = R.rate(seen, performance.now());
    seen = rt.seen;
    $("rate").textContent = rt.text;
    draw();
  }
  async function loop() { await poll(); setTimeout(loop, POLL_MS); }
  document.addEventListener("visibilitychange", () => { if (!document.hidden) etag = null; });
  loop();
})();
