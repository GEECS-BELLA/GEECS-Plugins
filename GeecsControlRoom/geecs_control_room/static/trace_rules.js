// The trace page's rules, pure: no DOM, no socket, no clock. trace.js hands what
// these say to Plotly through the theme's template (plot.js); tests/test_trace_rules_js.py
// runs each rule under node. The answer's decode and reshape, this viewer's average
// and window, the traces Plotly is given (decimated, a log scale's positive extrema),
// the axes' titles (their number format is the theme's GeecsPlot.axisFormat), which
// channel group the page hides. Values and words are the
// camera page's rules (fmt, control), reused, never copied; nothing here judges
// the device. window.GeecsTraceRules in the page, module.exports under node.
(function (rules, C) {
  "use strict";

  // The route's 200: float32 LE, the axis (samples) then y (channels × samples, channels fastest).
  // Each channel's series, and `off` for one all NaN (a disabled scope channel).
  function decodeTrace(buffer, h) {
    const channels = Number(h.channels), samples = Number(h.samples);
    if (!(channels >= 1) || !(samples >= 1)) return { ok: false, why: "the trace came without its shape" };
    if (buffer.byteLength !== 4 * samples * (channels + 1)) return { ok: false, why: `shape disagrees: ${buffer.byteLength} bytes for ${channels}×${samples}` };
    const all = new Float32Array(buffer), axis = all.subarray(0, samples), series = [];
    for (let c = 0; c < channels; c++) {
      const v = new Float32Array(samples);
      let finite = false;
      for (let s = 0; s < samples; s++) { v[s] = all[samples + s * channels + c]; finite = finite || Number.isFinite(v[s]); }
      series.push({ values: v, off: !finite });
    }
    return { ok: true, channels, samples, axis, series, tag: h.tag ?? null };
  }

  // This viewer's average: the last n captures of one shape and one axis (a new shape or a moved axis, a
  // new interval or pre-trigger, starts over). A capture already held (the same tag, asked again after an
  // error or a hidden tab) is not counted twice; an untagged one always is.
  const sameAxis = (a, b) => a.length === b.length && a[0] === b[0] && a[1] === b[1] && a[a.length - 1] === b[b.length - 1];
  function keep(held, trace, n) {
    if (trace.tag !== null && held.length && held[held.length - 1].tag === trace.tag) return held;
    const same = held.filter((t) => t.channels === trace.channels && t.samples === trace.samples && sameAxis(t.axis, trace.axis));
    return [...same, trace].slice(-Math.max(1, n));
  }
  // The mean of the held captures, sample by sample, over the finite values only (NaN where none is).
  function averaged(held) {
    const last = held[held.length - 1];
    if (held.length < 2) return last;
    const series = last.series.map((s, c) => {
      const v = new Float32Array(last.samples);
      for (let i = 0; i < last.samples; i++) {
        let sum = 0, n = 0;
        for (const t of held) { const x = t.series[c].values[i]; if (Number.isFinite(x)) { sum += x; n++; } }
        v[i] = n ? sum / n : NaN;
      }
      return { values: v, off: s.off };
    });
    return { ...last, series };
  }
  // The average count typed: a whole number in 1..100, else 1.
  const averageOf = (raw) => { const n = Math.round(Number(raw)); return Number.isFinite(n) ? Math.min(100, Math.max(1, n)) : 1; };

  // ---- what the plot is given (plot.js and Plotly draw it)
  // The x range decimated over: the viewer's window, else the axis' own extent.
  function xRange(axis, window) {
    if (window) return window;
    let lo = Infinity, hi = -Infinity;
    for (const x of axis) if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; }
    return lo <= hi ? [lo, hi === lo ? lo + 1 : hi] : [0, 1];
  }
  // At most two points per pixel column inside [x0, x1] (each column's min and max, in order); NaN breaks the line (null).
  // On a log scale (`log`) a column's extrema are over its positive samples only, and a column with none is a gap:
  // the signed extrema of a bipolar trace would otherwise all be dropped by the drawing, leaving nothing.
  function decimate(axis, values, x0, x1, columns, log = false) {
    const pts = [];
    let col = -1, lo = null, hi = null, skipped = false;
    const gap = () => { if (pts.length && pts[pts.length - 1] !== null) pts.push(null); };
    const flush = () => {
      if (lo === null) { if (skipped) gap(); skipped = false; return; }
      const [a, b] = lo[2] <= hi[2] ? [lo, hi] : [hi, lo];
      pts.push([a[0], a[1]]);
      if (b[2] !== a[2]) pts.push([b[0], b[1]]); // one sample both min and max: drawn once
      lo = hi = null;
      skipped = false;
    };
    for (let i = 0; i < axis.length; i++) {
      const x = axis[i], v = values[i];
      if (x < x0 || x > x1) continue;
      if (!Number.isFinite(v)) { flush(); if (!pts.length || pts[pts.length - 1] !== null) pts.push(null); continue; }
      const c = Math.min(columns - 1, Math.floor(((x - x0) / (x1 - x0 || 1)) * columns));
      if (c !== col) { flush(); col = c; }
      if (log && v <= 0) { skipped = true; continue; }
      if (lo === null || v < lo[1]) lo = [x, v, i];
      if (hi === null || v > hi[1]) hi = [x, v, i];
    }
    flush();
    return pts;
  }
  // A series' name: the trace's label for one channel, else the channel's letter (the scope contract's A, B, …).
  const seriesName = (i, channels, label) => (channels === 1 ? label : String.fromCharCode(65 + i));


  // An axis's title: its label and its units as the family names them ("time axis (s)", "spectrum (counts)", "samples").
  const axisTitle = (label, units) => (units ? `${label} (${units})` : label);
  // The viewer's window from a Plotly relayout event: [lo, hi] for an x zoom or pan, null for an autoscale or a
  // reset (the whole trace), undefined for an event that leaves the x axis alone (a y-only zoom, a legend click).
  function windowFromRelayout(ev) {
    if (!ev) return undefined;
    if (ev["xaxis.autorange"]) return null;
    const r = Array.isArray(ev["xaxis.range"]) ? ev["xaxis.range"] : [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
    const lo = Number(r[0]), hi = Number(r[1]);
    return r[0] === undefined || r[1] === undefined || !Number.isFinite(lo) || !Number.isFinite(hi) ? undefined : [Math.min(lo, hi), Math.max(lo, hi)];
  }
  // The Plotly traces for the shown capture inside [x0, x1]: one per channel, in channel order (so trace i keeps the
  // theme's --trace-(i+1) whether or not its neighbours are on), each decimated to `columns` (a NaN or a log-scale
  // column with nothing positive is a gap: null); an off channel is one null point named "<name> off" (Plotly lists a
  // trace with a point in the legend, never one with none). `drawn` counts the points given, so the page can say why
  // the plot is empty.
  function plotData(shown, x0, x1, columns, log, label) {
    let drawn = 0;
    const traces = shown.series.map((s, i) => {
      const name = seriesName(i, shown.channels, label), x = s.off ? [null] : [], y = s.off ? [null] : [];
      if (!s.off) for (const p of decimate(shown.axis, s.values, x0, x1, columns, log)) {
        if (p === null) { x.push(null); y.push(null); continue; }
        x.push(p[0]); y.push(p[1]); drawn++;
      }
      return { type: "scatter", mode: "lines", name: s.off ? `${name} off` : name, x, y, connectgaps: false, line: { width: 1.25 } };
    });
    return { traces, drawn };
  }
  // Why the plot is empty, in words ("" while it draws something).
  function emptyWhy(shown, drawn, log) {
    if (!shown) return "waiting for the first trace";
    if (drawn) return "";
    return shown.series.every((s) => s.off) || !log ? "every channel is off or empty" : "nothing positive to draw on a log scale";
  }

  // ---- which array the plot shows
  // The viewer's choice when made and still drawn, else the first array whose `show_while` role reads 1 (a
  // processed spectrum while background subtraction is on), else the axis' first. `ys` the template's
  // [{role, …}], `showWhile` the panel's {array: role}, `values` {role: its value} (absent: unknown).
  function shownY(ys, showWhile, values, chosen) {
    if (ys.some((y) => y.role === chosen)) return chosen;
    const on = ys.find((y) => showWhile[y.role] && values[showWhile[y.role]] === 1);
    return on ? on.role : ys.length ? ys[0].role : null;
  }
  const traceUrl = (base, y) => (y ? `${base}?y=${encodeURIComponent(y)}` : base);

  // The rate chip: captures per second over the new tags seen in the last `spanMs`, first to last (never a count
  // over the whole span, which a 1 Hz source reads as 1.0 or 1.2 by where the span falls).
  function rate(seen, now, spanMs = 5000) {
    const recent = seen.filter((t) => now - t <= spanMs), n = recent.length;
    const dt = n > 1 ? (recent[n - 1] - recent[0]) / 1000 : 0;
    const text = n > 1 && dt > 0 ? `${((n - 1) / dt).toFixed(1)} new/s` : n ? `1 new in ${spanMs / 1000} s` : "no new captures";
    return { seen: recent, text };
  }

  Object.assign(rules, { decodeTrace, keep, averaged, averageOf, xRange, decimate, seriesName, axisTitle, windowFromRelayout, plotData, emptyWhy, shownY, traceUrl, rate, rawAnswer: C.rawAnswer, ifNoneMatch: C.ifNoneMatch, within: C.within });
})(
  typeof module === "object" ? module.exports : (window.GeecsTraceRules = {}),
  typeof module === "object" ? require("./camera_rules.js") : window.GeecsCameraRules,
);
