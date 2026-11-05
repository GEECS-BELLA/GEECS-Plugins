// The trace page's rules, pure: no DOM, no socket, no clock. trace.js draws what
// these say; tests/test_trace_rules_js.py runs each rule under node. The answer's
// decode and reshape, this viewer's average and window, the scale and its ticks,
// the cursor, which channel group the page hides. Values and words are the
// camera page's rules (fmt, control), reused, never copied; nothing here judges
// the device. window.GeecsTraceRules in the page, module.exports under node.
(function (rules, C) {
  "use strict";
  const PREFIXES = { "-15": "f", "-12": "p", "-9": "n", "-6": "µ", "-3": "m", 0: "", 3: "k", 6: "M", 9: "G", 12: "T" };

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

  // This viewer's average: the last n captures of one shape (a new shape starts over).
  function keep(held, trace, n) {
    const same = held.filter((t) => t.channels === trace.channels && t.samples === trace.samples);
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

  // ---- the scale
  // The x range drawn: the viewer's window, else the axis' own extent.
  function xRange(axis, window) {
    if (window) return window;
    let lo = Infinity, hi = -Infinity;
    for (const x of axis) if (Number.isFinite(x)) { if (x < lo) lo = x; if (x > hi) hi = x; }
    return lo <= hi ? [lo, hi === lo ? lo + 1 : hi] : [0, 1];
  }
  // The y range over the finite (log: positive) values of the drawn series inside [x0, x1]; null when there is none.
  function yRange(axis, series, x0, x1, log) {
    let lo = Infinity, hi = -Infinity;
    for (const s of series) {
      if (s.off) continue;
      for (let i = 0; i < axis.length; i++) {
        const v = s.values[i];
        if (axis[i] < x0 || axis[i] > x1 || !Number.isFinite(v) || (log && v <= 0)) continue;
        if (v < lo) lo = v; if (v > hi) hi = v;
      }
    }
    if (!(lo <= hi)) return null;
    if (log) return lo === hi ? [lo / 10, hi * 10] : [lo, hi];
    const pad = (hi - lo) * 0.05 || Math.abs(hi) * 0.05 || 1;
    return [lo - pad, hi + pad];
  }
  // Linear ticks at 1, 2 or 5 × 10^k, about `count` of them, inside [lo, hi].
  function ticks(lo, hi, count = 6) {
    const span = hi - lo;
    if (!(span > 0)) return [lo];
    const raw = span / count, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag, out = [];
    // each tick as i × step, rounded to 12 digits: never a running sum's float noise
    for (let i = Math.ceil(lo / step - 1e-9); i <= Math.floor(hi / step + 1e-9); i++) out.push(i === 0 ? 0 : Number((i * step).toPrecision(12)));
    return out;
  }
  // Log ticks: every decade inside [lo, hi] (both > 0).
  function logTicks(lo, hi) {
    const out = [];
    for (let k = Math.ceil(Math.log10(lo) - 1e-9); k <= Math.floor(Math.log10(hi) + 1e-9); k++) out.push(Math.pow(10, k));
    return out;
  }
  // The SI prefix that shows values up to `maxAbs` with 1..999 before the point: its factor and letter.
  function si(maxAbs) {
    if (!(maxAbs > 0) || !Number.isFinite(maxAbs)) return { factor: 1, prefix: "" };
    const k = Math.min(12, Math.max(-15, 3 * Math.floor(Math.log10(maxAbs) / 3 + 1e-9)));
    return { factor: Math.pow(10, k), prefix: PREFIXES[k] };
  }
  // A tick's label in the scale's units: up to 4 significant digits, no float noise.
  const tickLabel = (v, factor) => String(Number((v / factor).toPrecision(4)));
  // An axis's title: its label and units, prefixed ("time axis (µs)", "capture (mV)", "samples").
  const axisTitle = (label, units, prefix) => (units ? `${label} (${prefix}${units})` : label);

  // ---- drawing and the cursor
  // At most two points per pixel column inside [x0, x1] (each column's min and max, in order); NaN breaks the line (null).
  function decimate(axis, values, x0, x1, columns) {
    const pts = [];
    let col = -1, lo = null, hi = null, gap = false;
    const flush = () => {
      if (lo === null) return;
      const [a, b] = lo[2] <= hi[2] ? [lo, hi] : [hi, lo];
      pts.push([a[0], a[1]]);
      if (b[2] !== a[2]) pts.push([b[0], b[1]]); // one sample both min and max: drawn once
      lo = hi = null;
    };
    for (let i = 0; i < axis.length; i++) {
      const x = axis[i], v = values[i];
      if (x < x0 || x > x1) continue;
      if (!Number.isFinite(v)) { flush(); if (!gap) pts.push(null); gap = true; continue; }
      gap = false;
      const c = Math.min(columns - 1, Math.floor(((x - x0) / (x1 - x0 || 1)) * columns));
      if (c !== col) { flush(); col = c; }
      if (lo === null || v < lo[1]) lo = [x, v, i];
      if (hi === null || v > hi[1]) hi = [x, v, i];
    }
    flush();
    return pts;
  }
  // The sample nearest x on an increasing axis (binary search); -1 for an empty one.
  function nearest(axis, x) {
    if (!axis.length) return -1;
    let a = 0, b = axis.length - 1;
    while (b - a > 1) { const m = (a + b) >> 1; if (axis[m] <= x) a = m; else b = m; }
    return Math.abs(axis[a] - x) <= Math.abs(axis[b] - x) ? a : b;
  }
  // The viewer's window from a drag between two pixel x's over [x0, x1] drawn in `width` px; null for a click (< 4 px).
  function windowFromDrag(px0, px1, width, x0, x1) {
    if (Math.abs(px1 - px0) < 4 || !(width > 0)) return null;
    const at = (px) => x0 + (Math.min(width, Math.max(0, px)) / width) * (x1 - x0);
    return [at(Math.min(px0, px1)), at(Math.max(px0, px1))];
  }
  // A series' name: the trace's label for one channel, else the channel's letter (the scope contract's A, B, …).
  const seriesName = (i, channels, label) => (channels === 1 ? label : String.fromCharCode(65 + i));

  // ---- the rail and the readings
  // The role whose value 0 hides a group: `<ch>_present` when every role of the group is `<ch>_…`
  // (the scope contract's per-channel roles) and the device has it; else none.
  function presentRole(roles, known) {
    const m = roles.length ? /^(ch_[a-z0-9]+)_/.exec(roles[0]) : null;
    if (!m || !roles.every((r) => r.startsWith(m[1] + "_"))) return null;
    const present = `${m[1]}_present`;
    return known.includes(present) && !roles.includes(present) ? present : null;
  }
  // A group (or a reading) is hidden only while its present role reads 0: unknown keeps it shown.
  const hidden = (frame) => !!frame && frame.value === 0;
  // The rate chip: captures per second from the tags seen in the last `spanMs`.
  function rate(seen, now, spanMs = 5000) {
    const recent = seen.filter((t) => now - t <= spanMs);
    return { seen: recent, text: recent.length ? `${(recent.length / (spanMs / 1000)).toFixed(1)} new/s` : "no new captures" };
  }

  Object.assign(rules, { decodeTrace, keep, averaged, averageOf, xRange, yRange, ticks, logTicks, si, tickLabel, axisTitle, decimate, nearest, windowFromDrag, seriesName, presentRole, hidden, rate, fmt: C.fmt, control: C.control, rawAnswer: C.rawAnswer, ifNoneMatch: C.ifNoneMatch, within: C.within, whole: C.whole, number: C.number });
})(
  typeof module === "object" ? module.exports : (window.GeecsTraceRules = {}),
  typeof module === "object" ? require("./camera_rules.js") : window.GeecsCameraRules,
);
