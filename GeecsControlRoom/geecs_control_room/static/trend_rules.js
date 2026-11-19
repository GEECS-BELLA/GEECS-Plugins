// The settables page's trend rules, pure: no DOM, no socket, no clock (every time is handed in, in
// milliseconds since the epoch). trend.js hands what these say to Plotly through the theme's template
// (plot.js); tests/test_trend_rules_js.py runs each rule under node. A point is [t, value], the value null
// for a gap: a role Hare gives no frame for, a disconnected or INVALID record, a value that is no finite
// number. Nothing here judges the device, averages or fills a value: a line is the values the page was
// sent, each held until the next (a monitored record posts on change), thinned to the plot's width.
// window.GeecsTrendRules in the page, module.exports under node.
(function (rules, C) {
  "use strict";
  // The most points kept per role (the oldest dropped): at Hare's fastest, a diff every 100 ms, a little
  // over half an hour; at a loop's 1 Hz, more than the longest window.
  const CAP = 20000;

  const finite = (v) => typeof v === "number" && Number.isFinite(v);

  // One of Hare's frames as a point at `nowMs`: its value at the record's own stamp (tsNs; never after now,
  // and now when the frame carries none), or a gap at now.
  function point(frame, nowMs) {
    if (!frame || frame.connected === false || frame.severity === "INVALID" || !finite(frame.value)) return [nowMs, null];
    const t = Number(frame.tsNs) / 1e6;
    return [Number.isFinite(t) && t > 0 ? Math.min(t, nowMs) : nowMs, frame.value];
  }

  // The live buffer with `p` added, in place (returned): never a leading gap (before the first value nothing
  // is known, and the history says the rest), never a second gap after a gap or the same sample twice; a
  // stamp older than the last point (the first frame after a reconnect carries the record's old stamp) is
  // drawn when it arrived, so the gap before it keeps its width. Capped: the oldest dropped.
  function append(buffer, p, nowMs, cap = CAP) {
    const last = buffer[buffer.length - 1];
    if (p[1] === null && (!last || last[1] === null)) return buffer;
    if (last && p[1] === last[1] && p[0] <= last[0]) return buffer;
    buffer.push([last && p[0] <= last[0] ? Math.max(nowMs, last[0]) : p[0], p[1]]);
    if (buffer.length > cap) buffer.splice(0, buffer.length - cap);
    return buffer;
  }

  // Vole's samples as the route gives them ([[ms, value | null], …]): the well-formed ones, oldest first.
  function past(samples) {
    const ok = (Array.isArray(samples) ? samples : []).filter((s) => Array.isArray(s) && finite(s[0]) && (s[1] === null || finite(s[1])));
    return ok.map((s) => [s[0], s[1]]).sort((a, b) => a[0] - b[0]);
  }

  // One line's points: the history, then the live points after its last sample (the history's newest
  // sample and the first live frame are one sample). A live buffer at its cap has dropped its oldest
  // points, so what lies between the history and it is no longer known: the line is then the live points
  // alone, never the history's last value held across the hole.
  function merge(history, live, cap = CAP) {
    if (live.length >= cap) return live.slice(-cap);
    const lastT = history.length ? history[history.length - 1][0] : -Infinity;
    return [...history, ...live.filter((p) => p[0] > lastT)].slice(-cap);
  }

  // Whether the line runs on to now: the newest live point is a value (Hare still holds the record at it).
  const holding = (live) => live.length > 0 && live[live.length - 1][1] !== null;

  // The points drawn inside [fromMs, toMs]: the value held on entering the window, the points inside it,
  // and with `hold` the last value carried to toMs; thinned to at most four points per pixel column (its
  // first, lowest, highest and last, in order, so a held level is the column's last value, never its
  // extreme); a gap is one null. `drawn` counts the values.
  function series(points, fromMs, toMs, hold, columns) {
    const inside = [];
    let before = null;
    for (const [t, v] of points) {
      if (t < fromMs) before = v;
      else if (t <= toMs) inside.push([t, v]);
    }
    if (before !== null) inside.unshift([fromMs, before]);
    const last = inside[inside.length - 1];
    if (hold && last && last[1] !== null && last[0] < toMs) inside.push([toMs, last[1]]);
    const x = [], y = [];
    let drawn = 0, col = -1, held = [];
    const flush = () => {
      if (!held.length) return;
      let lo = 0, hi = 0;
      held.forEach((p, i) => { if (p[1] < held[lo][1]) lo = i; if (p[1] > held[hi][1]) hi = i; });
      for (const i of [...new Set([0, lo, hi, held.length - 1])].sort((a, b) => a - b)) { x.push(held[i][0]); y.push(held[i][1]); drawn++; }
      held = [];
    };
    for (const p of inside) {
      if (p[1] === null) { flush(); col = -1; if (x.length && y[y.length - 1] !== null) { x.push(p[0]); y.push(null); } continue; }
      const c = Math.min(columns - 1, Math.floor(((p[0] - fromMs) / (toMs - fromMs || 1)) * columns));
      if (c !== col) { flush(); col = c; }
      held.push(p);
    }
    flush();
    return { x, y, drawn };
  }

  // A time as Plotly's date axis reads it, in the viewer's wall clock: "2026-11-18 14:03:07.250".
  // `offsetMin` is Date.getTimezoneOffset() at that time (minutes behind UTC). A gap's null stays null.
  function stampText(ms, offsetMin) {
    return ms === null ? null : new Date(ms - offsetMin * 60000).toISOString().replace("T", " ").replace("Z", "");
  }

  // The Plotly traces of one trend: one per role, in the panel's order (so line i keeps the theme's
  // --trace-(i+1)), named by the role's label, each value held until the next (`hv`), a gap never bridged.
  // A line with nothing to draw is one null point (Plotly lists a trace with a point in the legend).
  // `lines`: [{label, points, hold}]; x is left in milliseconds for the page to stamp.
  function traces(lines, fromMs, toMs, columns) {
    return lines.map((l) => {
      const s = series(l.points, fromMs, toMs, l.hold, columns);
      return { type: "scatter", mode: "lines", name: l.label, x: s.drawn ? s.x : [null], y: s.drawn ? s.y : [null], connectgaps: false, line: { width: 1.25, shape: "hv" } };
    });
  }

  // The y axis' unit: the roles' own when every line has one and they agree, else none.
  const sharedUnit = (units) => (units.length && units[0] && units.every((u) => u === units[0]) ? units[0] : "");

  // The viewer's window in seconds: the choice pressed when it is one offered, else the page's first.
  function windowOf(raw, choices, fallback) {
    const n = Number(raw);
    return choices.includes(n) ? n : fallback;
  }
  const trendsUrl = (base, seconds, axis) => `${base}?seconds=${seconds}${axis ? `&axis=${encodeURIComponent(axis)}` : ""}`;

  // The route's answer as the page keeps it: { ok, to, roles: {role: points} } and why each role has no
  // history, or { ok: false, why } for a refusal, a silent server or an answer that is no history.
  function answer(ok, body, status) {
    const b = body && typeof body === "object" ? body : {};
    if (!ok || !b.roles || typeof b.roles !== "object") return { ok: false, why: b.detail || (ok ? "no history in the answer" : `answered ${status}`) };
    const roles = {}, why = {};
    for (const [role, r] of Object.entries(b.roles)) { roles[role] = past(r && r.samples); if (!roles[role].length) why[role] = (r && r.why) || "nothing buffered yet"; }
    return { ok: true, to: finite(b.to) ? b.to : null, roles, why };
  }

  // The one line under the window choice: "" while every line has its history; else that the plots start
  // at page load, and why (`got`: answer()'s, null while the first ask is out; `wanted`: [{role, label}]).
  function historyNote(got, wanted) {
    if (!got) return "reading the history";
    if (!got.ok) return `no earlier history (${got.why}): the plots start at page load`;
    const without = wanted.filter((r) => !(got.roles[r.role] || []).length);
    if (!without.length) return "";
    const why = got.why[without[0].role] || "nothing buffered yet";
    if (without.length === wanted.length) return `no earlier history (${why}): the plots start at page load`;
    return `no earlier history for ${without.map((r) => r.label).join(", ")} (${why}): drawn from page load`;
  }

  Object.assign(rules, { CAP, point, append, past, merge, holding, series, stampText, traces, sharedUnit, windowOf, trendsUrl, answer, historyNote, within: C.within });
})(
  typeof module === "object" ? module.exports : (window.GeecsTrendRules = {}),
  typeof module === "object" ? require("./camera_rules.js") : window.GeecsCameraRules,
);
