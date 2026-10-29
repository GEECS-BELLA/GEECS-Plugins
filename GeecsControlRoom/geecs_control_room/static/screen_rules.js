// The screen's rules, pure: no DOM, no socket, no clock. screen.js draws
// what these say; tests/test_screen_rules_js.py runs each rule under node.
// window.GeecsScreenRules in the page, module.exports under node.
(function (rules) {
  "use strict";
  const OUT = { accepted: "ok", denied: "degraded", refused: "failed", unknown: "unknown", sending: "running" };
  const NOT_ANSWERING = [1, 9, 10]; // READ, COMM, TIMEOUT
  const UDF = 17; // never processed

  const key = (role, axis) => role + "|" + (axis || "");

  // The module's device route as the page keeps it: roles by role|axis.
  function deviceView(ok, body) {
    if (!ok) return { ok: false, roles: {} };
    const roles = {};
    for (const e of body.roles || []) roles[key(e.role, e.axis)] = e;
    return { ok: true, state: body.state, roles };
  }

  // A widget's word. A bare PV has no module verdict: its alarm rules, as
  // GeecsCanopy's freshness.judge has them. A device role: the module's verdict,
  // "unknown" whenever the module or Hare is silent.
  function judge({ hareUp, frame, bare, view, entry }) {
    if (bare) {
      if (!hareUp || !frame) return "unknown";
      if (!frame.connected) return "unreachable";
      if (frame.severity === "INVALID" && frame.alarmStatus === UDF) return "unknown";
      if (frame.severity === "INVALID" && NOT_ANSWERING.includes(frame.alarmStatus)) return "not answering";
      return frame.severity === "MAJOR" || frame.severity === "INVALID" ? "error" : "live";
    }
    if (!view || !view.ok || !entry) return "unknown"; // the module cannot say
    if (entry.read === "never") return view.state || "unknown"; // a command: the device's word
    if (!hareUp || !frame) return "unknown";
    if (!frame.connected) return "unreachable";
    return entry.verdict || "unknown";
  }

  // A value as shown: an enum's label, the widget's precision (else the record's), "—" for nothing.
  function show(frame, meta, precision) {
    if (!frame || frame.value === null || frame.value === undefined) return "—";
    if (meta.enumChoices && typeof frame.value === "number") return meta.enumChoices[frame.value] ?? String(frame.value);
    if (typeof frame.value !== "number") return String(frame.value);
    const digits = precision ?? meta.precision;
    return digits === undefined || digits === null ? String(frame.value) : frame.value.toFixed(Number(digits));
  }

  // The control limits, when both are set (±1e300 is EPICS for no limit).
  function bounds(meta) {
    const lo = meta.controlLow, hi = meta.controlHigh;
    const set = (x) => typeof x === "number" && Math.abs(x) < 1e300;
    return set(lo) && set(hi) && lo < hi ? `[${lo} … ${hi}]` : "";
  }

  // Writable only on the module's word, never on a bare PV, never while a write is in flight.
  const writable = (entry, bare, busy) => !bare && !!entry && !!entry.writable && !busy;

  // Setpoint and readback disagree at the precision shown (-0.000 is 0.000).
  function differs(shownSp, shownRb, bothRead) {
    const same = shownSp === shownRb || (shownSp !== "—" && Number(shownSp) === Number(shownRb));
    return bothRead && !same;
  }

  const outcomeState = (word) => OUT[word] || "unknown";

  // The header's source line: Hare's state, and the devices the module is slow to answer for.
  function sourceLine(hareState, slow) {
    const up = hareState === "open";
    const text = (up ? "live values" : `Hare ${hareState}`) + (slow.length ? " · module slow" : "");
    return { text, kit: !up ? "unknown" : slow.length ? "degraded" : "ok", title: slow.length ? `no answer for: ${slow.join(", ")}` : "" };
  }

  Object.assign(rules, { key, deviceView, judge, show, bounds, writable, differs, outcomeState, sourceLine });
})(typeof module === "object" ? module.exports : (window.GeecsScreenRules = {}));
