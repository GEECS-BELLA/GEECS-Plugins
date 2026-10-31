// The device pages' rules, pure: no DOM, no socket, no clock. device.js draws
// what these say; tests/test_device_rules_js.py runs each rule under node. The
// verdict word and the value shown are screen_rules.js's, reused, never copied.
// window.GeecsDeviceRules in the page, module.exports under node.
(function (rules, S) {
  "use strict";

  // The picker's options: by name (a device never jumps as its state changes),
  // each with the module's roster word; the current device selected.
  function pickerOptions(rows, current, kind) {
    const sorted = [...rows].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    const head = { value: "", text: sorted.length ? `choose a ${kind}…` : `no ${kind} in the registry`, selected: !current };
    return [head, ...sorted.map((d) => ({ value: d.name, text: `${d.name} · ${d.state || "unknown"}`, selected: d.name === current }))];
  }

  // The note beside the picker: why the states read unknown (the module silent, the files unreadable).
  const pickerNote = (body) => (body && body.detail) || "";

  // A row is read from Hare when the module names its PV and reads it (an array,
  // on request, is the kind's component's; a command has no value).
  const watched = (entry) => !!entry && !!entry.pv && (entry.read === "monitor" || entry.read === "once");

  // A role row's word: the screen's judge for a device role; an array is "on request".
  function rowWord({ hareUp, frame, view, entry }) {
    if (view && view.ok && entry && entry.read === "on_request") return "on request";
    return S.judge({ hareUp, frame, bare: false, view, entry });
  }

  // A row's units: the family's, else the record's (EGU) for a value only (a motor's bits carry its EGU too).
  const units = (type, family, meta) => family || (["readback", "setpoint"].includes(type) && (meta || {}).units) || "";

  // The device route as the page keeps it: the screen's view, and the heartbeat word.
  function deviceView(ok, body) {
    const v = S.deviceView(ok, body);
    return ok ? { ...v, heartbeat: body.heartbeat || null } : v;
  }

  // The header: the module's device state and heartbeat, "unknown" while it is silent.
  function header(view) {
    if (!view || !view.ok) return { state: "unknown", heartbeat: "heartbeat unknown" };
    return { state: view.state || "unknown", heartbeat: `heartbeat ${view.heartbeat || "—"}` };
  }

  Object.assign(rules, { pickerOptions, pickerNote, watched, rowWord, units, deviceView, header });
})(
  typeof module === "object" ? module.exports : (window.GeecsDeviceRules = {}),
  typeof module === "object" ? require("./screen_rules.js") : window.GeecsScreenRules,
);
