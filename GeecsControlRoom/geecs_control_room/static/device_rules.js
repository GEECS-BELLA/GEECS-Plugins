// The device pages' rules, pure: no DOM, no socket, no clock. device.js draws
// what these say; tests/test_device_rules_js.py runs each rule under node. The
// verdict word and the value shown are screen_rules.js's, a field's number camera_rules.js's: reused, never copied.
// window.GeecsDeviceRules in the page, module.exports under node.
(function (rules, S, C) {
  "use strict";

  // The picker's options: by name (a device never jumps as its state changes),
  // each with the module's roster word; the current device selected.
  function pickerOptions(rows, current, kind) {
    const sorted = [...rows].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    const head = { value: "", text: sorted.length ? `choose a ${kind}…` : `no ${kind} in the registry`, selected: !current };
    const options = sorted.map((d) => ({ value: d.name, text: `${d.name} · ${d.state || "unknown"}`, selected: d.name === current }));
    const typed = current && !sorted.some((d) => d.name === current); // a typed name the registry lacks stays chosen
    return [head, ...options, ...(typed ? [{ value: current, text: `${current} · typed`, selected: true }] : [])];
  }

  // A stale page reloads once the module answers with the device (200) or without it (404); any other answer keeps it asking.
  const recovered = (status) => status === 200 || status === 404;

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

  // ---- the rail and the readings (device_rail.js: every page drawn from a panel)
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

  // A control's value as written: a checkbox 1 or 0, a select its choice's index, a whole-number field a whole
  // number, a text field its text, any other field a number; null (nothing sent) for a field that does not parse.
  function writeValue({ type, tag, step, checked, raw }) {
    if (type === "checkbox") return checked ? 1 : 0;
    if (tag === "SELECT") return raw === "" ? null : Number(raw);
    if (type === "text") return raw;
    return step === "1" ? C.whole(raw) : C.number(raw);
  }

  // The roles the module judges `error` on one axis (none named: every role), by label: a pane's alarm line.
  function alarmed(view, axis) {
    if (!view || !view.ok) return [];
    const on = (e) => axis === undefined || axis === null || e.axis === axis;
    return Object.values(view.roles).filter((e) => e.verdict === "error" && on(e)).map((e) => e.label || e.role);
  }

  Object.assign(rules, { pickerOptions, recovered, pickerNote, watched, rowWord, units, deviceView, header, presentRole, hidden, writeValue, alarmed });
})(
  typeof module === "object" ? module.exports : (window.GeecsDeviceRules = {}),
  typeof module === "object" ? require("./screen_rules.js") : window.GeecsScreenRules,
  typeof module === "object" ? require("./camera_rules.js") : window.GeecsCameraRules,
);
