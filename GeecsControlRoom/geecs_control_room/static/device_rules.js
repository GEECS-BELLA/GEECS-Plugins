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

  // The device route as the page keeps it: the screen's view, the heartbeat word, and its source (a DB device's "gateway").
  function deviceView(ok, body) {
    const v = S.deviceView(ok, body);
    return ok ? { ...v, heartbeat: body.heartbeat || null, source: body.source || null } : v;
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

  // ---- the setpoint control: what the field tracks, what stands beside it, the draft, the mark
  // Where a setpoint field's value comes from and whether the setpoint is shown beside it. The setpoint is the
  // role's write record (the module's `writePv`): areaDetector's X beside the X_RBV the role reads, the motor's
  // .VAL, a supply's current:SP. `readback` is the measured role (the family's `readback`, else the role itself).
  // field "write": the write record, when it is not the role's own record; "self": the role's record. One
  // exception: a DB device's write record is the gateway's `:SP`, which holds the last put it forwarded (0
  // until the first since the gateway started), never the device's setting, so its field tracks the device's
  // value. set: the setpoint record is shown beside the field, distinct from the measured one.
  function sources(entry, readback, gateway) {
    if (!entry) return { field: "self", set: false };
    const apart = !!entry.writePv && entry.writePv !== entry.pv;
    return { field: apart && !gateway ? "write" : "self", set: apart || (!!readback && readback !== entry.role) };
  }

  // A tracked value as the field shows it: text as it is, a number at a screen widget's own `digits`, else to six
  // significant digits; null leaves the field.
  function fieldText(v, digits) {
    if (typeof v === "string") return v;
    if (typeof v !== "number" || !Number.isFinite(v)) return null;
    if (Number.isInteger(digits)) return v.toFixed(digits);
    return String(Number.isInteger(v) ? v : Number(v.toPrecision(6)));
  }

  // A write list's answer from the module: accepted, or it stops there (device_writes.js).
  const accepted = (answer) => !!answer && answer.outcome === "accepted";
  // A typed draft is kept until its write is accepted: a refusal, an unsent list (no operator, not confirmed,
  // another write in flight) and a blur keep it. One rule for a field's Enter and a tweak's step.
  const keepsDraft = (ok) => ok !== true;

  // How far the measured value may sit from the setpoint: the family's number, else its tolerance role's value
  // (`by`: undefined when the family names none; unread, nothing is judged), else one unit in the last digit
  // the record shows (its precision), else exact (a whole number, or no precision).
  function tolerance({ hint, by, precision, integer }) {
    if (typeof hint === "number") return hint;
    if (by !== undefined) return typeof by === "number" && Number.isFinite(by) && by >= 0 ? by : null;
    if (integer) return 0;
    const p = Number(precision);
    return precision !== null && precision !== undefined && Number.isInteger(p) && p >= 0 ? 10 ** -p : 0;
  }

  // A done flag (a setpoint's `settled_by`, the motor's DMOV) reads done at 1; unread is not done.
  const done = (frame) => !!frame && frame.value === 1;

  // The mark beside a setpoint after this page's accepted write (`armedAt`, ms): "waiting" until it settles
  // (SETTLE_MS after the acceptance, and its done flag reading done when it has one, `isDone` false while not)
  // and while either value is unread; then "differs" while the measured value sits outside the tolerance of
  // the setpoint, "agrees" once within it (the page then disarms); "none" when unarmed.
  const SETTLE_MS = 1000;
  function mark({ armedAt, now, isDone, sp, mv, tol }) {
    if (typeof armedAt !== "number") return "none";
    if (now - armedAt < SETTLE_MS || isDone === false) return "waiting";
    const num = (x) => typeof x === "number" && Number.isFinite(x);
    if (!num(sp) || !num(mv) || tol === null) return "waiting";
    const slack = 1e-9 * Math.max(1, Math.abs(sp), Math.abs(mv)); // the float error of "one unit in the last digit"
    return Math.abs(sp - mv) - tol > slack ? "differs" : "agrees";
  }

  // A setpoint field's mark state: "none" unarmed, or where the setpoint and the measured value are one record
  // (`set` false); "waiting" while the module is silent (`known` false: no route entry, so nothing is known and
  // nothing is judged; the arming is kept and the mark hidden until the route answers again); else mark()'s.
  function settleMark({ known, set, ...judged }) {
    if (typeof judged.armedAt !== "number") return "none";
    if (!known) return "waiting";
    return set ? mark(judged) : "none";
  }
  // The page disarms a field on verified agreement, or where there is nothing to judge; never on a silence.
  const disarms = (state) => state === "agrees" || state === "none";

  // A key in a field holding a draft: Enter writes it, Escape drops it; any other key, or no draft, nothing.
  const commitKey = (key, draft) => (!draft ? null : key === "Enter" ? "write" : key === "Escape" ? "revert" : null);

  // A tweak's press: the command alone, or (its step holding a draft) the step's value first, in one list,
  // so the record never steps by the step the field no longer shows; null for a draft that is no number.
  function tweakList(command, step, drafted, value) {
    if (!drafted) return [[command, 1]];
    return value === null ? null : [[step, value], [command, 1]];
  }

  // The roles the module judges `error` on one axis (none named: every role), by label: a pane's alarm line.
  // camera_rules.js's `alarms` over the axis' roles: one rule for every page's alarm line.
  function alarmed(view, axis) {
    if (axis === undefined || axis === null || !view || !view.ok) return C.alarms(view);
    return C.alarms({ ...view, roles: Object.fromEntries(Object.entries(view.roles).filter(([, e]) => e.axis === axis)) });
  }

  Object.assign(rules, { pickerOptions, recovered, pickerNote, watched, rowWord, units, deviceView, header, presentRole, hidden, sources, fieldText, accepted, keepsDraft, tolerance, done, SETTLE_MS, mark, settleMark, disarms, writeValue, commitKey, tweakList, alarmed });
})(
  typeof module === "object" ? module.exports : (window.GeecsDeviceRules = {}),
  typeof module === "object" ? require("./screen_rules.js") : window.GeecsScreenRules,
  typeof module === "object" ? require("./camera_rules.js") : window.GeecsCameraRules,
);
