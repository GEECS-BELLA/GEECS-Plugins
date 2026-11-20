// One device as a page reads it, DOM free: the module's view (the caller's), and per role its entry, Hare's frame
// (watched on the first ask), meta and word, and its write record (`target`: the module's `writePv` when it is not
// the role's own record, areaDetector's X beside the X_RBV the role reads, a DB device's :SP; judged as a bare PV,
// Hare's connection and alarm, since the module judges the role's own record). The device page's GeecsDevicePage
// and every device a screen binds are made here: one reading of a device for both.
// window.GeecsDeviceSource.make({ name, root, hare, view, hareUp, wire, changed, read }) → a page (device_rail.js reads it).
(function () {
  "use strict";
  const D = window.GeecsDeviceRules, S = window.GeecsScreenRules;

  // view(): the module's device route as kept (D.deviceView); wire: pv → Hare's frame, shared with the caller.
  // read(pv): a role's frame when the caller watches the roles itself (the device page's rows): one subscription each.
  function make({ name, root, hare, view, hareUp, wire, changed, read }) {
    const watched = new Set();
    const watch = (pv) => {
      if (!watched.has(pv)) { watched.add(pv); hare.watch([pv], (n, f) => { wire[n] = f; changed(); }); }
      return wire[pv];
    };
    // `axis`: a per-axis family's axis (a motor's m1); omitted for every other device.
    const entry = (role, axis) => { const v = view(); return v && v.roles[S.key(role, axis)]; };
    const frame = (role, axis) => { const e = entry(role, axis); return e && D.watched(e) ? (read || watch)(e.pv) : undefined; };
    function target(role, axis) {
      const e = entry(role, axis);
      if (!e || !e.writePv || e.writePv === e.pv) return null;
      const f = watch(e.writePv);
      return { frame: f, meta: hare.meta(e.writePv), word: S.judge({ hareUp: hareUp(), frame: f, bare: true }) };
    }
    return {
      name, root, view, entry, frame, target,
      meta: (role, axis) => { const e = entry(role, axis); return e && e.pv ? hare.meta(e.pv) : {}; },
      word: (role, axis) => D.rowWord({ hareUp: hareUp(), frame: frame(role, axis), view: view(), entry: entry(role, axis) }),
    };
  }
  window.GeecsDeviceSource = { make };
})();
