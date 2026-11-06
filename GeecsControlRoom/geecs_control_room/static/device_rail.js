// A panel page's rail and readings strip, DOM only: the one copy for every page drawn from a panel
// (_panel.html: the settables page in device.html, the trace page). Each control is filled from Hare unless
// focused or holding a draft, enabled on the module's word (camera_rules.js's `control`), written through device_writes.js (the
// module's put route, the operator's name, sent once; a per-axis control's write carries its axis); each reading
// shows its value and the module's word; a value with no family units takes the record's (Hare's EGU); a group whose `<ch>_present` reads 0 is hidden; each pane's alarms are
// the roles the module judges `error`. On the settables page (#settables) it starts itself and drives the axis
// selector. A number or text field is a draft once typed in (marked, kept when the field loses focus, the
// screen renderer's rule): Enter writes it, Esc drops it (an arrow key or a spinner click never writes a
// move); a tweak's − / + with its step a draft sends the step, then the press (one list, stopped at a refusal);
// a select, a checkbox or a button writes when chosen. The rules are device_rules.js's and camera_rules.js's (tested under node).
// window.GeecsDeviceRail.make(page, box, writes) → { fill }.
(function () {
  "use strict";
  const C = window.GeecsCameraRules, D = window.GeecsDeviceRules;
  const KIT = JSON.parse(document.getElementById("kit-state").textContent);
  const focused = (el) => document.activeElement === el;
  const axisOf = (el) => el.dataset.axis || undefined;
  const paneAxis = (el) => { const p = el.closest("[data-axis-pane]"); return (p && p.dataset.axisPane) || undefined; };

  function make(P, box, writes) {
    const value = (role, axis) => { const f = P.frame(role, axis); return f ? f.value : undefined; };
    const known = () => { const v = P.view(); return v && v.ok ? Object.values(v.roles).map((e) => e.role) : []; };
    const controls = [...box.querySelectorAll(".panel-rail [data-role]")];
    const out = document.getElementById("write-out");
    const say = (text) => { if (out) { out.dataset.state = "unknown"; out.textContent = text; } };
    function fill() {
      for (const el of controls) {
        const role = el.dataset.role, axis = axisOf(el), v = value(role, axis);
        const c = C.control(P.view(), P.entry(role, axis), writes.busy());
        el.disabled = !c.enabled;
        el.title = c.why;
        if (el.dataset.pair !== undefined) el.setAttribute("aria-pressed", String(C.pressed(v, Number(el.dataset.value))));
        if (el.tagName === "BUTTON") continue;
        if (el.type === "checkbox") { el.checked = !!v; continue; }
        if (el.tagName === "SELECT") {
          const choices = P.meta(role, axis).enumChoices || []; // the PV's own choices, never a list of the page's
          if (el.options.length !== choices.length) el.replaceChildren(...choices.map((t, i) => new Option(t, String(i))));
          if (!focused(el) && typeof v === "number") el.value = String(v);
          continue;
        }
        if (focused(el) || el.dataset.draft) continue;
        if (typeof v === "string") { el.value = v; continue; } // a text setting
        if (typeof v === "number") el.value = Number.isInteger(v) ? v : Number(v.toPrecision(6));
      }
      for (const el of box.querySelectorAll("[data-rb]")) el.textContent = `now ${C.fmt(P.frame(el.dataset.rb, axisOf(el)), P.meta(el.dataset.rb, axisOf(el)))}`;
      for (const el of box.querySelectorAll("[data-ro]")) el.textContent = C.fmt(P.frame(el.dataset.ro, axisOf(el)), P.meta(el.dataset.ro, axisOf(el)));
      for (const el of box.querySelectorAll("[data-egu]")) el.textContent = D.units(el.dataset.type, "", P.meta(el.dataset.egu, axisOf(el)));
      const names = known();
      for (const g of box.querySelectorAll(".panel-rail .group")) {
        const present = D.presentRole(JSON.parse(g.dataset.roles), names);
        g.hidden = !!present && D.hidden(P.frame(present, paneAxis(g)));
      }
      for (const el of box.querySelectorAll(".readings .live")) {
        const role = el.dataset.role, axis = axisOf(el), present = D.presentRole([role], names), word = P.word(role, axis);
        el.hidden = !!present && D.hidden(P.frame(present, axis));
        el.querySelector(".v").textContent = C.fmt(P.frame(role, axis), P.meta(role, axis));
        el.querySelector(".dot").dataset.state = KIT[word] || "unknown";
        el.querySelector(".word").textContent = word;
      }
      const view = P.view();
      for (const el of box.querySelectorAll(".alarms")) el.textContent = D.alarmed(view, paneAxis(el)).join(" · ");
    }
    for (const el of controls) {
      const write = (v) => writes.write([[el.dataset.role, v, axisOf(el)]]).then(fill);
      const fieldValue = (f) => D.writeValue({ type: f.type, tag: f.tagName, step: f.step, checked: f.checked, raw: f.value });
      if (el.tagName === "BUTTON" && el.dataset.step) { // a tweak's − / +: its step's draft first, when it has one
        const step = el.closest(".tweak").querySelector("input[data-role]");
        el.addEventListener("click", () => {
          const list = D.tweakList(el.dataset.role, step.dataset.role, !!step.dataset.draft, fieldValue(step));
          if (!list) { say("unsent · the step is not a number"); return; }
          writes.write(list.map(([role, v]) => [role, v, axisOf(el)])).then((ok) => { if (ok) delete step.dataset.draft; fill(); });  // a refusal keeps the draft
        });
        continue;
      }
      if (el.tagName === "BUTTON") { el.addEventListener("click", () => write(Number(el.dataset.value))); continue; }
      if (el.tagName === "INPUT" && el.type !== "checkbox") {
        el.addEventListener("input", () => { el.dataset.draft = "1"; });
        el.addEventListener("keydown", (ev) => {
          const act = D.commitKey(ev.key, !!el.dataset.draft);
          if (!act) return;
          ev.preventDefault();
          if (act === "revert") { delete el.dataset.draft; el.blur(); fill(); return; }
          const v = fieldValue(el);
          if (v === null) { say("unsent · not a number"); return; } // the draft stays, for the operator to fix
          delete el.dataset.draft;
          write(v);
        });
        continue;
      }
      el.addEventListener("change", () => { const v = fieldValue(el); if (v !== null) write(v); });
    }
    P.on(fill);
    fill();
    return { fill };
  }
  window.GeecsDeviceRail = { make };

  // The settables page: the rail and readings of every axis pane, one shown at a time (the axis selector).
  const box = document.getElementById("settables"), P = window.GeecsDevicePage;
  if (!box || !P) return;
  let rail = null;
  const writes = window.GeecsDeviceWrites.make(P, () => rail && rail.fill());
  rail = make(P, box, writes);
  const picks = [...box.querySelectorAll("[data-axis-pick]")];
  for (const b of picks) {
    b.addEventListener("click", () => {
      for (const o of picks) o.setAttribute("aria-pressed", String(o === b));
      for (const pane of box.querySelectorAll("[data-axis-pane]")) pane.hidden = pane.dataset.axisPane !== b.dataset.axisPick;
    });
  }
})();
