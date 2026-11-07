// A panel page's rail and readings strip, DOM only: the one copy for every page drawn from a panel
// (_panel.html: the settables page in device.html, the trace page, the camera page; a screen's widgets too). Each control is enabled on the module's word
// (camera_rules.js's `control`), written through device_writes.js (the module's put route, the operator's name,
// sent once; a per-axis control's write carries its axis); each reading shows its value and the module's word; a
// value with no family units takes the record's (Hare's EGU); a group whose `<ch>_present` reads 0 is hidden; each
// pane's alarms are the roles the module judges `error`. On the settables page (#settables) it starts itself and
// drives the axis selector.
// A setpoint field tracks the device's setpoint (its write record, device_rules.sources) whenever it holds no
// draft and is not focused (a value landing under the caret would be typed into); beside it the setpoint, where it is a record of its own, and the measured value, each with
// its word. A number or text field is a draft once typed in (marked): Enter writes it, Esc drops it, and it is
// kept through a blur, a refusal and an unsent write until a write of it is accepted (device_rules.keepsDraft,
// the tweak's step alike); an arrow key or a spinner click never writes a move. A tweak's − / + with its step a
// draft sends the step, then the press (one list, stopped at a refusal). After an accepted write of a setpoint
// (or its tweak) the field is armed: once the write settles (device_rules.mark) a `differs` mark stands while the
// measured value is outside the tolerance, cleared (and disarmed) when they agree. A select, a checkbox or a
// button writes when chosen. The rules are device_rules.js's and camera_rules.js's (tested under node).
// A screen draws its widgets through the same controls (screen.js): its `page` and `writes` then answer `for(el)`,
// the widget's own (a device, a bare PV, a DB device judged against its kind), and a write says its outcome on the
// widget's line and asks with the widget's own hazard note. A pair labelled by the record (`data-enum-label`) takes
// the enum's own words; a select is narrowed to `data-choices` (a family role's); a field the module serves as text
// (a DB device's) takes text; the module's `tolerance` is the hint when the page was drawn with none.
// window.GeecsDeviceRail.make(page, box, writes) → { fill }.
(function () {
  "use strict";
  const C = window.GeecsCameraRules, D = window.GeecsDeviceRules, S = window.GeecsScreenRules;
  const KIT = JSON.parse(document.getElementById("kit-state").textContent);
  const focused = (el) => document.activeElement === el;
  const axisOf = (el) => el.dataset.axis || undefined;
  const paneAxis = (el) => { const p = el.closest("[data-axis-pane]"); return (p && p.dataset.axisPane) || undefined; };

  function make(P, box, writes) {
    const pg = (el) => (P.for ? P.for(el) : P), wr = (el) => (writes.for ? writes.for(el) : writes);
    const opts = (el) => { const w = el.closest(".w"); return w ? { out: w.querySelector(".out"), note: w.dataset.confirm } : {}; };
    const value = (el, role, axis) => { const f = pg(el).frame(role, axis); return f ? f.value : undefined; };
    const known = () => { const v = P.view(); return v && v.ok ? Object.values(v.roles).map((e) => e.role) : []; };
    const controls = [...box.querySelectorAll(".field [data-role]")];
    const say = (el, text) => { const out = opts(el).out || document.getElementById("write-out"); if (out) { out.dataset.state = "unknown"; out.textContent = text; } };
    const paint = (line, text, word) => {
      line.querySelector("b").textContent = text;
      line.querySelector(".dot").dataset.state = KIT[word] || "unknown";
      line.querySelector(".word").textContent = word;
    };
    // A setpoint field: the tracked value (no draft), the setpoint and the measured value beside it, the mark.
    function setpoint(el) {
      const P = pg(el), role = el.dataset.role, axis = axisOf(el), rb = el.dataset.readback || role, view = P.view(), e = P.entry(role, axis);
      if (e && e.text && el.type === "number") { el.type = "text"; el.removeAttribute("step"); } // a DB device's text, as the module serves it
      const src = D.sources(P.entry(role, axis), rb, !!view && view.source === "gateway");
      const t = P.target(role, axis), own = { frame: P.frame(role, axis), meta: P.meta(role, axis), word: P.word(role, axis) };
      const sp = t || own, field = src.field === "write" ? sp : own;
      const text = D.fieldText(field.frame && field.frame.value, field.meta.digits);
      if (!el.dataset.draft && !focused(el) && text !== null && el.value !== text) el.value = text;
      const host = el.closest(".field"), shown = host.querySelector(".sv"), now = host.querySelector(".rb"), mark = host.querySelector(".differs");
      shown.hidden = !src.set;
      if (src.set) paint(shown, C.fmt(sp.frame, sp.meta), sp.word);
      const mv = P.frame(rb, axis), mvMeta = P.meta(rb, axis);
      paint(now, C.fmt(mv, mvMeta), P.word(rb, axis));
      const by = el.dataset.tolBy;
      const hint = el.dataset.tol === undefined ? (e && typeof e.tolerance === "number" ? e.tolerance : undefined) : Number(el.dataset.tol);
      el.title = el.title || (S.bounds(P.meta(role, axis)) && `limits ${S.bounds(P.meta(role, axis))}`);
      const tol = D.tolerance({ hint, by: by ? (P.frame(by, axis) || {}).value ?? null : undefined, precision: mvMeta.precision ?? sp.meta.precision, integer: el.step === "1" });
      const settledBy = el.dataset.settledBy;
      const m = D.settleMark({ known: !!P.entry(role, axis), set: src.set, armedAt: el.dataset.armed === undefined ? undefined : Number(el.dataset.armed), now: Date.now(), isDone: settledBy ? D.done(P.frame(settledBy, axis)) : undefined, sp: sp.frame && sp.frame.value, mv: mv && mv.value, tol });
      if (D.disarms(m)) delete el.dataset.armed; // never on a silent module: a failed poll keeps the arming
      mark.hidden = m !== "differs";
      mark.title = m === "differs" ? `the measured value is more than ${tol} from the setpoint since the write settled` : "";
    }
    // After an accepted write of a setpoint (or its tweak's press): the mark is judged once it settles.
    function arm(el) {
      if (!el || el.type === "text") return;
      el.dataset.armed = String(Date.now());
      setTimeout(fill, D.SETTLE_MS + 50);
    }
    function fill() {
      for (const el of controls) {
        const P = pg(el), role = el.dataset.role, axis = axisOf(el), v = value(el, role, axis);
        const c = C.control(P.view(), P.entry(role, axis), wr(el).busy());
        el.disabled = !c.enabled;
        el.title = c.why;
        if (el.dataset.pair !== undefined) {
          el.setAttribute("aria-pressed", String(C.pressed(v, Number(el.dataset.value))));
          const word = el.dataset.enumLabel !== undefined && (P.meta(role, axis).enumChoices || [])[Number(el.dataset.value)];
          if (word && el.textContent !== word) el.textContent = word; // the record's own word for the state
        }
        if (el.tagName === "BUTTON") continue;
        if (el.type === "checkbox") { el.checked = !!v; continue; }
        if (el.tagName === "SELECT") {
          const only = el.dataset.choices ? JSON.parse(el.dataset.choices) : null; // the PV's own choices, never a list of the page's
          const choices = (P.meta(role, axis).enumChoices || []).map((t, i) => [t, i]).filter(([t]) => !only || only.includes(t));
          const sig = choices.join("\n");
          if (el.dataset.sig !== sig) { el.replaceChildren(...choices.map(([t, i]) => new Option(t, String(i)))); el.dataset.sig = sig; }
          if (!choices.length) el.disabled = true; // unread: the module would refuse
          if (!focused(el) && typeof v === "number") el.value = String(v);
          continue;
        }
        if (el.closest(".field.setpoint")) { setpoint(el); continue; }
        const text = D.fieldText(v); // a tweak's step: its own record
        if (!el.dataset.draft && !focused(el) && text !== null && el.value !== text) el.value = text;
      }
      for (const el of box.querySelectorAll("[data-ro]")) {
        const P = pg(el), role = el.dataset.ro, axis = axisOf(el), word = P.word(role, axis), ws = el.parentElement.querySelector(".ws");
        el.textContent = C.fmt(P.frame(role, axis), P.meta(role, axis));
        if (ws) { ws.querySelector(".dot").dataset.state = KIT[word] || "unknown"; ws.querySelector(".word").textContent = word; }
      }
      for (const el of box.querySelectorAll("[data-egu]")) el.textContent = D.units(el.dataset.type, "", pg(el).meta(el.dataset.egu, axisOf(el)));
      const names = known();
      for (const g of box.querySelectorAll(".panel-rail .group[data-roles]")) { // a panel group: a page's own groups beside it carry none
        const present = D.presentRole(JSON.parse(g.dataset.roles), names);
        g.hidden = !!present && D.hidden(pg(g).frame(present, paneAxis(g)));
      }
      for (const el of box.querySelectorAll(".readings .live")) {
        const P = pg(el), role = el.dataset.role, axis = axisOf(el), present = D.presentRole([role], names), word = P.word(role, axis);
        el.hidden = !!present && D.hidden(P.frame(present, axis));
        el.querySelector(".v").textContent = C.fmt(P.frame(role, axis), P.meta(role, axis));
        el.querySelector(".dot").dataset.state = KIT[word] || "unknown";
        el.querySelector(".word").textContent = word;
      }
      const view = P.view();
      for (const el of box.querySelectorAll(".alarms")) el.textContent = D.alarmed(view, paneAxis(el)).join(" · ");
    }
    for (const el of controls) {
      const write = (v) => wr(el).write([[el.dataset.role, v, axisOf(el)]], false, opts(el)).then(fill);
      const fieldValue = (f) => D.writeValue({ type: f.type, tag: f.tagName, step: f.step, checked: f.checked, raw: f.value });
      if (el.tagName === "BUTTON" && el.dataset.step) { // a tweak's − / +: its step's draft first, when it has one
        const step = el.closest(".tweak").querySelector("input[data-role]");
        el.addEventListener("click", () => {
          const list = D.tweakList(el.dataset.role, step.dataset.role, !!step.dataset.draft, fieldValue(step));
          if (!list) { say(el, "unsent · the step is not a number"); return; }
          wr(el).write(list.map(([role, v]) => [role, v, axisOf(el)]), false, opts(el)).then((ok) => {
            if (!D.keepsDraft(ok)) { delete step.dataset.draft; arm(el.closest(".row").querySelector(`.field.setpoint input[data-role="${CSS.escape(el.closest(".tweak").dataset.tweaks)}"]`)); }
            fill();
          });
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
          if (v === null) { say(el, "unsent · not a number"); return; } // the draft stays, for the operator to fix
          wr(el).write([[el.dataset.role, v, axisOf(el)]], false, opts(el)).then((ok) => {
            if (!D.keepsDraft(ok)) { delete el.dataset.draft; if (el.closest(".field.setpoint")) arm(el); }
            fill();
            setTimeout(() => { // after device_writes re-enables: the kept draft focused again, for Esc or a fix, unless focus went elsewhere (the operator's name)
              const at = document.activeElement;
              if (el.dataset.draft && !el.disabled && (!at || at === document.body || at === el)) el.focus();
            });
          });
        });
        continue;
      }
      el.addEventListener("change", () => { // a select or a checkbox: an unsent or refused choice goes back to what the device reads
        const v = fieldValue(el);
        if (v === null) return;
        wr(el).write([[el.dataset.role, v, axisOf(el)]], false, opts(el)).then((ok) => { if (D.keepsDraft(ok)) el.blur(); fill(); });
      });
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
