// The screen renderer's live half. Each device's PVs and verdicts come from
// the module's device route (polled); values, connection and alarms from
// Hare. A widget's word claims no more than that evidence: the module's
// verdict for a device role, Hare's connection and alarm for a bare PV,
// "unknown" while either source is silent. A changing value is never read
// as freshness. Writes go to the module's put route once, and its outcome
// word and code stay on the widget; nothing is retried.
(function () {
  "use strict";
  const POLL_MS = 2000;
  const OP_KEY = "geecs-control-room-operator";
  const root = document.body.dataset.root || "";
  const KIT = JSON.parse(document.getElementById("kit-state").textContent);
  const OUT = { accepted: "ok", denied: "degraded", refused: "failed", unknown: "unknown", sending: "running" };
  const hare = window.GeecsHare.fromPage();
  const wire = {};           // pv -> Hare's latest frame (null while the socket is down)
  const views = {};          // device -> {ok, state, roles: {"role|axis": entry}}
  const users = {};          // pv -> Set(element)
  let hareUp = false;

  const widgets = [...document.querySelectorAll(".w")];
  const key = (role, axis) => role + "|" + (axis || "");
  const entryOf = (el, role) => {
    const v = views[el.dataset.device];
    return v && v.roles[key(role || el.dataset.role, el.dataset.axis)];
  };

  // ---- the operator, remembered in this browser
  const opInput = document.getElementById("operator");
  try { opInput.value = localStorage.getItem(OP_KEY) || ""; } catch (_) { /* no storage */ }
  opInput.addEventListener("change", () => {
    try { localStorage.setItem(OP_KEY, opInput.value.trim()); } catch (_) { /* no storage */ }
  });

  // ---- reading
  function use(pv, el) {
    if (!pv || (users[pv] && users[pv].has(el))) return;
    (users[pv] = users[pv] || new Set()).add(el);
    hare.watch([pv], (name, frame) => {
      wire[name] = frame;
      for (const user of users[name]) draw(user.closest(".w"));
    });
  }

  function pvOf(el, role) {
    if (el.dataset.pv && !role) return el.dataset.pv;
    const e = entryOf(el, role);
    return e && e.read !== "never" ? e.pv : null;
  }

  function show(pv, el) {
    const f = pv && wire[pv];
    if (!f || f.value === null || f.value === undefined) return "—";
    const meta = hare.meta(pv), host = el.closest(".w");
    if (meta.enumChoices && typeof f.value === "number") return meta.enumChoices[f.value] ?? String(f.value);
    if (typeof f.value !== "number") return String(f.value);
    const digits = host.dataset.precision ?? meta.precision;
    return digits === undefined || digits === null ? String(f.value) : f.value.toFixed(Number(digits));
  }

  function judge(el) {
    const pv = pvOf(el), f = pv && wire[pv];
    if (el.dataset.pv) {
      if (!hareUp || !f) return "unknown";
      if (!f.connected) return "unreachable";
      return f.severity === "MAJOR" || f.severity === "INVALID" ? "error" : "live";
    }
    const view = views[el.dataset.device], e = entryOf(el);
    if (!view || !view.ok || !e) return "unknown"; // the module cannot say
    if (e.read === "never") return view.state || "unknown"; // a command: the device's word
    if (!hareUp || !f) return "unknown";
    if (!f.connected) return "unreachable";
    return e.verdict || "unknown";
  }

  function draw(el) {
    const word = judge(el), state = el.querySelector(".ws");
    state.querySelector(".dot").dataset.state = KIT[word] || "unknown";
    state.querySelector(".word").textContent = word;
    const pv = pvOf(el), meta = pv ? hare.meta(pv) : {};
    el.querySelector(".u").textContent = el.dataset.units || meta.units || "";
    const bounds = el.querySelector(".bounds"), lo = meta.controlLow, hi = meta.controlHigh;
    const set = (x) => typeof x === "number" && Math.abs(x) < 1e300; // ±1e300: no limit set
    if (bounds) bounds.textContent = set(lo) && set(hi) && lo < hi ? `[${lo} … ${hi}]` : "";
    const value = el.querySelector(".v"), sp = el.querySelector(".sp"), act = el.querySelector(".act");
    if (value) value.textContent = show(pv, el);
    if (sp && !sp.dataset.draft) sp.value = show(pv, el) === "—" ? "" : show(pv, el);
    if (act && el.dataset.kind !== "button") act.textContent = show(pv, el);
    const e = entryOf(el);
    for (const c of [sp, act]) if (c) c.disabled = !el.dataset.pv && !!e && !e.writable;
    if (el.dataset.pv && (sp || act)) for (const c of [sp, act]) if (c) c.disabled = true;
    for (const ind of el.querySelectorAll(".ind")) ind.querySelector("b").textContent = show(pvOf(ind), ind);
    const rb = el.querySelector(".rb");
    if (rb) {
      const rbPv = pvOf(el, el.dataset.readback);
      rb.textContent = show(rbPv, el);
      const a = pv && wire[pv], b = rbPv && wire[rbPv];
      // differs: the two disagree at the precision shown (-0.000 is 0.000)
      const [x, y] = [show(pv, el), show(rbPv, el)];
      const same = x === y || (x !== "—" && Number(x) === Number(y));
      el.querySelector(".differs").hidden = !(a && b) || same;
    }
  }

  async function poll(device) {
    let view = { ok: false, roles: {} };
    try {
      const r = await fetch(`${root}/api/devices/${encodeURIComponent(device)}`);
      const body = await r.json();
      if (r.ok) {
        view = { ok: true, state: body.state, roles: {} };
        for (const e of body.roles || []) view.roles[key(e.role, e.axis)] = e;
      }
    } catch (_) { /* the module is silent: every word is unknown */ }
    views[device] = view;
    for (const el of widgets.filter((w) => w.dataset.device === device)) {
      use(pvOf(el), el);
      if (el.dataset.readback) use(pvOf(el, el.dataset.readback), el);
      draw(el);
    }
    for (const ind of document.querySelectorAll(`.ind[data-device="${CSS.escape(device)}"]`)) use(pvOf(ind), ind);
  }

  // ---- writing: once, through the module, the outcome left on the widget
  const dialog = document.getElementById("confirm");
  function confirmWrite(el, value) {
    if (!el.dataset.hazard && !el.dataset.confirm) return Promise.resolve(true);
    document.getElementById("confirm-title").textContent = `Write ${el.querySelector(".lbl").textContent}?`;
    document.getElementById("confirm-what").textContent = `${el.dataset.device} · ${el.dataset.role} ← ${value}`;
    document.getElementById("confirm-note").textContent = el.dataset.confirm || "This role needs the hazard confirm.";
    return new Promise((done) => {
      dialog.onclick = (ev) => { if (ev.target.value) { dialog.close(); done(ev.target.value === "write"); } };
      dialog.oncancel = () => done(false);
      dialog.showModal();
    });
  }

  function outcome(el, word, code, detail) {
    const out = el.querySelector(".out");
    out.dataset.state = OUT[word] || "unknown";
    out.textContent = code ? `${word} · ${code}` : word;
    out.title = detail || "";
  }

  async function write(el, value) {
    const operator = opInput.value.trim();
    if (!operator) { opInput.focus(); return outcome(el, "unsent", "", "name yourself in the header first"); }
    if (!(await confirmWrite(el, value))) return outcome(el, "unsent", "", "not confirmed");
    const body = { role: el.dataset.role, value, operator, confirm: true };
    if (el.dataset.axis) body.axis = el.dataset.axis;
    if (!el.dataset.hazard && !el.dataset.confirm) delete body.confirm;
    outcome(el, "sending");
    const controls = el.querySelectorAll("input, button");
    controls.forEach((c) => { c.disabled = true; });
    try {
      const r = await fetch(`${root}/api/devices/${encodeURIComponent(el.dataset.device)}/put`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const a = await r.json();
      outcome(el, a.outcome || "unknown", a.code, a.detail);
    } catch (err) {
      outcome(el, "unknown", "no_answer", "read the device and the journal before repeating it");
    }
    controls.forEach((c) => { c.disabled = false; });
    draw(el);
  }

  const asValue = (text) => (text.trim() !== "" && Number.isFinite(Number(text)) ? Number(text) : text);

  for (const el of widgets) {
    if (el.dataset.pv) use(el.dataset.pv, el);
    const sp = el.querySelector(".sp"), act = el.querySelector(".act");
    if (sp) {
      sp.addEventListener("input", () => { sp.dataset.draft = "1"; });
      sp.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") { delete sp.dataset.draft; draw(el); }
        if (ev.key === "Enter" && sp.dataset.draft) { delete sp.dataset.draft; write(el, asValue(sp.value)); }
      });
    }
    if (act) act.addEventListener("click", () => {
      if (el.dataset.kind === "button") return write(el, 1);
      const f = wire[pvOf(el)];
      if (f && typeof f.value === "number") write(el, f.value ? 0 : 1);
    });
  }

  hare.onStatus((s) => {
    hareUp = s === "open";
    document.getElementById("hare-state").textContent = hareUp ? "live values" : `Hare ${s}`;
    document.querySelector(".hare .dot").dataset.state = hareUp ? "ok" : "unknown";
    widgets.forEach(draw);
  });
  const devices = [...new Set(widgets.map((w) => w.dataset.device).filter(Boolean))];
  const tick = () => devices.forEach(poll);
  tick();
  setInterval(tick, POLL_MS);
})();
