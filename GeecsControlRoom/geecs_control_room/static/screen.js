// The screen renderer's live half. Each device's PVs and verdicts come from
// the module's device route (polled); values, connection and alarms from
// Hare. A widget's word claims no more than that evidence: the module's
// verdict for a device role, Hare's connection and alarm for a bare PV,
// "unknown" while either source is silent. A changing value is never read
// as freshness. Writes go to the module's put route once, and its outcome
// word and code stay on the widget; nothing is retried. A deferred widget
// (a DB device, data-fits) is judged by the module's word alone. The rules are
// screen_rules.js's (tested under node), the polling poller.js's; this file
// keeps the DOM work.
(function () {
  "use strict";
  const POLL_MS = 2000;
  const OP_KEY = "geecs-control-room-operator";
  const root = document.body.dataset.root || "";
  const KIT = JSON.parse(document.getElementById("kit-state").textContent);
  const R = window.GeecsScreenRules;
  const hare = window.GeecsHare.fromPage();
  const wire = {};           // pv -> Hare's latest frame (null while the socket is down)
  const views = {};          // device -> {ok, state, roles: {"role|axis": entry}}
  const users = {};          // pv -> Set(element)
  const slow = new Set();    // devices the module is slow to answer for
  let hareState = "connecting";

  const widgets = [...document.querySelectorAll(".w")];
  const entryOf = (el, role) => {
    const v = views[el.dataset.device];
    return v && v.roles[R.key(role || el.dataset.role, el.dataset.axis)];
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

  const fitsOf = (el) => (el.dataset.fits === undefined ? undefined : el.dataset.fits.split(" ").filter(Boolean));
  const hazardOf = (el) => R.hazard(el.dataset.hazard || el.dataset.confirm, entryOf(el));
  const show = (pv, el) => R.show(pv && wire[pv], hare.meta(pv), el.closest(".w").dataset.precision);

  function judge(el) {
    const pv = pvOf(el), bare = !!el.dataset.pv;
    return R.judge({ hareUp: hareState === "open", frame: pv && wire[pv], bare, view: views[el.dataset.device], entry: bare ? null : entryOf(el), fits: fitsOf(el) });
  }

  function draw(el) {
    const word = judge(el), state = el.querySelector(".ws");
    state.querySelector(".dot").dataset.state = KIT[word] || "unknown";
    state.querySelector(".word").textContent = word;
    const pv = pvOf(el), meta = pv ? hare.meta(pv) : {};
    el.querySelector(".u").textContent = el.dataset.units || meta.units || "";
    const bounds = el.querySelector(".bounds");
    if (bounds) bounds.textContent = R.bounds(meta);
    const value = el.querySelector(".v"), sp = el.querySelector(".sp"), act = el.querySelector(".act");
    if (value) value.textContent = show(pv, el);
    if (sp && !sp.dataset.draft) sp.value = show(pv, el) === "—" ? "" : show(pv, el);
    if (act && el.dataset.kind !== "button") act.textContent = show(pv, el);
    const open = R.writable(entryOf(el), !!el.dataset.pv, !!el.dataset.busy, fitsOf(el));
    for (const c of [sp, act]) if (c) c.disabled = !open;
    for (const ind of el.querySelectorAll(".ind")) ind.querySelector("b").textContent = show(pvOf(ind), ind);
    const rb = el.querySelector(".rb");
    if (rb) {
      const rbPv = pvOf(el, el.dataset.readback);
      rb.textContent = show(rbPv, el);
      const bothRead = !!(pv && wire[pv] && rbPv && wire[rbPv]);
      el.querySelector(".differs").hidden = !R.differs(show(pv, el), show(rbPv, el), bothRead);
    }
  }

  // a 5xx is the module silent (back off); any other answer is the module's word
  async function readDevice(device, signal) {
    const r = await fetch(`${root}/api/devices/${encodeURIComponent(device)}`, { signal });
    if (r.status >= 500) throw new Error(`answered ${r.status}`);
    return R.deviceView(r.ok, await r.json());
  }

  function source() {
    const line = R.sourceLine(hareState, [...slow]);
    const host = document.getElementById("hare-state");
    host.textContent = line.text;
    host.title = line.title;
    document.querySelector(".hare .dot").dataset.state = line.kit;
  }

  function seen(device, got) {
    views[device] = got.ok ? got.value : { ok: false, roles: {} }; // silent: every word is unknown
    if (got.slow) slow.add(device); else slow.delete(device);
    source();
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
    out.dataset.state = R.outcomeState(word);
    out.textContent = code ? `${word} · ${code}` : word;
    out.title = detail || "";
  }

  async function write(el, value) {
    const operator = opInput.value.trim();
    if (!operator) { opInput.focus(); return outcome(el, "unsent", "", "name yourself in the header first"); }
    // one decision: the confirm sent is the dialog the operator answered (the module's mark may change meanwhile)
    const asked = hazardOf(el);
    if (asked && !(await confirmWrite(el, value))) return outcome(el, "unsent", "", "not confirmed");
    const body = { role: el.dataset.role, value, operator };
    if (asked) body.confirm = true;
    if (el.dataset.axis) body.axis = el.dataset.axis;
    outcome(el, "sending");
    el.dataset.busy = "1";
    draw(el);
    try {
      const r = await fetch(`${root}/api/devices/${encodeURIComponent(el.dataset.device)}/put`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      const a = await r.json();
      outcome(el, a.outcome || "unknown", a.code, a.detail);
    } catch (err) {
      outcome(el, "unknown", "no_answer", "read the device and the journal before repeating it");
    }
    delete el.dataset.busy;
    draw(el);
  }

  const asValue = (text) => (text.trim() !== "" && Number.isFinite(Number(text)) ? Number(text) : text);

  for (const ind of document.querySelectorAll(".ind[data-pv]")) use(ind.dataset.pv, ind);
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
    hareState = s;
    source();
    widgets.forEach(draw);
  });
  // every device a widget or an indicator names (a bare-PV flipper's lamp may be a device's role)
  const devices = [...new Set([...document.querySelectorAll(".w[data-device], .ind[data-device]")].map((e) => e.dataset.device))];
  const poller = window.GeecsPoller.makePoller({ fetch: readDevice, pollMs: POLL_MS, onView: seen });
  const tick = () => poller.tick(devices);
  tick();
  setInterval(tick, POLL_MS);
})();
