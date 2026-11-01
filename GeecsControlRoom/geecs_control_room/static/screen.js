// The screen renderer's live half. Each device's PVs and verdicts come from
// the module's device route (polled); values, connection and alarms from
// Hare. A widget's word claims no more than that evidence: the module's
// verdict for a device role, Hare's connection and alarm for a bare PV,
// "unknown" while either source is silent. A changing value is never read
// as freshness. Writes go to the module's put route once, and its outcome
// word and code stay on the widget; nothing is retried. A deferred widget
// (a DB device, data-fits) is judged by the module's word alone. The rules are
// screen_rules.js's (tested under node), the polling poller.js's; this file
// keeps the DOM work. A button naming an action plan asks the module to run
// it (once, confirmed when the plan writes a hazard role) and shows the run
// as the module reports it, polled until it ends, with Stop while it runs.
// A toggle with values writes values.on/off and reads on, off or unknown by
// matching the readback; a select offers the enum's choices from Hare.
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

  const valuesOf = (el) => (el.dataset.on === undefined ? null : { on: Number(el.dataset.on), off: Number(el.dataset.off) });
  const fitsOf = (el) => (el.dataset.fits === undefined ? undefined : el.dataset.fits.split(" ").filter(Boolean));
  const hazardOf = (el) => R.hazard(el.dataset.hazard || el.dataset.confirm, entryOf(el));
  const show = (pv, el) => R.show(pv && wire[pv], hare.meta(pv), el.closest(".w").dataset.precision);

  function judge(el) {
    const pv = pvOf(el), bare = !!el.dataset.pv;
    return R.judge({ hareUp: hareState === "open", frame: pv && wire[pv], bare, view: views[el.dataset.device], entry: bare ? null : entryOf(el), fits: fitsOf(el) });
  }

  function draw(el) {
    if (el.dataset.action) return drawAction(el);
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
    const values = valuesOf(el), state = values && R.valued(pv && wire[pv], values);
    if (act && el.dataset.kind !== "button") act.textContent = values ? state : show(pv, el);
    const sel = el.querySelector(".sel");
    if (sel && !sel.dataset.draft) fillSelect(sel, meta, pv && wire[pv]);
    const open = R.writable(entryOf(el), !!el.dataset.pv, !!el.dataset.busy, fitsOf(el));
    for (const c of [sp, act, sel]) if (c) c.disabled = !open;
    if (sel && !R.options(meta).length) sel.disabled = true; // choices unread: the module would refuse
    if (act && values) {
      if (R.flipTo(state, values) === null) act.disabled = true; // unknown: nothing to toggle from
      act.title = state === "unknown" ? `reads ${show(pv, el)}: neither on (${values.on}) nor off (${values.off})` : "";
    }
    for (const ind of el.querySelectorAll(".ind")) ind.querySelector("b").textContent = show(pvOf(ind), ind);
    const rb = el.querySelector(".rb");
    if (rb) {
      const rbPv = pvOf(el, el.dataset.readback);
      rb.textContent = show(rbPv, el);
      const bothRead = !!(pv && wire[pv] && rbPv && wire[rbPv]);
      el.querySelector(".differs").hidden = !R.differs(show(pv, el), show(rbPv, el), bothRead);
    }
  }

  // a select's options from the enum's choices (rebuilt only when they change), its value the readback's
  function fillSelect(sel, meta, frame) {
    const opts = R.options(meta), sig = opts.map((o) => o.label).join("\n");
    if (sel.dataset.sig !== sig) {
      const none = new Option("—", "");
      none.disabled = true;
      sel.replaceChildren(none, ...opts.map((o) => new Option(o.label, String(o.value))));
      sel.dataset.sig = sig;
    }
    sel.value = R.chosen(frame, meta);
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
  function ask(verb, what, note, el) {
    document.getElementById("confirm-title").textContent = `${verb} ${el.querySelector(".lbl").textContent}?`;
    document.getElementById("confirm-what").textContent = what;
    document.getElementById("confirm-note").textContent = note;
    document.getElementById("confirm-go").textContent = verb;
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

  async function write(el, value, shown) {
    const operator = opInput.value.trim();
    if (!operator) { opInput.focus(); return outcome(el, "unsent", "", "name yourself in the header first"); }
    // one decision: the confirm sent is the dialog the operator answered (the module's mark may change meanwhile)
    const asked = hazardOf(el);
    const what = `${el.dataset.device} · ${el.dataset.role} ← ${shown ?? value}`;
    if (asked && !(await ask("Write", what, el.dataset.confirm || "This role needs the hazard confirm.", el))) return outcome(el, "unsent", "", "not confirmed");
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

  // ---- action plans: the module's library and its runs, by plan
  const plans = {}, runs = {}, runOf = {}, watching = new Set();
  let plansError = "not read yet";
  const ofPlan = (plan) => widgets.filter((w) => w.dataset.action === plan);

  function drawAction(el) {
    const plan = plans[el.dataset.action], run = runs[runOf[el.dataset.action]];
    const live = !!run && !R.runEnded(run);
    const state = el.querySelector(".ws"), act = el.querySelector(".act");
    state.querySelector(".word").textContent = run ? R.runWord(run) : plan ? "ready" : plansError ? "unknown" : "no such plan";
    state.querySelector(".dot").dataset.state = run ? R.runState(run) : plan ? "ok" : "unknown";
    state.title = run ? R.runDetail(run) : "";
    act.disabled = !plan || live || !!el.dataset.busy;
    act.title = plan ? plan.description : plansError || "not in the module's library";
    el.querySelector(".stop").hidden = !live;
  }

  function track(plan, run) {
    runs[run.id] = run;
    runOf[plan] = run.id;
    if (!R.runEnded(run)) watching.add(run.id);
    ofPlan(plan).forEach(drawAction);
  }

  async function readPlans() {
    try {
      const r = await fetch(`${root}/api/actions`);
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || `answered ${r.status}`);
      plansError = null;
      for (const p of body.plans) plans[p.name] = p;
      for (const p of body.plans) if (p.running && !runs[p.running]) track(p.name, { id: p.running, plan: p.name, state: "running", steps: [] });
    } catch (err) {
      plansError = `the module's actions: ${err.message}`;
    }
    actionWidgets.forEach(drawAction);
  }

  async function runAction(el) {
    const operator = opInput.value.trim();
    if (!operator) { opInput.focus(); return outcome(el, "unsent", "", "name yourself in the header first"); }
    await readPlans(); // the hazard steps as the module sees them now
    const plan = plans[el.dataset.action];
    if (!plan) return outcome(el, "unsent", "", plansError || "not in the module's library");
    const hazards = (plan.hazardSteps || []).map((h) => `step ${h.index}: ${h.device} · ${h.role} ← ${h.value}`);
    const asked = hazards.length > 0 || !!el.dataset.confirm;
    const note = [el.dataset.confirm, hazards.length ? `Hazard ${hazards.join("; ")}` : ""].filter(Boolean).join(" — ");
    if (asked && !(await ask("Run", plan.description || plan.name, note, el))) return outcome(el, "unsent", "", "not confirmed");
    outcome(el, "sending");
    el.dataset.busy = "1";
    drawAction(el);
    try {
      const r = await fetch(`${root}/api/actions/${encodeURIComponent(plan.name)}/run`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ operator, confirm: asked }),
      });
      const a = await r.json();
      if (r.status === 202) { outcome(el, ""); track(plan.name, a); }
      else outcome(el, [403, 409, 428].includes(r.status) ? "denied" : "refused", a.code, a.detail);
    } catch (err) {
      outcome(el, "unknown", "no_answer", "read the run before running it again");
    }
    delete el.dataset.busy;
    ofPlan(plan.name).forEach(drawAction);
  }

  // a 404: the module no longer holds the run (restarted, or past the last 100): it is lost, not running
  async function readRun(id, signal) {
    const r = await fetch(`${root}/api/actions/runs/${encodeURIComponent(id)}`, { signal });
    if (r.status === 404) return { ...runs[id], state: "lost" };
    if (!r.ok) throw new Error(`answered ${r.status}`);
    return r.json();
  }

  function seenRun(id, got) {
    if (!got.ok) return;
    if (R.runEnded(got.value)) watching.delete(id);
    const plan = got.value.plan || Object.keys(runOf).find((p) => runOf[p] === id);
    if (plan) track(plan, got.value);
  }

  const actionWidgets = widgets.filter((w) => w.dataset.action);
  for (const el of actionWidgets) {
    el.querySelector(".stop").addEventListener("click", () => {
      const id = runOf[el.dataset.action];
      if (id) fetch(`${root}/api/actions/runs/${encodeURIComponent(id)}/stop`, { method: "POST" }).catch(() => {});
    });
  }
  if (actionWidgets.length) {
    readPlans();
    const runPoller = window.GeecsPoller.makePoller({ fetch: readRun, pollMs: POLL_MS / 2, onView: seenRun });
    setInterval(() => runPoller.tick([...watching]), POLL_MS / 2);
    setInterval(readPlans, 5 * POLL_MS); // a module back up, a library edited, a run started elsewhere
  }

  const asValue = (text) => (text.trim() !== "" && Number.isFinite(Number(text)) ? Number(text) : text);

  for (const ind of document.querySelectorAll(".ind[data-pv]")) use(ind.dataset.pv, ind);
  for (const el of widgets) {
    if (el.dataset.pv) use(el.dataset.pv, el);
    const sp = el.querySelector(".sp"), act = el.querySelector(".act"), sel = el.querySelector(".sel");
    if (sel) sel.addEventListener("change", async () => {
      if (sel.value === "") return;
      sel.dataset.draft = "1"; // the chosen option stays while the dialog asks and the write runs
      await write(el, Number(sel.value), sel.selectedOptions[0].textContent);
      delete sel.dataset.draft;
      draw(el);
    });
    if (sp) {
      sp.addEventListener("input", () => { sp.dataset.draft = "1"; });
      sp.addEventListener("keydown", (ev) => {
        if (ev.key === "Escape") { delete sp.dataset.draft; draw(el); }
        if (ev.key === "Enter" && sp.dataset.draft) { delete sp.dataset.draft; write(el, asValue(sp.value)); }
      });
    }
    if (act) act.addEventListener("click", () => {
      if (el.dataset.action) return runAction(el);
      if (el.dataset.kind === "button") return write(el, 1);
      const f = wire[pvOf(el)], values = valuesOf(el);
      if (values) {
        const to = R.flipTo(R.valued(f, values), values);
        return to === null ? undefined : write(el, to);
      }
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
