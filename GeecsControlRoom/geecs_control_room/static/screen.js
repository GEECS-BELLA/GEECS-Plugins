// The screen renderer's live half. Each device's PVs and verdicts come from
// the module's device route (polled); values, connection and alarms from
// Hare. A widget's word claims no more than that evidence: the module's
// verdict for a device role, Hare's connection and alarm for a bare PV,
// "unknown" while either source is silent. A changing value is never read
// as freshness. Every control is the rail's (device_rail.js over each
// widget's page: its device's, GeecsDeviceSource, or its bare PV's; a
// deferred widget, a DB device with data-fits, refuses a role the module
// types otherwise, "wrong kind") and every write device_writes.js's (the
// module's put route, the operator's name, sent once, the outcome on the
// widget's own line, the widget's hazard note asked first). This file keeps
// the polling, the words, a flipper's lamps and the action buttons: a button
// naming an action plan asks the module to run it (once, confirmed when the
// plan writes a hazard role) and shows the run as the module reports it,
// polled until it ends, with Stop while it runs. The rules are
// screen_rules.js's and device_rules.js's (tested under node).
(function () {
  "use strict";
  const POLL_MS = 2000;
  const root = document.body.dataset.root || "";
  const KIT = JSON.parse(document.getElementById("kit-state").textContent);
  const R = window.GeecsScreenRules, D = window.GeecsDeviceRules;
  const hare = window.GeecsHare.fromPage();
  const wire = {};           // pv -> Hare's latest frame (null while the socket is down)
  const views = {};          // device -> the module's device route as kept (D.deviceView)
  const slow = new Set();    // devices the module is slow to answer for
  const listeners = [];
  let hareState = "connecting";
  const hareUp = () => hareState === "open";
  const changed = () => listeners.forEach((f) => f());
  const opInput = document.getElementById("operator");

  const widgets = [...document.querySelectorAll(".w")];
  const sources = {}, writes = {};
  const deviceOf = (name) => (sources[name] = sources[name] || window.GeecsDeviceSource.make({ name, root, hare, view: () => views[name] || null, hareUp, wire, changed }));  // fmt: skip
  const readOnly = window.GeecsDeviceWrites.make(null, changed); // a bare PV's (never written: its controls are disabled)
  const writesOf = (name) => (name ? (writes[name] = writes[name] || window.GeecsDeviceWrites.make(deviceOf(name), changed)) : readOnly);
  const heard = new Set();
  const watch = (pv) => { if (!heard.has(pv)) { heard.add(pv); hare.watch([pv], (n, f) => { wire[n] = f; changed(); }); } return wire[pv]; };
  // A bare PV: no module entry (its controls disabled), Hare's connection and alarm.
  const bare = (pv) => ({
    name: null, root, view: () => ({ ok: true, roles: {} }), entry: () => undefined, target: () => null,
    frame: () => watch(pv), meta: () => hare.meta(pv), word: () => R.judge({ hareUp: hareUp(), frame: watch(pv), bare: true }),
  });
  // A widget's page: its device's or its bare PV's; a deferred widget's role hidden from the controls when the
  // module types it otherwise, its word the module's ("unknown role", "wrong kind"); its own precision shown.
  function pageOf(w) {
    if (w.page) return w.page;
    const base = w.dataset.pv ? bare(w.dataset.pv) : deviceOf(w.dataset.device);
    const fits = w.dataset.fits === undefined ? undefined : w.dataset.fits.split(" ").filter(Boolean);
    const digits = w.dataset.precision === undefined ? undefined : Number(w.dataset.precision);
    const shown = (meta) => (digits === undefined ? meta : { ...meta, digits, precision: digits });
    return (w.page = {
      ...base,
      entry: (role, axis) => { const e = base.entry(role, axis); return R.misfit(e, fits) ? undefined : e; },
      word: (role, axis) => (fits ? R.judge({ hareUp: hareUp(), frame: base.frame(role, axis), bare: false, view: base.view(), entry: base.entry(role, axis), fits }) : base.word(role, axis)),
      meta: (role, axis) => shown(base.meta(role, axis)),
      target: (role, axis) => { const t = base.target(role, axis); return t && { ...t, meta: shown(t.meta) }; },
    });
  }
  const page = { for: (el) => pageOf(el.closest(".w")), on: (f) => listeners.push(f), view: () => null };
  window.GeecsDeviceRail.make(page, document.querySelector("main.screen"), { for: (el) => writesOf(el.closest(".w").dataset.device) });

  // ---- the words beside a control, and a flipper's lamps
  function draw() {
    for (const w of widgets) {
      if (w.dataset.action) continue;
      const ws = w.querySelector(":scope > .ws"), P = pageOf(w);
      if (ws) {
        const word = P.word(w.dataset.role || w.dataset.pv, w.dataset.axis);
        ws.querySelector(".dot").dataset.state = KIT[word] || "unknown";
        ws.querySelector(".word").textContent = word;
      }
      for (const ind of w.querySelectorAll(".ind")) {
        const L = ind.dataset.pv ? bare(ind.dataset.pv) : deviceOf(ind.dataset.device), role = ind.dataset.role || ind.dataset.pv;
        ind.querySelector("b").textContent = R.show(L.frame(role, ind.dataset.axis), L.meta(role, ind.dataset.axis), w.dataset.precision);
      }
    }
  }
  listeners.push(draw);

  // a 5xx is the module silent (back off); any other answer is the module's word
  async function readDevice(device, signal) {
    const r = await fetch(`${root}/api/devices/${encodeURIComponent(device)}`, { signal });
    if (r.status >= 500) throw new Error(`answered ${r.status}`);
    return D.deviceView(r.ok, await r.json());
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
    changed();
  }

  // ---- an action run asks once, behind its own dialog; its outcome stays on the widget
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
    el.querySelector(".act").addEventListener("click", () => runAction(el));
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

  hare.onStatus((s) => {
    hareState = s;
    source();
    changed();
  });
  // every device a widget or an indicator names (a bare-PV flipper's lamp may be a device's role)
  const devices = [...new Set([...document.querySelectorAll(".w[data-device], .ind[data-device]")].map((e) => e.dataset.device))];
  const poller = window.GeecsPoller.makePoller({ fetch: readDevice, pollMs: POLL_MS, onView: seen });
  const tick = () => poller.tick(devices);
  tick();
  setInterval(tick, POLL_MS);
})();
