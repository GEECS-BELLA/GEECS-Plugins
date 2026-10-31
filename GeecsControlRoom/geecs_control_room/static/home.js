// The Control Room's live half: the strip from /api/home/strip (the scanner's
// words), the two lists from /api/home/devices (the module's words, sorted
// there), the device pickers from the same roster, the presets from
// /api/home/presets (Squirrel's snapshots through the module: a save, a diff
// drawer, a restore as a run shown on the row, the golden mark) and the strip's
// baseline chip from /api/home/baseline, polled slowly. Nothing here judges a
// device: a list the module could not give reads "unknown", never "none".
// The rules are home_rules.js's and screen_rules.js's (tested under node), the polling poller.js's.
(function () {
  "use strict";
  const POLL_MS = 2000, MAX_POINTS = 600;
  const root = document.body.dataset.root || "";
  const $ = (id) => document.getElementById(id);
  const deviceHref = (name) => `${root}/api/devices/${encodeURIComponent(name)}`;

  function cell(text) {
    const td = document.createElement("td");
    td.textContent = text === null || text === undefined || text === "" ? "—" : text;
    return td;
  }

  function lastCommand(row) {
    if (!row) return "—";
    const axis = row.axis ? `[${row.axis}]` : "", code = row.canopy_code ? ` · ${row.canopy_code}` : "";
    const when = new Date(row.time).toLocaleTimeString();
    return `${row.outcome}${code} · ${row.role}${axis} ← ${JSON.stringify(row.value)} · ${row.operator} · ${when}`;
  }

  function list(key, body) {
    const journal = body.journal !== false; // a journal the module could not give is unknown, not "no command"
    const panel = document.querySelector(`[data-list="${key}"]`), v = window.GeecsHomeRules.listView(key, body);
    const chip = panel.querySelector("header .chip"), pane = panel.querySelector(".state"), scroll = panel.querySelector(".tscroll");
    $(`n-${key}`).textContent = v.count;
    chip.dataset.state = v.chipState;
    chip.textContent = v.chipText;
    chip.title = v.title;
    scroll.hidden = !v.rows.length;
    pane.hidden = !scroll.hidden;
    pane.dataset.state = v.none ? "empty" : "error";
    pane.querySelector(".t").textContent = v.none ? "none" : "unknown";
    pane.querySelector(".m").textContent = v.note;
    panel.querySelector("tbody").replaceChildren(...v.rows.map((d) => {
      const tr = document.createElement("tr"), name = cell(""), link = document.createElement("a");
      link.href = deviceHref(d.name);
      link.textContent = d.name;
      name.className = "id";
      name.replaceChildren(link);
      const state = cell(""), dot = document.createElement("span");
      dot.className = "dot";
      dot.dataset.state = d.kit;
      state.replaceChildren(dot, ` ${d.state}`);
      tr.append(name, cell(d.family), state, cell(d.cause), cell(d.ioc), cell(journal ? lastCommand(d.last) : "unknown"));
      return tr;
    }));
  }

  let filled = "", families = [];
  function pickers(devices) {
    families = [...new Set(devices.map((d) => d.family).filter(Boolean))].sort();
    const signature = JSON.stringify(devices);
    if (signature === filled) return;
    filled = signature;
    for (const select of document.querySelectorAll("select[data-families]")) {
      if (select.dataset.reason) continue; // disabled by the launcher, with its reason
      const families = select.dataset.families.split(" ");
      const names = devices.filter((d) => families.includes(d.family)).map((d) => d.name);
      select.replaceChildren(
        new Option(names.length ? "choose a device…" : "no such device", ""),
        ...names.map((n) => new Option(n, n)),
      );
      select.disabled = !names.length;
    }
  }
  document.addEventListener("change", (ev) => {
    const select = ev.target.closest("select[data-families]");
    if (select && select.value) window.location.href = deviceHref(select.value);
  });

  let points = [], scan = null;
  function strip(s) {
    const run = $("run");
    run.dataset.state = s.kit;
    run.textContent = s.run;
    $("scanner").textContent = s.scanner === "ok" ? "scanner" : `scanner ${s.scanner}`;
    const known = s.scan_number !== null && s.scan_number !== undefined;
    $("scan").textContent = known ? String(s.scan_number) : "—";
    $("shot").textContent = known ? `${s.shots_done ?? 0}${s.planned_total ? ` / ${s.planned_total}` : ""}` : "—";
    if (s.scan_number !== scan) { scan = s.scan_number; points = []; }
    // the chip is the manager's word; a scan is open only while its documents say so (not an mv)
    const open = s.state === "running" || s.state === "paused";
    const active = open && (s.run === "running" || s.run === "paused");
    if (active) points = [...points, [Date.now(), s.shots_done || 0]].slice(-MAX_POINTS);
    $("plot-fig").hidden = !active || points.length < 2;
    if ($("plot-fig").hidden) return;
    const t0 = points[0][0], span = points[points.length - 1][0] - t0 || 1;
    const top = Math.max(1, s.planned_total || 0, ...points.map((p) => p[1]));
    $("plot").setAttribute("points", points
      .map(([t, y]) => `${(((t - t0) / span) * 240).toFixed(1)},${(46 - (y / top) * 44).toFixed(1)}`)
      .join(" "));
  }

  // ---- presets: Squirrel's snapshots through the module; a restore is a run, shown on its row
  const H = window.GeecsHomeRules, R = window.GeecsScreenRules;
  const PRESETS = "/api/home/presets", BASELINE = "/api/home/baseline";
  let presets = [], screens = [], selection = null, drawn = "", runs = {}, runOf = {}, watching = new Set();
  const opInput = $("operator"), OP_KEY = "geecs-control-room-operator"; // the header's, remembered as on a screen
  try { opInput.value = localStorage.getItem(OP_KEY) || ""; } catch (_) { /* no storage */ }
  opInput.addEventListener("change", () => { try { localStorage.setItem(OP_KEY, opInput.value.trim()); } catch (_) { /* no storage */ } });
  const json = async (path, body) => {
    const r = await fetch(root + path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : undefined);
    return [r.status, await r.json()];
  };
  const ask = (dialog, yes) => new Promise((done) => {
    // the dialog's buttons decide; an <option> picked in the scope list has a value too, and must not
    dialog.onclick = (ev) => { if (ev.target.tagName === "BUTTON" && ev.target.value) { dialog.close(); done(ev.target.value === yes); } };
    dialog.oncancel = () => done(false);
    dialog.showModal();
  });

  function drawPresets(body) {
    const chip = $("presets-chip"), pane = $("presets-state"), scroll = document.querySelector("#presets .tscroll");
    if (!body.ok) {
      chip.dataset.state = "unknown"; chip.textContent = "unknown";
      pane.hidden = false; pane.dataset.state = "error"; pane.querySelector(".t").textContent = "unknown"; pane.querySelector(".m").textContent = body.detail || "";
      scroll.hidden = true;
      drawn = ""; // the next good list must redraw, whatever it holds
      return;
    }
    presets = body.presets; screens = body.screens || [];
    chip.dataset.state = "ok"; chip.textContent = String(presets.length);
    // redraw only on a change: a rebuilt row loses the operator's hover and focus
    const signature = JSON.stringify([presets, Object.values(runs).map((r) => [r.id, r.state, (r.steps || []).map((x) => x.state)])]);
    if (signature === drawn) return;
    drawn = signature;
    pane.hidden = presets.length > 0; scroll.hidden = !presets.length;
    pane.dataset.state = "empty"; pane.querySelector(".t").textContent = "none saved"; pane.querySelector(".m").textContent = "Save… keeps the live configuration as a preset.";
    $("preset-rows").replaceChildren(...presets.map((p) => {
      const v = H.presetRow(p), tr = document.createElement("tr"), run = runs[runOf[p.id]];
      const name = cell(p.name); if (v.golden) { const g = document.createElement("span"); g.className = "chip"; g.dataset.state = "ok"; g.textContent = "golden"; name.append(" ", g); }
      const runCell = cell(""), word = document.createElement("span"); word.className = "chip";
      word.dataset.state = run ? R.runState(run) : "unknown"; word.textContent = run ? R.runWord(run) : "—"; word.title = run ? R.runDetail(run) : "";
      if (run) runCell.replaceChildren(word);
      const acts = cell(""), row = document.createElement("div"); row.className = "row";
      for (const [label, act] of [["Diff", () => openDiff(p)], ["Restore", () => openDiff(p, true)], ["Golden", () => golden(p)]]) {
        const b = document.createElement("button"); b.className = "btn sm"; b.type = "button"; b.textContent = label; b.disabled = label === "Golden" && v.golden; b.addEventListener("click", act); row.append(b);
      }
      acts.replaceChildren(row);
      tr.append(name, cell(v.when), cell(v.by), cell(v.scope), runCell, acts);
      return tr;
    }));
  }

  function scopeOptions() {
    const select = $("preset-scope");
    const group = (label, kind, names) => { const g = document.createElement("optgroup"); g.label = label; for (const n of names) { const o = new Option(n, `${kind}:${n}`); g.append(o); } return g; };
    select.replaceChildren(group("Families", "families", families), group("Screens", "screen", screens));
  }

  async function savePreset() {
    scopeOptions(); // afresh: no name and no picks carried from the last save
    $("preset-name").value = "";
    const operator = opInput.value.trim();
    if (!operator) { opInput.focus(); return note("unsent", "name yourself in the header first"); }
    if (!(await ask($("preset-save"), "save"))) return;
    const name = $("preset-name").value.trim();
    if (!name) return note("unsent", "a preset needs a name");
    const scope = { devices: [], families: [], screen: [] };
    for (const o of $("preset-scope").selectedOptions) { const [kind, n] = o.value.split(/:(.*)/); scope[kind].push(n); }
    note("saving…");
    const [status, a] = await json("/api/presets", { name, operator, scope });
    note(status === 201 ? "saved" : `${a.code || status}`, a.detail || "", status === 201 ? "ok" : "failed");
    slow.tick([PRESETS]);
  }
  // the presets chip carries the last action's word until the next list read
  function note(word, title, state) {
    const chip = $("presets-chip");
    chip.dataset.state = state || "unknown"; chip.textContent = word; chip.title = title || "";
  }

  async function openDiff(p, restoring) {
    selection = H.openSelection(selection, p);
    const generation = selection.generation; // this open's; a later open or close supersedes it
    $("preset-diff-title").textContent = `${p.name} · saved vs live`;
    $("preset-diff-note").textContent = "reading…";
    $("preset-diff-rows").replaceChildren();
    const drawer = window.GeecsKit.drawer($("preset-diff"));
    drawer.open();
    let status, body;
    try {
      [status, body] = await json(`/api/presets/${encodeURIComponent(p.id)}/diff`);
    } catch (err) {
      [status, body] = [0, { code: "no_answer", detail: "the Control Room did not answer; open the diff again" }];
    }
    if (!H.stillCurrent(selection, generation, drawer.isOpen)) return; // closed or superseded: not this drawer's answer any more
    if (status !== 200) { $("preset-diff-note").textContent = `${body.code || status}: ${body.detail || ""}`; return; }
    selection = H.takeDiff(selection, generation, body.rows, drawer.isOpen);
    $("preset-diff-note").textContent = H.diffSummary(body.rows);
    $("preset-diff-rows").replaceChildren(...body.rows.map((r) => {
      const tr = document.createElement("tr"), pick = cell(""), box = document.createElement("input");
      box.type = "checkbox"; box.value = r.device; box.disabled = r.word !== "differs"; box.checked = r.word === "differs"; box.setAttribute("aria-label", `restore ${r.device}`);
      pick.replaceChildren(box);
      const word = cell(""), dot = document.createElement("span"); dot.className = "dot"; dot.dataset.state = H.diffState(r.word);
      word.replaceChildren(dot, ` ${H.diffNote(r.word)}`);
      tr.append(pick, cell(r.device), cell(r.axis ? `${r.role}[${r.axis}]` : r.role), cell(JSON.stringify(r.saved)), cell(JSON.stringify(r.live)), cell(r.tolerance), word);
      return tr;
    }));
    if (restoring) restore(generation);
  }

  let restoreInFlight = false; // a restore in flight: the drawer sends one at a time, once
  async function restore(asked) {
    const drawer = window.GeecsKit.drawer($("preset-diff"));
    const generation = asked === undefined ? selection.generation : asked; // the Restore button's: the drawer's own now
    if (!H.stillCurrent(selection, generation, drawer.isOpen)) return; // the drawer closed or moved on
    if (restoreInFlight) { $("preset-diff-note").textContent = "a restore is in flight; Restore… again once it answers"; return; }
    const p = selection.preset, operator = opInput.value.trim();
    if (!operator) { drawer.close(); opInput.focus(); return note("unsent", "name yourself in the header first"); }
    const picked = [...new Set([...document.querySelectorAll("#preset-diff-rows input:checked")].map((b) => b.value))];
    const plan = H.restorePlan(selection.rows, picked);
    if (!plan.rows.length) { $("preset-diff-note").textContent = `${plan.text}; nothing to restore`; return; }
    $("preset-restore-what").textContent = `${p.name} → ${plan.text}`;
    $("preset-restore-note").textContent = "Each row goes through the module's gates; a hazard role is confirmed here, once.";
    if (!(await ask($("preset-restore"), "restore"))) return;
    restoreInFlight = true;
    $("preset-restore-open").disabled = true;
    let status, a;
    try {
      [status, a] = await json(`/api/presets/${encodeURIComponent(p.id)}/restore`, { operator, confirm: true, devices: plan.devices });
    } catch (err) {
      [status, a] = [0, { code: "no_answer", detail: "read the journal before restoring again" }];
    } finally {
      restoreInFlight = false;
      $("preset-restore-open").disabled = false;
    }
    if (status === 202) track(p.id, a); // the run is the row's, whatever the drawer shows now
    if (!H.stillCurrent(selection, generation, drawer.isOpen)) {
      // the drawer moved on: its note is not this answer's, so a refusal is said on the presets chip instead
      if (status !== 202) note(a.code || String(status), `${p.name}: ${a.detail || ""}`, "failed");
      return;
    }
    if (status === 202) drawer.close();
    else $("preset-diff-note").textContent = `${a.code || status}: ${a.detail || ""}`;
  }

  async function golden(p) {
    const operator = opInput.value.trim();
    if (!operator) { opInput.focus(); return note("unsent", "name yourself in the header first"); }
    const [status, a] = await json(`/api/presets/${encodeURIComponent(p.id)}/golden`, { operator });
    if (status !== 200) note(a.code || String(status), a.detail || "", "failed");
    slow.tick([PRESETS, BASELINE]);
  }

  function track(id, run) { runs[run.id] = run; runOf[id] = run.id; if (!R.runEnded(run)) watching.add(run.id); drawPresets({ ok: true, presets, screens }); }
  async function readRun(id, signal) {
    const r = await fetch(`${root}/api/actions/runs/${encodeURIComponent(id)}`, { signal });
    if (r.status === 404) return { ...runs[id], state: "lost" };
    if (!r.ok) throw new Error(`answered ${r.status}`);
    return r.json();
  }
  function seenRun(id, got) {
    if (!got.ok) return;
    if (R.runEnded(got.value)) { watching.delete(id); refreshBaseline(); }
    const preset = Object.keys(runOf).find((k) => runOf[k] === id);
    if (preset) track(preset, got.value);
  }

  // a run just ended: the baseline is read fresh, past the Control Room's 10 s cache
  const refreshBaseline = () => read(BASELINE + "?fresh=1").then(baseline, () => {});

  function baseline(body) {
    const v = H.baselineChip(body), chip = $("baseline");
    chip.closest(".live").hidden = v.hidden;
    chip.dataset.state = v.state; chip.textContent = v.text; chip.title = v.title;
  }

  // a read that fails or times out backs off; while it does, the page says the module is slow
  const DEVICES = "/api/home/devices", STRIP = "/api/home/strip", SLOW_MS = 10000;
  const draw = {
    [DEVICES]: (body) => {
      list("not_live", body);
      list("in_alarm", body);
      if (body.ok) pickers(body.devices);
    },
    [STRIP]: strip,
    [PRESETS]: drawPresets,
    [BASELINE]: baseline,
  };
  const silent = {
    [DEVICES]: (why) => ({ ok: false, detail: `the Control Room did not answer (${why}); asking again shortly` }),
    [STRIP]: () => ({ scanner: "unreachable", run: "unknown", kit: "unknown" }),
    [PRESETS]: (why) => ({ ok: false, detail: `the Control Room did not answer (${why})` }),
    [BASELINE]: (why) => ({ ok: false, detail: `the Control Room did not answer (${why})` }),
  };
  const read = async (path, signal) => (await fetch(root + path, { signal })).json();
  const onView = (path, got) => draw[path](got.ok ? got.value : silent[path](got.error));
  const poller = window.GeecsPoller.makePoller({ fetch: read, pollMs: POLL_MS, onView });
  // the presets and the baseline: the golden snapshots' diff reads every saved row live, so a slow cadence
  const slow = window.GeecsPoller.makePoller({ fetch: read, pollMs: SLOW_MS, onView });
  const runPoller = window.GeecsPoller.makePoller({ fetch: readRun, pollMs: POLL_MS / 2, onView: seenRun });
  const tick = () => poller.tick([DEVICES, STRIP]);
  tick();
  setInterval(tick, POLL_MS);
  slow.tick([PRESETS, BASELINE]);
  setInterval(() => slow.tick([PRESETS, BASELINE]), SLOW_MS);
  setInterval(() => runPoller.tick([...watching]), POLL_MS / 2);
  $("preset-save-open").addEventListener("click", savePreset);
  $("preset-restore-open").addEventListener("click", () => restore());
})();
