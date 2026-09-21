/* GEECS Scanner — the page's one script.
 *
 * Fills every value from /api/* and keeps it live over /api/events (SSE).
 * No build step, no framework: the kit does the layout, this does the
 * data. Every fetch goes through ROOT so the page works under a proxy
 * prefix; every error the API returns ({error:{kind,message,...}}) is
 * shown in words, never swallowed.
 */
(function () {
  "use strict";

  var ROOT = document.body.dataset.root || "";
  var PORTAL = (document.body.dataset.portal || "").replace(/\/$/, "");
  var $ = function (id) { return document.getElementById(id); };
  var STALE_DOCS_S = 5;      // a running scan whose documents go quiet
  var STALE_MANAGER_S = 4;   // the status poll is every second

  /* ---------------------------------------------------------------- api */

  function api(path, opts) {
    return fetch(ROOT + path, opts).then(function (r) {
      return r.text().then(function (t) {
        var body = {};
        try { body = t ? JSON.parse(t) : {}; } catch (e) { body = {}; }
        if (!r.ok) {
          var err = new Error((body.error && body.error.message) || (r.status + " " + r.statusText));
          err.status = r.status;
          err.payload = body.error || {};
          throw err;
        }
        return body;
      });
    });
  }
  function post(path, data) {
    return api(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(data || {})
    });
  }

  /* -------------------------------------------------------------- state */

  var S = {
    status: null, statusAt: 0,
    progress: null, optimization: null, optimizer: null, optimizerRequest: 0,
    queue: null,
    presets: [], presetName: null, presetDoc: null, loadedName: null,
    variables: [], triggers: [],
    mode: "", acq: "strict",
    consoleSeq: 0, epoch: null,
    formable: true, formableNote: "",
    pendingPreset: null, pendingAck: [],
    devices: [], actions: [], actionName: null, armed: false, calibration: null,
    settables: [], settablesNote: "", readbackVar: null, readbackTimer: null,
    tail: "scanlog", logFolder: null, logLines: [], consoleLines: []
  };

  var composer = window.GEECS_SWEEP.create($("sweep-composer"), function (payload, signal) {
    return api("/api/trajectory", {method: "POST", headers: {"content-type": "application/json"}, body: JSON.stringify(payload), signal: signal});
  }, recalc);

  /* ------------------------------------------------------------- errors */

  function showError(text) {
    $("page-error-text").textContent = text;
    $("page-error").hidden = false;
  }
  $("page-error-close").addEventListener("click", function () { $("page-error").hidden = true; });

  /* ---------------------------------------------------------- operator */

  var OP_KEY = "geecs.operator";
  try { $("operator").value = window.localStorage.getItem(OP_KEY) || ""; } catch (e) { /* private window */ }
  $("operator").addEventListener("change", function () {
    try { window.localStorage.setItem(OP_KEY, $("operator").value.trim()); } catch (e) { /* ignore */ }
  });
  function operator() { return $("operator").value.trim() || null; }

  /* ---------------------------------------------------------- chips */

  // Every kit status word this page can set, spelled once; tests pin these
  // against geecs_web_theme.STATES, and setChip never takes a literal.
  var K = { ok: "ok", running: "running", paused: "paused", degraded: "degraded", failed: "failed", unknown: "unknown", queued: "queued" };

  function setChip(el, state, word, title) {
    el.setAttribute("data-state", state);
    el.textContent = word;
    if (title) el.title = title;
  }

  /* ------------------------------------------------------------- now */

  function fmtAge(s) { return s < 0 ? "" : s.toFixed(1) + " s"; }
  function nowS() { return Date.now() / 1000; }

  function renderStatus() {
    var st = S.status;
    if (!st) return;
    var running = st.re_state === "running" || st.re_state === "paused";
    var runWord = st.re_state === "paused" ? K.paused : K.running;
    if (!st.connected) {
      setChip($("chip-manager"), K.failed, "manager", st.detail || "not answering");
      setChip($("now-chip"), K.failed, "unreachable");
    } else if (st.readiness !== "ready") {
      setChip($("chip-manager"), K.degraded, "manager", st.readiness_detail || st.readiness);
      setChip($("now-chip"), running ? runWord : K.degraded, running ? st.re_state : st.readiness);
    } else {
      setChip($("chip-manager"), K.ok, "manager", "RE Manager answers; " + st.readiness_detail);
      if (running) setChip($("now-chip"), runWord, st.re_state);
      else setChip($("now-chip"), K.unknown, "idle");
    }
    $("btn-pause").disabled = st.re_state !== "running";
    $("btn-pause").hidden = st.re_state === "paused";
    $("btn-resume").hidden = st.re_state !== "paused";
    $("btn-stop").disabled = !running;
    $("btn-clear").disabled = !(st.items_in_queue > 0);
    renderIdleGates();
    $("lv-manager").textContent = st.re_state || (st.connected ? "?" : "down");
    $("verb-note").textContent = st.re_state === "paused"
      ? "Paused. Resume continues at the next step."
      : "Pause takes effect after the current step.";
    if (!running) $("paused-reason").textContent = "";
    updateStartGate();
  }

  function renderDayLink() {
    var p = S.progress, h = $("recent-h");
    h.textContent = "";
    var day = p && p.day;
    if (!day) { h.textContent = "Recent"; return; }
    // The last run's day is "today" only until midnight.
    var now = new Date(), iso = now.getFullYear() + "-" + String(now.getMonth() + 1).padStart(2, "0") + "-" + String(now.getDate()).padStart(2, "0");
    var label = (day === iso ? "Today" : "Last run") + " · " + day;
    if (PORTAL) {
      var a = document.createElement("a"); a.href = PORTAL + "/day/" + day; a.textContent = label; a.title = "the day in the Data Portal";
      h.appendChild(a);
    } else h.textContent = label;
  }
  function renderProgress() {
    var p = S.progress;
    if (!p) return;
    renderDayLink();
    renderOptimization();
    var total = p.planned_total, done = p.shots_done || 0;
    var pct = total ? Math.min(100, Math.round(100 * done / total)) : 0;
    $("meter-fill").style.width = pct + "%";
    var meter = $("meter");
    meter.setAttribute("aria-valuemax", total || 1);
    meter.setAttribute("aria-valuenow", done);
    var st = S.status || {};
    // Idle: the history's word, not the documents' — RunEngine.stop() marks a
    // stopped run's documents "success"; the manager's history says "stopped".
    var last = (!st.re_state || st.re_state === "idle") && S.queue && S.queue.finished[0];
    var word = st.re_state === "paused" ? K.paused
      : last ? last.state
      : (p.state === "done" ? K.ok : (p.state === "aborted" ? K.failed : ""));
    meter.setAttribute("data-state", word);
    $("lv-shots").textContent = done;
    $("lv-planned").textContent = total == null ? "—" : total;
    $("ag-planned").textContent = total == null ? (p.scan_number ? "not in the start document" : "") : "";
    $("meter-l").textContent = total ? done + " / " + (p.plan_name === "optimize" ? "≤ " : "") + total + " shots" : (done ? done + " shots" : "—");
    var right = "";
    if (st.re_state === "paused") right = "paused at shot " + done;
    else if (p.state === "running" && total) right = Math.max(0, total - done) + " shots left";
    else if (last) right = last.word + (last.state === "ok" ? "" : " at shot " + done);
    else if (p.state === "done") right = "done";
    else if (p.state === "aborted") right = "stopped at shot " + done;
    $("meter-r").textContent = right;
    $("now-scan").textContent = p.scan_number ? "Scan " + String(p.scan_number).padStart(3, "0") : "—";
    if (p.paused_reason) $("paused-reason").textContent = "paused: " + p.paused_reason;
    $("lv-docs").textContent = p.available ? (done ? "event #" + done : (p.scan_number ? "start" : "quiet")) : "off";
    if (!p.available) setChip($("chip-docs"), K.degraded, "doc stream", p.detail || "not consuming");
  }

  function tickAges() {
    var now = nowS();
    var p = S.progress, st = S.status;
    if (S.statusAt) {
      var ma = now - S.statusAt;
      $("ag-manager").textContent = fmtAge(ma);
      $("live-manager").toggleAttribute("data-age", ma > STALE_MANAGER_S);
      if (ma > STALE_MANAGER_S) $("live-manager").setAttribute("data-age", "stale");
    }
    if (p && p.updated_at) {
      var da = now - p.updated_at;
      var running = st && st.re_state === "running";
      $("ag-docs").textContent = fmtAge(da);
      var stale = running && p.available && da > STALE_DOCS_S;
      if (stale) { $("live-docs").setAttribute("data-age", "stale"); setChip($("chip-docs"), K.degraded, "doc stream", "last document " + da.toFixed(0) + " s ago while running"); }
      else { $("live-docs").removeAttribute("data-age"); if (p.available) setChip($("chip-docs"), K.ok, "doc stream", "documents flowing"); }
    }
  }
  setInterval(tickAges, 500);

  /* ------------------------------------------------------------- queue */

  function chip(state, word) {
    var s = document.createElement("span");
    s.className = "chip"; s.setAttribute("data-state", state); s.textContent = word;
    return s;
  }
  function td(text, cls) {
    var c = document.createElement("td");
    if (cls) c.className = cls;
    if (text instanceof Node) c.appendChild(text); else c.textContent = text == null ? "" : text;
    return c;
  }
  function row(r, selected) {
    var tr = document.createElement("tr");
    if (selected) tr.setAttribute("aria-selected", "true");
    tr.appendChild(td(chip(r.state, r.word)));
    tr.appendChild(td((r.position ? "#" + r.position + " " : "") + r.summary, "id"));
    tr.appendChild(td(r.user));
    var detail = (r.scan_numbers && r.scan_numbers.length ? "Scan " + r.scan_numbers.map(function (n) { return String(n).padStart(3, "0"); }).join(", ") + (r.detail ? " · " : "") : "") + (r.detail || "");
    tr.appendChild(td(detail, "dim"));
    return tr;
  }

  function renderQueue() {
    var q = S.queue;
    if (!q) return;
    var body = $("qbody");
    body.textContent = "";
    if (q.running) body.appendChild(row(q.running, true));
    q.waiting.forEach(function (r) { body.appendChild(row(r)); });
    q.finished.forEach(function (r) { body.appendChild(row(r)); });
    if (!body.children.length) body.appendChild(td("Nothing queued, nothing finished today.", "dim"));
    $("q-summary").textContent = q.summary;
    renderProgress();
    $("dlg-clear-n").textContent = q.waiting.length + " waiting item" + (q.waiting.length === 1 ? "" : "s");
    if (q.running) {
      $("now-plan").textContent = q.running.summary;
      $("now-by").textContent = q.running.user ? "submitted as " + q.running.user : "";
      $("dlg-stop-scan").textContent = S.progress && S.progress.scan_number ? "Scan " + String(S.progress.scan_number).padStart(3, "0") : "the running scan";
    } else {
      $("now-plan").textContent = q.finished.length ? "Idle. Last: " + q.finished[0].summary : "Idle.";
      $("now-by").textContent = "";
    }
    var recent = $("recent");
    recent.textContent = "";
    // Scans only: a move, an action or a calibration opens no run and has no
    // scan number, so it belongs in the queue table, not in "Recent".
    q.finished.filter(function (r) { return r.scan_numbers && r.scan_numbers.length; }).slice(0, 6).forEach(function (r) {
      var a = document.createElement("a");
      // The portal's run page is keyed by the run uid the history row carries.
      a.href = PORTAL && r.run_uids && r.run_uids.length ? PORTAL + "/run/" + encodeURIComponent(r.run_uids[0]) : "#queue";
      if (a.href.indexOf("#") === -1) a.title = "open in the Data Portal";
      var n = r.scan_numbers && r.scan_numbers.length ? "Scan " + String(r.scan_numbers[0]).padStart(3, "0") : r.plan;
      a.innerHTML = "<span></span><span class=\"dim\"></span>";
      a.firstChild.textContent = n;
      a.lastChild.textContent = r.word;
      recent.appendChild(a);
    });
  }

  function refreshQueue() {
    return api("/api/queue").then(function (q) { S.queue = q; renderQueue(); })
      .catch(function (e) { $("q-summary").textContent = e.message; });
  }

  /* --------------------------------------------------------------- SSE */

  var lastKey = "";
  function connect() {
    // The browser reconnects by itself and sends Last-Event-ID (the console
    // frames carry id: "<epoch>:<seq>"), so a blip resumes where it left
    // off; a new epoch means the scanner restarted, and the tail starts over.
    var es = new EventSource(ROOT + "/api/events");
    es.addEventListener("status", function (ev) {
      S.status = JSON.parse(ev.data); S.statusAt = nowS();
      renderStatus();
      var key = [S.status.re_state, S.status.items_in_queue, S.status.running_item_uid].join("|");
      if (key !== lastKey) { lastKey = key; refreshQueue(); }
    });
    es.addEventListener("optimization", function (e) { S.optimization = JSON.parse(e.data); renderOptimization(); });
    es.addEventListener("progress", function (ev) {
      var p = JSON.parse(ev.data);
      var finished = S.progress && S.progress.state !== p.state && (p.state === "done" || p.state === "aborted");
      S.progress = p;
      renderProgress();
      if (finished) refreshQueue();
    });
    es.addEventListener("console", function (ev) {
      var line = JSON.parse(ev.data);
      if (S.epoch !== null && line.epoch !== S.epoch) { S.consoleLines = []; }
      S.epoch = line.epoch;
      S.consoleSeq = line.seq;
      pushLine(S.consoleLines, { at: line.at, text: line.text });
      if (S.tail === "manager") appendTail(line);
    });
    es.addEventListener("log", function (ev) {
      var chunk = JSON.parse(ev.data);
      if (chunk.folder !== S.logFolder) { S.logFolder = chunk.folder; S.logLines = []; if (S.tail === "scanlog") $("tail").textContent = ""; }
      if (!chunk.available) {
        $("tail-note").textContent = chunk.detail || "scan.log not readable from this host";
        return;
      }
      $("tail-note").textContent = "";
      chunk.lines.forEach(function (text) {
        var line = { at: logStamp(text), text: text };
        pushLine(S.logLines, line);
        if (S.tail === "scanlog") appendTail(line);
      });
    });
    es.onerror = function () {
      setChip($("chip-manager"), K.failed, "manager", "event stream lost; reconnecting");
    };
  }

  function pushLine(buf, line) { buf.push(line); while (buf.length > 300) buf.shift(); }
  // scan.log lines start "YYYY-MM-DD HH:MM:SS.mmm LEVEL logger [thread] scan=ScanNNN - message":
  // keep the clock, drop the plumbing.
  function logStamp(text) {
    var m = /^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})/.exec(text);
    return m ? m[2] : null;
  }
  function logBody(text) {
    var m = /^\S+ \S+ (\w+) \S+ \[[^\]]*\] scan=\S+ - (.*)$/.exec(text);
    return m ? (m[1] === "INFO" ? "" : m[1] + " ") + m[2] : text;
  }
  function renderTail() {
    var tail = $("tail"); tail.textContent = "";
    (S.tail === "scanlog" ? S.logLines : S.consoleLines).forEach(appendTail);
    Array.prototype.forEach.call($("tail-seg").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.tail === S.tail));
    });
    $("tail").dataset.empty = S.tail === "scanlog"
      ? (S.logFolder ? "scan.log is empty so far." : "No scan.log yet — the run's folder arrives with the start document.")
      : "No console output yet.";
  }
  $("tail-seg").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-tail]");
    if (!b) return;
    S.tail = b.dataset.tail;
    renderTail();
  });
  function appendTail(line) {
    var tail = $("tail");
    var ln = document.createElement("div");
    ln.className = "ln" + (/FAILED|error|refus|WARNING/i.test(line.text) ? " warn" : "");
    var t = document.createElement("span"); t.className = "t";
    t.textContent = typeof line.at === "number" ? new Date(line.at * 1000).toTimeString().slice(0, 8) : (line.at || "");
    var body = document.createElement("span"); body.textContent = S.tail === "scanlog" ? logBody(line.text) : line.text;
    ln.appendChild(t); ln.appendChild(body);
    tail.appendChild(ln);
    while (tail.children.length > 300) tail.removeChild(tail.firstChild);
    tail.scrollTop = tail.scrollHeight;
  }

  /* ------------------------------------------------------------ configs */

  function option(value, label, disabled, title) {
    var o = document.createElement("option");
    o.value = value; o.textContent = label;
    if (disabled) o.disabled = true;
    if (title) o.title = title;
    return o;
  }

  function renderOptimizerAvailability(listing) {
      var optimizeButton = document.querySelector('[data-mode="optimize"]');
      optimizeButton.disabled = !listing.names.length;
      optimizeButton.title = listing.detail || (listing.names.length ? "" : "No compatible optimizer configs available");
      var unavailable = listing.unavailable || {};
      var availability = $("optimizer-availability");
      var issues = Object.keys(unavailable).map(function (n) { return n + ": " + unavailable[n]; });
      availability.textContent = listing.detail ? "Unable to load optimizer configs: " + listing.detail :
        (issues.length ? "Unavailable optimizer configs — " + issues.join("; ") : optimizeButton.title);
      availability.hidden = !availability.textContent;
  }

  function loadConfigs() {
    return Promise.all([
      api("/api/configs/presets"),
      api("/api/scan-variables"),
      api("/api/configs/trigger_profiles"),
      api("/api/devices").catch(function () { return []; }),
      api("/api/actions").catch(function (e) { return { error: e.message }; }),
      api("/api/calibration").catch(function (e) { return { stored: false, detail: e.message }; }),
      api("/api/settables").catch(function (e) { return { items: [], source: "?", detail: e.message }; }),
      api("/api/configs/optimizer_configs").catch(function (e) { return { names: [], detail: e.message }; })
    ]).then(function (res) {
      S.presets = res[0].names; S.variables = res[1]; S.triggers = res[2].names;
      S.devices = res[3]; S.actions = res[4].error ? [] : res[4]; S.calibration = res[5];
      var optimizers = $("optimizer-config");
      optimizers.textContent = ""; optimizers.appendChild(option("", "Choose an optimizer…"));
      res[7].names.forEach(function (n) { optimizers.appendChild(option(n, n)); });
      renderOptimizerAvailability(res[7]);
      S.settables = res[6].items || []; S.settablesNote = res[6].detail || "";
      renderMoveVars(); renderDeviceList(""); renderActions(res[4].error || null); renderCalibration();
      renderPresetList();
      composer.variables(S.variables.filter(function (v) { return v.scannable; }).concat(S.settables));
      var tr = $("trig"); tr.textContent = "";
      tr.appendChild(option("", "— none —"));
      S.triggers.forEach(function (n) { tr.appendChild(option(n, n)); });
      recalc();
    }).catch(function (e) { showError("Loading configs failed: " + e.message); });
  }

  // The picker is a verb, not the form's state: it reads "Load preset…" and
  // snaps back after a load, so the same preset can be reloaded over an
  // edited form; #preset-name in the footer is the provenance.
  function renderPresetList() {
    var sel = $("preset"); sel.textContent = "";
    sel.appendChild(option("", "Load preset…"));
    S.presets.forEach(function (name) { sel.appendChild(option(name, name)); });
    sel.disabled = !S.presets.length;
    $("presets-note").textContent = S.presets.length ? "" : "No presets in the configs tree.";
  }
  $("preset").addEventListener("change", function () {
    var name = this.value; this.value = "";
    if (name) selectPreset(name);
  });

  function selectPreset(name) {
    S.presetName = name;  // claimed now, so a slower default cannot override the pick
    api("/api/configs/presets/" + encodeURIComponent(name)).then(function (doc) {
      if (S.presetName !== name) return;  // a later click won
      S.presetDoc = doc; S.loadedName = name;  // the LISTING name: a document's inner name need not match its file
      $("preset-name").textContent = "preset " + name;
      $("devices-eyebrow").textContent = "devices · preset " + name;
      fillFormFromPreset(doc);
    }).catch(function (e) {
      // The claim above is provenance only once the document arrived: a
      // preset deleted since the listing must not stamp its name on the form.
      if (S.presetName === name) { S.presetName = S.loadedName; recalc(); }
      showError("Preset " + name + ": " + e.message);
    });
  }

  function formShape(plan) {
    if ((plan.args || []).length) return null;
    if (plan.name === "sweep" && !(plan.kwargs && plan.kwargs.sweep && plan.kwargs.sweep.trajectory)) return null;
    return ["count", "sweep", "optimize"].indexOf(plan.name) >= 0 ? plan.name : null;
  }

  function fillFormFromPreset(doc) {
    var plan = doc.plan || {name: "count", args: [], kwargs: {}};
    var kw = plan.kwargs || {}, shape = formShape(plan);
    S.formable = shape !== null;
    S.formableNote = S.formable ? "" : "Retired or unsupported preset plan: " + plan.name;
    setMode(shape || "", true);
    setAcq(kw.acquisition || "strict");
    $("background").checked = !!doc.background;
    if (shape === "sweep") {
      try { composer.load(kw.sweep); }
      catch (e) { S.formable = false; S.formableNote = "Cannot load Sweep preset: " + e.message; showError(S.formableNote); }
    } else composer.reset();
    $("shots").value = shape === "count" ? (kw.num || 1) : (kw.shots_per_step || 1);
    $("period").value = kw.shot_period != null ? kw.shot_period : "";
    var trigger = Object.prototype.hasOwnProperty.call(kw, "trigger_profile") ? kw.trigger_profile : doc.trigger_profile;
    setSelect("trig", trigger || ""); $("desc").value = doc.description || "";
    // The run-level LabVIEW-files switch (#738): unset = the experiment default.
    $("native-save").value = doc.native_image_save === true ? "true" : doc.native_image_save === false ? "false" : "";
    var body = $("devs"); body.textContent = "";
    (doc.devices || []).forEach(function (d) { body.appendChild(deviceRow(d.device, d.save_images !== false, d.essential !== false)); });
    noDevicesNote();
    if (shape === "optimize") { setSelect("optimizer-config", kw.optimizer_config || ""); loadOptimizer(kw); }
    recalc(); renderCalibration();
  }
  function deviceRow(name, saveImages, essential) {
    var tr = document.createElement("tr");
    tr.dataset.device = name;
    tr.appendChild(td(name, "id"));
    tr.appendChild(td(checkbox(saveImages, "save images for " + name)));
    tr.appendChild(td(checkbox(essential, name + " essential")));
    var rm = document.createElement("button");
    rm.type = "button"; rm.className = "btn sm ghost rm"; rm.textContent = "remove";
    rm.setAttribute("aria-label", "remove " + name);
    rm.addEventListener("click", function () { tr.remove(); noDevicesNote(); recalc(); renderCalibration(); });
    tr.appendChild(td(rm));
    return tr;
  }
  function noDevicesNote() {
    var body = $("devs");
    if (!body.querySelector("tr[data-device]")) {
      body.textContent = "";
      var tr = document.createElement("tr"), c = td("No devices — add one.", "dim"); c.colSpan = 4; tr.appendChild(c); body.appendChild(tr);
    } else {
      Array.prototype.forEach.call(body.querySelectorAll("tr:not([data-device])"), function (tr) { tr.remove(); });
    }
  }
  function tableDevices() {
    var out = [];
    Array.prototype.forEach.call($("devs").querySelectorAll("tr[data-device]"), function (tr) {
      var boxes = tr.querySelectorAll("input[type=checkbox]");
      out.push({ device: tr.dataset.device, save_images: boxes[0].checked, essential: boxes[1].checked });
    });
    return out;
  }
  function checkbox(checked, label) {
    var c = document.createElement("input"); c.type = "checkbox"; c.checked = checked; c.setAttribute("aria-label", label);
    return c;
  }
  function setSelect(id, value) {
    var sel = $(id);
    if (![].some.call(sel.options, function (o) { return o.value === String(value); })) {
      sel.appendChild(option(String(value), String(value) + " · not in the catalog"));
    }
    sel.value = String(value);
  }
  /* --------------------------------------------------------------- form */

  function setMode(mode, silent) {
    S.mode = mode;
    Array.prototype.forEach.call($("mode").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
    });
    var count = mode === "count";
    $("sweep-composer").hidden = mode !== "sweep";
    $("count-options").hidden = !count;
    $("optimizer-form").hidden = mode !== "optimize";
    $("acq").hidden = mode === "optimize";
    if (mode === "optimize") setAcq("strict");
    lockOptimizerDevices();
    if (!silent) S.formable = true;
    $("shots-hint").textContent = count ? "num — the shots of the count" : "shots_per_step";
    // the lower half is named for what it holds, so "this changes with the mode"
    // is legible without switching modes to find out
    $("plan-eyebrow").textContent = count ? "Count · options"
      : mode === "sweep" ? "Sweep · trajectory"
      : mode === "optimize" ? "Optimize · optimizer"
      : "This plan";
    $("plan-empty").hidden = !!mode;   // at boot no mode is pressed: say so, don't show a bare rule
    if (!silent) recalc();
  }
  function setAcq(acq) {
    S.acq = acq;
    Array.prototype.forEach.call($("acq").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.acq === acq));
    });
    $("period").disabled = acq !== "strict";
    recalc();
  }
  $("mode").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-mode]");
    if (!b || b.getAttribute("aria-disabled") === "true") return;
    setMode(b.dataset.mode);
  });
  $("acq").addEventListener("click", function (e) {
    var b = e.target.closest("button[data-acq]");
    if (b) setAcq(b.dataset.acq);
  });
  ["shots", "period", "iterations"].forEach(function (id) {
    $(id).addEventListener("input", recalc);
  });

  function fmtSecs(secs) {
    var m = Math.floor(secs / 60), s = Math.round(secs % 60);
    return (m ? m + " min " : "") + s + " s";
  }
  function setInvalid(id, bad) {
    var el = $(id);
    if (bad) el.setAttribute("aria-invalid", "true"); else el.removeAttribute("aria-invalid");
  }

  var valid = false;
  function recalc() {
    var count = S.mode === "count";
    var shots = Number($("shots").value);
    var badShots = !Number.isInteger(shots) || shots < 1;
    setInvalid("shots", badShots);
    var ok = !badShots && !!S.mode, steps = 1;
    if (S.mode === "optimize") {
      steps = Number($("iterations").value);
      var iterationsOk = Number.isInteger(steps) && steps >= 1;
      setInvalid("iterations", !iterationsOk);
      ok = ok && iterationsOk && !!S.optimizer;
    } else if (S.mode === "sweep") {
      try { composer.value(); } catch (e) { ok = false; }
      var preview = composer.result(); steps = preview ? preview.total_steps : 0;
    }
    var total = count ? shots : steps * shots;
    var period = S.acq === "strict" && $("period").value !== "" ? Number($("period").value) : null;
    var periodOk = period === null || (Number.isFinite(period) && period > 0);
    setInvalid("period", !periodOk); ok = ok && periodOk;
    var time = period ? " · ~<b>" + fmtSecs(total * period) + "</b> at " + period + " s/shot" : "";
    $("est").innerHTML = count
      ? "<b>" + shots + "</b> shots" + time
      : (S.mode === "optimize" ? "≤ " : "") + "<b>" + steps + "</b> " + (S.mode === "optimize" ? "iteration" : "step") + (steps === 1 ? "" : "s") + " × <b>" + shots + "</b> shots = <b>" + total + "</b> shots" + time;
    // The form is the document: a preset only seeds it, so Start (and
    // Save as preset) need a valid form, never a loaded preset (#900).
    if (!S.mode) $("est").textContent = "Choose Count, Sweep or Optimize.";
    else if (S.mode === "sweep" && !composer.result()) $("est").textContent = ok ? "Shot estimate unavailable until preview completes. Start runs preflight validation." : "Complete the trajectory to continue.";
    valid = ok;
    updateStartGate();
  }

  function updateStartGate() {
    var st = S.status;
    // A running plan does not close the gate: the next scan queues behind
    // it (#905). A paused one does — the client removes an item added
    // then, so Start waits for resume or stop.
    var busy = !st || !st.connected || st.re_state === "paused";
    var btn = $("btn-start");
    btn.disabled = busy || !valid || !S.formable;
    $("btn-save-preset").disabled = !valid || !S.formable;
    btn.title = !st ? "waiting for the manager" : !st.connected ? "manager unreachable" : busy ? "a scan is paused — resume or stop it first" : !S.formable ? S.formableNote : !valid ? "fix the form first" : "";
    $("preset-name").textContent = S.presetName ? "preset " + S.presetName + (S.formable ? "" : " · " + S.formableNote) : "";
  }

  function buildPreset() {
    var doc = S.presetDoc || {}, previous = doc.plan || {}, name = S.mode;
    var kwargs = previous.name === name ? Object.assign({}, previous.kwargs || {}) : {};
    // The visible selectors own these values. Older presets may put them in
    // kwargs: expand_preset gives a trigger_profile copy precedence over the
    // field and refuses a native_image_save copy outright, so both are dropped.
    delete kwargs.trigger_profile;
    delete kwargs.native_image_save;
    var nativeSave = $("native-save").value;
    var shots = Number($("shots").value);
    if (name === "optimize") {
      kwargs.optimizer_config = $("optimizer-config").value; kwargs.max_iterations = Number($("iterations").value); kwargs.shots_per_step = shots;
    } else {
      kwargs.acquisition = S.acq;
      if (name === "count") kwargs.num = shots;
      else { kwargs.shots_per_step = shots; kwargs.sweep = composer.value(); }
    }
    if (S.acq === "strict" && $("period").value !== "") kwargs.shot_period = Number($("period").value);
    else delete kwargs.shot_period;
    return {name: S.presetName || "adhoc", description: $("desc").value.trim(), trigger_profile: $("trig").value || null,
      native_image_save: nativeSave === "" ? null : nativeSave === "true",
      background: name === "count" && $("background").checked,
      devices: tableDevices(), plan: {name: name, args: [], kwargs: kwargs}};
  }

  function lockOptimizerDevices() {
    var required = S.mode === "optimize" && S.optimizer ? S.optimizer.required_devices : [];
    var body = $("devs");
    Array.prototype.forEach.call(body.querySelectorAll("tr[data-device]"), function (tr) {
      var needed = required.some(function (name) { return name.toLowerCase() === tr.dataset.device.toLowerCase(); });
      var boxes = tr.querySelectorAll("input[type=checkbox]");
      if (needed && !tr.dataset.optimizerRequired) {
        tr.dataset.previousSave = boxes[0].checked; tr.dataset.previousEssential = boxes[1].checked;
        tr.dataset.optimizerRequired = "true";
      } else if (!needed && tr.dataset.optimizerRequired) {
        if (tr.dataset.optimizerAdded) { tr.remove(); return; }
        boxes[0].checked = tr.dataset.previousSave === "true";
        boxes[1].checked = tr.dataset.previousEssential === "true";
        delete tr.dataset.optimizerRequired;
      }
      Array.prototype.forEach.call(boxes, function (box) { box.disabled = needed; if (needed) box.checked = true; });
      var button = tr.querySelector("button"); button.disabled = needed;
      button.textContent = needed ? "required" : "remove";
    });
    required.forEach(function (name) {
      var exists = [].some.call(body.querySelectorAll("tr[data-device]"), function (tr) { return tr.dataset.device.toLowerCase() === name.toLowerCase(); });
      if (!exists) {
        var tr = deviceRow(name, true, true); tr.dataset.optimizerAdded = "true";
        body.appendChild(tr);
        tr.dataset.optimizerRequired = "true";
        Array.prototype.forEach.call(tr.querySelectorAll("input, button"), function (control) { control.disabled = true; });
        tr.querySelector("button").textContent = "required";
      }
    });
    noDevicesNote(); renderCalibration();
  }

  function loadOptimizer(overrides) {
    var name = $("optimizer-config").value, request = ++S.optimizerRequest;
    S.optimizer = null; lockOptimizerDevices(); recalc();
    $("optimizer-note").textContent = name ? "Loading optimizer…" : "Choose a config to see its required devices.";
    if (!name) return;
    api("/api/configs/optimizer_configs/" + encodeURIComponent(name)).then(function (config) {
      if (request !== S.optimizerRequest) return;
      S.optimizer = config;
      $("shots").value = overrides && overrides.shots_per_step || config.shots_per_step;
      $("iterations").value = overrides && overrides.max_iterations || config.max_iterations || "";
      $("optimizer-note").textContent = "Required: " + config.required_devices.join(", ") + ". Strict acquisition.";
      lockOptimizerDevices(); recalc();
    }).catch(function (e) {
      if (request !== S.optimizerRequest) return;
      $("optimizer-note").textContent = e.message; recalc();
    });
  }
  $("optimizer-config").addEventListener("change", function () { loadOptimizer(); });

  function renderOptimization() {
    var o = S.optimization, st = S.status || {};
    $("optimization-live").hidden = !o || !o.run_uid;
    if (!o || !o.run_uid) return;
    $("optimization-iteration").textContent = o.config + " · scan " + (o.scan_number || "—") + " · iteration " + o.iteration + " / " + (o.max_iterations || "—") + " · " + (o.exit_status || "running") + (o.completed_at ? " · " + new Date(o.completed_at * 1000).toLocaleString() : "") + (o.expired ? " · best offer expired" : "") + (o.invalidated_reason ? " · " + o.invalidated_reason : "");
    var body = $("optimization-values"); body.textContent = "";
    var values = Object.assign({}, o.measured, o.outputs);
    Object.keys(values).forEach(function (name) {
      var tr = document.createElement("tr");
      tr.appendChild(td(name)); tr.appendChild(td(values[name] == null ? "unavailable" : values[name].toPrecision(5)));
      tr.appendChild(td(o.best[name] == null ? "—" : o.best[name].toPrecision(5))); body.appendChild(tr);
    });
    $("optimization-shots").textContent = Object.keys(o.valid_shots).map(function (name) { return name + ": " + o.valid_shots[name] + " valid shots"; }).join(" · ");
    var targets = $("optimization-targets"); targets.textContent = "";
    Object.keys(o.best_moves).forEach(function (name) {
      var row = document.createElement("tr"); row.appendChild(td(name));
      row.appendChild(td(o.best_moves[name] == null ? "unavailable" : String(o.best_moves[name])));
      targets.appendChild(row);
    });
    var moves = Object.values(o.best_moves);
    $("btn-set-best").disabled = !st.connected || st.re_state !== "idle" || st.items_in_queue > 0 || !o.finished || o.exit_status !== "success" || o.expired || !!o.invalidated_reason || !moves.length || moves.some(function (v) { return v == null; });
    $("btn-set-best").title = moves.length ? Object.keys(o.best_moves).map(function (n) { return n + " = " + o.best_moves[n]; }).join(", ") : "No feasible best point";
  }
  $("btn-set-best").addEventListener("click", function () {
    if (!S.optimization) return;
    var dialog = $("dlg-set-best"); dialog.dataset.runUid = S.optimization.run_uid;
    $("dlg-set-best-text").textContent = Object.keys(S.optimization.best_moves).map(function (name) {
      return name + " = " + S.optimization.best_moves[name];
    }).join("; ");
    window.GeecsKit.confirm(dialog).open();
  });
  $("do-set-best").addEventListener("click", function () {
    var uid = $("dlg-set-best").dataset.runUid;
    window.GeecsKit.confirm($("dlg-set-best")).close();
    $("btn-set-best").disabled = true;
    post("/api/optimization/best", { run_uid: uid, operator: operator() }).then(function () { refreshQueue(); }).catch(function (e) { showError(e.message); renderOptimization(); });
  });

  /* --------------------------------------------------------- submission */

  $("btn-start").addEventListener("click", function () {
    var preset = buildPreset();
    S.pendingPreset = preset;
    $("btn-start").disabled = true;
    post("/api/preflight", preset).then(function (out) {
      if (out.refusal) { showError("Refused: " + out.refusal); updateStartGate(); return; }
      if (out.questions.length) { openAck(out.questions); return; }
      submit(preset, [], false);
    }).catch(function (e) { showError("Preflight failed: " + e.message + detailOf(e)); updateStartGate(); });
  });

  function detailOf(e) {
    var p = e.payload || {};
    if (p.errors && p.errors.length) return " — " + p.errors.map(function (x) { return x.loc + ": " + x.msg; }).join("; ");
    return "";
  }

  function openAck(questions) {
    S.pendingAck = questions.map(function (q) { return q.check; });
    var list = $("ack-list"); list.textContent = "";
    questions.forEach(function (q) {
      var li = document.createElement("li"), label = document.createElement("label");
      var box = document.createElement("input"); box.type = "checkbox"; box.className = "ackbox"; box.dataset.check = q.check;
      var text = document.createElement("span");
      text.textContent = q.title;
      var m = document.createElement("span"); m.className = "m"; m.textContent = q.message;
      text.appendChild(m);
      label.appendChild(box); label.appendChild(text); li.appendChild(label); list.appendChild(li);
    });
    $("dlg-ack-title").textContent = "Preflight found " + questions.length + " thing" + (questions.length === 1 ? "" : "s") + " to acknowledge";
    $("do-submit").disabled = true;
    window.GeecsKit.confirm($("dlg-ack")).open();
  }
  document.addEventListener("change", function (e) {
    if (!e.target.classList.contains("ackbox")) return;
    var all = Array.prototype.every.call(document.querySelectorAll(".ackbox"), function (c) { return c.checked; });
    $("do-submit").disabled = !all;
  });
  $("do-submit").addEventListener("click", function () {
    var acked = Array.prototype.filter.call(document.querySelectorAll(".ackbox"), function (c) { return c.checked; })
      .map(function (c) { return c.dataset.check; });
    window.GeecsKit.confirm($("dlg-ack")).close();
    submit(S.pendingPreset, acked, false);
  });
  $("do-replace").addEventListener("click", function () {
    window.GeecsKit.confirm($("dlg-pending")).close();
    submit(S.pendingPreset, S.pendingAckDone || [], true);
  });

  function submit(preset, acknowledged, clearPending) {
    S.pendingAckDone = acknowledged;
    post("/api/submit", { preset: preset, acknowledged: acknowledged, operator: operator(), clear_pending: clearPending })
      .then(function (out) {
        $("verb-note").textContent = "Queued: " + out.summary + (out.planned_shots ? " (" + out.planned_shots + " shots)" : "");
        refreshQueue();
      })
      .catch(function (e) {
        if (e.status === 409 && e.payload.pending_items) {
          var items = e.payload.pending_items;
          $("dlg-pending-text").textContent = "Waiting now: " + items.map(function (i) { return i.summary; }).join("; ") + ". Replace it with this scan?";
          window.GeecsKit.confirm($("dlg-pending")).open();
          return;
        }
        if (e.status === 409 && e.payload.needs_acknowledgement) { openAck(e.payload.needs_acknowledgement); return; }
        showError("Submit failed: " + e.message + detailOf(e));
      })
      .then(updateStartGate);
  }

  /* --------------------------------------------------------------- verbs */

  function verb(path, body, after) {
    return post(path, body || {}).then(function (out) {
      $("verb-note").textContent = out.message || (out.ok ? "ok" : "refused");
      if (after) after(out);
    }).catch(function (e) { showError(e.message); });
  }
  $("btn-pause").addEventListener("click", function () { verb("/api/pause", { operator: operator() }); });
  $("btn-resume").addEventListener("click", function () { verb("/api/resume", { operator: operator() }); });
  $("do-stop").addEventListener("click", function () {
    window.GeecsKit.confirm($("dlg-stop")).close();
    $("verb-note").textContent = "Stopping… (pauses at the shot boundary, then stops)";
    verb("/api/stop", { operator: operator() }, refreshQueue);
  });
  $("do-clear").addEventListener("click", function () {
    window.GeecsKit.confirm($("dlg-clear")).close();
    verb("/api/clear", {}, refreshQueue);
  });

  /* ------------------------------------------------- idle-only items */

  // A move, an action or a calibration is a queue item that runs by itself
  // as soon as it is added: allowed only while nothing runs or waits.
  function idle() {
    var st = S.status;
    return !!(st && st.connected && st.re_state === "idle" && !(st.items_in_queue > 0));
  }
  function idleTitle() {
    var st = S.status;
    return !st ? "waiting for the manager" : !st.connected ? "manager unreachable"
      : st.re_state !== "idle" ? "a plan is " + st.re_state : st.items_in_queue > 0 ? "items wait in the queue" : "";
  }
  function renderIdleGates() {
    var ok = idle(), why = idleTitle();
    var mvName = $("mv-var").value.trim();
    var mvKnown = !!settableMatch(mvName);
    $("btn-move").disabled = !ok || !mvKnown;
    $("btn-move").title = mvName && !mvKnown ? "no settable named " + mvName : why;
    var mvWord = ok ? "idle" : (why || "held");
    setChip($("mv-chip"), ok ? K.ok : K.unknown, mvWord, why);
    $("act-arm").disabled = !ok || !S.actionName || !!(currentAction() && currentAction().problem);
    $("act-run").disabled = !ok || !S.armed;
    $("act-run").title = why;
    if (!ok && S.armed) setArmed(false);
    var twoDevices = tableDevices().length >= 2;
    $("cal-check").disabled = !ok || !twoDevices; $("cal-measure").disabled = !ok || !twoDevices;
    $("cal-check").title = $("cal-measure").title = why || (twoDevices ? "" : "needs two devices in the New scan table");
  }
  function itemQueued(out) {
    $("verb-note").textContent = "Queued: " + out.summary;
    refreshQueue();
  }
  function itemRefused(e) {
    var p = e.payload || {};
    showError(e.message + (p.items_in_queue ? " (" + p.items_in_queue + " waiting)" : ""));
  }

  /* ---- devices · move */
  // The list is every numeric settable of the experiment, aliased ones first
  // (the DB's curated short names); the option VALUE is always the canonical
  // Device:Variable — what the request stores. The readback beside it is the
  // gateway's readback PV, never the :SP echo.
  function settableFor(name) {
    return S.settables.filter(function (s) { return s.name === name; })[0] || null;
  }
  // What the user typed, resolved to a settable. The picker is an input now,
  // so the text can be pasted off a log line or typed in the wrong case; only
  // the canonical name is ever sent.
  function settableMatch(text) {
    var name = (text || "").trim();
    if (!name) return null;
    var exact = settableFor(name);
    if (exact) return exact;
    var lower = name.toLowerCase();
    var byName = S.settables.filter(function (s) { return s.name.toLowerCase() === lower; })[0];
    if (byName) return byName;
    // The alias is what the labels show and what operators say out loud, so it
    // has to resolve too — but only when it names exactly one settable; an
    // alias the DB has put on two variables names neither.
    var byAlias = S.settables.filter(function (s) { return (s.alias || "").toLowerCase() === lower; });
    return byAlias.length === 1 ? byAlias[0] : null;
  }
  function renderMoveVars() {
    // A datalist, not a bare <select>: the experiment has hundreds of numeric
    // settables, and typing a fragment is the only sane way through them —
    // the same affordance the sweep composer's axis field has.  The option
    // VALUE stays the canonical Device:Variable; the alias is the label.
    var list = $("mv-variables"); list.textContent = "";
    S.settables.forEach(function (s) {
      var parts = [s.alias, s.units ? "(" + s.units + ")" : ""].filter(Boolean);
      list.appendChild(option(s.name, parts.join(" · ")));
    });
    var input = $("mv-var");
    input.disabled = !S.settables.length;
    input.placeholder = S.settables.length ? "Choose or type Device:Variable"
      : (S.settablesNote ? "settables unavailable" : "no numeric settables");
    var resolved = settableMatch(input.value);
    if (input.value && !resolved) input.value = "";
    $("mv-hint").textContent = S.settablesNote || "";
    watchReadback(resolved ? resolved.name : "");
    renderIdleGates();
  }
  function fmtVal(x) {
    var a = Math.abs(x);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return x.toExponential(3);
    return x.toFixed(a >= 100 ? 2 : a >= 1 ? 3 : 4);
  }
  function watchReadback(name) {
    S.readbackVar = name || null;
    if (S.readbackTimer) { clearInterval(S.readbackTimer); S.readbackTimer = null; }
    var live = $("mv-live");
    if (!name) { live.hidden = true; return; }
    var s = settableFor(name);
    $("mv-k").textContent = s && s.alias ? s.alias + " · " + name : name;
    $("mv-rb").textContent = "—"; $("mv-age").textContent = "reading…"; live.removeAttribute("data-age");
    live.hidden = false;
    pollReadback(true);
    S.readbackTimer = setInterval(pollReadback, 1000);
  }
  // The ticker skips a hidden tab (no point polling a page nobody sees); the
  // first read after a pick and the return to the tab always read.
  document.addEventListener("visibilitychange", function () { if (!document.hidden) pollReadback(true); });
  function pollReadback(force) {
    var name = S.readbackVar;
    if (!name || (document.hidden && force !== true)) return;
    var s = settableFor(name);
    api("/api/readback?variable=" + encodeURIComponent(name) + (s && s.units ? "&units=" + encodeURIComponent(s.units) : "")).then(function (r) {
      if (S.readbackVar !== name) return;
      var v = $("mv-rb"); v.textContent = "";
      if (r.ok && r.value != null) {
        v.appendChild(document.createTextNode(fmtVal(r.value)));
        if (r.units) { var u = document.createElement("small"); u.textContent = r.units; v.appendChild(u); }
      } else { v.textContent = "—"; }
      $("mv-age").textContent = r.age_s != null ? fmtAge(r.age_s) : (r.detail || "no reading");
      // stale = the gateway did not answer; a quiet device with an old stamp is
      // still a reading, and the age says how old.
      if (r.ok) $("mv-live").removeAttribute("data-age"); else $("mv-live").setAttribute("data-age", "stale");
    }).catch(function (e) {
      if (S.readbackVar !== name) return;
      $("mv-rb").textContent = "—"; $("mv-age").textContent = e.message; $("mv-live").setAttribute("data-age", "stale");
    });
  }
  ["change", "input"].forEach(function (ev) {
    $("mv-var").addEventListener(ev, function () {
      var match = settableMatch(this.value);
      // on commit, show the user the canonical name their text resolved to
      if (match && ev === "change" && this.value !== match.name) this.value = match.name;
      watchReadback(match ? match.name : "");
      renderIdleGates();
    });
  });
  $("btn-move").addEventListener("click", function () {
    var v = Number($("mv-val").value);
    var bad = $("mv-val").value === "" || !isFinite(v);
    setInvalid("mv-val", bad);
    if (bad) return;
    $("btn-move").disabled = true;
    var picked = settableMatch($("mv-var").value);
    if (!picked) { renderIdleGates(); return; }
    post("/api/move", { variable: picked.name, value: v, operator: operator() })
      .then(function (out) { $("mv-note").textContent = "Queued: " + out.summary + " (" + out.reference + ")"; refreshQueue(); })
      .catch(itemRefused).then(renderIdleGates);
  });

  /* ---- actions */
  function currentAction() {
    return S.actions.filter(function (a) { return a.name === S.actionName; })[0] || null;
  }
  function setArmed(on) {
    S.armed = on;
    var armWord = on ? "armed" : "disarmed";
    setChip($("act-chip"), on ? K.degraded : K.unknown, armWord);
    $("act-arm").textContent = on ? "Disarm" : "Arm";
    $("act-run").disabled = !on || !idle();
  }
  function renderActions(problem) {
    var sel = $("action"); sel.textContent = "";
    sel.appendChild(option("", S.actions.length ? "— pick an action —" : "no action plans in the configs tree", !S.actions.length));
    S.actions.forEach(function (a) {
      // A plan that cannot run stays pickable: the preview shows why.
      var label = a.name + " · " + (a.problem ? "cannot run" : a.steps + " step" + (a.steps === 1 ? "" : "s") + (a.nested.length ? " · runs " + a.nested.join(", ") : ""));
      sel.appendChild(option(a.name, label, false, a.problem || a.description || ""));
    });
    if (S.actionName) sel.value = S.actionName;
    $("actions-note").textContent = problem ? "Action library: " + problem
      : S.actions.length ? "" : "No action plans in the configs tree.";
    renderIdleGates();
  }
  $("action").addEventListener("change", function () {
    if (this.value) { selectAction(this.value); return; }
    S.actionName = null; setArmed(false);
    $("action-preview-group").hidden = true;
    renderIdleGates();
  });
  function selectAction(name) {
    S.actionName = name; setArmed(false);
    $("action").value = name;
    var a = currentAction();
    $("action-preview-group").hidden = false;
    $("action-preview-title").textContent = "preview · " + name;
    var ol = $("action-steps"); ol.textContent = "";
    if (a && a.problem) {
      var li = document.createElement("li"); li.className = "prob"; li.textContent = a.problem; ol.appendChild(li);
      $("action-writes").textContent = "";
      renderIdleGates();
      return;
    }
    api("/api/actions/" + encodeURIComponent(name)).then(function (d) {
      if (S.actionName !== name) return;
      d.steps.forEach(function (st) {
        var li = document.createElement("li");
        if (st.do === "set") li.className = "write";
        li.textContent = st.text;
        if (st.from_plan) { var f = document.createElement("span"); f.className = "from"; f.textContent = "← " + st.from_plan; li.appendChild(f); }
        ol.appendChild(li);
      });
      $("action-writes").textContent = d.writes + " write" + (d.writes === 1 ? "" : "s") + " · " + d.steps.length + " steps" + (d.description ? " · " + d.description : "");
      renderIdleGates();
    }).catch(function (e) { showError("Action " + name + ": " + e.message); });
  }
  $("act-arm").addEventListener("click", function () { setArmed(!S.armed); });
  $("act-run").addEventListener("click", function () {
    if (!S.armed || !S.actionName) return;
    var name = S.actionName;
    setArmed(false);
    post("/api/actions/" + encodeURIComponent(name) + "/run", { operator: operator() })
      .then(itemQueued).catch(itemRefused).then(renderIdleGates);
  });

  /* ---- calibration */
  function renderCalibration() {
    var c = S.calibration;
    var devs = tableDevices().map(function (d) { return d.device; });
    $("cal-set").textContent = devs.length ? "Over the " + devs.length + " device" + (devs.length === 1 ? "" : "s") + " in the New scan table: " + devs.join(", ") : "Over the devices in the New scan table (none yet).";
    if (!c) return;
    if (!c.stored) {
      setChip($("cal-chip"), K.unknown, "not measured", c.detail || "");
      $("cal-devices").textContent = "—"; $("cal-when").textContent = c.detail || "no shot_offsets.yaml";
      $("cal-max").textContent = "—"; $("cal-max-dev").textContent = "";
    } else {
      setChip($("cal-chip"), K.ok, "stored", c.path);
      $("cal-devices").textContent = c.devices.length;
      $("cal-when").textContent = (c.measured_at ? "measured " + c.measured_at.slice(0, 16).replace("T", " ") : "date unknown") + (c.trigger_profile ? " · " + c.trigger_profile : "");
      $("cal-max").textContent = c.max_offset_s == null ? "—" : c.max_offset_s.toFixed(3) + " s";
      $("cal-max-dev").textContent = c.max_offset_device || "";
    }
    renderIdleGates();
  }
  function calibrationBody() {
    return {
      devices: tableDevices().map(function (d) { return d.device; }),
      trigger_profile: $("trig").value || null,
      operator: operator()
    };
  }
  $("cal-check").addEventListener("click", function () {
    post("/api/calibration/check", calibrationBody()).then(itemQueued).catch(itemRefused).then(renderIdleGates);
  });
  $("cal-measure").addEventListener("click", function () {
    var n = parseInt($("cal-shots").value, 10) || 10, write = !!$("cal-write").value;
    $("dlg-measure-text").textContent = "Drives the trigger box OFF, waits the set quiet, then fires " + n + " single shots and reads every device's timestamp. "
      + (write ? "The result REPLACES shot_offsets.yaml in the configs tree (a commit is still yours) and reaches the worker at its NEXT environment open, not this session's scans." : "The result is reported only; nothing is written.");
  });
  $("do-measure").addEventListener("click", function () {
    window.GeecsKit.confirm($("dlg-measure")).close();
    var body = calibrationBody();
    body.shots = parseInt($("cal-shots").value, 10) || 10;
    body.write = !!$("cal-write").value;
    post("/api/calibration/measure", body).then(itemQueued).catch(itemRefused).then(renderIdleGates);
  });
  $("devs").addEventListener("change", renderCalibration);

  /* ------------------------------------------------------------ drawers */

  /* ---- add device */
  /* A trip to the drawer adds as many devices as you like: a click toggles a
     row, shift-click takes the range, Add commits them all.  Selection is
     kept by name, so narrowing the search and picking more does not lose
     what is already ticked. */
  var devPicked = {};        // name -> true, across re-renders and searches
  var devShown = [];         // the names currently rendered, for shift-ranges
  var devAnchor = null;      // the last row clicked, the other end of a range

  function devPickedNames() { return Object.keys(devPicked); }
  function renderDevicePicks() {
    var n = devPickedNames().length;
    $("btn-add-selected").disabled = !n;
    $("btn-add-selected").textContent = n ? "Add " + n + " device" + (n === 1 ? "" : "s") : "Add";
    $("btn-pick-none").disabled = !n;
    Array.prototype.forEach.call($("devlist").children, function (b) {
      b.setAttribute("aria-pressed", devPicked[b.dataset.device] ? "true" : "false");
    });
  }
  function pickDevice(name, index, range) {
    if (range && devAnchor !== null) {
      var lo = Math.min(devAnchor, index), hi = Math.max(devAnchor, index);
      // a range takes everything in it, skipping what is already in the table
      for (var i = lo; i <= hi; i++) if (devShown[i] && !devShown[i].have) devPicked[devShown[i].name] = true;
    } else {
      if (devPicked[name]) delete devPicked[name]; else devPicked[name] = true;
      devAnchor = index;
    }
    renderDevicePicks();
  }
  function renderDeviceList(q) {
    var list = $("devlist"); list.textContent = "";
    var have = {};
    tableDevices().forEach(function (d) { have[d.device] = true; });
    var needle = q.trim().toLowerCase();
    // The manager lists a device and its children (U_S1H, U_S1H.current …);
    // a preset names devices, so only the bare names are offered.
    var names = S.devices.filter(function (n) { return n.indexOf(".") === -1 && (!needle || n.toLowerCase().indexOf(needle) !== -1); });
    devShown = []; devAnchor = null;
    names.slice(0, 60).forEach(function (n, i) {
      devShown.push({ name: n, have: !!have[n] });
      var b = document.createElement("button");
      b.type = "button";
      b.dataset.device = n;
      b.setAttribute("aria-pressed", devPicked[n] ? "true" : "false");
      var a = document.createElement("span"); a.textContent = n;
      var d = document.createElement("span"); d.className = "sub"; d.textContent = have[n] ? "in the table" : "";
      b.appendChild(a); b.appendChild(d);
      b.disabled = !!have[n];
      b.addEventListener("click", function (ev) { pickDevice(n, i, ev.shiftKey); });
      list.appendChild(b);
    });
    $("devq-hint").textContent = S.devices.length
      ? names.length + " of " + S.devices.filter(function (n) { return n.indexOf(".") === -1; }).length + " devices" + (names.length > 60 ? " · type to narrow" : "")
      : "the manager's device list is empty or unreachable";
    renderDevicePicks();
  }
  $("devq").addEventListener("input", function () { renderDeviceList($("devq").value); });
  // Opening the drawer starts a fresh selection: Esc and the scrim are a
  // cancel, so picks that were never committed with Add do not come back.
  $("btn-add-device").addEventListener("click", function () { devPicked = {}; renderDeviceList($("devq").value); setTimeout(function () { $("devq").focus(); }, 50); });
  $("btn-pick-all").addEventListener("click", function () {
    devShown.forEach(function (d) { if (!d.have) devPicked[d.name] = true; });
    renderDevicePicks();
  });
  $("btn-pick-none").addEventListener("click", function () { devPicked = {}; devAnchor = null; renderDevicePicks(); });
  $("btn-add-selected").addEventListener("click", function () { addDevices(devPickedNames()); });
  function addDevices(names) {
    if (!names.length) return;
    noDevicesNote();
    if (!$("devs").querySelector("tr[data-device]")) $("devs").textContent = "";
    names.forEach(function (name) { $("devs").appendChild(deviceRow(name, true, true)); });
    noDevicesNote();
    $("devices-eyebrow").textContent = "devices · " + (S.presetName ? "preset " + S.presetName + " + " : "") + "edited";
    recalc(); renderCalibration();
    devPicked = {}; devAnchor = null;
    window.GeecsKit.drawer($("drw-devices")).close();
  }

  /* ---- save as preset */
  function yamlScalar(v) {
    if (v === null || v === undefined) return "null";
    if (typeof v === "number" || typeof v === "boolean") return String(v);
    return /^[A-Za-z0-9_][A-Za-z0-9_ .:-]*$/.test(v) && !/^(true|false|null|yes|no|on|off)$/i.test(v) && !/^\d/.test(v) ? v : JSON.stringify(v);
  }
  function toYaml(obj, indent) {
    // Enough YAML for a preset: mappings, lists of scalars, lists of flat
    // mappings. The server validates and writes the real document; this is
    // the operator's read-through of what will be saved.
    var pad = indent || "", out = "";
    Object.keys(obj).forEach(function (k) {
      var v = obj[k];
      if (Array.isArray(v)) {
        if (!v.length) { out += pad + k + ": []\n"; return; }
        out += pad + k + ":\n";
        v.forEach(function (it) {
          if (it && typeof it === "object" && !Array.isArray(it)) {
            var keys = Object.keys(it);
            out += pad + "  - " + keys.map(function (kk) { return kk + ": " + yamlScalar(it[kk]); }).join(", ").replace(/^/, "{") + "}\n";
          } else out += pad + "  - " + yamlScalar(it) + "\n";
        });
      } else if (v && typeof v === "object") {
        out += pad + k + ":\n" + toYaml(v, pad + "  ");
      } else out += pad + k + ": " + yamlScalar(v) + "\n";
    });
    return out;
  }
  function presetForSave() {
    var preset = buildPreset();
    preset.name = $("pname").value.trim();
    preset.description = $("pdesc").value.trim();
    return preset;
  }
  function renderPresetYaml() {
    var ok = /^[A-Za-z0-9_][A-Za-z0-9_.-]*$/.test($("pname").value.trim());
    setInvalid("pname", !ok);
    $("do-save-preset").disabled = !ok;
    $("preset-yaml").textContent = toYaml(presetForSave());
  }
  $("btn-save-preset").addEventListener("click", function () {
    $("pname").value = S.presetName ? S.presetName : "";
    $("pdesc").value = $("desc").value.trim() || (S.presetDoc && S.presetDoc.description) || "";
    $("preset-saved").textContent = "";
    renderPresetYaml();
    setTimeout(function () { $("pname").focus(); $("pname").select(); }, 50);
  });
  ["pname", "pdesc"].forEach(function (id) { $(id).addEventListener("input", renderPresetYaml); });
  function savePreset(overwrite) {
    var preset = presetForSave();
    post("/api/configs/presets/" + encodeURIComponent(preset.name), { preset: preset, overwrite: overwrite })
      .then(function (out) {
        $("preset-saved").textContent = out.message;
        return api("/api/configs/presets").then(function (r) {
          S.presets = r.names; S.presetName = out.name; S.loadedName = out.name; renderPresetList();
          $("preset-name").textContent = "preset " + out.name;
        });
      })
      .catch(function (e) {
        if (e.status === 409 && e.payload.exists) {
          $("dlg-overwrite-name").textContent = preset.name;
          window.GeecsKit.confirm($("dlg-overwrite")).open();
          return;
        }
        $("preset-saved").textContent = "Not saved: " + e.message + detailOf(e);
      });
  }
  $("do-save-preset").addEventListener("click", function () { savePreset(false); });
  $("do-overwrite").addEventListener("click", function () {
    window.GeecsKit.confirm($("dlg-overwrite")).close();
    savePreset(true);
  });

  /* ------------------------------------------------------------ keyboard */

  document.addEventListener("keydown", function (e) {
    if (e.target.matches("input,textarea,select") || document.querySelector("dialog[open]")) return;
    if (e.key === "p" || e.key === "P") {
      if (!$("btn-resume").hidden) $("btn-resume").click(); else if (!$("btn-pause").disabled) $("btn-pause").click();
    } else if ((e.key === "s" || e.key === "S") && !$("btn-stop").disabled) {
      window.GeecsKit.confirm($("dlg-stop")).open();
    } else if (e.key === "n" || e.key === "N") {
      $("submit").scrollIntoView({ behavior: "smooth", block: "start" });
      var variable = $("sweep-composer").querySelector("input[data-field=axis]");
      if (variable && S.mode === "sweep") variable.focus();
    }
  });

  /* ---------------------------------------------------------------- boot */

  setMode("", true);
  setAcq("strict");
  renderTail();
  loadConfigs();
  refreshQueue();
  api("/api/progress").then(function (p) { S.progress = p; renderProgress(); }).catch(function () { /* the stream will say */ });
  connect();
})();
