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
    progress: null,
    queue: null,
    presets: [], presetName: null, presetDoc: null, loadedName: null,
    variables: [], triggers: [],
    mode: "scan", acq: "strict",
    consoleSeq: 0, epoch: null,
    formable: true, formableNote: "",
    pendingPreset: null, pendingAck: [],
    devices: [], actions: [], actionName: null, armed: false, calibration: null,
    settables: [], settablesNote: "", readbackVar: null, readbackTimer: null,
    tail: "scanlog", logFolder: null, logLines: [], consoleLines: []
  };

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
    $("meter-l").textContent = total ? done + " / " + total + " shots" : (done ? done + " shots" : "—");
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

  function loadConfigs() {
    return Promise.all([
      api("/api/configs/presets"),
      api("/api/scan-variables"),
      api("/api/configs/trigger_profiles"),
      api("/api/devices").catch(function () { return []; }),
      api("/api/actions").catch(function (e) { return { error: e.message }; }),
      api("/api/calibration").catch(function (e) { return { stored: false, detail: e.message }; }),
      api("/api/settables").catch(function (e) { return { items: [], source: "?", detail: e.message }; })
    ]).then(function (res) {
      S.presets = res[0].names; S.variables = res[1]; S.triggers = res[2].names;
      S.devices = res[3]; S.actions = res[4].error ? [] : res[4]; S.calibration = res[5];
      S.settables = res[6].items || []; S.settablesNote = res[6].detail || "";
      renderMoveVars(); renderDeviceList(""); renderActions(res[4].error || null); renderCalibration();
      renderPresetList();
      ["var1", "var2"].forEach(function (id) {
        var sel = $(id); sel.textContent = "";
        S.variables.forEach(function (v) {
          sel.appendChild(option(v.name, v.name + (v.target ? " · " + v.target : " · pseudo"), !v.scannable, v.reason || v.target || ""));
        });
        if (!S.variables.length) sel.appendChild(option("", "no scan variables in the catalog", true));
      });
      var tr = $("trig"); tr.textContent = "";
      tr.appendChild(option("", "— none —"));
      S.triggers.forEach(function (n) { tr.appendChild(option(n, n)); });
      // Default to the first preset only if the operator has not already
      // picked one while the listing was loading.
      if (S.presets.length && !S.presetName) selectPreset(S.presets[0]);
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

  // The shapes the form can express. Anything else is shown, not guessed:
  // a preset that runs list_scan must not be resubmitted as a scan.
  function formShape(plan) {
    var args = plan.args || [];
    if (plan.name === "count" && args.length === 0) return "count";
    if (plan.name === "scan" && args.length === 4) return "scan";
    if (plan.name === "grid_scan" && args.length === 8) return "grid";
    return null;
  }

  function fillFormFromPreset(doc) {
    var plan = doc.plan || { name: "count", args: [], kwargs: {} };
    var kw = plan.kwargs || {}, args = plan.args || [];
    var shape = formShape(plan);
    S.formable = shape !== null;
    S.formableNote = S.formable ? "" : "this preset runs " + plan.name + " with " + args.length + " argument(s); no form for it yet";
    var mode = shape === "count" ? (doc.background ? "background" : "noscan") : (shape || "scan");
    setMode(mode, true);
    setAcq(kw.acquisition || "strict");
    if (plan.name === "scan" && args.length >= 4) {
      setSelect("var1", args[0]); $("start1").value = args[1]; $("stop1").value = args[2];
      $("step1").value = stepFor(args[1], args[2], args[3]);
    } else if (plan.name === "grid_scan" && args.length >= 8) {
      setSelect("var1", args[0]); $("start1").value = args[1]; $("stop1").value = args[2]; $("step1").value = stepFor(args[1], args[2], args[3]);
      setSelect("var2", args[4]); $("start2").value = args[5]; $("stop2").value = args[6]; $("step2").value = stepFor(args[5], args[6], args[7]);
    }
    $("shots").value = plan.name === "count" ? (kw.num || 1) : (kw.shots_per_step || 1);
    $("period").value = kw.shot_period != null ? kw.shot_period : "";
    setSelect("trig", doc.trigger_profile || "");
    $("desc").value = doc.description || "";
    var body = $("devs"); body.textContent = "";
    (doc.devices || []).forEach(function (d) { body.appendChild(deviceRow(d.device, d.save_images !== false, d.essential !== false)); });
    noDevicesNote();
    recalc();
    renderCalibration();  // the calibration set is this table
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
  function stepFor(start, stop, num) {
    var n = Number(num);
    if (!(n > 1)) return 0;
    return Math.abs((Number(stop) - Number(start)) / (n - 1));
  }

  /* --------------------------------------------------------------- form */

  function setMode(mode, silent) {
    S.mode = mode;
    Array.prototype.forEach.call($("mode").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.mode === mode));
    });
    var count = mode === "noscan" || mode === "background";
    $("axis1").hidden = count;
    $("axis2").hidden = mode !== "grid";
    $("shots-hint").textContent = count ? "num — the shots of the count" : "shots_per_step";
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
  ["start1", "stop1", "step1", "start2", "stop2", "step2", "shots", "period"].forEach(function (id) {
    $(id).addEventListener("input", recalc);
  });

  function fmtSecs(secs) {
    var m = Math.floor(secs / 60), s = Math.round(secs % 60);
    return (m ? m + " min " : "") + s + " s";
  }
  function points(a, b, s) {
    if (!(s > 0)) return 0;
    return Math.floor(Math.abs(b - a) / s + 1e-9) + 1;
  }
  function setInvalid(id, bad) {
    var el = $(id);
    if (bad) el.setAttribute("aria-invalid", "true"); else el.removeAttribute("aria-invalid");
  }

  function axis(n) {
    // A descending range is a scan like any other (the focus scan runs
    // -18 → -26); only the step has to be positive.
    var a = Number($("start" + n).value), b = Number($("stop" + n).value), s = Number($("step" + n).value);
    var badStep = !(s > 0);
    setInvalid("step" + n, badStep);
    var pts = badStep ? 0 : points(a, b, s);
    // The plan takes a point COUNT; when the step does not divide the range
    // the effective step differs from the one typed, so say so.
    var eff = pts > 1 ? Math.abs(b - a) / (pts - 1) : 0;
    var effNote = pts > 1 && Math.abs(eff - s) > 1e-9 ? " · effective step " + Number(eff.toFixed(6)) : "";
    $("pts" + n).textContent = pts ? pts + " point" + (pts === 1 ? "" : "s") + (a > b ? " · descending" : "") + effNote : "—";
    return { variable: $("var" + n).value, start: a, stop: b, num: pts, ok: !badStep && !!$("var" + n).value };
  }

  var valid = false;
  function recalc() {
    var count = S.mode === "noscan" || S.mode === "background";
    var shots = parseInt($("shots").value, 10);
    var badShots = !(shots >= 1);
    setInvalid("shots", badShots);
    var ok = !badShots, steps = 1;
    if (!count) {
      var a1 = axis(1); ok = ok && a1.ok; steps = a1.num || 0;
      if (S.mode === "grid") { var a2 = axis(2); ok = ok && a2.ok; steps *= (a2.num || 0); }
    }
    var total = count ? shots : steps * shots;
    var period = S.acq === "strict" && $("period").value !== "" ? Number($("period").value) : null;
    var time = period ? " · ~<b>" + fmtSecs(total * period) + "</b> at " + period + " s/shot" : "";
    $("est").innerHTML = count
      ? "<b>" + shots + "</b> shots" + time
      : "<b>" + steps + "</b> step" + (steps === 1 ? "" : "s") + " × <b>" + shots + "</b> shots = <b>" + total + "</b> shots" + time;
    // The form is the document: a preset only seeds it, so Start (and
    // Save as preset) need a valid form, never a loaded preset (#900).
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
    $("btn-save-preset").disabled = !valid;
    btn.title = !st ? "waiting for the manager" : !st.connected ? "manager unreachable" : busy ? "a scan is paused — resume or stop it first" : !S.formable ? S.formableNote : !valid ? "fix the form first" : "";
    $("preset-name").textContent = S.presetName ? "preset " + S.presetName + (S.formable ? "" : " · " + S.formableNote) : "";
  }

  function buildPreset() {
    var doc = S.presetDoc || {};
    var count = S.mode === "noscan" || S.mode === "background";
    var shots = parseInt($("shots").value, 10);
    var kwargs = { acquisition: S.acq };
    var plan;
    if (count) {
      kwargs.num = shots;
      plan = { name: "count", args: [], kwargs: kwargs };
    } else {
      kwargs.shots_per_step = shots;
      var a1 = axis(1);
      if (S.mode === "grid") {
        var a2 = axis(2);
        plan = { name: "grid_scan", args: [a1.variable, a1.start, a1.stop, a1.num, a2.variable, a2.start, a2.stop, a2.num], kwargs: kwargs };
      } else {
        plan = { name: "scan", args: [a1.variable, a1.start, a1.stop, a1.num], kwargs: kwargs };
      }
    }
    if (S.acq === "strict" && $("period").value !== "") kwargs.shot_period = Number($("period").value);
    var devices = tableDevices();
    return {
      name: S.presetName || "adhoc",
      description: $("desc").value.trim(),
      trigger_profile: $("trig").value || null,
      // A background flag on the preset survives a mode change; the
      // Background mode sets it for a count.
      background: S.mode === "background" || (S.mode !== "noscan" && !!doc.background),
      devices: devices,
      plan: plan
    };
  }

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
    $("btn-move").disabled = !ok || !$("mv-var").value; $("btn-move").title = why;
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
  function renderMoveVars() {
    var sel = $("mv-var"); sel.textContent = "";
    if (S.settables.length) sel.appendChild(option("", "— pick a variable —"));
    S.settables.forEach(function (s) {
      var label = (s.alias ? s.alias + " · " : "") + s.name + (s.units ? " (" + s.units + ")" : "");
      sel.appendChild(option(s.name, label, false, s.alias ? s.name : ""));
    });
    if (!S.settables.length) sel.appendChild(option("", S.settablesNote ? "settables unavailable" : "no numeric settables", true));
    $("mv-hint").textContent = S.settablesNote || "";
    watchReadback(sel.value);
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
  $("mv-var").addEventListener("change", function () { watchReadback(this.value); renderIdleGates(); });
  $("btn-move").addEventListener("click", function () {
    var v = Number($("mv-val").value);
    var bad = $("mv-val").value === "" || !isFinite(v);
    setInvalid("mv-val", bad);
    if (bad) return;
    $("btn-move").disabled = true;
    post("/api/move", { variable: $("mv-var").value, value: v, operator: operator() })
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
  function renderDeviceList(q) {
    var list = $("devlist"); list.textContent = "";
    var have = {};
    tableDevices().forEach(function (d) { have[d.device] = true; });
    var needle = q.trim().toLowerCase();
    // The manager lists a device and its children (U_S1H, U_S1H.current …);
    // a preset names devices, so only the bare names are offered.
    var names = S.devices.filter(function (n) { return n.indexOf(".") === -1 && (!needle || n.toLowerCase().indexOf(needle) !== -1); });
    names.slice(0, 60).forEach(function (n) {
      var b = document.createElement("button");
      b.type = "button";
      var a = document.createElement("span"); a.textContent = n;
      var d = document.createElement("span"); d.className = "sub"; d.textContent = have[n] ? "in the table" : "";
      b.appendChild(a); b.appendChild(d);
      b.disabled = !!have[n];
      b.addEventListener("click", function () { addDevice(n); });
      list.appendChild(b);
    });
    $("devq-hint").textContent = S.devices.length
      ? names.length + " of " + S.devices.filter(function (n) { return n.indexOf(".") === -1; }).length + " devices" + (names.length > 60 ? " · type to narrow" : "")
      : "the manager's device list is empty or unreachable";
  }
  $("devq").addEventListener("input", function () { renderDeviceList($("devq").value); });
  $("btn-add-device").addEventListener("click", function () { renderDeviceList($("devq").value); setTimeout(function () { $("devq").focus(); }, 50); });
  function addDevice(name) {
    noDevicesNote();
    if (!$("devs").querySelector("tr[data-device]")) $("devs").textContent = "";
    $("devs").appendChild(deviceRow(name, true, true));
    noDevicesNote();
    $("devices-eyebrow").textContent = "devices · " + (S.presetName ? "preset " + S.presetName + " + " : "") + "edited";
    recalc(); renderCalibration();
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
      $("var1").focus();
    }
  });

  /* ---------------------------------------------------------------- boot */

  setMode("scan", true);
  setAcq("strict");
  renderTail();
  loadConfigs();
  refreshQueue();
  api("/api/progress").then(function (p) { S.progress = p; renderProgress(); }).catch(function () { /* the stream will say */ });
  connect();
})();
