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
    presets: [], presetName: null, presetDoc: null,
    variables: [], triggers: [],
    mode: "scan", acq: "strict",
    consoleSeq: 0, epoch: null,
    formable: true, formableNote: "",
    pendingPreset: null, pendingAck: []
  };

  var NOTES = {
    noscan: "A count: no axis, the shots below at the current position. The old console's No-scan.",
    scan: "A 1D scan: one variable stepped from start to stop, shots-per-step at each. Free-run in the old console is gated here; strict is the default.",
    grid: "A grid_scan: axis 2 steps inside every axis-1 point. Steps multiply.",
    background: "A count flagged background in the preset, so analysis knows these are darks."
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
    $("lv-manager").textContent = st.re_state || (st.connected ? "?" : "down");
    $("verb-note").textContent = st.re_state === "paused"
      ? "Paused. Resume continues at the next step."
      : "Pause takes effect after the current step.";
    if (!running) $("paused-reason").textContent = "";
    updateStartGate();
  }

  function renderProgress() {
    var p = S.progress;
    if (!p) return;
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
    q.finished.slice(0, 6).forEach(function (r) {
      var a = document.createElement("a");
      a.href = "#queue";
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
      if (S.epoch !== null && line.epoch !== S.epoch) { $("tail").textContent = ""; }
      S.epoch = line.epoch;
      S.consoleSeq = line.seq;
      appendTail(line);
    });
    es.onerror = function () {
      setChip($("chip-manager"), K.failed, "manager", "event stream lost; reconnecting");
    };
  }

  function appendTail(line) {
    var tail = $("tail");
    var d = new Date(line.at * 1000);
    var ln = document.createElement("div");
    ln.className = "ln" + (/FAILED|error|refus/i.test(line.text) ? " warn" : "");
    var t = document.createElement("span"); t.className = "t";
    t.textContent = d.toTimeString().slice(0, 8);
    var body = document.createElement("span"); body.textContent = line.text;
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
      api("/api/configs/trigger_profiles")
    ]).then(function (res) {
      S.presets = res[0].names; S.variables = res[1]; S.triggers = res[2].names;
      var pl = $("presets"); pl.textContent = "";
      S.presets.forEach(function (name) {
        var b = document.createElement("button");
        b.type = "button"; b.textContent = name; b.dataset.preset = name;
        b.addEventListener("click", function () { selectPreset(name); });
        pl.appendChild(b);
      });
      $("presets-note").textContent = S.presets.length ? "presets/ in the configs tree" : "No presets in the configs tree.";
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

  function selectPreset(name) {
    S.presetName = name;  // claimed now, so a slower default cannot override the click
    Array.prototype.forEach.call($("presets").querySelectorAll("button"), function (b) {
      b.setAttribute("aria-pressed", String(b.dataset.preset === name));
    });
    api("/api/configs/presets/" + encodeURIComponent(name)).then(function (doc) {
      if (S.presetName !== name) return;  // a later click won
      S.presetDoc = doc;
      $("preset-name").textContent = "preset " + name;
      $("devices-eyebrow").textContent = "devices · preset " + name;
      fillFormFromPreset(doc);
    }).catch(function (e) { showError("Preset " + name + ": " + e.message); });
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
    (doc.devices || []).forEach(function (d) {
      var tr = document.createElement("tr");
      tr.appendChild(td(d.device, "id"));
      tr.appendChild(td(checkbox(d.save_images !== false, "save images for " + d.device)));
      tr.appendChild(td(checkbox(d.essential !== false, d.device + " essential")));
      body.appendChild(tr);
    });
    if (!(doc.devices || []).length) body.appendChild(td("This preset names no devices.", "dim"));
    recalc();
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
    $("mode-note").textContent = NOTES[mode] || "";
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
    valid = ok && !!S.presetDoc;
    updateStartGate();
  }

  function updateStartGate() {
    var st = S.status;
    var busy = !st || !st.connected || st.re_state === "running" || st.re_state === "paused";
    var btn = $("btn-start");
    btn.disabled = busy || !valid || !S.formable;
    btn.title = !st ? "waiting for the manager" : !st.connected ? "manager unreachable" : busy ? "a scan is running" : !S.presetDoc ? "pick a preset" : !S.formable ? S.formableNote : !valid ? "fix the form first" : "";
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
    var devices = [];
    Array.prototype.forEach.call($("devs").querySelectorAll("tr"), function (tr) {
      var boxes = tr.querySelectorAll("input[type=checkbox]");
      if (boxes.length !== 2) return;
      devices.push({ device: tr.firstChild.textContent, save_images: boxes[0].checked, essential: boxes[1].checked });
    });
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
  loadConfigs();
  refreshQueue();
  api("/api/progress").then(function (p) { S.progress = p; renderProgress(); }).catch(function () { /* the stream will say */ });
  connect();
})();
