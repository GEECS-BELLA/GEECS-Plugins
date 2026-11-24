/* The interlocks' web client: live verdict, table, inspector, events, and the
 * two actions on the server (Arm, Reload).
 *
 * The stream (/api/stream) carries every new snapshot. The page counts the
 * age of the last one itself: past STALE_S it shows the stale banner and
 * takes the verdict off the screen ("No signal", the last verdict and its
 * time in the detail line). A page that stopped hearing the server never
 * keeps saying "Safe".
 *
 * The selection lives in the URL (?interlock=<name>), so a link opens the
 * same inspector.
 *
 * An action needs the operator's name (kept in this browser's storage)
 * and a confirmation in the kit's dialog; it is sent once and the
 * page shows the server's own message, never a retry.
 */
(function () {
  "use strict";

  var STALE_S = 2.0;          // seconds of silence before the page says stale
  var DETAIL_EVERY_MS = 2000; // the inspector's trend and history refresh
  var root = document.body.dataset.root || "";
  var last = null;            // the latest snapshot
  var lastAt = 0;             // Date.now() when it arrived
  var expanded = {};          // group name -> shown open
  var wasStale = false;       // the rows were last drawn stale
  var selected = new URLSearchParams(location.search).get("interlock");
  var $ = function (id) { return document.getElementById(id); };

  function text(tag, content, cls) {
    var el = document.createElement(tag);
    if (content !== undefined && content !== null) el.textContent = content;
    if (cls) el.className = cls;
    return el;
  }

  function chip(word, kitState, extra) {
    var el = text("span", word, "chip" + (extra ? " " + extra : ""));
    el.dataset.state = kitState;
    return el;
  }

  function clock(seconds) {
    return new Date(seconds * 1000).toLocaleTimeString([], { hour12: false });
  }

  // -- the verdict and its freshness ------------------------------------------

  function drawVerdict() {
    var verdict = $("verdict");
    var age = last ? (Date.now() - lastAt) / 1000 : Infinity;
    var stale = age > STALE_S;
    $("stale").hidden = !stale || !last;
    if (!last) {
      $("freshness").textContent = "no update yet";
      return;
    }
    $("freshness").textContent = "updated " + age.toFixed(1) + " s ago";
    if (stale) {
      verdict.dataset.state = "unknown";
      verdict.textContent = "No signal";
      $("verdict-detail").textContent =
        "last verdict " + last.headline + " at " + clock(last.taken);
      $("stale-text").textContent =
        "No update from the server for " + age.toFixed(0) +
        " s. The verdict shown may be out of date.";
    } else {
      verdict.dataset.state = last.kit_state;
      verdict.textContent = last.headline;
      $("verdict-detail").textContent = last.detail;
    }
    if (stale !== wasStale) { drawRows(); drawDetail(); }  // grey every chip while stale
  }

  // -- the table ---------------------------------------------------------------

  function drawRows() {
    if (!last) return;
    // Stale: every chip goes grey (the word stays), so no row looks live.
    wasStale = (Date.now() - lastAt) / 1000 > STALE_S;
    // Arming only means something for settings checks: no button without one.
    $("btn-arm").hidden = !anyArmable();
    var body = $("rows");
    body.textContent = "";
    last.statuses.forEach(function (s) {
      var row = document.createElement("tr");
      row.setAttribute("aria-selected", String(s.name === selected));
      var state = document.createElement("td");
      state.appendChild(chip(s.state, wasStale ? "unknown" : s.kit_state));
      var name = document.createElement("td");
      if (s.checks.length > 1) {
        var toggle = text("button", expanded[s.name] ? "▾" : "▸", "btn sm ghost");
        toggle.type = "button";
        toggle.setAttribute("aria-expanded", String(!!expanded[s.name]));
        toggle.setAttribute("aria-label", "Show the checks of " + s.name);
        toggle.addEventListener("click", function (event) {
          event.stopPropagation();
          expanded[s.name] = !expanded[s.name];
          drawRows();
        });
        name.appendChild(toggle);
      }
      name.appendChild(text("span", s.name + (s.armable ? " (holds settings)" : "")));
      var value = text("td", s.checks.length > 1 ? s.checks.length + " checks" : s.value, "num mono");
      var reason = text("td", s.diagnostic);
      [state, name, value, reason].forEach(function (td) { row.appendChild(td); });
      row.addEventListener("click", function () { select(s.name); });
      body.appendChild(row);
      if (s.checks.length > 1 && expanded[s.name]) {
        s.checks.forEach(function (c) {
          var sub = document.createElement("tr");
          sub.appendChild(document.createElement("td"));
          sub.appendChild(text("td", c.condition, "dim"));
          sub.appendChild(text("td", c.value, "num mono"));
          sub.appendChild(text("td", c.unsafe ? c.diagnostic : "ok", c.unsafe ? "" : "dim"));
          body.appendChild(sub);
        });
      }
    });
  }

  // -- the inspector -----------------------------------------------------------

  function select(name) {
    selected = name;
    var url = new URL(location.href);
    url.searchParams.set("interlock", name);
    history.replaceState(null, "", url);
    drawRows();
    drawDetail();
    refreshDetail();
  }

  function drawDetail() {
    var status = last && last.statuses.filter(function (s) { return s.name === selected; })[0];
    $("detail-empty").hidden = !!status;
    $("detail-body").hidden = !status;
    if (!status) return;
    $("detail-name").textContent = status.name;
    var state = $("detail-state");
    state.dataset.state = wasStale ? "unknown" : status.kit_state;
    state.textContent = status.state;
    $("detail-since").textContent = "since " + clock(status.since);
    $("detail-description").textContent = status.description || status.condition;
    $("btn-arm-one").hidden = !status.armable;
  }

  function drawTrend(data) {
    var host = $("trend");
    if (!window.Plotly || !window.GeecsPlot) return;
    // One stacked plot per check, sharing the time axis: a group's checks
    // have unrelated scales (counts beside pixels), so one axis would
    // flatten all but the largest.
    var plotted = data.series.filter(function (s) { return s.t.length; });
    var traces = [];
    var over = {
      margin: { l: 56, r: 12, t: 22, b: 28 },
      showlegend: false,
      uirevision: data.name,
      grid: { rows: Math.max(1, plotted.length), columns: 1, pattern: "coupled", ygap: 0.35 },
      xaxis: { tickformat: "%H:%M:%S" },
      annotations: [],
    };
    plotted.forEach(function (s, i) {
      var axis = i === 0 ? "" : String(i + 1);
      var x = s.t.map(function (t) { return new Date(t * 1000); });
      traces.push({ x: x, y: s.v, name: s.condition, mode: "lines", yaxis: "y" + axis });
      if (s.limit !== null) {
        traces.push({
          x: [x[0], x[x.length - 1]], y: [s.limit, s.limit], name: "limit",
          mode: "lines", line: { dash: "dash" }, yaxis: "y" + axis,
        });
      }
      over.annotations.push({
        text: s.condition, xref: "paper", yref: "y" + axis + " domain",
        x: 0, y: 1, xanchor: "left", yanchor: "bottom", showarrow: false,
      });
    });
    $("trend").style.height = Math.max(1, plotted.length) * 150 + 40 + "px";
    $("trend-note").textContent = traces.length
      ? "Dashed: the limit. Settings and external checks have no number to plot."
      : "Nothing to plot for this interlock yet.";
    var layout = window.GeecsPlot.layout(over);
    window.Plotly.react(host, traces, layout, window.GeecsPlot.config());
    window.GeecsPlot.watch(host);
  }

  function drawHistory(events, host) {
    host.textContent = "";
    if (!events.length) { host.textContent = "No state changes yet."; return; }
    events.forEach(function (e) {
      var line = text("div", clock(e.at) + "  " + e.name + "  " + e.old + " → " + e.new +
        (e.diagnostic ? "  · " + e.diagnostic : ""));
      host.appendChild(line);
    });
  }

  function getJSON(path) {
    return fetch(root + path, { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error(r.status + " " + r.statusText);
      return r.json();
    });
  }

  function refreshDetail() {
    if (!selected) return;
    var q = encodeURIComponent(selected);
    getJSON("/api/trend?name=" + q).then(drawTrend).catch(function () {});
    getJSON("/api/events?limit=20&name=" + q)
      .then(function (events) { drawHistory(events, $("history")); })
      .catch(function () {});
  }

  function refreshEvents() {
    getJSON("/api/events?limit=30")
      .then(function (events) { drawHistory(events, $("events")); })
      .catch(function () {});
  }

  // -- the actions -------------------------------------------------------------

  var OP_KEY = "geecs.operator";
  try { $("operator").value = window.localStorage.getItem(OP_KEY) || ""; } catch (e) { /* private window */ }
  $("operator").addEventListener("change", function () {
    try { window.localStorage.setItem(OP_KEY, $("operator").value.trim()); } catch (e) { /* ignore */ }
  });
  function operator() { return $("operator").value.trim(); }

  var busy = false;           // one action in flight at a time
  var pending = null;         // the action the dialog is asking about

  function say(message, failed) {
    $("action").hidden = false;
    if (failed) $("action").dataset.state = "error";
    else delete $("action").dataset.state;
    $("action-text").textContent = clock(Date.now() / 1000) + "  " + message;
  }

  function ask(title, question, action) {
    if (busy) { say("Another action is still running.", true); return; }
    if (!operator()) {
      say("Enter your name (operator) before acting.", true);
      $("operator").focus();
      return;
    }
    pending = action;
    $("confirm-title").textContent = title;
    $("confirm-text").textContent = question;
    window.GeecsKit.confirm($("confirm")).open();
  }

  $("confirm-ok").addEventListener("click", function () {
    var action = pending;
    pending = null;
    window.GeecsKit.confirm($("confirm")).close();
    if (action) action();
  });

  function anyArmable() {
    return !!last && last.statuses.some(function (s) { return s.armable; });
  }

  function send(path, body) {
    busy = true;
    ["btn-arm", "btn-reload", "btn-arm-one"].forEach(function (id) { $(id).disabled = true; });
    body.operator = operator();
    fetch(root + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(function (r) {
      return r.json().then(function (data) {
        // 200/409 carry the engine's message; 400/422 FastAPI's detail.
        var message = data.message || (typeof data.detail === "string" ? data.detail : r.status + " " + r.statusText);
        say(message, !(r.ok && data.ok));
      }, function () { say(r.status + " " + r.statusText, true); });
    }).catch(function (e) {
      say("Not sent: " + e.message + ". Check the verdict before trying again.", true);
    }).then(function () {
      busy = false;
      ["btn-arm", "btn-reload", "btn-arm-one"].forEach(function (id) { $(id).disabled = false; });
      refreshEvents();
    });
  }

  $("btn-arm").addEventListener("click", function () {
    ask("Arm every settings check",
      "Record the settings every settings check sees now as the ones to hold. " +
      "Arm only when you know the settings are right.",
      function () { send("/api/arm", {}); });
  });
  $("btn-arm-one").addEventListener("click", function () {
    var name = selected;
    ask("Arm " + name,
      "Record the settings " + name + " sees now as the ones to hold. " +
      "Arm only when you know they are right.",
      function () { send("/api/arm", { interlock: name }); });
  });
  $("btn-reload").addEventListener("click", function () {
    ask("Reload the configuration",
      "Re-read the running configuration file. Every interlock then waits for fresh " +
      "values (unsafe meanwhile)" +
      (anyArmable() ? ", and every settings check must be armed again." : "."),
      function () { send("/api/reload", {}); });
  });

  // -- the stream --------------------------------------------------------------

  function take(snapshot) {
    last = snapshot;
    lastAt = Date.now();
    drawVerdict();
    drawRows();
    drawDetail();
  }

  var source = new EventSource(root + "/api/stream");
  source.onmessage = function (message) { take(JSON.parse(message.data)); };
  // EventSource reconnects on its own; the freshness timer reports the gap.

  setInterval(drawVerdict, 500);
  setInterval(function () { refreshDetail(); refreshEvents(); }, DETAIL_EVERY_MS);
  getJSON("/api/status").then(take).catch(function () {});
  refreshEvents();
  if (selected) refreshDetail();
})();
