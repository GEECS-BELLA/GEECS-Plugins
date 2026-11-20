/* The interlocks' web client: live verdict, table, inspector, events — read only.
 *
 * The stream (/api/stream) carries every new snapshot. The page counts the
 * age of the last one itself: past STALE_S it shows the stale banner and
 * takes the verdict off the screen ("No signal", the last verdict and its
 * time in the detail line). A page that stopped hearing the server never
 * keeps saying "Safe".
 *
 * The selection lives in the URL (?interlock=<name>), so a link opens the
 * same inspector.
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
