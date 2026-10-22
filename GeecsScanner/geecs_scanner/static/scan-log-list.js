/* GEECS Scanner — the scan-log list panel.
 *
 * One shared list per experiment (GET/PUT /api/scan-log-variables): every
 * scan submitted from this scanner records it, and the logbook shows each
 * scan's start values for it. The picker lists the experiment's variables
 * (/api/logged-variables); only logged ones (get = yes) can be added, since
 * only those are in every scan's rows. Each edit saves at once, naming the
 * version it was made from; a stale one comes back 409 with the current
 * list, which replaces the page's. The value beside each name is one
 * readback taken when the list loads or the name is added — never polled.
 */
(function () {
  "use strict";

  var ROOT = document.body.dataset.root || "";
  var $ = function (id) { return document.getElementById(id); };
  var MAX_SUGGESTIONS = 12;
  var K = { ok: "ok", failed: "failed", unknown: "unknown" };  // kit words only

  var L = { variables: [], version: "", catalog: [], catalogNote: "", readings: {}, active: -1, shown: [] };

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

  function setChip(el, state, word) {
    el.dataset.state = state;
    el.textContent = word;
  }

  function hint(text) {
    $("sl-hint").textContent = text || "Only logged variables (get = yes) can be picked — they are the ones every scan records.";
  }

  function renderList() {
    var ul = $("sl-chips");
    ul.textContent = "";
    L.variables.forEach(function (name) {
      var li = document.createElement("li");
      var nm = document.createElement("span");
      nm.className = "nm"; nm.textContent = name; nm.title = name;
      var rb = document.createElement("span");
      rb.className = "rb"; rb.textContent = L.readings[name] || "";
      rb.title = "Readback when the list loaded or this name was added";
      var x = document.createElement("button");
      x.type = "button"; x.className = "x"; x.textContent = "×";
      x.setAttribute("aria-label", "Remove " + name);
      x.addEventListener("click", function () { save(L.variables.filter(function (n) { return n !== name; })); });
      li.appendChild(nm); li.appendChild(rb); li.appendChild(x);
      ul.appendChild(li);
    });
    var n = L.variables.length;
    setChip($("sl-chip"), n ? K.ok : K.unknown, n + (n === 1 ? " variable" : " variables"));
  }

  function adopt(doc) {
    L.variables = doc.variables || [];
    L.version = doc.version || "";
    renderList();
  }

  function save(next) {
    return api("/api/scan-log-variables", {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ variables: next, version: L.version })
    }).then(function (doc) {
      hint("");
      adopt(doc);
    }).catch(function (e) {
      if (e.status === 409 && e.payload.current) {
        adopt(e.payload.current);
        hint("Someone else changed the list meanwhile — this is theirs now; make your change again.");
        L.variables.forEach(read);
      } else {
        hint("Not saved: " + e.message);
      }
    });
  }

  function fmt(v) {
    if (v === null || v === undefined) return "";
    var a = Math.abs(v);
    return (a !== 0 && (a < 1e-3 || a >= 1e6)) ? v.toExponential(3) : String(Number(v.toPrecision(6)));
  }

  function read(name) {
    api("/api/readback?variable=" + encodeURIComponent(name)).then(function (r) {
      L.readings[name] = r.ok ? fmt(r.value) + (r.units ? " " + r.units : "") : "—";
      renderList();
    }).catch(function () { L.readings[name] = "—"; renderList(); });
  }

  function add(name) {
    if (!name || L.variables.indexOf(name) >= 0) return;
    $("sl-q").value = "";
    closeSugg();
    save(L.variables.concat([name])).then(function () { read(name); });
  }

  function closeSugg() {
    var ul = $("sl-sugg");
    ul.hidden = true; ul.textContent = "";
    $("sl-q").setAttribute("aria-expanded", "false");
    L.active = -1; L.shown = [];
  }

  function renderSugg() {
    var q = $("sl-q").value.trim().toLowerCase();
    var ul = $("sl-sugg");
    ul.textContent = "";
    if (!q) { closeSugg(); return; }
    L.shown = L.catalog.filter(function (it) { return it.name.toLowerCase().indexOf(q) >= 0; }).slice(0, MAX_SUGGESTIONS);
    if (!L.shown.length) { closeSugg(); return; }
    L.shown.forEach(function (it, i) {
      var chosen = L.variables.indexOf(it.name) >= 0;
      var li = document.createElement("li");
      var b = document.createElement("button");
      b.type = "button"; b.setAttribute("role", "option");
      b.disabled = chosen || !it.logged;
      b.setAttribute("aria-selected", String(i === L.active));
      var nm = document.createElement("span"); nm.textContent = it.name;
      var why = document.createElement("span"); why.className = "why";
      why.textContent = chosen ? "in the list" : (it.logged ? "" : "not logged");
      b.appendChild(nm); b.appendChild(why);
      b.addEventListener("mousedown", function (ev) { ev.preventDefault(); });
      b.addEventListener("click", function () { add(it.name); });
      li.appendChild(b); ul.appendChild(li);
    });
    ul.hidden = false;
    $("sl-q").setAttribute("aria-expanded", "true");
  }

  function pickable(i) {
    var it = L.shown[i];
    return it && it.logged && L.variables.indexOf(it.name) < 0;
  }

  $("sl-q").addEventListener("input", function () { L.active = -1; renderSugg(); });
  $("sl-q").addEventListener("blur", closeSugg);
  $("sl-q").addEventListener("keydown", function (ev) {
    if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      if (!L.shown.length) return;
      ev.preventDefault();
      var step = ev.key === "ArrowDown" ? 1 : -1;
      for (var k = 0, i = L.active; k < L.shown.length; k++) {
        i = (i + step + L.shown.length) % L.shown.length;
        if (pickable(i)) { L.active = i; break; }
      }
      renderSugg();
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      if (pickable(L.active)) { add(L.shown[L.active].name); return; }
      var typed = $("sl-q").value.trim();
      // With no catalog (the DB did not answer) a typed Device:Variable is
      // taken as written; the server checks its shape.
      if (!L.catalog.length && typed.indexOf(":") > 0) add(typed);
    } else if (ev.key === "Escape") {
      closeSugg();
    }
  });

  api("/api/scan-log-variables").then(function (doc) {
    adopt(doc);
    L.variables.forEach(read);
  }).catch(function (e) { setChip($("sl-chip"), K.failed, "unavailable"); hint("The list could not be read: " + e.message); });

  api("/api/logged-variables").then(function (cat) {
    L.catalog = cat.items || [];
    if (!L.catalog.length) hint("No variable list from the DB" + (cat.detail ? " (" + cat.detail + ")" : "") + " — type a Device:Variable and press Enter.");
  }).catch(function (e) { hint("No variable list: " + e.message + " — type a Device:Variable and press Enter."); });
})();
