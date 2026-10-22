/* GEECS Scanner — the scan-log list panel.
 *
 * One shared list per experiment (GET/PUT /api/scan-log-variables): every
 * scan submitted from this scanner records it, and the logbook shows each
 * scan's start values for it. "Add variables…" opens a drawer shaped like
 * the device picker's — search, click to select, shift-click for a range,
 * one Add for the lot — over the experiment's variables
 * (/api/logged-variables); only logged ones (get = yes) can be picked,
 * since only those are in every scan's rows. Each edit saves at once,
 * naming the version it was made from; a stale one comes back 409 with the
 * current list, which replaces the page's. The value beside each name is
 * one readback taken when the list loads or the name is added — never polled.
 */
(function () {
  "use strict";

  var ROOT = document.body.dataset.root || "";
  var $ = function (id) { return document.getElementById(id); };
  var MAX_SHOWN = 80;
  // The server's bound (service/scan_log_list.MAX_VARIABLES): a batch that
  // would pass it is refused whole, so the picker never selects past it.
  var MAX_VARIABLES = 64;
  var K = { ok: "ok", failed: "failed", unknown: "unknown" };  // kit words only

  var L = { variables: [], version: "", catalog: [], catalogNote: "", readings: {} };
  var picked = {};   // name -> true, across searches, until Add or Clear
  var shown = [];    // [{name, pickable}] in render order, for shift-ranges
  var anchor = null; // the last row clicked, the other end of a range

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

  function hint(text) { $("sl-hint").textContent = text || ""; }

  /* --------------------------------------------------------- the panel */

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
      x.addEventListener("click", function () { save(function (list) { return list.filter(function (n) { return n !== name; }); }); });
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

  // Edits run one after another: each is a change to the list as the last
  // save left it, sent with that save's version. Two quick clicks are two
  // queued changes, never two saves of the same version — which would make
  // the page's own first save look like another operator's edit.
  var pending = Promise.resolve();

  function save(change) {
    pending = pending.then(function () {
      var next = change(L.variables.slice());
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
    });
    return pending;
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

  /* -------------------------------------------- the drawer (device picker's shape) */

  function pickedNames() { return Object.keys(picked); }
  function room() {
    return MAX_VARIABLES - L.variables.length - pickedNames().length;
  }
  /** Whether a catalog row may be picked: logged, and not in the list already. */
  function isPickable(item, have) {
    return !!item.logged && !have[item.name];
  }

  function renderPicks() {
    var n = pickedNames().length;
    $("btn-sl-add").disabled = !n;
    $("btn-sl-add").textContent = n ? "Add " + n + " variable" + (n === 1 ? "" : "s") : "Add";
    $("btn-sl-none").disabled = !n;
    $("btn-sl-room").textContent = room() > 0 ? "" : "The list is full (" + MAX_VARIABLES + " variables).";
    Array.prototype.forEach.call($("sl-list").children, function (b) {
      b.setAttribute("aria-pressed", picked[b.dataset.variable] ? "true" : "false");
    });
  }

  function pick(name, index, range) {
    if (range && anchor !== null) {
      var lo = Math.min(anchor, index), hi = Math.max(anchor, index);
      // a range takes everything pickable in it: logged, and not already listed
      for (var i = lo; i <= hi && room() > 0; i++) if (shown[i] && shown[i].pickable) picked[shown[i].name] = true;
    } else {
      if (picked[name]) delete picked[name]; else if (room() > 0) picked[name] = true;
      anchor = index;
    }
    renderPicks();
  }

  function renderPicker() {
    var list = $("sl-list"); list.textContent = "";
    var needle = $("slq").value.trim().toLowerCase();
    var have = {};
    L.variables.forEach(function (n) { have[n] = true; });
    var items = L.catalog;
    var typed = $("slq").value.trim();
    // With no catalog (the DB did not answer) a typed Device:Variable is
    // offered as written; the server checks its shape.
    if (!items.length) {
      items = typed.indexOf(":") > 0 ? [{ name: typed, logged: true }] : [];
      // Typed rows are not a catalog: a pick of "U_A:po" must not survive
      // the keystroke that makes it "U_A:pos".
      Object.keys(picked).forEach(function (n) { if (n !== typed) delete picked[n]; });
    }
    var matches = items.filter(function (it) { return !needle || it.name.toLowerCase().indexOf(needle) !== -1; });
    shown = []; anchor = null;
    matches.slice(0, MAX_SHOWN).forEach(function (it, i) {
      var pickable = isPickable(it, have);
      shown.push({ name: it.name, pickable: pickable });
      var b = document.createElement("button");
      b.type = "button";
      b.dataset.variable = it.name;
      b.setAttribute("aria-pressed", picked[it.name] ? "true" : "false");
      var a = document.createElement("span"); a.textContent = it.name;
      var d = document.createElement("span"); d.className = "sub";
      d.textContent = have[it.name] ? "in the list" : (it.logged ? "" : "not logged");
      b.appendChild(a); b.appendChild(d);
      b.disabled = !pickable;
      b.addEventListener("click", function (ev) { pick(it.name, i, ev.shiftKey); });
      list.appendChild(b);
    });
    var logged = L.catalog.filter(function (it) { return it.logged; }).length;
    $("slq-hint").textContent = L.catalog.length
      ? matches.length + " of " + L.catalog.length + " variables · " + logged + " logged" + (matches.length > MAX_SHOWN ? " · type to narrow" : "")
      : (L.catalogNote || "No variable list from the DB") + " — type a Device:Variable";
    renderPicks();
  }

  $("slq").addEventListener("input", renderPicker);
  // Opening the drawer starts a fresh selection: Esc and the scrim are a
  // cancel, so picks never committed with Add do not come back.
  $("btn-sl-open").addEventListener("click", function () {
    picked = {}; renderPicker(); setTimeout(function () { $("slq").focus(); }, 50);
  });
  $("btn-sl-all").addEventListener("click", function () {
    shown.forEach(function (s) { if (s.pickable && !picked[s.name] && room() > 0) picked[s.name] = true; });
    renderPicks();
  });
  // Enter adds what is picked — or, with nothing picked, the one row the
  // search leaves (the typed Device:Variable when the DB gave no list).
  $("slq").addEventListener("keydown", function (ev) {
    if (ev.key !== "Enter") return;
    ev.preventDefault();
    if (!pickedNames().length) {
      var only = shown.filter(function (s) { return s.pickable; });
      if (only.length !== 1 || room() <= 0) return;
      picked[only[0].name] = true;
    }
    $("btn-sl-add").click();
  });
  $("btn-sl-none").addEventListener("click", function () { picked = {}; anchor = null; renderPicks(); });
  $("btn-sl-add").addEventListener("click", function () {
    var names = pickedNames();
    if (!names.length) return;
    picked = {}; anchor = null;
    window.GeecsKit.drawer($("drw-scanlog")).close();
    save(function (list) {
      names.forEach(function (n) { if (list.indexOf(n) < 0) list.push(n); });
      return list;
    }).then(function () { names.forEach(read); });
  });

  /* ------------------------------------------------------------- load */

  api("/api/scan-log-variables").then(function (doc) {
    adopt(doc);
    L.variables.forEach(read);
  }).catch(function (e) { setChip($("sl-chip"), K.failed, "unavailable"); hint("The list could not be read: " + e.message); });

  api("/api/logged-variables").then(function (cat) {
    L.catalog = cat.items || [];
    L.catalogNote = cat.detail ? "No variable list from the DB (" + cat.detail + ")" : "";
  }).catch(function (e) { L.catalogNote = "No variable list: " + e.message; });
})();
