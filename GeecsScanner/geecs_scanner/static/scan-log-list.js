/* GEECS Scanner — the scan-log list panel.
 *
 * One shared list per experiment (GET/PUT /api/scan-log-variables): every
 * scan submitted from this scanner records it, and the logbook shows each
 * scan's start values for it. "Add variables…" opens the kit's drawer
 * picker (GeecsKit.picker — the device picker's component: search, click,
 * shift-click for a range, one Add for the lot) over the experiment's variables
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

  var L = { variables: [], version: "", catalog: [], catalogNote: "", catalogLoading: true, readings: {} };

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

  /* -------------------------------------------- the drawer: GeecsKit.picker */

  /** Whether a catalog row may be picked: logged, and not in the list already. */
  function isPickable(item, have) {
    return !!item.logged && !have[item.name];
  }

  function rows(query) {
    var needle = query.trim().toLowerCase();
    var have = {};
    L.variables.forEach(function (n) { have[n] = true; });
    var items = L.catalog;
    // With no catalog (the DB did not answer) a typed Device:Variable is
    // offered as written; the server checks its shape.
    if (!items.length) items = query.trim().indexOf(":") > 0 ? [{ name: query.trim(), logged: true }] : [];
    return items
      .filter(function (it) { return !needle || it.name.toLowerCase().indexOf(needle) !== -1; })
      .map(function (it) {
        return {
          name: it.name,
          pickable: isPickable(it, have),
          sub: have[it.name] ? "in the list" : (it.logged ? "" : "not logged")
        };
      });
  }

  var picker = window.GeecsKit && window.GeecsKit.picker({
    drawer: $("drw-scanlog"),
    opener: $("btn-sl-open"),
    search: $("slq"),
    list: $("sl-list"),
    add: $("btn-sl-add"),
    all: $("btn-sl-all"),
    none: $("btn-sl-none"),
    noun: ["variable", "variables"],
    maxShown: MAX_SHOWN,
    rows: rows,
    // The server's bound: a batch that would pass it is refused whole.
    room: function () { return MAX_VARIABLES - L.variables.length; },
    // Typed rows are not a catalog: a pick of "U_A:po" must not survive the
    // keystroke that makes it "U_A:pos".
    shownOnly: function () { return !L.catalog.length; },
    onRender: function (info) {
      var logged = L.catalog.filter(function (it) { return it.logged; }).length;
      $("slq-hint").textContent = L.catalog.length
        ? info.matches + " of " + L.catalog.length + " variables · " + logged + " logged" + (info.matches > info.shown ? " · type to narrow" : "")
        : (L.catalogLoading ? "Reading the variable list…" : (L.catalogNote || "No variable list from the DB") + " — type a Device:Variable");
      $("btn-sl-room").textContent = info.full ? "The list is full (" + MAX_VARIABLES + " variables)." : "";
    },
    onAdd: function (names) {
      save(function (list) {
        names.forEach(function (n) { if (list.indexOf(n) < 0) list.push(n); });
        return list;
      }).then(function () { names.forEach(read); });
    }
  });

  /* ------------------------------------------------------------- load */

  api("/api/scan-log-variables").then(function (doc) {
    adopt(doc);
    L.variables.forEach(read);
  }).catch(function (e) { setChip($("sl-chip"), K.failed, "unavailable"); hint("The list could not be read: " + e.message); });

  // The drawer may already be open when the list lands (a cold DB query):
  // refresh it then, keeping what is picked, rather than leaving it empty.
  api("/api/logged-variables").then(function (cat) {
    L.catalog = cat.items || [];
    L.catalogNote = cat.detail ? "No variable list from the DB (" + cat.detail + ")" : "";
  }).catch(function (e) {
    L.catalogNote = "No variable list: " + e.message;
  }).then(function () {
    L.catalogLoading = false;
    if (picker) picker.refresh();
  });
})();
