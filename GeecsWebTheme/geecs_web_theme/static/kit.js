/* GEECS surface kit — the behaviours kit.css cannot express.
 *
 * Load deferred, beside theme.js. Everything here is additive: with this
 * file absent the page still renders correctly, <details> still opens, and
 * the density stamped by theme-boot.js still applies — only the drawer,
 * the dialog helper, the density control and the drawer picker go missing.
 *
 *   <div data-density-picker></div>   the spacing control, built here
 *   GeecsKit.drawer(el)               rung 2 — open/close with Esc, scrim,
 *                                     focus return
 *   GeecsKit.confirm(el)              rung 3 — <dialog>.showModal() with a
 *                                     fallback for browsers without it
 *   GeecsKit.picker(opts)             a multi-select .picklist in a drawer:
 *                                     search, click / shift-click, select
 *                                     all shown, clear, one Add for the lot
 *
 * Why a drawer needs script at all, when rungs 0 and 3 do not: the browser
 * gives us <details> and <dialog> outright, but there is no element for "a
 * panel over the page that Esc dismisses and that hands focus back where
 * it came from". That is the whole of this file's drawer half, and it is
 * deliberately small — if it grows, the thing being built is probably a
 * route (rung 4) wearing a drawer's clothes.
 *
 * It dispatches `geecs:density` on window with {density} so anything that
 * measures itself — a Plotly figure sizing to its container — can relayout.
 */
(function () {
  "use strict";

  var CFG = window.GEECS_THEME;
  var root = document.documentElement;

  var DENSITY_LABEL = {
    comfortable: { label: "Comfortable", hint: "Roomier rows — reading and writing" },
    compact: { label: "Compact", hint: "Tighter rows — tables and live panels" }
  };

  function write(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* private window */ }
  }

  /* ---- density ---------------------------------------------------- */

  function setDensity(value) {
    if (!CFG || CFG.densities.indexOf(value) === -1) return;
    root.setAttribute("data-density", value);
    write(CFG.keys.density, value);
    sync();
    try {
      window.dispatchEvent(
        new CustomEvent("geecs:density", { detail: { density: value } })
      );
    } catch (e) { /* very old browser: the attribute is still stamped */ }
  }

  function sync() {
    var now = root.getAttribute("data-density");
    var buttons = document.querySelectorAll("[data-density-picker] button[data-density]");
    Array.prototype.forEach.call(buttons, function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-density") === now));
    });
  }

  function buildDensityPickers() {
    if (!CFG || !CFG.densities) return;
    var hosts = document.querySelectorAll("[data-density-picker]");
    Array.prototype.forEach.call(hosts, function (host) {
      // Not firstChild: `<div data-density-picker>\n</div>` — ordinary Jinja
      // formatting — has a whitespace text node and would silently ship no
      // control at all.
      if (host.querySelector("button")) return; // already built
      host.className = host.className ? host.className + " seg" : "seg";
      host.setAttribute("role", "group");
      host.setAttribute("aria-label", "Row density");
      CFG.densities.forEach(function (value) {
        var meta = DENSITY_LABEL[value] || { label: value, hint: value };
        var b = document.createElement("button");
        b.type = "button";
        b.setAttribute("data-density", value);
        b.title = meta.hint;
        b.textContent = meta.label;
        b.addEventListener("click", function () { setDensity(value); });
        host.appendChild(b);
      });
    });
    sync();
  }

  /* ---- rung 2: drawer --------------------------------------------- */

  //: The drawer currently open, so Esc and the scrim know what to close.
  //  One at a time by design: a drawer never opens another drawer.
  var openDrawer = null;
  var returnFocusTo = null;

  function scrimFor(el) {
    var id = el.getAttribute("data-scrim");
    if (id) return document.getElementById(id);
    var found = document.querySelector(".scrim");
    if (found) return found;
    found = document.createElement("div");
    found.className = "scrim";
    document.body.appendChild(found);
    return found;
  }

  function closeDrawer() {
    if (!openDrawer) return;
    var el = openDrawer;
    var scrim = scrimFor(el);
    openDrawer = null;
    el.setAttribute("data-open", "false");
    el.setAttribute("aria-hidden", "true");
    if (scrim) scrim.setAttribute("data-open", "false");
    // Focus goes back where it came from, or the opener is gone and the
    // page keeps it — never left on a hidden element.
    if (returnFocusTo && document.contains(returnFocusTo)) {
      try { returnFocusTo.focus(); } catch (e) { /* not focusable any more */ }
    }
    returnFocusTo = null;
  }

  function showDrawer(el, opener) {
    if (!el) return;
    if (openDrawer && openDrawer !== el) closeDrawer();
    returnFocusTo = opener || document.activeElement;
    openDrawer = el;
    var scrim = scrimFor(el);
    if (scrim) {
      scrim.setAttribute("data-open", "true");
      if (!scrim.getAttribute("data-kit-wired")) {
        scrim.setAttribute("data-kit-wired", "1");
        scrim.addEventListener("click", closeDrawer);
      }
    }
    el.setAttribute("data-open", "true");
    el.setAttribute("aria-hidden", "false");
    // A selector list carries no priority — querySelector returns the first
    // match in TREE order, which here is the header's Close button, so the
    // one-call form silently ignores [autofocus]. <dialog>.showModal() does
    // honour it, and two rungs disagreeing on identical markup gets debugged
    // as a browser quirk. Ask for the preferred element first, on its own.
    var first =
      el.querySelector("[autofocus]") ||
      el.querySelector("[data-drawer-close],button,a[href],input,select,textarea");
    if (first) {
      try { first.focus(); } catch (e) { /* nothing focusable: leave it */ }
    }
  }

  //: A missing target is always a typo in a template, and it fails by doing
  //  nothing at all — the exact "dead button" shape a past review caught in
  //  this repo before. Say so rather than returning quietly.
  function need(id, what) {
    var el = document.getElementById(id);
    if (!el && window.console && console.warn) {
      console.warn("[geecs-kit] no " + what + ' with id "' + id + '" — dead control');
    }
    return el;
  }

  function drawer(el) {
    return {
      open: function (opener) { showDrawer(el, opener); },
      close: closeDrawer,
      get isOpen() { return openDrawer === el; }
    };
  }

  /* ---- the drawer picker ------------------------------------------- */

  // "Add device…" was the first: a drawer with a search, a .picklist where a
  // click toggles a row and shift-click takes the range from the last row
  // clicked, Select all shown / Clear, and one Add for the lot. Every
  // drawer that adds several things to a list is this one component — the
  // surface says what the rows are, how many more fit, and what Add does.
  //
  //   GeecsKit.picker({
  //     drawer, search, list, add, all, none,   // elements
  //     noun: ["device", "devices"],             // the Add button's count
  //     rows: function (query) {                 // every row matching query:
  //       return [{name, sub, pickable}];        //   sub = the row's note
  //     },
  //     onAdd: function (names) {},              // after Add; the drawer closes
  //     room: function () { return n; },         // optional: how many more fit
  //     shownOnly: function () { return b; },    // optional: drop picks no
  //                                              //   longer in the rows
  //     maxShown: 60,                            // rows drawn; type to narrow
  //     onRender: function (info) {},            // after every redraw and change:
  //                                              //   {matches, shown, picked, full}
  //     opener: el,                              // optional: resets on open
  //   })
  //   → {open, refresh, pick, picked}
  //
  // Selection is kept by name, so narrowing the search and picking more does
  // not lose what is already ticked; opening the drawer starts afresh (Esc
  // and the scrim are a cancel). A pick never passes `room()`: a list with a
  // bound refuses a batch past it whole, so the picker stops at the bound.
  // Enter in the search adds the picks — or, with none, the one pickable row
  // the search leaves.
  function picker(o) {
    var picked = {};
    var shown = [];
    var anchor = null;
    var maxShown = o.maxShown || 60;
    var noun = o.noun || ["item", "items"];
    var matches = 0;   // rows the last refresh matched (shown may be fewer)

    function names() { return Object.keys(picked); }
    function room() { return (o.room ? o.room() : Infinity) - names().length; }

    function sync() {
      var n = names().length;
      o.add.disabled = !n;
      o.add.textContent = n ? "Add " + n + " " + (n === 1 ? noun[0] : noun[1]) : "Add";
      if (o.none) o.none.disabled = !n;
      Array.prototype.forEach.call(o.list.children, function (b) {
        b.setAttribute("aria-pressed", picked[b.getAttribute("data-name")] ? "true" : "false");
      });
      // After every change, not only a redraw: a "list is full" note must
      // appear on the pick that fills it and go on Clear.
      if (o.onRender) o.onRender({ matches: matches, shown: shown.length, picked: n, full: room() <= 0 });
    }

    function pick(name, index, range) {
      if (range && anchor !== null) {
        var lo = Math.min(anchor, index), hi = Math.max(anchor, index);
        for (var i = lo; i <= hi && room() > 0; i++) {
          if (shown[i] && shown[i].pickable) picked[shown[i].name] = true;
        }
      } else if (picked[name]) {
        delete picked[name];
        anchor = index;
      } else {
        if (room() > 0) picked[name] = true;
        anchor = index;
      }
      sync();
    }

    function refresh() {
      var query = o.search ? o.search.value : "";
      var rows = o.rows(query);
      matches = rows.length;
      if (o.shownOnly && o.shownOnly()) {
        var here = {};
        rows.forEach(function (r) { here[r.name] = true; });
        names().forEach(function (n) { if (!here[n]) delete picked[n]; });
      }
      o.list.textContent = "";
      shown = [];
      anchor = null;
      rows.slice(0, maxShown).forEach(function (r, i) {
        shown.push({ name: r.name, pickable: !!r.pickable });
        var b = document.createElement("button");
        b.type = "button";
        b.setAttribute("data-name", r.name);
        b.disabled = !r.pickable;
        var a = document.createElement("span");
        a.textContent = r.name;
        var d = document.createElement("span");
        d.className = "sub";
        d.textContent = r.sub || "";
        b.appendChild(a);
        b.appendChild(d);
        b.addEventListener("click", function (ev) { pick(r.name, i, ev.shiftKey); });
        o.list.appendChild(b);
      });
      sync();
    }

    function open() {
      picked = {};
      anchor = null;
      refresh();
      if (o.search) setTimeout(function () { try { o.search.focus(); } catch (e) { /* hidden */ } }, 50);
    }

    function commit() {
      var chosen = names();
      if (!chosen.length) return;
      picked = {};
      anchor = null;
      if (o.drawer && openDrawer === o.drawer) closeDrawer();
      o.onAdd(chosen);
    }

    if (o.search) {
      o.search.addEventListener("input", refresh);
      o.search.addEventListener("keydown", function (ev) {
        if (ev.key !== "Enter") return;
        ev.preventDefault();
        if (!names().length) {
          var only = shown.filter(function (s) { return s.pickable; });
          if (only.length !== 1 || room() <= 0) return;
          picked[only[0].name] = true;
        }
        commit();
      });
    }
    if (o.all) {
      o.all.addEventListener("click", function () {
        shown.forEach(function (s) { if (s.pickable && !picked[s.name] && room() > 0) picked[s.name] = true; });
        sync();
      });
    }
    if (o.none) o.none.addEventListener("click", function () { picked = {}; anchor = null; sync(); });
    o.add.addEventListener("click", commit);
    if (o.opener) o.opener.addEventListener("click", open);

    return { open: open, refresh: refresh, pick: pick, picked: names };
  }

  /* ---- rung 3: dialog --------------------------------------------- */

  function confirmDialog(el) {
    if (!el) return { open: function () {}, close: function () {} };
    return {
      open: function () {
        if (typeof el.showModal === "function") {
          if (!el.open) el.showModal();
        } else {
          // No <dialog> support: it still renders and is still dismissible,
          // it just does not make the background inert.
          el.setAttribute("open", "");
        }
      },
      close: function () {
        if (typeof el.close === "function") el.close();
        else el.removeAttribute("open");
      }
    };
  }

  /* ---- wiring ------------------------------------------------------ */

  // Declarative openers/closers, so a surface needs no script of its own
  // for the common case:
  //   <button data-drawer-open="save-set">Edit…</button>
  //   <button data-drawer-close>Cancel</button>
  //   <button data-dialog-open="clear-queue">Clear queue…</button>
  //   <button data-dialog-close>Keep them</button>
  function wire() {
    document.addEventListener("click", function (e) {
      var t = e.target && e.target.closest ? e.target.closest("[data-drawer-open],[data-drawer-close],[data-dialog-open],[data-dialog-close]") : null;
      if (!t) return;
      if (t.hasAttribute("data-drawer-open")) {
        showDrawer(need(t.getAttribute("data-drawer-open"), "drawer"), t);
      } else if (t.hasAttribute("data-drawer-close")) {
        closeDrawer();
      } else if (t.hasAttribute("data-dialog-open")) {
        confirmDialog(need(t.getAttribute("data-dialog-open"), "dialog")).open();
      } else if (t.hasAttribute("data-dialog-close")) {
        var host = t.closest("dialog");
        if (host) confirmDialog(host).close();
      }
    });

    // Esc closes the top rung. <dialog> handles its own Esc; this is the
    // drawer's, which the platform does not give us.
    document.addEventListener("keydown", function (e) {
      if (e.key !== "Escape" || !openDrawer) return;
      // A modal <dialog> above the drawer owns Esc and closes itself; taking
      // it here would dismiss the drawer underneath instead.
      if (document.querySelector("dialog[open]")) return;
      e.preventDefault();
      closeDrawer();
    });
  }

  function init() {
    buildDensityPickers();
    wire();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  window.GeecsKit = {
    setDensity: setDensity,
    drawer: drawer,
    confirm: confirmDialog,
    closeDrawer: closeDrawer,
    picker: picker
  };
})();
