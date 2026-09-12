/* GEECS surface kit — the three behaviours kit.css cannot express.
 *
 * Load deferred, beside theme.js. Everything here is additive: with this
 * file absent the page still renders correctly, <details> still opens, and
 * the density stamped by theme-boot.js still applies — only the drawer,
 * the dialog helper and the density control go missing.
 *
 *   <div data-density-picker></div>   the spacing control, built here
 *   GeecsKit.drawer(el)               rung 2 — open/close with Esc, scrim,
 *                                     focus return
 *   GeecsKit.confirm(el)              rung 3 — <dialog>.showModal() with a
 *                                     fallback for browsers without it
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
      if (host.firstChild) return; // already built
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
    var first = el.querySelector("[autofocus],[data-drawer-close],button,a[href],input,select,textarea");
    if (first) {
      try { first.focus(); } catch (e) { /* nothing focusable: leave it */ }
    }
  }

  function drawer(el) {
    return {
      open: function (opener) { showDrawer(el, opener); },
      close: closeDrawer,
      get isOpen() { return openDrawer === el; }
    };
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
        showDrawer(document.getElementById(t.getAttribute("data-drawer-open")), t);
      } else if (t.hasAttribute("data-drawer-close")) {
        closeDrawer();
      } else if (t.hasAttribute("data-dialog-open")) {
        confirmDialog(document.getElementById(t.getAttribute("data-dialog-open"))).open();
      } else if (t.hasAttribute("data-dialog-close")) {
        var host = t.closest("dialog");
        if (host) confirmDialog(host).close();
      }
    });

    // Esc closes the top rung. <dialog> handles its own Esc; this is the
    // drawer's, which the platform does not give us.
    document.addEventListener("keydown", function (e) {
      if (e.key !== "Escape" || !openDrawer) return;
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
    closeDrawer: closeDrawer
  };
})();
