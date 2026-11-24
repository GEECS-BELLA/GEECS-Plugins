/* GEECS web theme — boot. Load this in <head>, NOT deferred.
 *
 * Stamps the palette on the root element before the first paint, so a page
 * never renders in the default theme for a frame and then jumps. It is the
 * ONE definition of which themes exist and what the default is: theme.js
 * reads window.GEECS_THEME rather than carrying its own copy, and
 * geecs_web_theme/__init__.py is pinned to this file by a test.
 *
 * It resolves "follow the system" here, via matchMedia, and stamps the
 * EFFECTIVE mode. That is why theme.css needs exactly one dark block per
 * theme and no @media copy of it — the copy is what drifted before.
 *
 * It also stamps the spacing density the kit (kit.css) keys off, for the
 * same pre-paint reason.
 *
 * Everything is guarded: a private window, blocked storage, or a stale or
 * foreign value under our localStorage key (every page on this origin
 * shares it) must fall back to the default, never leave the page unstamped.
 *
 * It also defines window.GeecsOperator, the operator's name remembered in
 * this browser under ONE key every surface on the origin shares (behind the
 * front door, the scanner, the portal, the logbook, the Control Room and the
 * configurator are one origin): get() → the name or "", set(name). Here
 * rather than in kit.js because this script runs before every page script,
 * deferred or inline. A surface reads and writes the name only through it;
 * what a write sends, and when a name is required, stay each surface's own.
 */
(function () {
  "use strict";
  var CFG = {
    themes: ["bella", "laser", "plasma"],
    defaultTheme: "laser",
    modes: ["system", "light", "dark"],
    densities: ["comfortable", "compact"],
    defaultDensity: "comfortable",
    keys: {
      theme: "geecs.theme",
      mode: "geecs.mode",
      density: "geecs.density"
    }
  };
  window.GEECS_THEME = CFG;

  function read(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }

  var theme = read(CFG.keys.theme);
  if (CFG.themes.indexOf(theme) === -1) theme = CFG.defaultTheme;
  var mode = read(CFG.keys.mode);
  if (CFG.modes.indexOf(mode) === -1) mode = "system";

  var effective = mode;
  if (mode === "system") {
    var mq = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
    effective = mq && mq.matches ? "dark" : "light";
  }

  var density = read(CFG.keys.density);
  if (CFG.densities.indexOf(density) === -1) density = CFG.defaultDensity;

  var root = document.documentElement;
  root.setAttribute("data-theme", theme);
  // Spacing scale. Stamped here for the same reason as the palette: the
  // kit's padding and row heights key off it, so choosing it after first
  // paint would reflow every panel on the page.
  root.setAttribute("data-density", density);
  root.setAttribute("data-mode", effective);
  // What the viewer CHOSE, distinct from what is showing; theme.js reads it.
  root.setAttribute("data-mode-pref", mode);

  // The operator's name. OLD_OPERATOR_KEYS are the per-surface keys it
  // replaced (the Control Room's, the configurator's, the portal's, the
  // logbook's; the scanner's was already this key): while the shared key has
  // never been written, the first old one holding a name is copied to it, so
  // no remembered name is lost on the deploy. A name cleared on purpose is
  // the empty string, not missing, and is never refilled from an old key.
  // Delete the list once every operator's browser has opened a page since.
  var OPERATOR_KEY = "geecs.operator";
  var OLD_OPERATOR_KEYS = [
    "geecs-control-room-operator",
    "geecs.configurator.operator",
    "geecs-portal:author",
    "geecs.author"
  ];
  function write(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* private window: not remembered */ }
  }
  window.GeecsOperator = {
    get: function () {
      var name = read(OPERATOR_KEY);
      if (name !== null) return name;
      for (var i = 0; i < OLD_OPERATOR_KEYS.length; i++) {
        var old = read(OLD_OPERATOR_KEYS[i]);
        if (old && old.trim()) { write(OPERATOR_KEY, old.trim()); return old.trim(); }
      }
      return "";
    },
    set: function (name) { write(OPERATOR_KEY, String(name == null ? "" : name).trim()); }
  };
})();
