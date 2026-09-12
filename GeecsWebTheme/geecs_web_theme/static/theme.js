/* GEECS web theme — applies the stored palette and injects the picker.
 *
 * One implementation for every surface: the portal, the config editor and
 * the scan logbook all load this file, so the choice follows you across
 * /day, /run, /configs and /log without any of them knowing about the
 * others.
 *
 * The choice is a per-viewer convenience, so localStorage is the right
 * home — it never needs to reach another viewer or the server. It is also
 * read before first paint (see the inline bootstrap each surface includes)
 * so the page never flashes the wrong palette.
 *
 * Add <div data-theme-picker></div> where the control should appear. A
 * surface that wants the palette but not the control simply omits it.
 */
(function () {
  "use strict";

  var THEMES = [
    { id: "bella", label: "BELLA", hint: "BELLA Center — red on black" },
    { id: "laser", label: "Laser", hint: "Laser room — 532 nm pump green" },
    { id: "plasma", label: "Plasma", hint: "Hydrogen plasma — Balmer series" }
  ];
  var MODES = ["system", "light", "dark"];
  var MODE_LABEL = { system: "Auto", light: "Light", dark: "Dark" };
  var DEFAULT_THEME = "laser";

  function read(key, fallback) {
    try {
      return window.localStorage.getItem(key) || fallback;
    } catch (e) {
      // Private windows and blocked site data both throw here. A theme is
      // a convenience; losing it must never break the page.
      return fallback;
    }
  }

  function write(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (e) {
      /* see read() */
    }
  }

  var theme = read("geecs.theme", DEFAULT_THEME);
  var mode = read("geecs.mode", "system");
  if (!THEMES.some(function (t) { return t.id === theme; })) theme = DEFAULT_THEME;
  if (MODES.indexOf(mode) === -1) mode = "system";

  function apply() {
    var root = document.documentElement;
    root.setAttribute("data-theme", theme);
    if (mode === "system") root.removeAttribute("data-mode");
    else root.setAttribute("data-mode", mode);

    var buttons = document.querySelectorAll("[data-set-theme]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].classList.toggle("is-on", buttons[i].dataset.setTheme === theme);
      buttons[i].setAttribute(
        "aria-pressed", String(buttons[i].dataset.setTheme === theme)
      );
    }
    var mb = document.querySelector("[data-cycle-mode]");
    if (mb) mb.textContent = MODE_LABEL[mode];

    write("geecs.theme", theme);
    write("geecs.mode", mode);
  }

  function build(host) {
    host.className = "themepick";
    host.setAttribute("role", "group");
    host.setAttribute("aria-label", "Colour theme");
    var html = "";
    for (var i = 0; i < THEMES.length; i++) {
      var t = THEMES[i];
      html +=
        '<button type="button" class="th" data-set-theme="' + t.id +
        '" title="' + t.hint + '"><span class="sw sw-' + t.id +
        '"></span><span class="th-label">' + t.label + "</span></button>";
    }
    html +=
      '<span class="th-sep"></span>' +
      '<button type="button" class="th" data-cycle-mode title="Light, dark, ' +
      'or follow the system">Auto</button>';
    host.innerHTML = html;

    host.addEventListener("click", function (ev) {
      var b = ev.target.closest("button");
      if (!b) return;
      if (b.dataset.setTheme) theme = b.dataset.setTheme;
      else if (b.hasAttribute("data-cycle-mode"))
        mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
      else return;
      apply();
    });
  }

  function init() {
    var hosts = document.querySelectorAll("[data-theme-picker]");
    for (var i = 0; i < hosts.length; i++) build(hosts[i]);
    apply();
  }

  apply(); // before paint where possible
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
