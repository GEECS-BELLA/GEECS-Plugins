/* GEECS web theme — the picker, and re-theming after first paint.
 *
 * theme-boot.js has already stamped the palette before this runs; this
 * file adds the control and keeps the page right afterwards. One
 * implementation for every surface, so a viewer's choice follows them
 * across /day, /run, /configs and /log without any of those knowing about
 * the others.
 *
 * It reads the theme list and default from window.GEECS_THEME (set by
 * theme-boot.js) rather than carrying its own — one list, not two.
 *
 * On every change it dispatches `geecs:theme` on window with
 * {theme, mode} so anything that paints outside CSS — the run page's
 * Plotly figure — can re-read the tokens and repaint.
 *
 * Add <div data-theme-picker></div> where the control should appear. A
 * surface that wants the palette but not the control simply omits it.
 */
(function () {
  "use strict";

  var CFG = window.GEECS_THEME;
  if (!CFG) return; // theme-boot.js not loaded: nothing sensible to do

  var LABELS = {
    bella: { label: "BELLA", hint: "BELLA Center — red on black" },
    laser: { label: "Laser", hint: "Laser room — 532 nm pump green" },
    plasma: { label: "Plasma", hint: "Hydrogen plasma — Balmer series" }
  };
  var MODE_LABEL = { system: "Auto", light: "Light", dark: "Dark" };

  var root = document.documentElement;
  var theme = root.getAttribute("data-theme") || CFG.defaultTheme;
  var mode = root.getAttribute("data-mode-pref") || "system";
  var mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function write(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* private window */ }
  }

  function effective() {
    if (mode !== "system") return mode;
    return mq && mq.matches ? "dark" : "light";
  }

  function apply(persist) {
    root.setAttribute("data-theme", theme);
    root.setAttribute("data-mode", effective());
    root.setAttribute("data-mode-pref", mode);

    var buttons = document.querySelectorAll("[data-set-theme]");
    for (var i = 0; i < buttons.length; i++) {
      var on = buttons[i].dataset.setTheme === theme;
      buttons[i].classList.toggle("is-on", on);
      buttons[i].setAttribute("aria-pressed", String(on));
    }
    var mb = document.querySelector("[data-cycle-mode]");
    if (mb) mb.textContent = MODE_LABEL[mode];

    // Persist only a CHOICE. Writing on every load would pin today's
    // default for everyone who ever opened a page, so a later change of
    // the default would never reach them.
    if (persist) {
      write(CFG.keys.theme, theme);
      write(CFG.keys.mode, mode);
    }

    try {
      window.dispatchEvent(new CustomEvent("geecs:theme", {
        detail: { theme: theme, mode: effective(), pref: mode }
      }));
    } catch (e) { /* very old engines: no repaint hook, still themed */ }
  }

  function build(host) {
    host.className = "themepick";
    host.setAttribute("role", "group");
    host.setAttribute("aria-label", "Colour theme");
    var html = "";
    for (var i = 0; i < CFG.themes.length; i++) {
      var id = CFG.themes[i], meta = LABELS[id] || { label: id, hint: id };
      html +=
        '<button type="button" class="th" data-set-theme="' + id +
        '" title="' + meta.hint + '"><span class="sw sw-' + id +
        '"></span><span class="th-label">' + meta.label + "</span></button>";
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
        mode = CFG.modes[(CFG.modes.indexOf(mode) + 1) % CFG.modes.length];
      else return;
      apply(true);
    });
  }

  function init() {
    var hosts = document.querySelectorAll("[data-theme-picker]");
    for (var i = 0; i < hosts.length; i++) build(hosts[i]);
    apply(false);
  }

  // A viewer following the system gets re-stamped when the OS flips.
  if (mq && mq.addEventListener) mq.addEventListener("change", function () {
    if (mode === "system") apply(false);
  });

  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", init);
  else init();
})();
