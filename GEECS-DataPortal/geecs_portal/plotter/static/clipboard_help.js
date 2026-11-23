// Why a copy downloaded instead, said where it stays until read. A browser
// gives a page the clipboard for pictures only when the page is https or
// on localhost / 127.0.0.1; a portal reached as http://<host>:<port> has
// none, so Copy downloads the PNG and this dialog says why and what to do:
// open the portal on its own machine, or tell the browser to trust this
// address (Chrome and Edge: the "insecure origins treated as secure" flag).
// One dialog for every copy on the page (the plot's modebar, the strip's
// Copy grid), built on first use; the kit's rung-3 <dialog>.
"use strict";

window.ClipboardHelp = (function () {
  let dialog = null;

  // The flags page to paste, for the browsers that have the switch.
  function flagsPage() {
    const agent = navigator.userAgent || "";
    if (/\bEdg\//.test(agent)) return "edge://flags/#unsafely-treat-insecure-origin-as-secure";
    if (/\bChrome\//.test(agent)) return "chrome://flags/#unsafely-treat-insecure-origin-as-secure";
    return "";
  }

  function node(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) el.setAttribute(key, value);
    for (const child of children) el.append(child);
    return el;
  }
  // Text a viewer selects with one click to copy by hand (the clipboard
  // being exactly what this page lacks).
  const pasteable = (text) => node("code", { class: "clip-paste" }, text);

  function build() {
    const origin = location.origin;
    const flags = flagsPage();
    const steps = flags
      ? node("div", {},
        node("p", {}, "To copy from this address, tell the browser to trust it:"),
        node("ol", {},
        node("li", {}, "Paste ", pasteable(flags), " into the address bar of a new tab (the browser will not open it from a link)."),
        node("li", {}, "Add ", pasteable(origin), " to the box and set the flag to Enabled."),
        node("li", {}, "Click Relaunch. Copy works on this address from then on.")))
      :node("p", {}, "This browser has no simple switch for it; open the portal over https, or on the machine it runs on.");
    const body = node("div", { class: "body" },
      node("h4", { id: "clip-help-title" }, "Downloaded instead of copied"),
      node("p", {}, "Browsers only let a page put a picture on the clipboard when the page is https, or on localhost or 127.0.0.1. "
        + "This page is ", pasteable(origin), ", so the PNG was downloaded instead."),
      steps,
      node("p", {}, "Or open the portal on the machine it runs on, at http://127.0.0.1 and its port, where copy always works."),
      node("div", { class: "row end" },
        node("button", { class: "btn primary", type: "button", "data-dialog-close": "", autofocus: "" }, "OK")));
    const el = node("dialog", { id: "clip-help", "aria-labelledby": "clip-help-title" }, body);
    // Without kit.js, OK still closes it.
    el.querySelector("[data-dialog-close]").addEventListener("click", () => { if (el.open) el.close(); });
    document.body.append(el);
    return el;
  }

  return {
    /** Say why the copy downloaded, until the viewer closes it. */
    show() {
      dialog = dialog || build();
      if (window.GeecsKit) window.GeecsKit.confirm(dialog).open();
      else if (typeof dialog.showModal === "function") { if (!dialog.open) dialog.showModal(); }
      else dialog.setAttribute("open", "");
    },
  };
})();
