// Why a copy downloaded instead, said where it stays until read. A browser
// gives a page the clipboard for pictures only on a secure page (https, or
// localhost / 127.0.0.1); a portal reached as http://<host>:<port> has
// none, so Copy downloads the PNG and this dialog says why and what to do:
// open the portal's https address if it has one, or tell the browser to
// trust this address (Chrome and Edge: the "insecure origins treated as
// secure" flag). On a secure page the browser itself lacks picture copy
// (ClipboardItem), and the dialog says that instead.
// One dialog for every copy on the page (the plot's modebar, the strip's
// Copy grid), built on first use. It is the kit's rung-3 <dialog> used as
// an acknowledgement, one OK: a toast was missed, and the advice must stay
// until read.
"use strict";

window.ClipboardHelp = (function () {
  let dialog = null;

  // The flags page to paste, for the browsers that have the switch.
  function flagsPage(agent) {
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

  // What the dialog says: the reason, then what to do about it.
  function advice() {
    if (window.isSecureContext) {
      return [node("p", {}, "This browser cannot put pictures on the clipboard, so the PNG was downloaded instead. "
        + "A current Chrome, Edge, Safari or Firefox can.")];
    }
    const origin = location.origin;
    const flags = flagsPage(navigator.userAgent || "");
    const said = [
      node("p", {}, "Browsers only let an https page, or one on localhost, put a picture on the clipboard. "
        + "This page is ", pasteable(origin), ", so the PNG was downloaded instead."),
      node("p", {}, "If this portal has an https address, open it there and copy works."),
    ];
    if (flags) {
      said.push(node("p", {}, "Or let this browser trust this address:"), node("ol", {},
        node("li", {}, "Paste ", pasteable(flags), " into a new tab's address bar."),
        node("li", {}, "Add ", pasteable(origin), ", set it to Enabled, and Relaunch.")));
    }
    return said;
  }

  function build() {
    const ok = node("button", { class: "btn primary", type: "button", "data-dialog-close": "", autofocus: "" }, "OK");
    const body = node("div", { class: "body" },
      node("h4", { id: "clip-help-title" }, "Downloaded instead of copied"),
      ...advice(),
      node("div", { class: "row end" }, ok));
    const el = node("dialog", { id: "clip-help", "aria-labelledby": "clip-help-title" }, body);
    // Without kit.js, OK still closes it.
    ok.addEventListener("click", () => { if (el.open) el.close(); });
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
