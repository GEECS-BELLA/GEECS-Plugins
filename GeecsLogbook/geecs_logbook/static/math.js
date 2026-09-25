/* GeecsLogbook math — typeset the equations the server marked.
 *
 * render.py parses `$…$` and `$$…$$` and emits <span class="math-inline">
 * and <div class="math-display"> holding the TeX as plain text. This file
 * turns each mark into KaTeX output — and nothing else: no markdown is
 * parsed here, so there is no second renderer to drift from the server's.
 *
 * KaTeX is vendored under static/vendor/katex-<version>/ and loaded by
 * the page before this script (both deferred, so the order holds): no CDN,
 * because control-room machines may lack internet. Without it the marks
 * stay as TeX source, which is also what the markdown mirror reads as.
 *
 * Failure is honest. TeX that KaTeX rejects keeps its source on screen,
 * marked .math-error with the reason as its title; scanlog.css colours
 * that through the page's tokens — this file sets no colours.
 *
 * `trust` stays off (no \href, \url, \includegraphics from a note) and
 * `maxSize` bounds a rule or a delimiter, so a stray \rule{999em}{999em}
 * cannot paint the page.
 */
(function () {
  "use strict";

  const OPTIONS = { throwOnError: true, strict: "ignore", trust: false, maxSize: 50, maxExpand: 1000 };

  /** Typeset every unhandled mark under `root` (the document by default). */
  function typeset(root) {
    if (typeof katex === "undefined") return;
    (root || document).querySelectorAll(".math-inline, .math-display").forEach((el) => {
      if (el.dataset.tex !== undefined) return; // done on an earlier pass
      const tex = el.textContent;
      el.dataset.tex = tex;
      try {
        katex.render(tex, el, Object.assign({ displayMode: el.classList.contains("math-display") }, OPTIONS));
      } catch (err) {
        el.textContent = tex;
        el.classList.add("math-error");
        el.title = (err && err.message) || "Could not typeset this equation";
      }
    });
  }

  // editor.js calls this on a freshly rendered preview pane.
  window.logbookTypeset = typeset;

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => typeset());
  else typeset();
})();
