// The settables page's trends, DOM only: one Plotly plot per trend the family's panel names (_panel.html's
// `trends`), one line per role against wall-clock time, drawn through the theme's template
// (window.GeecsPlot). The past comes from this server's /api/devices/<name>/trends (Vole's buffer, asked
// once per window choice); each line is then extended from the frames the page already receives (device.js's
// GeecsDevicePage: Hare's one socket, no second stream). The window is this viewer's own, written nowhere.
// Vole silent or empty, the plots start at page load and one line says so. Nothing here may break the rest
// of the page: every entry point catches. The rules are trend_rules.js's.
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsTrendRules, D = window.GeecsDeviceRules, G = window.GeecsPlot;
  const boxes = [...document.querySelectorAll(".trends")];
  if (!P || !R || !D || !boxes.length) return;
  const DRAW_MS = 1000, TIMEOUT_MS = 10000;
  let skew = 0; // the server's clock less this browser's (the history's answer): "now" is the server's
  const now = () => Date.now() + skew;
  const safely = (work) => (...args) => { try { return work(...args); } catch (e) { console.error("trend", e); return undefined; } };

  function start(box) {
    const axis = box.dataset.axis || undefined, note = box.querySelector(".trend-note");
    const plots = [...box.querySelectorAll(".trend")].map((el) => ({ host: el.querySelector(".trend-plot"), roles: JSON.parse(el.dataset.trend), watched: false }));
    const wanted = [...new Map(plots.flatMap((p) => p.roles).map((r) => [r.role, r])).values()];
    const buttons = [...box.querySelectorAll("[data-window]")], choices = buttons.map((b) => Number(b.dataset.window));
    const pressed = buttons.find((b) => b.getAttribute("aria-pressed") === "true");
    let seconds = R.windowOf(pressed && pressed.dataset.window, choices, choices[0]), got = null, asked = 0;
    const live = {}, history = {}, seen = {};
    for (const r of wanted) { live[r.role] = []; history[r.role] = []; }

    // each new frame of a trend role, once: Hare hands a new object per frame, and null when the socket drops
    const take = safely(() => {
      const t = now();
      for (const r of wanted) {
        const f = P.frame(r.role, axis);
        if (f === seen[r.role]) continue;
        seen[r.role] = f;
        R.append(live[r.role], R.point(f, t), t);
      }
    });

    const draw = safely(() => {
      if (document.hidden || !window.Plotly || !G) return;
      const to = now(), from = to - seconds * 1000, offset = new Date(to).getTimezoneOffset();
      const stamp = (ms) => R.stampText(ms, offset);
      for (const p of plots) {
        const columns = Math.round(p.host.clientWidth);
        if (!columns) continue; // a hidden axis pane: drawn once it is shown
        const lines = p.roles.map((r) => ({ label: r.label, points: R.merge(history[r.role], live[r.role]), hold: R.holding(live[r.role]) }));
        const units = R.sharedUnit(p.roles.map((r) => D.units(r.type, r.units, P.meta(r.role, axis))));
        const data = R.traces(lines, from, to, columns).map((tr) => ({ ...tr, x: tr.x.map(stamp) }));
        // x is the window, always ending now (fixed: the choice above is the zoom); y is the viewer's to zoom, kept across redraws
        window.Plotly.react(p.host, data, G.layout({
          uirevision: "trend", showlegend: true,
          xaxis: { type: "date", range: [stamp(from), stamp(to)], fixedrange: true },
          yaxis: { title: { text: units }, ...G.axisFormat(units) },
        }), G.config());
        if (!p.watched) { p.watched = true; G.watch(p.host); }
      }
    });

    async function read() {
      const mine = ++asked;
      let next;
      try {
        next = await R.within(TIMEOUT_MS, async (signal) => {
          const r = await fetch(R.trendsUrl(box.dataset.trendsUrl, seconds, axis), { cache: "no-store", signal });
          return R.answer(r.ok, await r.json().catch(() => null), r.status);
        });
      } catch (e) { next = { ok: false, why: e && e.message === "timed out" ? "the Control Room did not answer in time" : "the Control Room did not answer" }; }
      if (mine !== asked) return; // the viewer chose another window meanwhile: that ask's answer is the one kept
      got = next;
      if (got.ok) {
        if (got.to !== null) skew = got.to - Date.now();
        for (const r of wanted) history[r.role] = got.roles[r.role] || [];
      }
      say();
      draw();
    }
    const say = safely(() => { note.textContent = !window.Plotly || !G ? "the plot library did not load" : R.historyNote(got, wanted); });

    for (const b of buttons) {
      b.addEventListener("click", () => {
        seconds = R.windowOf(b.dataset.window, choices, seconds);
        for (const o of buttons) o.setAttribute("aria-pressed", String(o === b));
        draw();
        read().catch(() => {});
      });
    }
    P.on(take);
    take();
    say();
    read().catch(() => {});
    setInterval(draw, DRAW_MS);
  }
  boxes.forEach(safely(start));
})();
