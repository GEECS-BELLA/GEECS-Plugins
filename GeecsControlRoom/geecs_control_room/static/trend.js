// The settables page's trends, DOM only: one Plotly plot per trend the family's panel names (_panel.html's
// `trends`), one line per role against wall-clock time, drawn through the theme's template
// (window.GeecsPlot). The past comes from this server's /api/devices/<name>/trends (Vole's buffer, asked when
// the plots are first shown, on each window choice, and again when the live buffer has dropped points the
// last answer does not cover); each line is then extended from the frames the page already receives
// (device.js's GeecsDevicePage: Hare's one socket, no second stream). The window is this viewer's own,
// written nowhere; a window of 30 s or less is redrawn five times a second, a longer one each second. Vole silent or empty, the plots start at page load and one line says so. Nothing here may
// break the rest of the page: every entry point (a frame, a redraw, a click, the read) catches. The rules are
// trend_rules.js's.
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsTrendRules, D = window.GeecsDeviceRules, G = window.GeecsPlot;
  const boxes = [...document.querySelectorAll(".trends")];
  if (!P || !R || !D || !boxes.length) return;
  const DRAW_MS = 1000, FAST_MS = 200, FAST_S = 30, TIMEOUT_MS = 10000, REREAD_MS = 30000; // a dropped stretch is read again at most this often
  let skew = 0; // the server's clock when it answered less this browser's on receiving it: "now" is the server's
  const now = () => Date.now() + skew;
  const safely = (work) => (...args) => { try { return work(...args); } catch (e) { console.error("trend", e); return undefined; } };

  function start(box) {
    const axis = box.dataset.axis || undefined, note = box.querySelector(".trend-note");
    const plots = [...box.querySelectorAll(".trend")].map((el) => ({ host: el.querySelector(".trend-plot"), roles: JSON.parse(el.dataset.trend), watched: false }));
    const wanted = [...new Map(plots.flatMap((p) => p.roles).map((r) => [r.role, r])).values()];
    const buttons = [...box.querySelectorAll("[data-window]")], choices = buttons.map((b) => Number(b.dataset.window));
    const pressed = buttons.find((b) => b.getAttribute("aria-pressed") === "true");
    let seconds = R.windowOf(pressed && pressed.dataset.window, choices, choices[0]), got = null, asked = 0, reading = false, readAt = null;
    const live = {}, history = {}, seen = {}, lost = {}, keepMs = Math.max(...choices) * 1000;
    let held = []; // the roles drawn with an earlier read's past, the last read of them having failed
    for (const r of wanted) { live[r.role] = []; history[r.role] = []; lost[r.role] = null; }
    const shown = () => plots.some((p) => p.host.clientWidth > 0); // false for a hidden axis pane

    // each new frame of a trend role, once: Hare hands a new object per frame, and null when the socket drops
    const take = safely(() => {
      const t = now();
      for (const r of wanted) {
        const f = P.frame(r.role, axis);
        if (f === seen[r.role]) continue;
        seen[r.role] = f;
        R.append(live[r.role], R.point(f, t), t);
        const gone = R.trim(live[r.role], t, keepMs);
        if (gone !== null) lost[r.role] = gone;
      }
    });

    const draw = safely(() => {
      if (document.hidden || !window.Plotly || !G || !shown()) return;
      // the first time the plots are shown (an axis pane is hidden until picked), and when a line's live buffer
      // has dropped points the last answer does not cover: the history is read (again)
      const holed = wanted.some((r) => R.hole(history[r.role], lost[r.role]));
      if (!reading && (readAt === null || (holed && Date.now() - readAt >= REREAD_MS))) read().catch(() => {});
      const to = now(), from = to - seconds * 1000, offset = new Date(to).getTimezoneOffset();
      const stamp = (ms) => R.stampText(ms, offset);
      for (const p of plots) {
        const columns = Math.round(p.host.clientWidth);
        if (!columns) continue; // a hidden axis pane: drawn once it is shown
        const lines = p.roles.map((r) => ({ label: r.label, points: R.merge(history[r.role], live[r.role], lost[r.role]), hold: R.holding(live[r.role]) }));
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
      reading = true;
      readAt = Date.now();
      let next;
      try {
        next = await R.within(TIMEOUT_MS, async (signal) => {
          const r = await fetch(R.trendsUrl(box.dataset.trendsUrl, seconds, axis), { cache: "no-store", signal });
          return R.answer(r.ok, await r.json().catch(() => null), r.status);
        });
      } catch (e) { next = { ok: false, why: e && e.message === "timed out" ? "the Control Room did not answer in time" : "the Control Room did not answer" }; }
      if (mine !== asked) return; // the viewer chose another window meanwhile: that ask's answer is the one kept
      reading = false;
      got = next;
      if (got.ok) {
        if (got.now !== null) skew = got.now - Date.now();
        for (const r of wanted) history[r.role] = R.kept(got, r.role, history[r.role]);
        held = wanted.filter((r) => got.state[r.role] === "failed" && history[r.role].length).map((r) => r.role);
      } else held = wanted.filter((r) => history[r.role].length).map((r) => r.role); // nothing read: what is drawn is the earlier read's
      say();
      draw();
    }
    const say = safely(() => { note.textContent = !window.Plotly || !G ? "the plot library did not load" : R.historyNote(got, wanted, held); });

    for (const b of buttons) {
      b.addEventListener("click", safely(() => {
        seconds = R.windowOf(b.dataset.window, choices, seconds);
        for (const o of buttons) o.setAttribute("aria-pressed", String(o === b));
        got = null; // the note reads "reading the history" until this window's answer; the lines keep the past they have
        say();
        if (shown()) read().catch(() => {}); // a hidden pane reads when it is first shown
        else readAt = null;
        draw();
      }));
    }
    P.on(take);
    take();
    say();
    draw(); // reads the history if the plots are shown
    // one timer: a window of FAST_S or less is redrawn on every tick (fast enough to watch a loop work), a longer one each second
    let drawnAt = 0;
    setInterval(() => { const t = Date.now(); if (seconds <= FAST_S || t - drawnAt >= DRAW_MS) { drawnAt = t; draw(); } }, FAST_MS);
  }
  boxes.forEach(safely(start));
})();
