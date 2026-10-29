// The 2-D map (the portal's old Grid tab): one scalar's average and error
// in each cell of two scan variables. The numbers and both figures come
// from /plot/api/map; this script draws them square, steps through the
// cells and saves the statistics. A statistic is a field of the mapcfg
// JSON (the page form's hidden field): a change writes it and submits the
// form, so it saves into the viewer's own set like every other control.
// The selected cell is the browser tab's own (sessionStorage, per scan), so
// it survives the reload a saved statistic makes; it is never the set's.
"use strict";

(function () {
  const host = document.getElementById("map");
  const form = document.getElementById("pick");
  // No URL: nothing to map yet (the page asks for a column).
  if (!host || !form || !host.dataset.url) return;
  const $ = (id) => document.getElementById(id);
  const message = $("map-message");
  const tools = window.PlotTools || { config: () => ({ responsive: true }), counted: () => {}, themed: (v) => v };
  const PLOT_CONFIG = tools.config();
  let data = null;

  // ---- the statistics: each control writes its key and saves ----
  const DEFAULTS = { lower: 0.25, upper: 0.75, min_count: 3 };
  const stats = () => { try { return JSON.parse(form.elements.mapcfg.value || "{}"); } catch (_) { return {}; } };
  function save(update) {
    const cfg = Object.assign(stats(), update);
    if (update.error === "iqr") Object.assign(cfg, { lower: 0.25, upper: 0.75 });
    form.elements.mapcfg.value = JSON.stringify(cfg);
    form.requestSubmit();
  }
  for (const control of document.querySelectorAll("[data-map-stat]")) {
    const key = control.dataset.mapStat;
    control.addEventListener("change", () => {
      if (control.type !== "number") {
        save({ [key]: key === "visit" ? Number(control.value) : control.value });
        return;
      }
      // An empty or bad number is not applied: the box goes back to the
      // value the map was drawn with, and the toast says what is allowed.
      const cfg = stats();
      const factor = "percent" in control.dataset ? 100 : 1;
      const restore = () => { control.value = (cfg[key] ?? DEFAULTS[key]) * factor; };
      if (!control.value.trim()) { restore(); return; }
      const value = Number(control.value) / factor;
      const lower = key === "lower" ? value : (cfg.lower ?? DEFAULTS.lower);
      const upper = key === "upper" ? value : (cfg.upper ?? DEFAULTS.upper);
      const valid = Number.isFinite(value) && (key === "min_count"
        ? Number.isInteger(value) && value >= 1
        : value >= 0 && value <= 1 && lower < upper);
      if (!valid) {
        restore();
        if (window.GeecsKit) window.GeecsKit.toast(key === "min_count"
          ? "Minimum shots must be a whole number of at least 1."
          : "Percentiles must be between 0 and 100, with lower below upper.");
        return;
      }
      save({ [key]: value });
    });
  }

  // ---- the selected cell ----
  const CELL_KEY = `geecs-plot:map-cell:${host.dataset.day}:${host.dataset.scan}`;
  const keptCell = () => { try { return sessionStorage.getItem(CELL_KEY) || ""; } catch (_) { return ""; } };
  const keepCell = (bin) => { try { sessionStorage.setItem(CELL_KEY, bin); } catch (_) { /* storage off */ } };
  const visitCells = () => data.cells.filter((c) => c.visit === data.config.visit);
  const fmt = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString(undefined, { maximumSignificantDigits: 5 }));
  function imagesUrl(cell) {
    // The scan page's Images tab, averaged over this cell's acquisition bin.
    const url = new URL(host.dataset.imagesUrl, location.origin);
    url.searchParams.set("view", "bin");
    url.searchParams.set("bincfg", JSON.stringify({ bin_col: data.bin_column, min_count: 1 }));
    url.searchParams.set("imagebin", String(cell.bin));
    const filters = form.elements.filters.value;
    if (filters) url.searchParams.set("filters", filters);
    return url.pathname + url.search;
  }
  function select(bin) {
    if (!data) return;
    const c = visitCells().find((cell) => cell.bin === Number(bin));
    if (!c) return;
    keepCell(String(c.bin));
    $("map-cell").value = String(c.bin);
    const cfg = data.config;
    const rows = [
      ["Coordinates", `${fmt(c.x)} / ${fmt(c.y)}`],
      ["Readbacks", `${fmt(c.x_readback)} / ${fmt(c.y_readback)}`],
      [cfg.average === "mean" ? "Mean" : "Median", fmt(c.center)],
      [data.error_label, fmt(c.error)],
      ["Lower / upper error", c.count >= cfg.min_count ? `${fmt(c.err_low)} / ${fmt(c.err_high)}` : "insufficient samples"],
      ["Valid / passing / acquired", `${c.count} / ${c.passing} / ${c.acquired}`],
    ];
    $("map-values").replaceChildren(...rows.map(([key, value]) => {
      const box = document.createElement("span");
      const label = document.createElement("span");
      label.className = "hint";
      label.textContent = key;
      box.append(label, document.createElement("br"), value);
      return box;
    }));
    $("map-members").textContent = c.shots.length ? c.shots.join(", ") : "No member shots pass the filters.";
    const link = $("map-images");
    link.href = imagesUrl(c);
    link.toggleAttribute("aria-disabled", !c.shots.length);
    // The marker trace (the figures' last) moves to the cell.
    for (const name of ["center", "uncertainty"]) {
      const plot = $(`map-${name}`);
      if (!plot.data) continue;
      Plotly.restyle(plot, { x: [[c.x]], y: [[c.y]] }, [plot.data.length - 1]);
    }
  }
  function step(delta) {
    const cells = visitCells();
    const index = cells.findIndex((c) => c.bin === Number($("map-cell").value));
    const next = cells[Math.max(0, Math.min(cells.length - 1, index + delta))];
    if (next) select(next.bin);
  }
  $("map-cell").addEventListener("change", (event) => select(event.target.value));
  for (const button of document.querySelectorAll("[data-map-step]"))
    button.addEventListener("click", () => step(Number(button.dataset.mapStep)));

  // ---- drawing: square plotting areas ----
  async function draw() {
    if (!data) return;
    const shown = data;
    const captions = {
      center: `${$("map-center-title").textContent} vs ${data.pretty[data.config.x] || data.config.x} / ${data.pretty[data.config.y] || data.config.y} · visit ${data.config.visit}`,
      uncertainty: `${data.pretty[data.config.value] || data.config.value} · ${data.error_label} vs ${data.pretty[data.config.x] || data.config.x} / ${data.pretty[data.config.y] || data.config.y} · visit ${data.config.visit}`,
    };
    await Promise.all([["center", "center"], ["error", "uncertainty"]].map(async ([figure, name]) => {
      const plot = $(`map-${name}`);
      const fig = tools.themed(JSON.parse(JSON.stringify(data.figures[figure])));
      // Equal pixel lengths of the plotting area, whatever the axis units.
      const width = plot.clientWidth;
      const m = fig.layout.margin;
      fig.layout.width = width;
      fig.layout.height = Math.max(200, width - m.l - m.r + m.t + m.b);
      plot.dataset.caption = captions[name];
      plot.dataset.square = "true";  // the send keeps the plotting area square
      await Plotly.react(plot, fig.data, fig.layout, PLOT_CONFIG);
      plot.removeAllListeners("plotly_click");
      plot.on("plotly_click", (event) => {
        const cd = event.points[0] && event.points[0].customdata;
        if (cd) select(cd[0]);
      });
    }));
    if (shown === data) select($("map-cell").value);
  }

  fetch(host.dataset.url)
    .then((response) => response.json().then((body) => ({ response, body })))
    .then(async ({ response, body }) => {
      if (!response.ok) { message.textContent = String(body.detail || response.statusText); return; }
      data = body;
      const cfg = data.config;
      $("map-visit").replaceChildren(...Array.from({ length: data.visits }, (_, i) => new Option(String(i + 1), String(i + 1))));
      $("map-visit").value = String(cfg.visit);
      const cells = visitCells();
      $("map-cell").replaceChildren(...cells.map((c) => new Option(`Bin ${c.bin} · ${c.x} / ${c.y}`, String(c.bin))));
      const kept = keptCell();
      const wanted = cells.some((c) => String(c.bin) === kept) ? kept : (cells.length ? String(cells[0].bin) : "");
      $("map-cell").value = wanted;
      message.textContent = cells.length
        ? (data.kind === "grid" ? "One cell = one scan position" : "Measured positions · no interpolation")
        : "No positions in this visit.";
      $("map-results").hidden = !cells.length;
      $("map-center-title").textContent = `${cfg.average === "mean" ? "Mean" : "Median"} · ${data.pretty[cfg.value] || cfg.value}`;
      $("map-error-title").textContent = data.error_label;
      $("map-notes").textContent = `${data.notes.join(" ")} Square plotting areas; independent colour scales.`;
      $("code").textContent = data.code;
      tools.counted(data);
      if (cells.length) await draw();
    })
    .catch((error) => { message.textContent = String(error); });

  let width = 0;
  new ResizeObserver((entries) => {
    const now = entries[0].contentRect.width;
    if (now > 0 && Math.abs(now - width) > 1) { width = now; draw(); }
  }).observe(host);
  window.addEventListener("geecs:theme", () => draw());
})();
