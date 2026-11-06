// The Images page's big view: the tile clicked in a camera's strip at full
// size, stepped with ‹ › along that camera's row (a click on another
// row's tile moves the big view to that camera). It is the tile's own picture (its ROI, levels and
// colour map, following the tile as they change); with a recipe picked, a
// shot is the portal's processing preview instead (image.png, the whole
// frame on its own levels). A trace device has no strip and draws one
// shot's trace (the portal's trace route). The selection is the browser tab's
// (sessionStorage, per scan, device and strip), never the set's; a 2-D map
// cell opens the page on its bin (?bin=).
"use strict";

(function () {
  const view = document.getElementById("bigview");
  if (!view) return;
  const $ = (id) => document.getElementById(id);
  const d = view.dataset;
  const mode = d.mode;  // "bins", "shots", "pick" (one shot per bin) or "trace"
  const single = mode === "shots" || mode === "pick";  // each tile one shot
  const tools = window.PlotTools || { config: () => ({ responsive: true }), themed: (v) => v };
  const strips = Array.from(document.querySelectorAll(".strip[data-cam]"));
  const ROW_KEY = `geecs-plot:big-row:${d.day}:${d.scan}:${mode}`;
  // The camera row shown: the one last clicked in this tab, else the first.
  let strip = strips.find((s) => s.dataset.cam === kept0(ROW_KEY)) || strips[0] || null;
  let tiles = strip ? Array.from(strip.querySelectorAll(".tile")) : [];
  const device = () => (strip ? strip.dataset.device : d.device);
  const total = () => (mode === "trace" ? Number(d.shots) || 0 : tiles.length);
  function kept0(key) { try { return sessionStorage.getItem(key) || ""; } catch (_) { return ""; } }
  const note = $("big-note");
  const caption = $("big-caption");
  const portal = d.inPortal === "true";
  const key = () => `geecs-plot:big:${d.day}:${d.scan}:${device()}:${mode}`;
  const PROCESSING_KEY = "geecs-plot:processing";
  const kept = (key) => { try { return sessionStorage.getItem(key) || ""; } catch (_) { return ""; } };
  const keep = (key, value) => { try { sessionStorage.setItem(key, value); } catch (_) { /* storage off */ } };
  if (!total()) { note.textContent = "Nothing to show: this scan has no shots."; return; }

  // ---- where to start: the map's bin, else this tab's last pick ----
  function start() {
    const wanted = new URLSearchParams(location.search).get("bin");
    if (wanted !== null && mode !== "trace") {
      const i = tiles.findIndex((t) => t.dataset.bin === wanted);
      if (i >= 0) return i;
    }
    const last = Number(kept(key()));
    return Number.isInteger(last) && last >= 0 && last < total() ? last : 0;
  }
  let current = start();

  // ---- the processing preview (a shot's pixels, ephemeral, write-free) ----
  // The choices are the device route's: recipes, for a device that renders.
  // Asked per camera (each has its own recipes), once each.
  const processing = $("big-processing");
  const offered = new Map();  // device -> its recipe names
  function offerProcessing() {
    if (!(portal && single)) return;
    const name = device();
    const fill = (names) => {
      if (device() !== name) return;
      while (processing.options.length > 1) processing.remove(1);
      for (const n of names) processing.add(new Option(n, n));
      const last = kept(PROCESSING_KEY);
      processing.value = names.includes(last) ? last : "";
      $("big-processing-box").hidden = !names.length;
      if (processing.value) show(current);
    };
    if (offered.has(name)) { fill(offered.get(name)); return; }
    const query = new URLSearchParams({ device: name, day: d.day });
    fetch(`${d.portalRoot}/api/run/${encodeURIComponent(d.uid)}/device?${query}`)
      .then((r) => (r.ok ? r.json() : { processing_options: [] }))
      .then(({ processing_options: names }) => { offered.set(name, names); fill(names); })
      .catch(() => { /* no choice offered */ });
  }
  if (portal && single) {
    processing.addEventListener("change", () => { keep(PROCESSING_KEY, processing.value); show(current); });
  }

  // ---- drawing one ----
  function shotUrl(row) {
    // The colour map as it is now (the rail changes it without a reload).
    const field = strip && strip.dataset.cam !== "1" ? `cmap@${strip.dataset.cam}` : "cmap";
    const cmap = (document.querySelector(`[name="${field}"]`) || {}).value || d.cmap;
    const query = new URLSearchParams({ device: device(), shot: String(row), day: d.day });
    query.set("display", JSON.stringify({ cmap }));
    query.set("processing", processing.value);
    return `${d.portalRoot}/run/${encodeURIComponent(d.uid)}/image.png?${query}`;
  }
  async function why(url) {
    // A missed shot or a bad file answers with its reason; say it.
    try {
      const body = await (await fetch(url)).json();
      return body.detail || "no image";
    } catch (_) { return "no image"; }
  }
  function picture(i) {
    const tile = tiles[i];
    const image = $("big-image");
    const thumb = tile.querySelector("img");
    const processed = single && portal && processing.value;
    // The tile's picture, drawn with a colour bar beside it.
    // (without the tile's slim bar: the big view draws its own, with axes)
    const own = (thumb.getAttribute("src") || "").replace(/&cbar=true/, "");
    const src = processed ? shotUrl(tile.dataset.index) : own && `${own}&bar=true`;
    if (!src) {
      // A shared z scale: the strip's levels are still being read; the
      // tile's picture arriving redraws this (the observer below).
      note.textContent = "Waiting for the strip's levels…";
      return;
    }
    note.textContent = processed
      ? `${processing.value}, previewed: the whole frame on its own levels.`
      : mode === "bins" ? "The bin's average, as its tile draws it." : "";
    image.alt = `${device()} ${tile.dataset.caption}`;
    image.onerror = async () => { if (current === i) note.textContent = await why(src); };
    image.src = src;
  }
  function trace(shot) {
    const host = $("big-trace");
    const query = new URLSearchParams({ device: device(), shot: String(shot), day: d.day });
    host.dataset.caption = `${device()} · shot ${shot}`;
    note.textContent = "";
    fetch(`${d.portalRoot}/api/run/${encodeURIComponent(d.uid)}/trace?${query}`)
      .then((r) => r.json().then((body) => ({ r, body })))
      .then(({ r, body }) => {
        if (current !== shot - 1) return;
        if (!r.ok) { Plotly.purge(host); note.textContent = String(body.detail || r.statusText); return; }
        const fig = tools.themed(JSON.parse(JSON.stringify(body.figure)));
        Plotly.react(host, fig.data, fig.layout, tools.config());
      })
      .catch((error) => { note.textContent = String(error); });
  }
  function show(i) {
    current = Math.max(0, Math.min(total() - 1, i));
    keep(key(), String(current));
    if (mode === "trace") {
      caption.textContent = `shot ${current + 1} of ${total()}`;
      trace(current + 1);
      return;
    }
    for (const s of strips) {
      for (const t of s.querySelectorAll(".tile")) t.toggleAttribute("aria-current", t === tiles[current]);
    }
    caption.textContent = strips.length > 1
      ? `${device()} · ${tiles[current].dataset.caption}` : tiles[current].dataset.caption;
    picture(current);
  }
  // A click on another camera's tile moves the big view to that row.
  function useRow(next) {
    if (next === strip) return;
    strip = next;
    tiles = Array.from(strip.querySelectorAll(".tile"));
    keep(ROW_KEY, strip.dataset.cam);
    // A recipe is one camera's: never draw another camera's shot through
    // it. The new camera's own list (offerProcessing) may pick one again.
    processing.value = "";
    $("big-processing-box").hidden = true;
    offerProcessing();
  }

  for (const s of strips) {
    Array.from(s.querySelectorAll(".tile")).forEach((tile, i) => tile.addEventListener("click", () => { useRow(s); show(i); }));
  }
  offerProcessing();
  // The tile's picture changes under the big view (a shared z scale read,
  // a new colour map, levels or ROI): follow it.
  if (typeof MutationObserver !== "undefined") {
    const observer = new MutationObserver((changes) => {
      if (changes.some((c) => c.target === tiles[current]?.querySelector("img"))) picture(current);
    });
    for (const img of document.querySelectorAll(".strip .tile img")) observer.observe(img, { attributes: true, attributeFilter: ["src"] });
  }
  for (const button of document.querySelectorAll("[data-big-step]"))
    button.addEventListener("click", () => show(current + Number(button.dataset.bigStep)));
  window.addEventListener("geecs:theme", () => { if (mode === "trace") show(current); });
  show(current);
})();
