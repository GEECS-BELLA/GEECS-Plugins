// The Images page's big view: the tile clicked in the strip at full size,
// stepped with ‹ ›. It is the tile's own picture (its ROI, levels and
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
  const mode = d.mode;  // "bins", "shots" or "trace"
  const tools = window.PlotTools || { config: () => ({ responsive: true }), themed: (v) => v };
  const tiles = Array.from(document.querySelectorAll("#strip .tile"));
  const total = mode === "trace" ? Number(d.shots) || 0 : tiles.length;
  const note = $("big-note");
  const caption = $("big-caption");
  const portal = d.inPortal === "true";
  const KEY = `geecs-plot:big:${d.day}:${d.scan}:${d.device}:${mode}`;
  const PROCESSING_KEY = "geecs-plot:processing";
  const kept = (key) => { try { return sessionStorage.getItem(key) || ""; } catch (_) { return ""; } };
  const keep = (key, value) => { try { sessionStorage.setItem(key, value); } catch (_) { /* storage off */ } };
  if (!total) { note.textContent = "Nothing to show: this scan has no shots."; return; }

  // ---- where to start: the map's bin, else this tab's last pick ----
  function start() {
    const wanted = new URLSearchParams(location.search).get("bin");
    if (wanted !== null && mode !== "trace") {
      const i = tiles.findIndex((t) => t.dataset.bin === wanted);
      if (i >= 0) return i;
    }
    const last = Number(kept(KEY));
    return Number.isInteger(last) && last >= 0 && last < total ? last : 0;
  }
  let current = start();

  // ---- the processing preview (a shot's pixels, ephemeral, write-free) ----
  // The choices are the device route's: recipes, for a device that renders.
  const processing = $("big-processing");
  if (portal && mode === "shots") {
    const query = new URLSearchParams({ device: d.device, day: d.day });
    fetch(`${d.portalRoot}/api/run/${encodeURIComponent(d.uid)}/device?${query}`)
      .then((r) => (r.ok ? r.json() : { processing_options: [] }))
      .then(({ processing_options: names }) => {
        if (!names.length) return;
        for (const name of names) processing.add(new Option(name, name));
        const last = kept(PROCESSING_KEY);
        processing.value = names.includes(last) ? last : "";
        $("big-processing-box").hidden = false;
        if (processing.value) show(current);
      })
      .catch(() => { /* no choice offered */ });
    processing.addEventListener("change", () => { keep(PROCESSING_KEY, processing.value); show(current); });
  }

  // ---- drawing one ----
  function shotUrl(row) {
    // The colour map as it is now (the rail changes it without a reload).
    const cmap = (document.getElementById("cmap") || {}).value || d.cmap;
    const query = new URLSearchParams({ device: d.device, shot: String(row), day: d.day });
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
    const processed = mode === "shots" && portal && processing.value;
    const src = processed ? shotUrl(tile.dataset.index) : thumb.getAttribute("src") || "";
    if (!src) {
      // A shared z scale: the strip's levels are still being read; the
      // tile's picture arriving redraws this (the observer below).
      note.textContent = "Waiting for the strip's levels…";
      return;
    }
    note.textContent = processed
      ? `${processing.value}, previewed: the whole frame on its own levels.`
      : mode === "bins" ? "The bin's average, as its tile draws it." : "";
    image.alt = `${d.device} ${tile.dataset.caption}`;
    image.onerror = async () => { if (current === i) note.textContent = await why(src); };
    image.src = src;
  }
  function trace(shot) {
    const host = $("big-trace");
    const query = new URLSearchParams({ device: d.device, shot: String(shot), day: d.day });
    host.dataset.caption = `${d.device} · shot ${shot}`;
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
    current = Math.max(0, Math.min(total - 1, i));
    keep(KEY, String(current));
    if (mode === "trace") {
      caption.textContent = `shot ${current + 1} of ${total}`;
      trace(current + 1);
      return;
    }
    tiles.forEach((t, j) => t.toggleAttribute("aria-current", j === current));
    caption.textContent = tiles[current].dataset.caption;
    picture(current);
  }

  tiles.forEach((tile, i) => tile.addEventListener("click", () => show(i)));
  // The tile's picture changes under the big view (a shared z scale read,
  // a new colour map, levels or ROI): follow it.
  if (typeof MutationObserver !== "undefined") {
    const observer = new MutationObserver((changes) => {
      if (changes.some((c) => c.target === tiles[current]?.querySelector("img"))) picture(current);
    });
    for (const tile of tiles) observer.observe(tile.querySelector("img"), { attributes: true, attributeFilter: ["src"] });
  }
  for (const button of document.querySelectorAll("[data-big-step]"))
    button.addEventListener("click", () => show(current + Number(button.dataset.bigStep)));
  window.addEventListener("geecs:theme", () => { if (mode === "trace") show(current); });
  show(current);
})();
