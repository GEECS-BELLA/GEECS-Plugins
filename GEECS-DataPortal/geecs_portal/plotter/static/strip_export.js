// The Images page's camera rows as one picture: Copy puts it on the
// clipboard, Save downloads it as a PNG. It is drawn from the tiles' own
// pictures (their ROI, levels and colour map as shown) at their full
// resolution, each camera laid out as its strip is (the tiles shown, as
// many across), each tile with its caption (and its colour bar, drawn into
// it), a camera's one colour bar (a shared or fixed scale) in a narrow
// column at the right beside its last row, and with several cameras each row under its camera's name;
// all under the panel's title line and the scan's conditions (the page's
// list above the strip, on one line) and the picked scan scalars (a line
// under them). White background and dark text whatever
// the page's theme, so it pastes cleanly into slides and the logbook.
// Each tile is drawn at the size it has on screen (times the screen's pixel
// density: sharp where it is pasted, a fraction of the file); the "full res"
// switch beside the buttons draws the pictures at their own resolution
// instead, remembered per browser.
"use strict";

(function () {
  const holder = document.getElementById("strips");
  const copyButton = document.querySelector("[data-strip-copy]");
  const saveButton = document.querySelector("[data-strip-save]");
  if (!holder || !copyButton || !saveButton) return;
  const toast = (text) => { if (window.GeecsKit) window.GeecsKit.toast(text); };
  const CLIPBOARD_OK = !!(navigator.clipboard && window.ClipboardItem);
  const MAX_WIDTH = 6000;  // px; a wider grid is scaled down to this
  const FULL_KEY = "geecs.strip.fullres";
  const fullButton = document.querySelector("[data-strip-full]");
  let full = false;
  try { full = localStorage.getItem(FULL_KEY) === "1"; } catch (_) { full = false; }
  const showFull = () => { if (fullButton) fullButton.setAttribute("aria-pressed", String(full)); };
  if (fullButton) {
    showFull();
    fullButton.addEventListener("click", () => {
      full = !full;
      showFull();
      try { localStorage.setItem(FULL_KEY, full ? "1" : "0"); } catch (_) { /* this tab only */ }
      toast(full ? "Grid exports at the pictures' full resolution" : "Grid exports at screen size");
    });
  }
  const FONT = 'px system-ui, -apple-system, "Segoe UI", sans-serif';

  // Tiles across as a strip lays them out now: in one row, every tile shown;
  // in a grid, the tiles sharing the first one's row.
  function across(strip, tiles) {
    if (strip.classList.contains("row")) return tiles.length;
    const top = tiles[0].offsetTop;
    return Math.max(1, tiles.filter((t) => t.offsetTop === top).length);
  }

  async function loaded(src) {
    if (!src) return null;
    const image = new Image();
    image.src = src;
    try { await image.decode(); return image; } catch (_) { return null; }
  }

  // One camera's part: its shown tiles and their pictures (loaded now: the
  // strip loads them as they scroll into view), its bar, its name.
  async function section(strip) {
    const tiles = Array.from(strip.querySelectorAll(".tile")).filter((t) => !t.hidden);
    if (!tiles.length) return null;
    if (tiles.some((t) => !t.querySelector("img").getAttribute("src")))
      throw new Error("a strip's levels are still being read");
    const images = await Promise.all(tiles.map((tile) =>
      tile.classList.contains("missing") ? null : loaded(tile.querySelector("img").getAttribute("src"))));
    const cell = strip.querySelector(".strip-bar-cell");
    const bar = cell && !cell.hidden ? await loaded(cell.querySelector("img").getAttribute("src")) : null;
    const cols = across(strip, tiles);
    return { tiles, images, bar, cols, rows: Math.ceil(tiles.length / cols), name: strip.dataset.device || "" };
  }

  // Every camera's part drawn on one canvas.
  async function render() {
    const strips = Array.from(holder.querySelectorAll(".strip[data-cam]"));
    const parts = (await Promise.all(strips.map(section))).filter(Boolean);
    const pictures = parts.flatMap((p) => p.images).filter(Boolean);
    if (!pictures.length) throw new Error("no tile has a picture");
    const named = parts.length > 1;
    const cols = Math.max(...parts.map((p) => p.cols));
    let cellW = Math.max(...pictures.map((i) => i.naturalWidth));
    let cellH = Math.max(...pictures.map((i) => i.naturalHeight));
    const gap0 = Math.max(4, Math.round(cellW * 0.03));
    let scale = Math.min(1, (MAX_WIDTH - gap0 * (cols + 1)) / (cellW * cols));
    if (!full) {
      // the widest tile as shown, in device pixels; never above the picture's own size
      const shown = Math.max(...parts.map((p) => p.tiles[0].getBoundingClientRect().width));
      if (shown > 0) scale = Math.min(scale, (shown * (window.devicePixelRatio || 1)) / cellW);
    }
    cellW = Math.round(cellW * scale);
    cellH = Math.round(cellH * scale);
    const gap = Math.max(4, Math.round(cellW * 0.03));
    const font = Math.max(11, Math.round(cellW * 0.07));
    const captionH = Math.round(font * 1.5);
    const titleH = Math.round(font * 2.2);
    // the scan's conditions (the Images page's list), on one line under the title
    let conditions = [];
    try { conditions = JSON.parse(holder.dataset.conditions || "[]"); } catch (e) { conditions = []; }
    const lineH = Math.round(font * 1.4);
    // the picked scan scalars, one line under the conditions
    let scalars = "";
    try { scalars = JSON.parse(holder.dataset.scalars || '""'); } catch (e) { scalars = ""; }
    const lines = (conditions.length ? 1 : 0) + (scalars ? 1 : 0);
    const condH = lines ? lines * lineH + Math.round(font * 0.4) : 0;
    const nameH = named ? Math.round(font * 1.8) : 0;
    const rowH = cellH + captionH + gap;
    // A camera's one colour bar (a shared or fixed scale) stands in a
    // narrow column at the right, beside its last row, as tall as a picture.
    const barW = Math.max(0, ...parts.filter((p) => p.bar)
      .map((p) => Math.round(p.bar.naturalWidth * (cellH / p.bar.naturalHeight))));
    const canvas = document.createElement("canvas");
    canvas.width = gap + cols * (cellW + gap) + (barW ? barW + gap : 0);
    canvas.height = titleH + condH + parts.reduce((h, p) => h + nameH + p.rows * rowH, 0) + gap;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#111";
    ctx.textBaseline = "middle";
    ctx.font = `600 ${font}${FONT}`;
    ctx.fillText(holder.dataset.title || "", gap, titleH / 2, canvas.width - 2 * gap);
    ctx.font = `${font}${FONT}`;
    let line = 0;
    if (conditions.length) {
      ctx.fillText(conditions.join("  ·  "), gap, titleH + lineH / 2, canvas.width - 2 * gap);
      line += 1;
    }
    if (scalars) ctx.fillText(scalars, gap, titleH + line * lineH + lineH / 2, canvas.width - 2 * gap);
    const fit = (image, x, y) => {
      // Fit the cell, keeping the picture's shape.
      const f = Math.min(cellW / image.naturalWidth, cellH / image.naturalHeight);
      const w = image.naturalWidth * f;
      const h = image.naturalHeight * f;
      ctx.drawImage(image, x + (cellW - w) / 2, y + (cellH - h) / 2, w, h);
    };
    let top = titleH + condH;
    for (const part of parts) {
      if (named) {
        ctx.font = `600 ${font}${FONT}`;
        ctx.textAlign = "left";
        ctx.fillText(part.name, gap, top + nameH / 2, canvas.width - 2 * gap);
        top += nameH;
      }
      ctx.font = `${font}${FONT}`;
      ctx.textAlign = "center";
      const at = (k) => [gap + (k % part.cols) * (cellW + gap), top + Math.floor(k / part.cols) * rowH];
      part.tiles.forEach((tile, k) => {
        const [x, y] = at(k);
        if (part.images[k]) fit(part.images[k], x, y);
        else {
          ctx.fillStyle = "#eee";
          ctx.fillRect(x, y, cellW, cellH);
          ctx.fillStyle = "#111";
        }
        ctx.fillText(tile.dataset.caption || "", x + cellW / 2, y + cellH + captionH / 2, cellW);
      });
      if (part.bar) {
        const f = cellH / part.bar.naturalHeight;
        const y = top + (part.rows - 1) * rowH;
        ctx.drawImage(part.bar, gap + cols * (cellW + gap), y, part.bar.naturalWidth * f, cellH);
      }
      top += part.rows * rowH;
    }
    return new Promise((resolve, reject) =>
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("no PNG"))), "image/png"));
  }

  function filename() {
    const name = (holder.dataset.filename || "geecs-images").replace(/[^\w.-]+/g, "_");
    return `${name}.png`;
  }
  function download(blob) {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename();
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  }
  const reason = (error) => (error && error.message) || String(error);

  saveButton.addEventListener("click", async () => {
    try {
      const blob = await render();
      download(blob);
      toast(`Saved ${filename()} (${(blob.size / 1e6).toFixed(1)} MB${full ? ", full resolution" : ""})`);
    }
    catch (error) { toast(`Could not save the grid: ${reason(error)}`); }
  });
  copyButton.addEventListener("click", async () => {
    // A bitmap reaches the clipboard only on a secure (https or localhost)
    // page; elsewhere the grid downloads, and the toast says why.
    const png = render();
    if (!CLIPBOARD_OK) {
      try { download(await png); toast("Clipboard needs an https or localhost page: downloaded the PNG instead"); }
      catch (error) { toast(`Could not copy the grid: ${reason(error)}`); }
      return;
    }
    try {
      // The promise goes in as is: Safari wants the clipboard call made
      // while the click is still the user's.
      await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
      toast("Grid copied to the clipboard");
    } catch (error) {
      try { download(await png); toast(`Copy failed (${(error && error.name) || "error"}): downloaded the PNG instead`); }
      catch (inner) { toast(`Could not copy the grid: ${reason(inner)}`); }
    }
  });
})();
