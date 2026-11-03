// The Images page's strip as one picture: Copy puts it on the clipboard,
// Save downloads it as a PNG. It is drawn from the tiles' own pictures
// (their ROI, levels and colour map as shown) at their full resolution,
// laid out as the strip is (its tiles across), each with its caption, under
// the panel's title line. White background and dark text whatever the
// page's theme, so it pastes cleanly into slides and the logbook.
"use strict";

(function () {
  const strip = document.getElementById("strip");
  const copyButton = document.querySelector("[data-strip-copy]");
  const saveButton = document.querySelector("[data-strip-save]");
  if (!strip || !copyButton || !saveButton) return;
  const toast = (text) => { if (window.GeecsKit) window.GeecsKit.toast(text); };
  const CLIPBOARD_OK = !!(navigator.clipboard && window.ClipboardItem);
  const MAX_WIDTH = 6000;  // px; a wider grid is scaled down to this
  const FONT = 'px system-ui, -apple-system, "Segoe UI", sans-serif';

  // Tiles across as the strip lays them out now: the tiles sharing the
  // first one's row (a "row" layout is one row however many fit on screen).
  function across(tiles) {
    if (strip.classList.contains("row")) return tiles.length;
    const top = tiles[0].offsetTop;
    const n = tiles.filter((t) => t.offsetTop === top).length;
    return Math.max(1, n);
  }

  // Each tile's picture, loaded now (the strip loads them as they scroll
  // into view). A tile still waiting for shared levels has no picture yet.
  async function pictures(tiles) {
    return Promise.all(tiles.map(async (tile) => {
      const src = tile.querySelector("img").getAttribute("src");
      if (!src || tile.classList.contains("missing")) return null;
      const image = new Image();
      image.src = src;
      try { await image.decode(); return image; } catch (_) { return null; }
    }));
  }

  // The whole grid drawn on one canvas.
  async function render() {
    const tiles = Array.from(strip.querySelectorAll(".tile"));
    if (!tiles.length) throw new Error("no tiles");
    if (tiles.some((t) => !t.querySelector("img").getAttribute("src")))
      throw new Error("the strip's levels are still being read");
    const images = await pictures(tiles);
    const loaded = images.filter(Boolean);
    if (!loaded.length) throw new Error("no tile has a picture");
    const cols = across(tiles);
    const rows = Math.ceil(tiles.length / cols);
    let cellW = Math.max(...loaded.map((i) => i.naturalWidth));
    let cellH = Math.max(...loaded.map((i) => i.naturalHeight));
    const gap0 = Math.max(4, Math.round(cellW * 0.03));
    const scale = Math.min(1, (MAX_WIDTH - gap0 * (cols + 1)) / (cellW * cols));
    cellW = Math.round(cellW * scale);
    cellH = Math.round(cellH * scale);
    const gap = Math.max(4, Math.round(cellW * 0.03));
    const font = Math.max(11, Math.round(cellW * 0.07));
    const captionH = Math.round(font * 1.5);
    const titleH = Math.round(font * 2.2);
    const canvas = document.createElement("canvas");
    canvas.width = gap + cols * (cellW + gap);
    canvas.height = titleH + rows * (cellH + captionH + gap) + gap;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#111";
    ctx.textBaseline = "middle";
    ctx.font = `600 ${font}${FONT}`;
    ctx.fillText(strip.dataset.title || "", gap, titleH / 2, canvas.width - 2 * gap);
    ctx.font = `${font}${FONT}`;
    ctx.textAlign = "center";
    tiles.forEach((tile, k) => {
      const x = gap + (k % cols) * (cellW + gap);
      const y = titleH + Math.floor(k / cols) * (cellH + captionH + gap);
      const image = images[k];
      if (image) {
        // Fit the cell, keeping the picture's shape.
        const f = Math.min(cellW / image.naturalWidth, cellH / image.naturalHeight);
        const w = image.naturalWidth * f;
        const h = image.naturalHeight * f;
        ctx.drawImage(image, x + (cellW - w) / 2, y + (cellH - h) / 2, w, h);
      } else {
        ctx.fillStyle = "#eee";
        ctx.fillRect(x, y, cellW, cellH);
        ctx.fillStyle = "#111";
      }
      ctx.fillText(tile.dataset.caption || "", x + cellW / 2, y + cellH + captionH / 2, cellW);
    });
    return new Promise((resolve, reject) =>
      canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("no PNG"))), "image/png"));
  }

  function filename() {
    const name = (strip.dataset.filename || "geecs-images").replace(/[^\w.-]+/g, "_");
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
    try { download(await render()); toast(`Saved ${filename()}`); }
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
