// The camera panel: the frame loop, the viewer (zoom and pan), this viewer's
// colour map, levels and scale, and the live readings from the SSE stream.
//
// Frames are PULLED one at a time (never more than one in flight), so the
// picture lags by one frame whatever the network does; the loop pauses while
// the tab is hidden, and a frame the camera has not replaced (the service's
// X-Frame-Counter is unchanged) is not decoded again. Colour map, levels and
// scale are applied here, to the IOC's 8-bit greyscale stream: nothing is
// written to the camera.
//
// Coordinates: "image" pixels are the frame's own (PROC1, after any
// transform); "view" pixels are CSS pixels in the viewport. view = image *
// scale + offset. Everything drawn over the image goes through toView /
// toImage, so later layers (crosshairs, the ROI box) zoom and pan with it.
(() => {
  "use strict";

  const root = document.getElementById("camera");
  const frameUrl = root.dataset.frameUrl;
  const eventsUrl = root.dataset.eventsUrl;
  const maxFps = parseFloat(root.dataset.maxFps) || 10;
  const storeKey = "geecs-panels:" + root.dataset.device + ":display";

  const $ = (id) => document.getElementById(id);
  const viewport = $("viewport");
  const canvas = $("image");
  const ctx = canvas.getContext("2d");
  // The colour-mapped frame at its own size; drawn into the view at the zoom.
  const frame = document.createElement("canvas");
  const fctx = frame.getContext("2d", { willReadFrequently: true });

  const cmapSel = $("cmap"), scaleSel = $("scale");
  const blackIn = $("level-lo"), whiteIn = $("level-hi");
  const autoLive = $("auto-live"), liveStrip = $("live-strip");

  // ---- colour maps: control points [t, r, g, b], interpolated to 256 -----
  const MAPS = {
    grey: [[0, 0, 0, 0], [1, 255, 255, 255]],
    viridis: [[0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140],
              [0.75, 94, 201, 98], [1, 253, 231, 37]],
    inferno: [[0, 0, 0, 4], [0.25, 87, 16, 110], [0.5, 188, 55, 84],
              [0.75, 249, 142, 9], [1, 252, 255, 164]],
    jet: [[0, 0, 0, 128], [0.125, 0, 0, 255], [0.375, 0, 255, 255],
          [0.625, 255, 255, 0], [0.875, 255, 0, 0], [1, 128, 0, 0]],
  };
  // The scale bends t in 0..1 after the levels: dim features rise with sqrt
  // and log. It stretches the 8-bit stream; it adds no dynamic range.
  const SCALES = {
    linear: (t) => t,
    sqrt: (t) => Math.sqrt(t),
    log: (t) => Math.log1p(1000 * t) / Math.log1p(1000),
  };

  function mapColour(points, t) {
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i];
      if (t <= b[0]) {
        const f = (t - a[0]) / (b[0] - a[0] || 1);
        return [a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * (b[3] - a[3])];
      }
    }
    const last = points[points.length - 1];
    return [last[1], last[2], last[3]];
  }

  // grey level 0..255 → rgb, through the levels, the scale, then the map.
  const lut = new Uint8ClampedArray(256 * 3);
  function buildLut() {
    const points = MAPS[cmapSel.value] || MAPS.grey;
    const bend = SCALES[scaleSel.value] || SCALES.linear;
    const black = clampLevel(blackIn.value, 0, 254);
    const white = Math.max(black + 1, clampLevel(whiteIn.value, 1, 255));
    for (let v = 0; v < 256; v++) {
      const t = bend(Math.min(1, Math.max(0, (v - black) / (white - black))));
      const c = mapColour(points, t);
      lut[3 * v] = c[0]; lut[3 * v + 1] = c[1]; lut[3 * v + 2] = c[2];
    }
  }
  function clampLevel(raw, lo, hi) {
    const n = Math.round(Number(raw));
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : lo;
  }

  // ---- per-viewer settings, remembered per camera -------------------------
  function save() {
    try {
      localStorage.setItem(storeKey, JSON.stringify({
        cmap: cmapSel.value, scale: scaleSel.value, black: blackIn.value,
        white: whiteIn.value, auto: autoLive.checked, live: liveStrip.open,
      }));
    } catch (_) { /* storage unavailable: settings last for this page only */ }
  }
  function restore() {
    try {
      const s = JSON.parse(localStorage.getItem(storeKey) || "null");
      if (!s) return;
      if (MAPS[s.cmap]) cmapSel.value = s.cmap;
      if (SCALES[s.scale]) scaleSel.value = s.scale;
      if (s.black !== undefined) blackIn.value = s.black;
      if (s.white !== undefined) whiteIn.value = s.white;
      autoLive.checked = !!s.auto;
      if (s.live === false) liveStrip.open = false;
    } catch (_) { /* nothing remembered */ }
  }

  // ---- the frame: grey levels kept, colour-mapped into `frame` ------------
  let grey = null;  // the last frame's grey levels: recolouring, auto levels, cursor
  let fw = 0, fh = 0;

  function autoLevels() {
    if (!grey) return;
    const hist = new Uint32Array(256);
    for (let i = 0; i < grey.length; i++) hist[grey[i]]++;
    const lo = grey.length * 0.001, hi = grey.length * 0.999;
    let acc = 0, black = 0, white = 255;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > lo) { black = v; break; } }
    acc = 0;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= hi) { white = v; break; } }
    if (white <= black) white = Math.min(255, black + 1);
    blackIn.value = black; whiteIn.value = white;
    buildLut();
  }

  function recolour() {
    if (!grey) return;
    const img = fctx.createImageData(fw, fh);
    const px = img.data;
    for (let i = 0, j = 0; i < grey.length; i++, j += 4) {
      const k = 3 * grey[i];
      px[j] = lut[k]; px[j + 1] = lut[k + 1]; px[j + 2] = lut[k + 2]; px[j + 3] = 255;
    }
    fctx.putImageData(img, 0, 0);
    draw();
  }

  function takeFrame(bitmap) {
    const sized = bitmap.width !== fw || bitmap.height !== fh;
    if (sized) {
      fw = frame.width = bitmap.width;
      fh = frame.height = bitmap.height;
    }
    fctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const px = fctx.getImageData(0, 0, fw, fh).data;
    if (!grey || grey.length !== fw * fh) grey = new Uint8Array(fw * fh);
    for (let i = 0, j = 0; i < grey.length; i++, j += 4) grey[i] = px[j];
    if (autoLive.checked) autoLevels();
    if (sized && !view.moved) fit();
    recolour();
  }

  // ---- the viewer: zoom and pan -------------------------------------------
  const view = { scale: 1, ox: 0, oy: 0, moved: false };
  const MAX_ZOOM = 40;
  let vw = 0, vh = 0;  // viewport size, CSS pixels

  const toView = (x, y) => [x * view.scale + view.ox, y * view.scale + view.oy];
  const toImage = (x, y) => [(x - view.ox) / view.scale, (y - view.oy) / view.scale];

  function fitScale() { return fw && fh ? Math.min(vw / fw, vh / fh) : 1; }

  function fit() {
    view.scale = fitScale();
    view.ox = (vw - fw * view.scale) / 2;
    view.oy = (vh - fh * view.scale) / 2;
    view.moved = false;
    draw();
  }

  // Zoom by `factor` keeping the image point under (cx, cy) where it is.
  function zoomAt(cx, cy, factor) {
    const next = Math.min(MAX_ZOOM, Math.max(fitScale() / 2, view.scale * factor));
    const [ix, iy] = toImage(cx, cy);
    view.scale = next;
    view.ox = cx - ix * next;
    view.oy = cy - iy * next;
    view.moved = true;
    draw();
  }

  function resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = viewport.getBoundingClientRect();
    vw = r.width; vh = r.height;
    canvas.width = Math.round(vw * dpr);
    canvas.height = Math.round(vh * dpr);
    if (!view.moved) fit(); else draw();
  }

  function draw() {
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!fw) return;
    ctx.setTransform(dpr * view.scale, 0, 0, dpr * view.scale, dpr * view.ox, dpr * view.oy);
    // Magnified: show the pixels as squares, never blurred into each other.
    ctx.imageSmoothingEnabled = view.scale * dpr < 1;
    ctx.drawImage(frame, 0, 0);
    $("zoom-level").textContent = Math.round(view.scale * 100) + "%";
  }

  viewport.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = viewport.getBoundingClientRect();
    zoomAt(ev.clientX - r.left, ev.clientY - r.top, Math.exp(-ev.deltaY * 0.0015));
  }, { passive: false });

  let drag = null;
  viewport.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    drag = { x: ev.clientX, y: ev.clientY, ox: view.ox, oy: view.oy };
    viewport.setPointerCapture(ev.pointerId);
    viewport.classList.add("panning");
  });
  viewport.addEventListener("pointermove", (ev) => {
    const r = viewport.getBoundingClientRect();
    showCursor(ev.clientX - r.left, ev.clientY - r.top);
    if (!drag) return;
    view.ox = drag.ox + (ev.clientX - drag.x);
    view.oy = drag.oy + (ev.clientY - drag.y);
    view.moved = true;
    draw();
  });
  const endDrag = () => { drag = null; viewport.classList.remove("panning"); };
  viewport.addEventListener("pointerup", endDrag);
  viewport.addEventListener("pointercancel", endDrag);
  viewport.addEventListener("pointerleave", () => { $("cursor").textContent = ""; });
  viewport.addEventListener("dblclick", fit);
  $("zoom-fit").addEventListener("click", fit);
  $("zoom-one").addEventListener("click", () => zoomAt(vw / 2, vh / 2, 1 / view.scale));
  new ResizeObserver(resize).observe(viewport);

  // Cursor readout in image pixels (PROC1 coordinates, after any transform).
  function showCursor(cx, cy) {
    if (!grey) return;
    const [ix, iy] = toImage(cx, cy).map(Math.floor);
    $("cursor").textContent = ix < 0 || iy < 0 || ix >= fw || iy >= fh ? ""
      : "x " + ix + "  y " + iy + "  level " + grey[iy * fw + ix];
  }

  // ---- the frame loop -----------------------------------------------------
  const errBox = $("image-error"), errText = $("image-error-text");
  const shown = [];  // new-frame arrival times, for the fps readout
  let lastTag = null;  // the service's frame counter of the frame on screen
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const visible = () => new Promise((r) => {
    if (!document.hidden) return r();
    document.addEventListener("visibilitychange", function on() {
      if (!document.hidden) { document.removeEventListener("visibilitychange", on); r(); }
    });
  });

  // Fetching and drawing are decoupled: the next request goes out on the
  // poll's clock while the last frame is still being decoded and drawn, so
  // the frame rate is limited by the cap, not by fetch time + draw time.
  // Still one request in flight; a frame that arrives while another is
  // being drawn replaces any frame waiting behind it (latest wins).
  let waiting = null, drawing = false;

  async function drawLatest() {
    if (drawing) return;
    drawing = true;
    try {
      while (waiting) {
        const blob = waiting;
        waiting = null;
        takeFrame(await createImageBitmap(blob));
        shown.push(performance.now());
      }
    } finally {
      drawing = false;
    }
  }

  async function loop() {
    for (;;) {
      await visible();
      const t0 = performance.now();
      try {
        const r = await fetch(frameUrl, { cache: "no-store" });
        if (!r.ok) throw new Error((await r.text()) || ("HTTP " + r.status));
        const tag = r.headers.get("X-Frame-Counter");
        const blob = await r.blob();
        // The same frame again (a camera slower than the poll): no redraw.
        if (tag === null || tag !== lastTag) {
          lastTag = tag;
          waiting = blob;
          drawLatest();
        }
        errBox.hidden = true;
      } catch (e) {
        errText.textContent = String(e.message || e);
        errBox.hidden = false;
        await sleep(1000);
      }
      while (shown.length && shown[0] < performance.now() - 2000) shown.shift();
      $("fps").textContent = (shown.length / 2).toFixed(1) + " fps";
      await sleep(Math.max(0, 1000 / maxFps - (performance.now() - t0)));
    }
  }

  // ---- live readings ------------------------------------------------------
  const SCALE = { exposure: 1000 };  // seconds on the wire, ms on the page
  // ADDriver DetectorState words → kit words; anything else is "unknown".
  const DETECTOR = {
    acquire: "running", readout: "running", correct: "running", saving: "running",
    waiting: "running", initializing: "running",
    idle: "paused", error: "failed", aborted: "failed", disconnected: "failed",
  };
  // What the folded Live strip still shows in its summary line.
  const SUMMARY = [["cx", "centroid "], ["cy", ", "], ["max", "  max "]];

  function fmt(key, value) {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "number") return String(value);
    const v = value * (SCALE[key] || 1);
    if (Number.isInteger(v) && Math.abs(v) < 1e7) return String(v);
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) return v.toExponential(3);
    return String(Number(v.toPrecision(4)));
  }
  function ageText(age) {
    if (age === null || age === undefined) return "";
    return age < 60 ? age.toFixed(1) + " s" : Math.round(age / 60) + " min";
  }

  function apply(snap) {
    const byKey = {};
    for (const r of snap.readings) {
      byKey[r.key] = r;
      const el = root.querySelector('.live[data-key="' + r.key + '"]');
      if (!el) continue;
      // The value is the text before the unit's <small>; the unit stays.
      el.querySelector(".v").firstChild.nodeValue =
        r.state === "disconnected" ? "—" : fmt(r.key, r.value);
      const age = el.querySelector(".age");
      if (r.state === "fresh") {
        el.removeAttribute("data-age");
        age.textContent = ageText(r.age);
      } else if (r.state === "off") {
        el.removeAttribute("data-age");
        age.textContent = "not computed";
      } else {
        el.setAttribute("data-age", "stale");
        age.textContent = r.state === "disconnected" ? "disconnected"
          : (r.age === null ? "no frames yet" : ageText(r.age));
      }
      if (r.key === "state") {
        const chip = $("detector");
        const word = r.state === "disconnected" ? "disconnected" : String(r.value || "").toLowerCase();
        chip.textContent = word || "unknown";
        chip.dataset.state = r.state === "disconnected" ? "unknown" : (DETECTOR[word] || "unknown");
      }
    }
    $("live-summary").textContent = SUMMARY.map(([k, label]) =>
      label + (byKey[k] && byKey[k].state === "fresh" ? fmt(k, byKey[k].value) : "—")).join("");
    const img = $("image-state");
    if (snap.image.state === "fresh") {
      img.dataset.state = "ok"; img.textContent = "live";
    } else if (snap.image.state === "stale") {
      img.dataset.state = "degraded";
      img.textContent = snap.image.age === null ? "no new frames" : "stale · " + ageText(snap.image.age);
    } else {
      img.dataset.state = "failed"; img.textContent = "disconnected";
    }
    $("missing").hidden = !snap.missing.length;
    $("missing-list").textContent = snap.missing.join("  ");
  }

  // The readings are only as fresh as the stream that carries them: if no
  // snapshot arrives for a while (the service restarted, the laptop slept),
  // every value and the image chip go stale here rather than keep their
  // last "fresh" look.
  const STREAM_STALE_MS = 3000;
  let lastSnapshot = 0;

  function streamLost() {
    for (const el of root.querySelectorAll(".live")) {
      el.setAttribute("data-age", "stale");
      el.querySelector(".age").textContent = "no stream";
    }
    $("live-summary").textContent = "no stream";
    const chip = $("detector");
    chip.dataset.state = "unknown"; chip.textContent = "reconnecting";
    const img = $("image-state");
    img.dataset.state = "unknown"; img.textContent = "no update";
  }

  function connect() {
    const es = new EventSource(eventsUrl);
    es.addEventListener("state", (ev) => {
      lastSnapshot = performance.now();
      apply(JSON.parse(ev.data));
    });
    es.onerror = streamLost;
    setInterval(() => {
      if (lastSnapshot && performance.now() - lastSnapshot > STREAM_STALE_MS) streamLost();
    }, 1000);
  }

  // ---- wiring -------------------------------------------------------------
  restore();
  buildLut();
  for (const el of [cmapSel, scaleSel, blackIn, whiteIn]) {
    el.addEventListener("change", () => { buildLut(); recolour(); save(); });
  }
  autoLive.addEventListener("change", save);
  liveStrip.addEventListener("toggle", save);
  $("auto-once").addEventListener("click", () => { autoLevels(); recolour(); save(); });
  $("reset-levels").addEventListener("click", () => {
    blackIn.value = 0; whiteIn.value = 255; scaleSel.value = "linear";
    autoLive.checked = false; buildLut(); recolour(); save();
  });
  resize();
  connect();
  loop();
})();
