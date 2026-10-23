// The camera panel: the frame loop, this viewer's colour map and levels,
// and the live readings from the service's SSE stream.
//
// Frames are PULLED one at a time (never more than one in flight), so the
// picture lags by one frame whatever the network does; the loop pauses while
// the tab is hidden. Colour map and levels are applied here, to the IOC's
// 8-bit greyscale stream: nothing is written to the camera.
(() => {
  "use strict";

  const root = document.getElementById("camera");
  const frameUrl = root.dataset.frameUrl;
  const eventsUrl = root.dataset.eventsUrl;
  const maxFps = parseFloat(root.dataset.maxFps) || 10;
  const storeKey = "geecs-panels:" + root.dataset.device + ":display";

  const canvas = document.getElementById("image");
  const ctx = canvas.getContext("2d");
  const scratch = document.createElement("canvas");
  const sctx = scratch.getContext("2d", { willReadFrequently: true });

  const $ = (id) => document.getElementById(id);
  const cmapSel = $("cmap"), blackIn = $("level-lo"), whiteIn = $("level-hi");
  const autoLive = $("auto-live");

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

  // grey level 0..255 → rgb, through the levels then the map.
  let lut = new Uint8ClampedArray(256 * 3);
  function buildLut() {
    const points = MAPS[cmapSel.value] || MAPS.grey;
    const black = clampLevel(blackIn.value, 0, 254);
    const white = Math.max(black + 1, clampLevel(whiteIn.value, 1, 255));
    for (let v = 0; v < 256; v++) {
      const t = Math.min(1, Math.max(0, (v - black) / (white - black)));
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
        cmap: cmapSel.value, black: blackIn.value, white: whiteIn.value, auto: autoLive.checked,
      }));
    } catch (_) { /* storage unavailable: settings last for this page only */ }
  }
  function restore() {
    try {
      const s = JSON.parse(localStorage.getItem(storeKey) || "null");
      if (!s) return;
      if (MAPS[s.cmap]) cmapSel.value = s.cmap;
      blackIn.value = s.black; whiteIn.value = s.white; autoLive.checked = !!s.auto;
    } catch (_) { /* nothing remembered */ }
  }

  // ---- levels from the frame's histogram ----------------------------------
  let grey = null;  // the last frame's grey levels, for auto levels and the cursor
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

  // ---- drawing ------------------------------------------------------------
  function render(bitmap) {
    const w = bitmap.width, h = bitmap.height;
    if (scratch.width !== w || scratch.height !== h) {
      scratch.width = canvas.width = w;
      scratch.height = canvas.height = h;
    }
    sctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const img = sctx.getImageData(0, 0, w, h);
    const px = img.data;
    const n = w * h;
    if (!grey || grey.length !== n) grey = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) grey[i] = px[j];
    if (autoLive.checked) autoLevels();
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const k = 3 * grey[i];
      px[j] = lut[k]; px[j + 1] = lut[k + 1]; px[j + 2] = lut[k + 2];
    }
    ctx.putImageData(img, 0, 0);
  }

  // ---- the frame loop -----------------------------------------------------
  const errBox = $("image-error"), errText = $("image-error-text");
  const shown = [];  // arrival times, for the fps readout
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const visible = () => new Promise((r) => {
    if (!document.hidden) return r();
    document.addEventListener("visibilitychange", function on() {
      if (!document.hidden) { document.removeEventListener("visibilitychange", on); r(); }
    });
  });

  async function loop() {
    for (;;) {
      await visible();
      const t0 = performance.now();
      try {
        const r = await fetch(frameUrl, { cache: "no-store" });
        if (!r.ok) throw new Error((await r.text()) || ("HTTP " + r.status));
        render(await createImageBitmap(await r.blob()));
        errBox.hidden = true;
        shown.push(performance.now());
        while (shown.length && shown[0] < performance.now() - 2000) shown.shift();
        $("fps").textContent = (shown.length / 2).toFixed(1) + " fps";
      } catch (e) {
        errText.textContent = String(e.message || e);
        errBox.hidden = false;
        await sleep(1000);
      }
      await sleep(Math.max(0, 1000 / maxFps - (performance.now() - t0)));
    }
  }

  // ---- cursor readout (image = PROC1 coordinates, after any transform) ----
  canvas.addEventListener("mousemove", (ev) => {
    if (!grey) return;
    const r = canvas.getBoundingClientRect();
    const x = Math.floor((ev.clientX - r.left) * canvas.width / r.width);
    const y = Math.floor((ev.clientY - r.top) * canvas.height / r.height);
    if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return;
    $("cursor").textContent = "x " + x + "  y " + y + "  level " + grey[y * canvas.width + x];
  });
  canvas.addEventListener("mouseleave", () => { $("cursor").textContent = ""; });

  // ---- live readings ------------------------------------------------------
  const SCALE = { exposure: 1000 };  // seconds on the wire, ms on the page
  const DETECTOR = { acquire: "running", idle: "paused", error: "failed" };

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
    for (const r of snap.readings) {
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
        chip.dataset.state = r.state === "disconnected" ? "unknown" : (DETECTOR[word] || "running");
      }
    }
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

  function connect() {
    const es = new EventSource(eventsUrl);
    es.addEventListener("state", (ev) => apply(JSON.parse(ev.data)));
    es.onerror = () => {
      const chip = $("detector");
      chip.dataset.state = "unknown"; chip.textContent = "reconnecting";
    };
  }

  // ---- wiring -------------------------------------------------------------
  restore();
  buildLut();
  for (const el of [cmapSel, blackIn, whiteIn]) {
    el.addEventListener("change", () => { buildLut(); save(); });
  }
  autoLive.addEventListener("change", save);
  $("auto-once").addEventListener("click", () => { autoLevels(); save(); });
  $("reset-levels").addEventListener("click", () => {
    blackIn.value = 0; whiteIn.value = 255; autoLive.checked = false; buildLut(); save();
  });
  connect();
  loop();
})();
