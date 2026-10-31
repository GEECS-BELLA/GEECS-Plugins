// The camera page's rules, pure: no DOM, no socket, no clock. camera.js and
// camera_controls.js draw what these say; tests/test_camera_rules_js.py runs
// each rule under node. The view's coordinates, this viewer's colour map and
// levels, how a value is shown, which control is enabled: nothing here judges
// the device (verdicts are the module's, values Hare's).
// window.GeecsCameraRules in the page, module.exports under node.
(function (rules) {
  "use strict";

  // ---- colour maps: control points [t, r, g, b], interpolated to 256 levels
  const MAPS = {
    grey: [[0, 0, 0, 0], [1, 255, 255, 255]],
    viridis: [[0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140], [0.75, 94, 201, 98], [1, 253, 231, 37]],
    inferno: [[0, 0, 0, 4], [0.25, 87, 16, 110], [0.5, 188, 55, 84], [0.75, 249, 142, 9], [1, 252, 255, 164]],
    jet: [[0, 0, 0, 128], [0.125, 0, 0, 255], [0.375, 0, 255, 255], [0.625, 255, 255, 0], [0.875, 255, 0, 0], [1, 128, 0, 0]],
  };
  // The scale bends t in 0..1 after the levels; it stretches the 8-bit stream, it adds no range.
  const SCALES = { linear: (t) => t, sqrt: (t) => Math.sqrt(t), log: (t) => Math.log1p(1000 * t) / Math.log1p(1000) };

  function mapColour(points, t) {
    for (let i = 1; i < points.length; i++) {
      const a = points[i - 1], b = points[i];
      if (t <= b[0]) {
        const f = (t - a[0]) / (b[0] - a[0] || 1);
        return [1, 2, 3].map((k) => a[k] + f * (b[k] - a[k]));
      }
    }
    return points[points.length - 1].slice(1);
  }

  function clampLevel(raw, lo, hi) {
    const n = Math.round(Number(raw));
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : lo;
  }

  // The levels as applied: black in 0..254, white above it.
  function levels(blackRaw, whiteRaw) {
    const black = clampLevel(blackRaw, 0, 254);
    return { black, white: Math.max(black + 1, clampLevel(whiteRaw, 1, 255)) };
  }

  // grey level 0..255 → [r, g, b] x 256, through the levels, the scale, then the map.
  function lut(cmap, scale, blackRaw, whiteRaw) {
    const points = MAPS[cmap] || MAPS.grey, bend = SCALES[scale] || SCALES.linear;
    const { black, white } = levels(blackRaw, whiteRaw);
    const out = [];
    for (let v = 0; v < 256; v++) {
      const t = bend(Math.min(1, Math.max(0, (v - black) / (white - black))));
      out.push(...mapColour(points, t).map(Math.round));
    }
    return out;
  }

  // Auto levels from a 256-bin histogram of the frame's grey levels: the 0.1 % and 99.9 % points.
  function autoLevels(hist) {
    const n = hist.reduce((a, b) => a + b, 0);
    let acc = 0, black = 0, white = 255;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc > n * 0.001) { black = v; break; } }
    acc = 0;
    for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= n * 0.999) { white = v; break; } }
    return { black, white: white <= black ? Math.min(255, black + 1) : white };
  }

  // The image in image pixels (MJPG1's ArraySize): the IOC's size when it agrees with
  // the JPEG's shape (ffmpegServer scales a large frame by ONE factor, each side
  // rounded), else the JPEG's own. The JPEG is stretched over it.
  function imageSize(fw, fh, trueW, trueH) {
    if (!trueW || !trueH) return [fw, fh];
    const ok = (s) => s <= 1 + 1e-9 && Math.abs(trueW * s - fw) <= 1.5 && Math.abs(trueH * s - fh) <= 1.5;
    return ok(fw / trueW) || ok(fh / trueH) ? [trueW, trueH] : [fw, fh];
  }

  // ---- the view: view = image * scale + offset (CSS pixels in the viewport)
  const toView = (v, x, y) => [x * v.scale + v.ox, y * v.scale + v.oy];
  const toImage = (v, x, y) => [(x - v.ox) / v.scale, (y - v.oy) / v.scale];
  const FIT_MARGIN = 16, MAX_ZOOM = 40;
  const fitScale = (vw, vh, iw, ih) => (!iw || !ih ? 1 : Math.max(0.01, Math.min((vw - 2 * FIT_MARGIN) / iw, (vh - 2 * FIT_MARGIN) / ih)));

  function fit(vw, vh, iw, ih) {
    const scale = fitScale(vw, vh, iw, ih);
    return { scale, ox: (vw - iw * scale) / 2, oy: (vh - ih * scale) / 2, moved: false };
  }

  // Zoom by `factor`, the image point under (cx, cy) staying put; down to half the fit (or 100 %).
  function zoomAt(v, cx, cy, factor, fitted) {
    const scale = Math.min(MAX_ZOOM, Math.max(Math.min(1, fitted / 2), v.scale * factor));
    const [ix, iy] = toImage(v, cx, cy);
    return { scale, ox: cx - ix * scale, oy: cy - iy * scale, moved: true };
  }

  // ---- the analysis ROI's own pixels ↔ image pixels (STAT's results are in the ROI's)
  // One axis: {min, size, bin, reverse}; output pixel i covers MinX + i·BinX … (+BinX − 1), mirrored when reversed.
  const span = (a) => Math.floor(a.size / Math.max(a.bin, 1)) * Math.max(a.bin, 1);
  function roiToImage(a, i) {
    const b = Math.max(a.bin, 1), offset = i * b + (b - 1) / 2;
    return a.reverse ? a.min + span(a) - 1 - offset : a.min + offset;
  }
  function imageToRoi(a, x) {
    const b = Math.max(a.bin, 1), offset = a.reverse ? a.min + span(a) - 1 - x : x - a.min;
    return (offset - (b - 1) / 2) / b;
  }

  // An analysis-ROI drag: the edges the handle names follow the pointer, the others stay; inside the frame, at least 1 px.
  function dragRoi(start, handle, ix, iy, fw, fh) {
    const c = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(v)));
    let l = start.minx, t = start.miny, r = start.minx + start.sizex, b = start.miny + start.sizey;
    if (handle.includes("w")) l = c(ix, 0, r - 1);
    if (handle.includes("e")) r = c(ix, l + 1, fw);
    if (handle.includes("n")) t = c(iy, 0, b - 1);
    if (handle.includes("s")) b = c(iy, t + 1, fh);
    return { minx: l, miny: t, sizex: r - l, sizey: b - t };
  }

  // Which ROI edges a press at (cx, cy) moves ("nw", "e", "" for none), the box in view pixels.
  function roiHandle(q, cx, cy, grab, edge) {
    const near = (a, b) => Math.abs(a - b) <= grab;
    const inX = cx >= q.l - edge && cx <= q.r + edge, inY = cy >= q.t - edge && cy <= q.b + edge;
    let h = "";
    if (inX && near(cy, q.t)) h += "n"; else if (inX && near(cy, q.b)) h += "s";
    if (inY && near(cx, q.l)) h += "w"; else if (inY && near(cx, q.r)) h += "e";
    return h;
  }

  // ---- values and controls
  // A reading as shown: an enum's label, else up to 4 significant digits, exponent outside 1e-3 … 1e5.
  function fmt(frame, meta) {
    if (!frame || frame.value === null || frame.value === undefined) return "—";
    const v = frame.value;
    if (meta && meta.enumChoices && typeof v === "number") return meta.enumChoices[v] ?? String(v);
    if (typeof v !== "number") return String(v);
    if (Number.isInteger(v) && Math.abs(v) < 1e7) return String(v);
    const a = Math.abs(v);
    return a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(3) : String(Number(v.toPrecision(4)));
  }

  // A control: enabled on the module's word only (the role named, writable, the module answering, no write in
  // flight). A run-locked role stays enabled: the module refuses it while a run holds the camera, and says so.
  function control(view, entry, busy) {
    if (!view || !view.ok) return { enabled: false, why: "the module cannot say" };
    if (!entry) return { enabled: false, why: "the module names no such role for this camera" };
    if (!entry.writable) return { enabled: false, why: "read only (the module's policy)" };
    const why = entry.locked ? "refused while a run holds the camera (the run lock)" : "";
    return { enabled: !busy, why: busy ? "a write is in flight" : why };
  }

  // The image chip: role image_counter's verdict (the image's own freshness, by MJPG1's counter).
  function imageChip(word) {
    const text = word === "live" ? "image live" : word === "stale" ? "image stale" : `image ${word || "unknown"}`;
    return { text, word: word || "unknown" };
  }

  // The device's errors: every role the module calls `error`, by label.
  const alarms = (view) => (!view || !view.ok ? [] : Object.values(view.roles).filter((e) => e.verdict === "error").map((e) => e.label || e.role));

  // An array's first n values (the waveform's valid length, a role of its own); all of them when n is unread.
  const valid = (values, n) => (typeof n === "number" && n >= 0 ? (values || []).slice(0, n) : values || []);

  // A whole number typed into a field, or null for an empty or non-numeric one (which writes nothing).
  function whole(raw) {
    const s = String(raw).trim(), n = Number(s);
    return s === "" || !Number.isFinite(n) ? null : Math.round(n);
  }
  function number(raw) {
    const s = String(raw).trim(), n = Number(s);
    return s === "" || !Number.isFinite(n) ? null : n;
  }

  Object.assign(rules, { MAPS, levels, lut, autoLevels, imageSize, toView, toImage, fitScale, fit, zoomAt, roiToImage, imageToRoi, dragRoi, roiHandle, fmt, control, imageChip, alarms, valid, whole, number });
})(typeof module === "object" ? module.exports : (window.GeecsCameraRules = {}));
