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
  // The scale bends t in 0..1 after the levels; it stretches the values shown, it adds no range.
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

  // The levels as applied at a full scale (255 for the JPEG, the raw frame's): black in 0..full-1, white above it.
  function levels(blackRaw, whiteRaw, full = 255) {
    const black = clampLevel(blackRaw, 0, full - 1);
    return { black, white: Math.max(black + 1, clampLevel(whiteRaw, 1, full)) };
  }

  // value 0..full → [r, g, b] x (full + 1), through the levels, the scale, then the map.
  function lut(cmap, scale, blackRaw, whiteRaw, full = 255) {
    const points = MAPS[cmap] || MAPS.grey, bend = SCALES[scale] || SCALES.linear;
    const { black, white } = levels(blackRaw, whiteRaw, full);
    const out = new Array(3 * (full + 1));
    for (let v = 0; v <= full; v++) {
      const c = mapColour(points, bend(Math.min(1, Math.max(0, (v - black) / (white - black)))));
      out[3 * v] = Math.round(c[0]); out[3 * v + 1] = Math.round(c[1]); out[3 * v + 2] = Math.round(c[2]);
    }
    return out;
  }

  // A frame's histogram: full + 1 bins, a value above the full scale counted in the top bin.
  function histogram(lum, full) {
    const hist = new Array(full + 1).fill(0);
    for (let i = 0; i < lum.length; i++) hist[lum[i] > full ? full : lum[i]]++;
    return hist;
  }

  // Auto levels from a histogram of 0..full (full + 1 bins): the 0.1 % and 99.9 % points.
  function autoLevels(hist, full = hist.length - 1) {
    const n = hist.reduce((a, b) => a + b, 0);
    let acc = 0, black = 0, white = full;
    for (let v = 0; v <= full; v++) { acc += hist[v]; if (acc > n * 0.001) { black = v; break; } }
    acc = 0;
    for (let v = 0; v <= full; v++) { acc += hist[v]; if (acc >= n * 0.999) { white = v; break; } }
    return { black, white: white <= black ? Math.min(full, black + 1) : white };
  }

  // ---- levels per mode, remembered as fractions of the mode's full scale, so a stored 8-bit white
  // never saturates the raw view and a 12-bit black never zeroes the JPEG
  const levelsToStore = (black, white, full) => { const l = levels(black, white, full); return { black: l.black / full, white: l.white / full }; };
  const levelsFromStore = (f, full) => (f && Number.isFinite(f.black) && Number.isFinite(f.white) ? levels(f.black * full, f.white * full, full) : { black: 0, white: full });
  // A mode's stored fractions ("jpeg" | "raw"): a store from before the modes held 8-bit counts, the JPEG's.
  function storedLevels(saved, mode) {
    if (!saved) return null;
    if (saved.levels && saved.levels[mode]) return saved.levels[mode];
    return mode === "jpeg" && saved.black !== undefined && saved.white !== undefined ? { black: Number(saved.black) / 255, white: Number(saved.white) / 255 } : null;
  }

  // The image in image pixels (MJPG1's ArraySize): the IOC's size when it agrees with
  // the JPEG's shape (ffmpegServer scales a large frame by ONE factor, each side
  // rounded), else the JPEG's own. The JPEG is stretched over it.
  function imageSize(fw, fh, trueW, trueH) {
    if (!trueW || !trueH) return [fw, fh];
    const ok = (s) => s <= 1 + 1e-9 && Math.abs(trueW * s - fw) <= 1.5 && Math.abs(trueH * s - fh) <= 1.5;
    return ok(fw / trueW) || ok(fh / trueH) ? [trueW, trueH] : [fw, fh];
  }

  // ---- the raw frame (/api/devices/<name>/frame.raw): ARR's frame, the driver's, before TRANS1
  const RAW_FULL_ASSUMED = 4095;
  // TRANS1:Type, in the IOC's order; each [swap, flip x, flip y] of NDPluginTransform's mapping
  // (a source pixel (x, y) of a w×h frame lands at (u, v) = swap ? (y, x) : (x, y), then mirrored on
  // the output's axes): Rot90 is (h-1-y, x), Rot270 (y, w-1-x), Mirror (w-1-x, y), Rot90Mirror (y, x).
  const TRANSFORMS = {
    None: [false, false, false], Rot90: [true, true, false], Rot180: [false, true, true], Rot270: [true, false, true],
    Mirror: [false, true, false], Rot90Mirror: [true, false, false], Rot180Mirror: [false, false, true], Rot270Mirror: [true, true, true],
  };

  // The transform role's value as a name (an enum's index through its choices), and whether the page knows it.
  function transformOf(frame, meta) {
    const v = frame ? frame.value : undefined;
    const name = typeof v === "number" && meta && meta.enumChoices ? meta.enumChoices[v] : typeof v === "string" ? v : undefined;
    if (name !== undefined && Object.prototype.hasOwnProperty.call(TRANSFORMS, name)) return { name, known: true, word: "" };
    const word = v === undefined || v === null ? "transform unread" : `transform ${JSON.stringify(name ?? v)} unknown`;
    return { name: null, known: false, word: `${word}: the driver's frame as read, overlays hidden` };
  }

  // A w×h frame transformed as TRANS1 would (null for a name it does not have): {data, width, height}.
  function transformFrame(src, w, h, name) {
    const t = TRANSFORMS[name];
    if (!t) return null;
    const [swap, fx, fy] = t, ow = swap ? h : w, oh = swap ? w : h;
    if (!swap && !fx && !fy) return { data: src, width: w, height: h };
    const out = new src.constructor(w * h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let u = swap ? y : x, v = swap ? x : y;
        if (fx) u = ow - 1 - u;
        if (fy) v = oh - 1 - v;
        out[v * ow + u] = src[y * w + x];
      }
    }
    return { data: out, width: ow, height: oh };
  }

  // A 200's body (uint16 little-endian, row-major) and its headers (get(name) → string | null) as a frame.
  function decodeRaw(buffer, get) {
    const w = Number(get("X-Frame-Width")), h = Number(get("X-Frame-Height"));
    if (!(Number.isInteger(w) && w > 0 && Number.isInteger(h) && h > 0)) throw new Error("the frame came without its geometry");
    if (buffer.byteLength !== 2 * w * h) throw new Error(`geometry disagrees: ${buffer.byteLength / 2} values for ${w}×${h}`);
    const fs = Number(get("X-Frame-Full-Scale")), assumed = !(get("X-Frame-Full-Scale") && Number.isInteger(fs) && fs > 0);
    return { data: new Uint16Array(buffer), width: w, height: h, full: assumed ? RAW_FULL_ASSUMED : fs, assumed, tag: get("X-Frame-Tag"), etag: get("ETag") };
  }

  // One poll of frame.raw: {fresh: true, frame} for a new frame, {fresh: false} for a 304 (nothing new,
  // never an error); any other answer throws its detail. fetchFn is the page's fetch (a fake one under node).
  async function fetchRaw(fetchFn, url, etag, signal) {
    const r = await fetchFn(url, { cache: "no-store", signal, headers: etag ? { "If-None-Match": etag } : {} });
    if (r.status === 304) return { fresh: false };
    if (!r.ok) { let d = `answered ${r.status}`; try { d = (await r.json()).detail || d; } catch (_) { /* not JSON */ } throw new Error(d); }
    return { fresh: true, frame: decodeRaw(await r.arrayBuffer(), (k) => r.headers.get(k)) };
  }

  // The words the Raw toggle sets: the toggle's own, the levels' hint and full scale, and whether
  // the module's image chip (MJPG1's counter) is shown.
  function rawToggle(entry) {
    return entry ? { enabled: true, why: "the camera's 16-bit frame, before the transform" } : { enabled: false, why: "the module names no raw frame for this camera" };
  }
  function modeWords(raw, full, assumed) {
    if (!raw) return { hint: "The 8-bit stream, stretched in this browser: nothing is written to the camera.", scale: "", chip: true, full: 255 };
    return { hint: "The camera's 16-bit frame, levels applied in this browser; nothing is written to the camera.", scale: assumed ? `counts of ${full}, full scale assumed` : `counts of ${full}`, chip: false, full };
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
  // The roles of a write list the module marks as hazards (its device route's `confirm: "hazard"`, the policy's):
  // the page asks once for the whole list before sending any of it, so a refusal never half-applies it.
  const hazards = (pairs, entries) => pairs.map(([role]) => role).filter((role) => !!entries[role] && entries[role].confirm === "hazard");
  // A plot whose STAT computation is off (its flag read 0) is not drawn: STAT keeps the last array it computed.
  const notComputed = (on, what) => (on === 0 ? `${what} not computed` : null);
  // One write list at a time for the whole page (the rail and the image's edits): a call while one runs
  // is refused (resolves null) without starting; busy() says whether one runs.
  const single = (run) => {
    let running = false;
    const call = async (...args) => {
      if (running) return null;
      running = true;
      try { return await run(...args); } finally { running = false; }
    };
    call.busy = () => running;
    return call;
  };
  const valid = (values, n) => (typeof n === "number" && n >= 0 ? (values || []).slice(0, n) : values || []);

  // A pointer gesture commits its edit only when it completes: a cancelled one
  // (the browser or OS took the pointer) discards the preview and writes nothing.
  const gestureEnds = (eventType) => (eventType === "pointerup" ? "commit" : "discard");

  // work(signal) under a deadline covering the whole exchange, the body's read
  // included: past ms the request is aborted and the promise rejects "timed out",
  // so a stalled connection to this service never holds a loop open.
  const within = (ms, work) => {
    const ctl = new AbortController();
    let timer;
    const deadline = new Promise((_, reject) => { timer = setTimeout(() => { ctl.abort(); reject(new Error("timed out")); }, ms); });
    return Promise.race([work(ctl.signal), deadline]).finally(() => clearTimeout(timer));
  };

  // A whole number typed into a field, or null for an empty or non-numeric one (which writes nothing).
  function whole(raw) {
    const s = String(raw).trim(), n = Number(s);
    return s === "" || !Number.isFinite(n) ? null : Math.round(n);
  }
  function number(raw) {
    const s = String(raw).trim(), n = Number(s);
    return s === "" || !Number.isFinite(n) ? null : n;
  }

  Object.assign(rules, { MAPS, TRANSFORMS, RAW_FULL_ASSUMED, levels, lut, histogram, autoLevels, levelsToStore, levelsFromStore, storedLevels, transformOf, transformFrame, decodeRaw, fetchRaw, rawToggle, modeWords, imageSize, toView, toImage, fitScale, fit, zoomAt, roiToImage, imageToRoi, dragRoi, roiHandle, fmt, control, imageChip, alarms, hazards, notComputed, single, valid, whole, number, gestureEnds, within });
})(typeof module === "object" ? module.exports : (window.GeecsCameraRules = {}));
