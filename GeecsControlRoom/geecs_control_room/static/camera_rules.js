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

  // The levels as applied, in counts of the mode's full scale (255 for the JPEG): black below it, white above black.
  function levels(blackRaw, whiteRaw, full = 255) {
    const black = clampLevel(blackRaw, 0, full - 1);
    return { black, white: Math.max(black + 1, clampLevel(whiteRaw, 1, full)) };
  }

  // value 0..full → [r, g, b] x (full + 1), through the levels, the scale, then the map.
  function lut(cmap, scale, blackRaw, whiteRaw, full = 255) {
    const points = MAPS[cmap] || MAPS.grey, bend = SCALES[scale] || SCALES.linear;
    const { black, white } = levels(blackRaw, whiteRaw, full);
    const out = [];
    for (let v = 0; v <= full; v++) {
      const t = bend(Math.min(1, Math.max(0, (v - black) / (white - black))));
      out.push(...mapColour(points, t).map(Math.round));
    }
    return out;
  }

  // The frame's values counted in full + 1 bins; a value above the full scale lands in the top bin.
  function histogram(values, full) {
    const hist = new Array(full + 1).fill(0);
    for (let i = 0; i < values.length; i++) { const v = values[i]; hist[v > full ? full : v]++; }
    return hist;
  }

  // Auto levels from a full + 1 bin histogram (0..full): the 0.1 % and 99.9 % points.
  function autoLevels(hist, full = hist.length - 1) {
    const n = hist.reduce((a, b) => a + b, 0);
    let acc = 0, black = 0, white = full;
    for (let v = 0; v <= full; v++) { acc += hist[v]; if (acc > n * 0.001) { black = v; break; } }
    acc = 0;
    for (let v = 0; v <= full; v++) { acc += hist[v]; if (acc >= n * 0.999) { white = v; break; } }
    return { black, white: white <= black ? Math.min(full, black + 1) : white };
  }

  // ---- levels per mode, stored as fractions of the mode's full scale: a stored 8-bit white never
  // saturates the raw view, a 12-bit black never zeroes the JPEG. `stored` is the page's display record.
  function storedLevels(stored, mode, full) {
    const f = stored && stored.levels && stored.levels[mode];
    if (f && Number.isFinite(f.black) && Number.isFinite(f.white)) return levels(f.black * full, f.white * full, full);
    // before levels were per mode: the JPEG's own black and white, in counts of 255
    if (mode === "mjpeg" && stored && stored.black !== undefined) return levels(stored.black, stored.white ?? 255, 255);
    return { black: 0, white: full };
  }
  function storeLevels(stored, mode, black, white, full) {
    const l = levels(black, white, full), was = (stored && stored.levels) || {};
    return { ...was, [mode]: { black: l.black / full, white: l.white / full } };
  }

  // ---- the raw frame (/api/devices/<name>/frame.raw: uint16 LE row-major, geometry, full scale and tag in headers)
  const ASSUMED_FULL = 4095;
  // What a poll's answer means, raw or JPEG: 200 a frame, 304 nothing new (never an error), anything else the error.
  const rawAnswer = (status) => (status === 200 ? "frame" : status === 304 ? "same" : "error");

  // The body (uint16 LE, row-major, height rows of width, gzip undone by fetch) and its headers, by name.
  function decodeRaw(buffer, h) {
    const w = Number(h.width), ht = Number(h.height), fs = Number(h.fullScale);
    if (!(Number.isInteger(w) && w > 0 && Number.isInteger(ht) && ht > 0)) return { ok: false, why: "the frame came without its geometry" };
    if (buffer.byteLength !== w * ht * 2) return { ok: false, why: `the frame is ${buffer.byteLength} bytes for ${w}×${ht}` };
    const assumed = !(h.fullScale !== null && h.fullScale !== undefined && Number.isInteger(fs) && fs > 0 && fs <= 65535);
    return { ok: true, data: new Uint16Array(buffer), width: w, height: ht, full: assumed ? ASSUMED_FULL : fs, assumed, tag: h.tag ?? null };
  }

  // TRANS1:Type → where the driver's pixel (x, y) of a w×h frame lands, as ADCore's NDPluginTransform
  // writes a mono frame (transformNDArray): Rot90 is clockwise, Mirror flips left-right.
  const TRANSFORMS = {
    None: (x, y) => [x, y],
    Rot90: (x, y, w, h) => [h - 1 - y, x],
    Rot180: (x, y, w, h) => [w - 1 - x, h - 1 - y],
    Rot270: (x, y, w) => [y, w - 1 - x],
    Mirror: (x, y, w) => [w - 1 - x, y],
    Rot90Mirror: (x, y) => [y, x],
    Rot180Mirror: (x, y, w, h) => [x, h - 1 - y],
    Rot270Mirror: (x, y, w, h) => [h - 1 - y, w - 1 - x],
  };
  const SWAPS = new Set(["Rot90", "Rot270", "Rot90Mirror", "Rot270Mirror"]);
  // The transform role's name from Hare's value: an enum's label, a string as is; null when unread.
  function transformName(frame, meta) {
    const v = frame ? frame.value : undefined;
    if (typeof v === "string") return v;
    return typeof v === "number" && meta && meta.enumChoices && meta.enumChoices[v] !== undefined ? meta.enumChoices[v] : null;
  }
  // The frame as TRANS1 would hand it on (the MJPEG's, the crosshairs' and the ROI's geometry); null for an unknown name.
  function transformFrame(name, data, w, h) {
    const at = Object.prototype.hasOwnProperty.call(TRANSFORMS, name) ? TRANSFORMS[name] : null;
    if (!at) return null;
    if (name === "None") return { data, width: w, height: h };
    const ow = SWAPS.has(name) ? h : w, oh = SWAPS.has(name) ? w : h;
    const idx = (p) => p[1] * ow + p[0];
    // the mapping is affine, so out index = o0 + x·dx + y·dy
    const o0 = idx(at(0, 0, w, h)), dx = idx(at(1, 0, w, h)) - o0, dy = idx(at(0, 1, w, h)) - o0;
    const out = new data.constructor(data.length);
    for (let y = 0, i = 0; y < h; y++) for (let x = 0, o = o0 + y * dy; x < w; x++, i++, o += dx) out[o] = data[i];
    return { data: out, width: ow, height: oh };
  }

  // The page's words for the mode: the source note, the hint, the value word, and whether
  // MJPG1's image_counter chip means anything (in raw mode it does not, and is hidden).
  const HINTS = {
    mjpeg: "The 8-bit stream, stretched in this browser: nothing is written to the camera.",
    raw: "The camera's 16-bit frame, levels applied in this browser; nothing is written to the camera. Raw is the driver's frame: no background subtracted, rotate / flip applied here.",
  };
  function modeWords(raw, s = {}) {
    if (!raw) return { note: "", hint: HINTS.mjpeg, value: "level", counterChip: true, overlays: true };
    const full = s.assumed ? `${ASSUMED_FULL} · full scale assumed` : s.full ? `full scale ${s.full}` : "";
    const known = s.transform !== undefined && Object.prototype.hasOwnProperty.call(TRANSFORMS, s.transform);
    const how = !known ? "rotate / flip unknown: the driver's frame as is, overlays hidden" : s.transform === "None" ? "" : `${s.transform} applied`;
    return { note: [how, full].filter(Boolean).join(" · "), hint: HINTS.raw, value: "counts", counterChip: false, overlays: known };
  }

  // The image in image pixels (MJPG1's ArraySize): the IOC's size when it agrees with
  // the JPEG's shape (ffmpegServer scales a large frame by ONE factor, each side
  // rounded), else the JPEG's own. The JPEG is stretched over it. A raw frame is its true size.
  function imageSize(fw, fh, trueW, trueH, raw = false) {
    if (raw || !trueW || !trueH) return [fw, fh];
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

  // A button pair's state (a family panel's `buttons`, e.g. acquire's Stop / Start): the button whose value the
  // role reads is pressed; unknown, neither.
  const pressed = (value, buttonValue) => typeof value === "number" && value === buttonValue;

  // A control: enabled on the module's word only (the role named, writable, the module answering, no write in
  // flight). A run-locked role stays enabled: the module refuses it while a run holds the device, and says so.
  // Shared: the trace page (trace_rules.js) takes it as is.
  function control(view, entry, busy) {
    if (!view || !view.ok) return { enabled: false, why: "the module cannot say" };
    if (!entry) return { enabled: false, why: "the module names no such role for this device" };
    if (!entry.writable) return { enabled: false, why: "read only (the module's policy)" };
    const why = entry.locked ? "refused while a run holds the device (the run lock)" : "";
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
  // The histogram plot's series: Raw's own histogram of the frame drawn (counts 0..full, bin i centred on i;
  // STAT's histogram_on is not Raw's), else STAT's histogram over hist_min..hist_max, nothing while not computed.
  // lo is bin 0's low edge, step a bin's width: bin i is centred on lo + (i + 0.5) * step; empty is the word
  // said for a series not drawn (false: drawn).
  function histogramPlot(rawHist, iocHist, histogramOn, histMin, histMax) {
    if (rawHist) return { values: rawHist.values, lo: -0.5, step: 1, empty: false };
    const hist = iocHist || [], off = notComputed(histogramOn, "histogram"), lo = histMin || 0;
    return { values: off ? [] : hist, lo, step: ((histMax || hist.length) - lo) / (hist.length || 1), empty: off || false };
  }
  // A raw answer is drawn when its ETag (the representation: the counter and the full scale) is not the one
  // drawn last; an answer with no tag or no ETag is always drawn.
  const rawIsNew = (etag, lastEtag, tag) => tag === null || etag === null || etag !== lastEtag;
  // A JPEG answer is decoded when its X-Frame-Tag (the bytes' checksum) is not the one drawn last; no tag, always.
  const jpegIsNew = (tag, lastTag) => tag === null || tag !== lastTag;
  // A poll's conditional headers: If-None-Match the held ETag, none while nothing is held.
  const ifNoneMatch = (etag) => (etag ? { "If-None-Match": etag } : {});
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

  Object.assign(rules, { MAPS, levels, lut, histogram, autoLevels, storedLevels, storeLevels, ASSUMED_FULL, rawAnswer, decodeRaw, transformName, transformFrame, modeWords, imageSize, toView, toImage, fitScale, fit, zoomAt, roiToImage, imageToRoi, dragRoi, roiHandle, fmt, control, pressed, imageChip, alarms, hazards, notComputed, histogramPlot, rawIsNew, jpegIsNew, ifNoneMatch, single, valid, whole, number, gestureEnds, within });
})(typeof module === "object" ? module.exports : (window.GeecsCameraRules = {}));
