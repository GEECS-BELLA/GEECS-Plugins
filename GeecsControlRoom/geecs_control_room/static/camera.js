// The camera page's viewer, DOM only: the frame loop, zoom and pan, this
// viewer's colour map and levels, the crosshairs and analysis ROI drawn and
// dragged over the image, and the live readings. Moved from GeecsPanels and
// adapted: the PVs and words are the module's (window.GeecsDevicePage, from
// device.js), the values Hare's, every write the module's put route.
//
// Frames are PULLED one at a time (never more than one in flight); the loop
// pauses while the tab is hidden, asks with If-None-Match the held ETag (MJPG1's
// counter: a 304 is nothing new), and a frame whose X-Frame-Tag is unchanged
// is not decoded again. Coordinates: image pixels are MJPG1's (roles width and
// height; the JPEG is stretched over them); view pixels are CSS pixels in the
// viewport. With Raw on, the driver's 16-bit frame (frame.raw) is polled at
// 2 Hz instead, rotated / flipped by role `transform` as TRANS1 would, so
// every overlay keeps MJPEG mode's coordinates; levels are per mode. The rules
// are camera_rules.js's (tested under node).
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsCameraRules, S = window.GeecsScreenRules;
  const box = document.getElementById("camera");
  if (!P || !box) return;
  const $ = (id) => document.getElementById(id);
  const KIT = JSON.parse($("kit-state").textContent);
  const value = (role) => { const f = P.frame(role); return f ? f.value : undefined; };
  const MAX_FPS = 10, HOLD_MS = 2000, GRAB_PX = 9, EDGE_PX = 6;
  const FRAME_TIMEOUT_MS = 5000; // a frame request past this is abandoned: the loop never waits on a stalled connection
  const storeKey = `geecs-control-room:${P.name}:display`;

  const viewport = $("viewport"), canvas = $("image"), ctx = canvas.getContext("2d");
  const frame = document.createElement("canvas"), fctx = frame.getContext("2d", { willReadFrequently: true });
  const cmapSel = $("cmap"), scaleSel = $("scale"), blackIn = $("level-lo"), whiteIn = $("level-hi"), autoLive = $("auto-live"), rawIn = $("raw-mode");
  // lum: the frame's values (8-bit from the JPEG, 16-bit raw); full: the mode's full scale (255, or the raw frame's)
  let lut = [], lum = null, fw = 0, fh = 0, iw = 0, ih = 0, full = 255;
  let view = { scale: 1, ox: 0, oy: 0, moved: false }, vw = 0, vh = 0;
  // raw: the mode (MJPEG by default, never remembered); gen: bumped on every toggle, so an answer of the other mode is dropped
  let raw = false, gen = 0, rawState = { transform: null }; // the drawn raw frame's transform and full scale (none before the first)
  rawIn.checked = false; // a browser may restore the box on reload; the page starts in MJPEG, so the box says so
  const mode = () => (raw ? "raw" : "mjpeg");
  const transformNow = () => R.transformName(P.frame("transform"), P.meta("transform"));
  const words = () => R.modeWords(raw, rawState);
  const overlaysOn = () => words().overlays;

  // ---- this viewer's display, remembered per camera; the levels per mode, as fractions of its full scale
  let stored = {};
  const buildLut = () => { lut = R.lut(cmapSel.value, scaleSel.value, blackIn.value, whiteIn.value, full); };
  function save() {
    stored = { cmap: cmapSel.value, scale: scaleSel.value, auto: autoLive.checked, levels: R.storeLevels(stored, mode(), blackIn.value, whiteIn.value, full) };
    try { localStorage.setItem(storeKey, JSON.stringify(stored)); } catch (_) { /* no storage */ }
  }
  // the mode's stored levels into the inputs, whose range is the mode's full scale
  function showLevels() {
    const l = R.storedLevels(stored, mode(), full);
    blackIn.max = full - 1; whiteIn.max = full;
    blackIn.value = l.black; whiteIn.value = l.white;
    buildLut();
  }
  try {
    const s = JSON.parse(localStorage.getItem(storeKey) || "null");
    if (s) { stored = s; if (R.MAPS[s.cmap]) cmapSel.value = s.cmap; scaleSel.value = s.scale || "linear"; autoLive.checked = !!s.auto; }
  } catch (_) { /* nothing remembered */ }
  showLevels();

  function autoLevels() {
    if (!lum) return;
    // raw: the held frame's histogram (counted once per frame in takeRaw); MJPEG: the JPEG's, counted here
    const l = R.autoLevels(raw && rawHist ? rawHist.values : R.histogram(lum, full), full);
    blackIn.value = l.black; whiteIn.value = l.white;
    buildLut();
  }

  function recolour() {
    if (!lum) return;
    const img = fctx.createImageData(fw, fh), px = img.data;
    for (let i = 0, j = 0; i < lum.length; i++, j += 4) {
      const v = lum[i], k = 3 * (v > full ? full : v); // a raw value past the full scale is drawn as the full scale
      px[j] = lut[k]; px[j + 1] = lut[k + 1]; px[j + 2] = lut[k + 2]; px[j + 3] = 255;
    }
    fctx.putImageData(img, 0, 0);
    draw();
  }

  function takeFrame(bitmap) {
    if (bitmap.width !== fw || bitmap.height !== fh) { fw = frame.width = bitmap.width; fh = frame.height = bitmap.height; }
    fctx.drawImage(bitmap, 0, 0);
    bitmap.close();
    const px = fctx.getImageData(0, 0, fw, fh).data;
    if (!lum || lum.length !== fw * fh) lum = new Uint8Array(fw * fh);
    for (let i = 0, j = 0; i < lum.length; i++, j += 4) lum[i] = px[j];
    if (autoLive.checked) autoLevels();
    sized();
    recolour();
  }

  // The driver's frame, rotated / flipped as TRANS1 would (an unknown transform: as is, the overlays hidden).
  // rawHist: the drawn raw frame's histogram, full + 1 bins of its transformed values (the plot's and auto-levels')
  let heldRaw = null, rawHist = null;
  function takeRaw(f) {
    heldRaw = f;
    const name = transformNow();
    const t = R.transformFrame(name, f.data, f.width, f.height) || { data: f.data, width: f.width, height: f.height };
    rawState = { transform: name, full: f.full, assumed: f.assumed };
    if (f.full !== full) { full = f.full; showLevels(); }
    if (t.width !== fw || t.height !== fh) { fw = frame.width = t.width; fh = frame.height = t.height; }
    lum = t.data;
    rawHist = { values: R.histogram(lum, full), full };
    if (autoLive.checked) autoLevels();
    sized();
    recolour();
    sayMode();
  }

  // The image's size from the module's roles (MJPG1 ArraySize), checked against the JPEG's shape; raw: the frame's own.
  function sized() {
    if (!lum) return; // between modes: the last frame's geometry holds, so a toggle keeps the view
    const [nw, nh] = R.imageSize(fw, fh, value("width"), value("height"), raw);
    if (nw !== iw || nh !== ih) { iw = nw; ih = nh; if (!view.moved) fit(); }
  }

  // ---- the viewer
  const toView = (x, y) => R.toView(view, x, y), toImage = (x, y) => R.toImage(view, x, y);
  const fit = () => { view = R.fit(vw, vh, iw, ih); draw(); };
  const zoomAt = (cx, cy, factor) => { view = R.zoomAt(view, cx, cy, factor, R.fitScale(vw, vh, iw, ih)); draw(); };
  const layers = [];
  function draw() {
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!fw) return;
    ctx.setTransform(dpr * view.scale, 0, 0, dpr * view.scale, dpr * view.ox, dpr * view.oy);
    ctx.imageSmoothingEnabled = view.scale * dpr < 1; // magnified: square pixels
    ctx.drawImage(frame, 0, 0, iw, ih);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (overlaysOn()) for (const layer of layers) layer(ctx);
    $("zoom-level").textContent = Math.round(view.scale * 100) + "%";
  }
  function resize() {
    const dpr = window.devicePixelRatio || 1, r = viewport.getBoundingClientRect();
    vw = r.width; vh = r.height;
    canvas.width = Math.round(vw * dpr); canvas.height = Math.round(vh * dpr);
    if (!view.moved) fit(); else draw();
  }

  // ---- the layer: crosshairs (stored colour) and the analysis ROI (accent), this viewer's edit held until stored
  const token = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  const held = { xh: {}, roi: null };
  const can = (role) => R.control(P.view(), P.entry(role), write.busy()).enabled;
  const hex2 = (v) => Math.max(0, Math.min(255, Number(v) | 0)).toString(16).padStart(2, "0");
  function crosshair(n) {
    const g = (f) => value(`crosshair${n}_${f}`);
    const c = { n, use: !!g("use"), x: g("x"), y: g("y"), sx: g("sx") || 0, sy: g("sy") || 0, name: g("name") || String(n) };
    const h = held.xh[n];
    if (h && (h.dragging || (performance.now() < h.until && (h.x !== c.x || h.y !== c.y)))) return { ...c, x: h.x, y: h.y };
    delete held.xh[n];
    return { ...c, colour: `#${hex2(g("r"))}${hex2(g("g"))}${hex2(g("b"))}` };
  }
  function roi() {
    const b = { minx: value("roi_minx"), miny: value("roi_miny"), sizex: value("roi_sizex"), sizey: value("roi_sizey") };
    const h = held.roi, same = h && ["minx", "miny", "sizex", "sizey"].every((k) => h[k] === b[k]);
    if (h && (h.dragging || (performance.now() < h.until && !same))) return h;
    held.roi = null;
    return [b.minx, b.miny, b.sizex, b.sizey].every((v) => typeof v === "number") ? b : null;
  }
  function roiRect() {
    const b = roi();
    if (!b) return null;
    const [l, t] = toView(b.minx, b.miny), [r, bt] = toView(b.minx + b.sizex, b.miny + b.sizey);
    return { l, t, r, b: bt };
  }
  layers.push((c) => {
    const q = $("roi-show").checked && roiRect();
    if (q) {
      c.save(); c.strokeStyle = token("--accent"); c.lineWidth = 1.5; c.setLineDash([6, 4]);
      c.strokeRect(q.l, q.t, q.r - q.l, q.b - q.t); c.restore();
    }
    for (const n of [1, 2]) {
      const x = crosshair(n);
      if (!x.use || typeof x.x !== "number" || typeof x.y !== "number") continue;
      const [vx, vy] = toView(x.x + 0.5, x.y + 0.5);
      const ax = Math.max(12, (x.sx / 2) * view.scale), ay = Math.max(12, (x.sy / 2) * view.scale);
      c.save(); c.lineWidth = 1.5; c.strokeStyle = c.fillStyle = x.colour || token("--accent");
      c.beginPath(); c.moveTo(vx - ax, vy); c.lineTo(vx + ax, vy); c.moveTo(vx, vy - ay); c.lineTo(vx, vy + ay);
      c.arc(vx, vy, 5, 0, 2 * Math.PI); c.stroke();
      c.font = "11px " + (token("--ff-mono") || "monospace"); c.fillText(x.name, vx + 7, vy - 7); c.restore();
    }
  });
  function crosshairAt(cx, cy) {
    if (!overlaysOn()) return null; // hidden overlays are never dragged
    for (const n of [1, 2]) {
      const x = crosshair(n);
      if (!x.use || typeof x.x !== "number" || !can(`crosshair${n}_x`)) continue;
      const [vx, vy] = toView(x.x + 0.5, x.y + 0.5);
      if (Math.hypot(cx - vx, cy - vy) <= GRAB_PX) return x;
    }
    return null;
  }
  const roiHandleAt = (cx, cy) => { const q = overlaysOn() && $("roi-show").checked && can("roi_minx") && roiRect(); return q ? R.roiHandle(q, cx, cy, GRAB_PX, EDGE_PX) : ""; };
  const CURSORS = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize" };

  function grab(cx, cy) {
    if (!fw) return null;
    const x = crosshairAt(cx, cy);
    if (x) {
      const h = held.xh[x.n] = { x: x.x, y: x.y, dragging: true, until: 0 };
      return {
        move(ix, iy) { h.x = Math.floor(ix); h.y = Math.floor(iy); draw(); },
        end() { h.dragging = false; h.until = performance.now() + HOLD_MS; W.write([[`crosshair${x.n}_x`, h.x], [`crosshair${x.n}_y`, h.y]]); },
        cancel() { delete held.xh[x.n]; draw(); },
      };
    }
    const handle = roiHandleAt(cx, cy);
    if (!handle) return null;
    const start = roi();
    const h = held.roi = { ...start, dragging: true, until: 0 };
    return {
      move(ix, iy) { Object.assign(h, R.dragRoi(start, handle, ix, iy, iw, ih)); draw(); },
      end() { h.dragging = false; h.until = performance.now() + HOLD_MS; W.write(["minx", "miny", "sizex", "sizey"].filter((k) => h[k] !== start[k]).map((k) => [`roi_${k}`, h[k]])); },
      cancel() { held.roi = null; draw(); },
    };
  }

  // ---- pointer: a layer's edit, else a pan; a short press released in place is a click (after a double-click is ruled out)
  let drag = null, edit = null, clickTimer = null;
  const clicks = [];
  const local = (ev) => { const r = viewport.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };
  viewport.addEventListener("wheel", (ev) => { ev.preventDefault(); const [cx, cy] = local(ev); zoomAt(cx, cy, Math.exp(-ev.deltaY * 0.0015)); }, { passive: false });
  viewport.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    const [cx, cy] = local(ev);
    viewport.setPointerCapture(ev.pointerId);
    edit = grab(cx, cy);
    if (!edit) { drag = { x: ev.clientX, y: ev.clientY, ox: view.ox, oy: view.oy, cx, cy, t: performance.now() }; viewport.classList.add("panning"); }
  });
  viewport.addEventListener("pointermove", (ev) => {
    const [cx, cy] = local(ev);
    showCursor(cx, cy);
    if (edit) return edit.move(...toImage(cx, cy));
    if (!drag) { viewport.style.cursor = crosshairAt(cx, cy) ? "move" : CURSORS[roiHandleAt(cx, cy)] || ""; return; }
    view = { ...view, ox: drag.ox + ev.clientX - drag.x, oy: drag.oy + ev.clientY - drag.y, moved: true };
    draw();
  });
  const endDrag = (ev) => {
    if (edit) { (R.gestureEnds(ev.type) === "commit" ? edit.end : edit.cancel)(); edit = null; } // a cancelled gesture writes nothing
    if (drag && overlaysOn() && ev.type === "pointerup" && Math.hypot(ev.clientX - drag.x, ev.clientY - drag.y) < 4 && performance.now() - drag.t < 400) {
      const [ix, iy] = toImage(drag.cx, drag.cy).map(Math.floor);
      clearTimeout(clickTimer);
      clickTimer = setTimeout(() => clicks.forEach((f) => f(ix, iy)), 500);
    }
    drag = null;
    viewport.classList.remove("panning");
  };
  viewport.addEventListener("pointerup", endDrag);
  viewport.addEventListener("pointercancel", endDrag);
  viewport.addEventListener("pointerleave", () => { $("cursor").textContent = ""; });
  viewport.addEventListener("dblclick", () => { clearTimeout(clickTimer); fit(); });
  $("zoom-fit").addEventListener("click", fit);
  $("zoom-one").addEventListener("click", () => zoomAt(vw / 2, vh / 2, 1 / view.scale));
  new ResizeObserver(resize).observe(viewport);

  function showCursor(cx, cy) {
    if (!lum) return;
    const [ix, iy] = toImage(cx, cy).map(Math.floor);
    if (ix < 0 || iy < 0 || ix >= iw || iy >= ih) { $("cursor").textContent = ""; return; }
    const jx = Math.min(fw - 1, Math.floor((ix * fw) / iw)), jy = Math.min(fh - 1, Math.floor((iy * fh) / ih));
    $("cursor").textContent = `x ${ix}  y ${iy}  ${words().value} ${lum[jy * fw + jx]}`;
  }

  // ---- the frame loop: one request in flight, latest wins, paused while hidden
  const shown = [];
  // lastTag: the drawn JPEG's X-Frame-Tag; jpegEtag: the last answer's ETag (none while the counter is unread)
  let lastTag = null, jpegEtag = null, waiting = null, drawing = false;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const visible = () => new Promise((r) => {
    if (!document.hidden) return r();
    document.addEventListener("visibilitychange", function on() { if (!document.hidden) { document.removeEventListener("visibilitychange", on); r(); } });
  });
  async function drawLatest() {
    if (drawing) return;
    drawing = true;
    try {
      while (waiting) {
        const blob = waiting, g = gen;
        waiting = null;
        const bitmap = await createImageBitmap(blob);
        if (g !== gen) { bitmap.close(); continue; } // toggled while decoding: not this mode's frame
        takeFrame(bitmap); $("image-error").hidden = true; shown.push(performance.now());
      }
    } catch (e) {
      $("image-error-text").textContent = "the frame could not be decoded: " + (e.message || e);
      $("image-error").hidden = false;
      lastTag = null; jpegEtag = null; // the next ask fetches the bytes again, never a 304 with nothing drawn
    } finally { drawing = false; }
  }
  // the two loops wait out the other mode; a toggle wakes them
  const toggled = [];
  const untilToggle = () => new Promise((r) => toggled.push(r));
  const failed = (e) => { $("image-error-text").textContent = String(e.message || e); $("image-error").hidden = false; };
  async function loop() {
    for (;;) {
      while (raw) await untilToggle();
      await visible();
      const t0 = performance.now(), g = gen;
      try {
        const got = await R.within(FRAME_TIMEOUT_MS, async (signal) => {
          const r = await fetch(box.dataset.frameUrl, { cache: "no-store", headers: R.ifNoneMatch(jpegEtag), signal });
          const answer = R.rawAnswer(r.status);
          if (answer === "same") return null;
          if (answer === "error") { let d = `answered ${r.status}`; try { d = (await r.json()).detail || d; } catch (_) { /* not JSON */ } throw new Error(d); }
          return [r.headers.get("ETag"), r.headers.get("X-Frame-Tag"), await r.blob()];
        });
        if (g !== gen) continue; // toggled while asking
        $("image-error").hidden = true; // a good answer ends a transport error, new frame or not (a decode failure says so again)
        if (got) {
          const [e, tag, blob] = got;
          jpegEtag = e;
          if (R.jpegIsNew(tag, lastTag)) { lastTag = tag; waiting = blob; drawLatest(); }
        }
      } catch (e) {
        if (g !== gen) continue;
        jpegEtag = null; // the next ask is a full one
        failed(e);
        await sleep(1000);
      }
      await sleep(Math.max(0, 1000 / MAX_FPS - (performance.now() - t0)));
    }
  }

  // ---- the raw loop: 2 Hz, If-None-Match the held ETag; a 304 is "nothing new", never the error branch
  const RAW_PERIOD_MS = 500, RAW_TIMEOUT_MS = 5000;
  let etag = null, lastRawEtag = null;
  async function rawLoop() {
    for (;;) {
      while (!raw) await untilToggle();
      await visible();
      const t0 = performance.now(), g = gen;
      try {
        const got = await R.within(RAW_TIMEOUT_MS, async (signal) => {
          const r = await fetch(box.dataset.rawUrl, { cache: "no-store", headers: R.ifNoneMatch(etag), signal });
          const answer = R.rawAnswer(r.status);
          if (answer === "same") return null;
          if (answer === "error") { let d = `answered ${r.status}`; try { d = (await r.json()).detail || d; } catch (_) { /* not JSON */ } throw new Error(d); }
          const h = (k) => r.headers.get(k);
          const f = R.decodeRaw(await r.arrayBuffer(), { width: h("X-Frame-Width"), height: h("X-Frame-Height"), fullScale: h("X-Frame-Full-Scale"), tag: h("X-Frame-Tag") });
          if (!f.ok) throw new Error(f.why);
          return [h("ETag"), f];
        });
        if (g !== gen) continue;
        $("image-error").hidden = true;
        if (got) {
          etag = got[0];
          const f = got[1];
          // deduped on the representation: the same tag with a full scale newly read is drawn again at that scale
          if (R.rawIsNew(etag, lastRawEtag, f.tag)) {
            const moved = f.tag === null || !heldRaw || f.tag !== heldRaw.tag;
            lastRawEtag = etag; takeRaw(f);
            if (moved) shown.push(performance.now()); // the rate counts new frames, not a scale landing
          }
        }
      } catch (e) {
        if (g !== gen) continue;
        etag = null; // the next ask is a full one
        failed(e);
        await sleep(1000);
      }
      await sleep(Math.max(0, RAW_PERIOD_MS - (performance.now() - t0)));
    }
  }

  // The mode's words: the note beside the toggle, the levels' hint, MJPG1's counter chip shown in MJPEG mode only.
  function sayMode() {
    const w = words();
    $("raw-note").textContent = w.note;
    $("raw-note").dataset.state = w.overlays ? "" : "degraded";
    $("levels-hint").textContent = w.hint;
    $("image-state").hidden = !w.counterChip;
  }
  rawIn.addEventListener("change", () => {
    save(); // the leaving mode's levels
    raw = rawIn.checked;
    gen++;
    // nothing of the other mode is drawn again: its lum goes, its last picture stays until this mode's first frame
    lum = null; waiting = null; heldRaw = null; rawHist = null; lastTag = null; jpegEtag = null; etag = null; lastRawEtag = null; shown.length = 0;
    rawState = { ...rawState, transform: transformNow() };
    full = raw ? rawState.full || R.ASSUMED_FULL : 255;
    showLevels();
    sayMode();
    draw();
    $("image-error").hidden = true;
    toggled.splice(0).forEach((r) => r());
    modes.forEach((f) => f()); // the plots drawn from this mode's histogram now, not at their next poll
  });
  // the rate ages on its own clock, so a request still waiting reads as frames stopping, not as the last rate
  setInterval(() => {
    while (shown.length && shown[0] < performance.now() - 2000) shown.shift();
    $("fps").textContent = `${(shown.length / 2).toFixed(1)} new frames/s`;
  }, 500);

  // ---- the readings, the image chip, the device's errors: the module's words, Hare's values
  let queued = false;
  function refresh() {
    queued = false;
    for (const el of box.querySelectorAll(".live[data-role]")) {
      const role = el.dataset.role, word = P.entry(role) ? P.word(role) : "not served";
      el.querySelector(".v").textContent = R.fmt(P.frame(role), P.meta(role));
      el.querySelector(".word").textContent = word;
      el.querySelector(".dot").dataset.state = KIT[word] || "unknown";
    }
    // the transform changed under a held raw frame: drawn again with the new one (before the first, only said)
    if (raw && heldRaw && transformNow() !== rawState.transform) takeRaw(heldRaw);
    else if (raw && !heldRaw) rawState = { ...rawState, transform: transformNow() };
    sayMode();
    const chip = R.imageChip(P.entry("image_counter") ? P.word("image_counter") : "unknown");
    $("image-state").textContent = chip.text;
    $("image-state").dataset.state = KIT[chip.word] || "unknown";
    const bad = R.alarms(P.view());
    $("alarms").textContent = bad.length ? `in error: ${bad.join(", ")}` : "";
    sized();
    draw();
    hooks.forEach((f) => f());
  }
  const hooks = [], modes = [];
  P.on(() => { if (!queued) { queued = true; requestAnimationFrame(refresh); } });

  // ---- wiring; what camera_controls.js uses
  buildLut();
  for (const el of [cmapSel, scaleSel, blackIn, whiteIn]) el.addEventListener("change", () => { buildLut(); recolour(); save(); });
  autoLive.addEventListener("change", save);
  $("auto-once").addEventListener("click", () => { autoLevels(); recolour(); save(); });
  $("reset-levels").addEventListener("click", () => { blackIn.value = 0; whiteIn.value = full; scaleSel.value = "linear"; autoLive.checked = false; buildLut(); recolour(); save(); });
  $("roi-show").addEventListener("change", draw);

  // ---- writes: device_writes.js (the module's put route, sent once, a hazard asked first); the rail drawn
  // disabled while a list runs, and a refusal drops this viewer's held edits (the image snaps back to the readbacks)
  const write = window.GeecsDeviceWrites.make(P, () => hooks.forEach((f) => f()), () => { held.xh = {}; held.roi = null; draw(); });
  const writeOrSay = write.write, hazardsOf = write.hazardsOf;
  const W = window.GeecsCameraView = { write: writeOrSay, busy: write.busy, hazardsOf, value, hooks, modes, clicks, layers, toView, redraw: draw, rawHistogram: () => (raw && heldRaw && rawHist ? rawHist : null), frameSize: () => (overlaysOn() ? [iw, ih] : [0, 0]) }; // an unknown transform's geometry is not the ROI's
  resize();
  sayMode();
  loop();
  rawLoop();
})();
