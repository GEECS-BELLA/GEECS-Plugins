// The camera page's viewer, DOM only: the frame loop, zoom and pan, this
// viewer's colour map and levels, the crosshairs and analysis ROI drawn and
// dragged over the image, and the live readings. Moved from GeecsPanels and
// adapted: the PVs and words are the module's (window.GeecsDevicePage, from
// device.js), the values Hare's, every write the module's put route.
//
// Frames are PULLED one at a time (never more than one in flight); the loop
// pauses while the tab is hidden, and a frame whose X-Frame-Tag is unchanged
// is not decoded again. Coordinates: image pixels are MJPG1's (roles width and
// height; the JPEG is stretched over them); view pixels are CSS pixels in the
// viewport. The rules are camera_rules.js's (tested under node).
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsCameraRules, S = window.GeecsScreenRules;
  const box = document.getElementById("camera");
  if (!P || !box) return;
  const $ = (id) => document.getElementById(id);
  const KIT = JSON.parse($("kit-state").textContent);
  const value = (role) => { const f = P.frame(role); return f ? f.value : undefined; };
  const MAX_FPS = 10, HOLD_MS = 2000, GRAB_PX = 9, EDGE_PX = 6;
  const storeKey = `geecs-control-room:${P.name}:display`;

  const viewport = $("viewport"), canvas = $("image"), ctx = canvas.getContext("2d");
  const frame = document.createElement("canvas"), fctx = frame.getContext("2d", { willReadFrequently: true });
  const cmapSel = $("cmap"), scaleSel = $("scale"), blackIn = $("level-lo"), whiteIn = $("level-hi"), autoLive = $("auto-live");
  let lut = [], lum = null, fw = 0, fh = 0, iw = 0, ih = 0;
  let view = { scale: 1, ox: 0, oy: 0, moved: false }, vw = 0, vh = 0;

  // ---- this viewer's display, remembered per camera
  const buildLut = () => { lut = R.lut(cmapSel.value, scaleSel.value, blackIn.value, whiteIn.value); };
  function save() {
    try { localStorage.setItem(storeKey, JSON.stringify({ cmap: cmapSel.value, scale: scaleSel.value, black: blackIn.value, white: whiteIn.value, auto: autoLive.checked })); } catch (_) { /* no storage */ }
  }
  try {
    const s = JSON.parse(localStorage.getItem(storeKey) || "null");
    if (s) { if (R.MAPS[s.cmap]) cmapSel.value = s.cmap; scaleSel.value = s.scale || "linear"; blackIn.value = s.black ?? 0; whiteIn.value = s.white ?? 255; autoLive.checked = !!s.auto; }
  } catch (_) { /* nothing remembered */ }

  function autoLevels() {
    if (!lum) return;
    const hist = new Array(256).fill(0);
    for (let i = 0; i < lum.length; i++) hist[lum[i]]++;
    const l = R.autoLevels(hist);
    blackIn.value = l.black; whiteIn.value = l.white;
    buildLut();
  }

  function recolour() {
    if (!lum) return;
    const img = fctx.createImageData(fw, fh), px = img.data;
    for (let i = 0, j = 0; i < lum.length; i++, j += 4) {
      const k = 3 * lum[i];
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

  // The image's size from the module's roles (MJPG1 ArraySize), checked against the JPEG's shape.
  function sized() {
    const [nw, nh] = R.imageSize(fw, fh, value("width"), value("height"));
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
    for (const layer of layers) layer(ctx);
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
  const can = (role) => R.control(P.view(), P.entry(role), false).enabled;
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
    for (const n of [1, 2]) {
      const x = crosshair(n);
      if (!x.use || typeof x.x !== "number" || !can(`crosshair${n}_x`)) continue;
      const [vx, vy] = toView(x.x + 0.5, x.y + 0.5);
      if (Math.hypot(cx - vx, cy - vy) <= GRAB_PX) return x;
    }
    return null;
  }
  const roiHandleAt = (cx, cy) => { const q = $("roi-show").checked && can("roi_minx") && roiRect(); return q ? R.roiHandle(q, cx, cy, GRAB_PX, EDGE_PX) : ""; };
  const CURSORS = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize" };

  function grab(cx, cy) {
    if (!fw) return null;
    const x = crosshairAt(cx, cy);
    if (x) {
      const h = held.xh[x.n] = { x: x.x, y: x.y, dragging: true, until: 0 };
      return {
        move(ix, iy) { h.x = Math.floor(ix); h.y = Math.floor(iy); draw(); },
        end() { h.dragging = false; h.until = performance.now() + HOLD_MS; W.write([[`crosshair${x.n}_x`, h.x], [`crosshair${x.n}_y`, h.y]]); },
      };
    }
    const handle = roiHandleAt(cx, cy);
    if (!handle) return null;
    const start = roi();
    const h = held.roi = { ...start, dragging: true, until: 0 };
    return {
      move(ix, iy) { Object.assign(h, R.dragRoi(start, handle, ix, iy, iw, ih)); draw(); },
      end() { h.dragging = false; h.until = performance.now() + HOLD_MS; W.write(["minx", "miny", "sizex", "sizey"].filter((k) => h[k] !== start[k]).map((k) => [`roi_${k}`, h[k]])); },
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
    if (edit) { edit.end(); edit = null; }
    if (drag && ev.type === "pointerup" && Math.hypot(ev.clientX - drag.x, ev.clientY - drag.y) < 4 && performance.now() - drag.t < 400) {
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
    $("cursor").textContent = `x ${ix}  y ${iy}  level ${lum[jy * fw + jx]}`;
  }

  // ---- the frame loop: one request in flight, latest wins, paused while hidden
  const shown = [];
  let lastTag = null, waiting = null, drawing = false;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const visible = () => new Promise((r) => {
    if (!document.hidden) return r();
    document.addEventListener("visibilitychange", function on() { if (!document.hidden) { document.removeEventListener("visibilitychange", on); r(); } });
  });
  async function drawLatest() {
    if (drawing) return;
    drawing = true;
    try {
      while (waiting) { const blob = waiting; waiting = null; takeFrame(await createImageBitmap(blob)); $("image-error").hidden = true; shown.push(performance.now()); }
    } catch (e) {
      $("image-error-text").textContent = "the frame could not be decoded: " + (e.message || e);
      $("image-error").hidden = false;
      lastTag = null;
    } finally { drawing = false; }
  }
  async function loop() {
    for (;;) {
      await visible();
      const t0 = performance.now();
      try {
        const r = await fetch(box.dataset.frameUrl, { cache: "no-store" });
        if (!r.ok) { let d = `answered ${r.status}`; try { d = (await r.json()).detail || d; } catch (_) { /* not JSON */ } throw new Error(d); }
        const tag = r.headers.get("X-Frame-Tag"), blob = await r.blob();
        if (tag === null || tag !== lastTag) { lastTag = tag; waiting = blob; drawLatest(); }
      } catch (e) {
        $("image-error-text").textContent = String(e.message || e);
        $("image-error").hidden = false;
        await sleep(1000);
      }
      while (shown.length && shown[0] < performance.now() - 2000) shown.shift();
      $("fps").textContent = `${(shown.length / 2).toFixed(1)} new frames/s`;
      await sleep(Math.max(0, 1000 / MAX_FPS - (performance.now() - t0)));
    }
  }

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
    const chip = R.imageChip(P.entry("image_counter") ? P.word("image_counter") : "unknown");
    $("image-state").textContent = chip.text;
    $("image-state").dataset.state = KIT[chip.word] || "unknown";
    const bad = R.alarms(P.view());
    $("alarms").textContent = bad.length ? `in error: ${bad.join(", ")}` : "";
    sized();
    draw();
    hooks.forEach((f) => f());
  }
  const hooks = [];
  P.on(() => { if (!queued) { queued = true; requestAnimationFrame(refresh); } });

  // ---- wiring; what camera_controls.js uses
  buildLut();
  for (const el of [cmapSel, scaleSel, blackIn, whiteIn]) el.addEventListener("change", () => { buildLut(); recolour(); save(); });
  autoLive.addEventListener("change", save);
  $("auto-once").addEventListener("click", () => { autoLevels(); recolour(); save(); });
  $("reset-levels").addEventListener("click", () => { blackIn.value = 0; whiteIn.value = 255; scaleSel.value = "linear"; autoLive.checked = false; buildLut(); recolour(); save(); });
  $("roi-show").addEventListener("change", draw);

  // ---- writes: POST /api/devices/<name>/put {role, value, operator[, confirm]}, each sent once; a list stops at its first refusal
  const OP_KEY = "geecs-control-room-operator", opInput = $("operator");
  try { opInput.value = localStorage.getItem(OP_KEY) || ""; } catch (_) { /* no storage */ }
  opInput.addEventListener("change", () => { try { localStorage.setItem(OP_KEY, opInput.value.trim()); } catch (_) { /* no storage */ } });
  const out = $("write-out");
  const say = (word, text, detail) => { out.dataset.state = S.outcomeState(word); out.textContent = text; out.title = detail || ""; };
  // The roles in `pairs` the module marks as hazards (asked before anything is sent).
  const hazardsOf = (pairs) => R.hazards(pairs, Object.fromEntries(pairs.map(([role]) => [role, P.entry(role)])));
  const hazardDialog = $("hazard-dialog");
  function askHazard(pairs, roles) {
    $("hazard-dialog-text").textContent = pairs.map(([role, v]) => `${P.name} · ${role} ← ${v}`).join("\n");
    $("hazard-dialog-roles").textContent = `Hazard: ${roles.join(", ")}.`;
    return new Promise((done) => {
      hazardDialog.onclick = (ev) => { if (ev.target.value) { hazardDialog.close(); done(ev.target.value === "write"); } };
      hazardDialog.oncancel = () => done(false);
      hazardDialog.showModal();
    });
  }
  // `confirmed`: the caller's own dialog, naming these writes and their hazards, was answered (the camera ROI's).
  async function write(pairs, confirmed = false) {
    const operator = opInput.value.trim();
    if (!operator) { opInput.focus(); say("unknown", "unsent · name yourself in the header first"); return false; }
    // one decision for the whole list: the confirm sent is the dialog the operator answered
    const marked = hazardsOf(pairs);
    if (marked.length && !confirmed && !(await askHazard(pairs, marked))) { say("unknown", "unsent · not confirmed"); return false; }
    const confirm = confirmed || marked.length > 0;
    for (const [role, v] of pairs) {
      say("sending", `${role} ← ${v} · sending`);
      let a;
      try {
        const body = { role, value: v, operator };
        if (confirm) body.confirm = true;
        const r = await fetch(`${P.root}/api/devices/${encodeURIComponent(P.name)}/put`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        a = await r.json();
      } catch (_) { a = { outcome: "unknown", code: "no_answer", detail: "read the device and the journal before repeating it" }; }
      const word = a.outcome || "unknown";
      say(word, `${role} ← ${v} · ${word}${a.code ? " · " + a.code : ""}`, a.detail);
      if (word !== "accepted") { held.xh = {}; held.roi = null; draw(); return false; }
    }
    return true;
  }
  const W = window.GeecsCameraView = { write, hazardsOf, value, hooks, clicks, layers, toView, redraw: draw, frameSize: () => [iw, ih] };
  resize();
  loop();
})();
