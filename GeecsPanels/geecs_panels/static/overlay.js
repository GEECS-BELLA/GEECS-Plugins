// The layer over the camera image: the two crosshairs and the analysis ROI.
//
// The IOC STORES them (OVER1:1/2 and ROI1: shared by every viewer,
// autosaved); this layer DRAWS them — crosshairs in their stored colour,
// over whatever colour map this viewer chose — and edits them through the
// service's allow-listed writes. Positions are image pixels (PROC1), so
// everything here goes through the viewer's toView/toImage and zooms and
// pans with the image. It plugs into camera.js through window.GeecsCameraView
// (draw, grab, hover and snapshot hooks) and owns nothing else.
//
// An edit shows at once: while dragging, and for a moment after the write,
// this viewer's own value is drawn instead of the stored one, until the
// IOC's readback agrees (or HOLD_MS passes) — no jump back mid-drag.
(() => {
  "use strict";

  const V = window.GeecsCameraView;
  if (!V) return;
  const root = V.root;
  const $ = (id) => document.getElementById(id);
  const writable = root.dataset.writable === "true";
  const putUrl = root.dataset.putUrl;
  // Per-crosshair routes: ".../crosshair/<n>/to-centroid" and ".../colour".
  const crosshairUrl = (n, verb) =>
    root.dataset.centroid1.replace(/\/1\/to-centroid$/, "/" + n + "/" + verb);
  const HOLD_MS = 2000;
  // A first colour for a crosshair switched on with the IOC's near-black default.
  const DEFAULT_COLOURS = { 1: { r: 0, g: 255, b: 0 }, 2: { r: 255, g: 0, b: 255 } };
  const GRAB_PX = 9;   // how near the pointer must be to grab a centre or a handle
  const EDGE_PX = 6;

  let crosshairs = [];  // the stored state, from the latest snapshot
  let roi = null;
  const held = { xh: {}, roi: null };  // this viewer's edits, shown until stored

  // ---- writes -------------------------------------------------------------
  let errTimer = null;
  function sayError(text) {
    const el = $("write-error");
    el.textContent = text;
    clearTimeout(errTimer);
    errTimer = setTimeout(() => { el.textContent = ""; }, 6000);
  }

  async function post(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!r.ok) {
      let detail = "HTTP " + r.status;
      try {
        const d = (await r.json()).detail;
        // A validation refusal is a list of {msg}; everything else a string.
        detail = Array.isArray(d) ? d.map((x) => x.msg).join("; ") : (d || detail);
      } catch (_) { /* not JSON */ }
      throw new Error(detail);
    }
    return r.json();
  }

  async function put(pv, value) {
    try {
      await post(putUrl, { pv: pv, value: value });
    } catch (e) {
      sayError(pv + ": " + e.message);
      throw e;
    }
  }

  async function putAll(pairs) {
    for (const [pv, value] of pairs) await put(pv, value);
  }

  // ---- what to draw: the stored value, or this viewer's edit --------------
  function position(c) {
    const h = held.xh[c.n];
    if (h) {
      const settled = !h.dragging && (performance.now() > h.until || (h.x === c.x && h.y === c.y));
      if (!settled) return h;
      delete held.xh[c.n];
    }
    return { x: c.x, y: c.y };
  }

  function bounds() {
    const h = held.roi;
    if (h) {
      const same = roi && h.minx === roi.minx && h.miny === roi.miny &&
        h.sizex === roi.sizex && h.sizey === roi.sizey;
      if (h.dragging || (performance.now() < h.until && !same)) return h;
      held.roi = null;
    }
    return roi;
  }

  // ---- drawing ------------------------------------------------------------
  const hex2 = (v) => Math.max(0, Math.min(255, v | 0)).toString(16).padStart(2, "0");
  const colourOf = (c) => "#" + hex2(c.r) + hex2(c.g) + hex2(c.b);
  const token = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();

  function drawCrosshair(ctx, c) {
    if (!c.use || c.x === null || c.y === null) return;
    const p = position(c);
    const [vx, vy] = V.toView(p.x, p.y);
    const s = V.scale();
    // Arms in image pixels (the stored size), never shorter than 12 px on screen.
    const ax = Math.max(12, ((c.sx || 0) / 2) * s), ay = Math.max(12, ((c.sy || 0) / 2) * s);
    const colour = colourOf(c);
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = colour;
    ctx.fillStyle = colour;
    ctx.beginPath();
    if (c.shape === "Rectangle") {
      ctx.rect(vx - ax, vy - ay, 2 * ax, 2 * ay);
    } else {
      ctx.moveTo(vx - ax, vy); ctx.lineTo(vx + ax, vy);
      ctx.moveTo(vx, vy - ay); ctx.lineTo(vx, vy + ay);
    }
    ctx.stroke();
    if (writable) {
      ctx.beginPath();
      ctx.arc(vx, vy, 5, 0, 2 * Math.PI);
      ctx.stroke();
    }
    ctx.font = "11px " + (token("--ff-mono") || "monospace");
    ctx.fillText(c.name || String(c.n), vx + 7, vy - 7);
    ctx.restore();
  }

  function roiRect() {
    const b = bounds();
    if (!b || b.minx === null || b.sizex === null) return null;
    const [l, t] = V.toView(b.minx, b.miny);
    const [r, bt] = V.toView(b.minx + b.sizex, b.miny + b.sizey);
    return { l: l, t: t, r: r, b: bt };
  }

  function drawRoi(ctx) {
    if (!$("roi-show").checked) return;
    const q = roiRect();
    if (!q) return;
    ctx.save();
    ctx.strokeStyle = token("--accent");
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(q.l, q.t, q.r - q.l, q.b - q.t);
    if (writable) {
      ctx.setLineDash([]);
      ctx.fillStyle = token("--accent");
      for (const [x, y] of [[q.l, q.t], [q.r, q.t], [q.l, q.b], [q.r, q.b]]) {
        ctx.fillRect(x - 3, y - 3, 6, 6);
      }
    }
    ctx.restore();
  }

  V.hooks.draw.push((ctx) => {
    drawRoi(ctx);
    for (const c of crosshairs) drawCrosshair(ctx, c);
  });

  // ---- hit testing: what a press here would grab -------------------------
  function crosshairAt(cx, cy) {
    for (const c of crosshairs) {
      if (!c.use || c.x === null) continue;
      const p = position(c);
      const [vx, vy] = V.toView(p.x, p.y);
      if (Math.hypot(cx - vx, cy - vy) <= GRAB_PX) return c;
    }
    return null;
  }

  // Which ROI edges a press here would move: e.g. "nw", "e"; "" for none.
  function roiHandleAt(cx, cy) {
    if (!$("roi-show").checked) return "";
    const q = roiRect();
    if (!q) return "";
    const near = (a, b, d) => Math.abs(a - b) <= d;
    const inX = cx >= q.l - EDGE_PX && cx <= q.r + EDGE_PX;
    const inY = cy >= q.t - EDGE_PX && cy <= q.b + EDGE_PX;
    let h = "";
    if (inX && near(cy, q.t, GRAB_PX)) h += "n";
    else if (inX && near(cy, q.b, GRAB_PX)) h += "s";
    if (inY && near(cx, q.l, GRAB_PX)) h += "w";
    else if (inY && near(cx, q.r, GRAB_PX)) h += "e";
    return h;
  }

  const CURSORS = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
    nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize" };

  V.hooks.hover.push((cx, cy) => {
    if (!writable) return "";
    if (crosshairAt(cx, cy)) return "move";
    return CURSORS[roiHandleAt(cx, cy)] || "";
  });

  V.hooks.grab.push((cx, cy) => {
    if (!writable) return null;
    const c = crosshairAt(cx, cy);
    if (c) return dragCrosshair(c);
    const handle = roiHandleAt(cx, cy);
    if (handle) return dragRoi(handle);
    return null;
  });

  function dragCrosshair(c) {
    const p = position(c);
    const h = held.xh[c.n] = { x: p.x, y: p.y, dragging: true, until: 0 };
    return {
      move(ix, iy) {
        h.x = Math.round(ix); h.y = Math.round(iy);
        V.redraw();
      },
      end() {
        h.dragging = false;
        h.until = performance.now() + HOLD_MS;
        putAll([["OVER1:" + c.n + ":CenterX", h.x], ["OVER1:" + c.n + ":CenterY", h.y]])
          .catch(() => { delete held.xh[c.n]; V.redraw(); });
      },
    };
  }

  function dragRoi(handle) {
    const b = bounds();
    const [fw, fh] = V.frameSize();
    // The edges that do not move stay where they were.
    let l = b.minx, t = b.miny, r = b.minx + b.sizex, bt = b.miny + b.sizey;
    const h = held.roi = { minx: l, miny: t, sizex: b.sizex, sizey: b.sizey, dragging: true, until: 0 };
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(v)));
    return {
      move(ix, iy) {
        if (handle.includes("w")) l = clamp(ix, 0, r - 1);
        if (handle.includes("e")) r = clamp(ix, l + 1, fw);
        if (handle.includes("n")) t = clamp(iy, 0, bt - 1);
        if (handle.includes("s")) bt = clamp(iy, t + 1, fh);
        h.minx = l; h.miny = t; h.sizex = r - l; h.sizey = bt - t;
        V.redraw();
      },
      end() {
        h.dragging = false;
        h.until = performance.now() + HOLD_MS;
        putRoi(h).catch(() => { held.roi = null; V.redraw(); });
      },
    };
  }

  function putRoi(b) {
    return putAll([["ROI1:MinX", b.minx], ["ROI1:MinY", b.miny],
      ["ROI1:SizeX", b.sizex], ["ROI1:SizeY", b.sizey]]);
  }

  // ---- the rail's controls -----------------------------------------------
  const focused = (el) => document.activeElement === el;

  function fillControls() {
    for (const c of crosshairs) {
      const box = root.querySelector('.crosshair[data-n="' + c.n + '"]');
      if (!box) continue;
      const p = position(c);
      box.querySelector(".xh-use").checked = !!c.use;
      box.querySelector(".xh-name").textContent = c.name || String(c.n);
      if (c.r !== null && !focused(box.querySelector(".xh-colour"))) {
        box.querySelector(".xh-colour").value = colourOf(c);
      }
      for (const [cls, v] of [[".xh-x", p.x], [".xh-y", p.y]]) {
        const input = box.querySelector(cls);
        if (!focused(input) && v !== null) input.value = v;
      }
    }
    const b = bounds();
    if (b) {
      for (const k of ["minx", "miny", "sizex", "sizey"]) {
        const input = $("roi-" + k);
        if (!focused(input) && b[k] !== null) input.value = b[k];
      }
    }
  }

  V.hooks.snapshot.push((snap) => {
    crosshairs = snap.crosshairs || [];
    roi = snap.roi || null;
    fillControls();
    V.redraw();
  });

  for (const box of root.querySelectorAll(".crosshair")) {
    const n = Number(box.dataset.n);
    const pv = (f) => "OVER1:" + n + ":" + f;
    box.querySelector(".xh-use").addEventListener("change", (ev) => {
      const on = ev.target.checked;
      const c = crosshairs.find((x) => x.n === n) || {};
      const steps = [];
      // A never-configured overlay (the IOC's default is a 1-pixel
      // rectangle): make it a visible cross first.
      if (on && (c.shape !== "Cross" || !(c.sx > 1))) {
        const [fw, fh] = V.frameSize();
        const arm = Math.max(40, Math.round(Math.min(fw, fh) / 5)) || 100;
        steps.push([pv("Shape"), "Cross"], [pv("SizeX"), arm], [pv("SizeY"), arm]);
        if (c.x === null || c.x === undefined || c.x === 0) {
          steps.push([pv("CenterX"), Math.round(fw / 2)], [pv("CenterY"), Math.round(fh / 2)]);
        }
      }
      steps.push([pv("Use"), on ? 1 : 0]);
      // The IOC's default colour is near-black, invisible on a dark frame.
      const dark = on && (c.r || 0) + (c.g || 0) + (c.b || 0) < 60;
      putAll(steps)
        .then(() => dark && post(crosshairUrl(n, "colour"), DEFAULT_COLOURS[n] || DEFAULT_COLOURS[1]))
        .catch(() => {});
    });
    box.querySelector(".xh-colour").addEventListener("change", (ev) => {
      const v = ev.target.value;  // "#rrggbb"
      const ch = (i) => parseInt(v.slice(i, i + 2), 16);
      post(crosshairUrl(n, "colour"), { r: ch(1), g: ch(3), b: ch(5) })
        .catch((e) => sayError("crosshair " + n + ": " + e.message));
    });
    for (const [cls, field] of [[".xh-x", "CenterX"], [".xh-y", "CenterY"]]) {
      box.querySelector(cls).addEventListener("change", (ev) => {
        put(pv(field), Math.round(Number(ev.target.value))).catch(() => {});
      });
    }
    box.querySelector(".xh-centroid").addEventListener("click", () => {
      post(crosshairUrl(n, "to-centroid")).catch((e) => sayError("crosshair " + n + ": " + e.message));
    });
  }

  for (const k of ["minx", "miny", "sizex", "sizey"]) {
    $("roi-" + k).addEventListener("change", (ev) => {
      const field = { minx: "MinX", miny: "MinY", sizex: "SizeX", sizey: "SizeY" }[k];
      put("ROI1:" + field, Math.round(Number(ev.target.value))).catch(() => {});
    });
  }
  $("roi-full").addEventListener("click", () => {
    const [fw, fh] = V.frameSize();
    if (fw) putRoi({ minx: 0, miny: 0, sizex: fw, sizey: fh }).catch(() => {});
  });
  $("roi-show").addEventListener("change", () => V.redraw());

  if (!writable) {
    for (const el of root.querySelectorAll(".writes input, .writes button")) el.disabled = true;
  }
})();
