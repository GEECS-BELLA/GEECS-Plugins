// The plots beside the image: X and Y lineouts (the ROI's average, and the
// line through the cursor), the histogram, and the centroid/mean history.
//
// Every number comes from the IOC's STAT plugin (full precision, not the
// 8-bit picture); the service reads the arrays on request (at most twice a
// second, shared by viewers) and keeps the history. The stream carries them
// only while the plots panel is open. Click the image to put the lineout
// cursor there; a marker shows where it is.
//
// One small drawer (lines) serves every plot: custom for now, to be weighed
// against a plotting library once the panel's design settles.
(() => {
  "use strict";

  const V = window.GeecsCameraView;
  if (!V) return;
  const $ = (id) => document.getElementById(id);
  const panel = $("plots");
  const cursorUrl = V.root.dataset.cursorUrl;
  const writable = V.root.dataset.writable === "true";
  const token = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  let latest = null;

  // ---- one plot: lines over a shared x axis --------------------------------
  // series: [{values: [y | null], colour: token name}]; x: the i-th value's x.
  function drawLines(canvas, series, opts) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pad = { l: 44, r: 6, t: 6, b: 16 };
    const ys = [], xs = [];
    for (const s of series) {
      s.values.forEach((v, i) => {
        if (v === null || !Number.isFinite(v) || (opts.log && v <= 0)) return;
        ys.push(opts.log ? Math.log10(v) : v);
        xs.push(opts.x(i));
      });
    }
    ctx.font = "10px " + (token("--ff-mono") || "monospace");
    ctx.fillStyle = token("--muted");
    if (!ys.length) {
      ctx.fillText(opts.empty || "no data", pad.l, h / 2);
      return;
    }
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (y1 === y0) { y0 -= 1; y1 += 1; }
    const x0 = Math.min(...xs), x1 = Math.max(...xs) || x0 + 1;
    const px = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r);
    const py = (y) => h - pad.b - ((y - y0) / (y1 - y0)) * (h - pad.t - pad.b);
    // Frame and range labels (the axis is the data's own span).
    ctx.strokeStyle = token("--rule-soft");
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.l + 0.5, pad.t + 0.5, w - pad.l - pad.r - 1, h - pad.t - pad.b - 1);
    const fmtY = (y) => {
      const v = opts.log ? Math.pow(10, y) : y;
      const a = Math.abs(v);
      return a !== 0 && (a >= 1e5 || a < 1e-2) ? v.toExponential(1) : String(Number(v.toPrecision(3)));
    };
    ctx.textAlign = "right";
    ctx.fillText(fmtY(y1), pad.l - 4, pad.t + 8);
    ctx.fillText(fmtY(y0), pad.l - 4, h - pad.b);
    ctx.textAlign = "left";
    ctx.fillText(opts.fmtX ? opts.fmtX(x0) : String(Math.round(x0)), pad.l, h - 3);
    ctx.textAlign = "right";
    ctx.fillText(opts.fmtX ? opts.fmtX(x1) : String(Math.round(x1)), w - pad.r, h - 3);
    ctx.textAlign = "left";
    for (const s of series) {
      ctx.strokeStyle = token(s.colour);
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      let pen = false;
      s.values.forEach((v, i) => {
        if (v === null || !Number.isFinite(v) || (opts.log && v <= 0)) { pen = false; return; }
        const X = px(opts.x(i)), Y = py(opts.log ? Math.log10(v) : v);
        if (pen) ctx.lineTo(X, Y); else ctx.moveTo(X, Y);
        pen = true;
      });
      ctx.stroke();
    }
  }

  // ---- the five plots ------------------------------------------------------
  const axisX = (a) => (i) => a.start + i * a.step;

  function render() {
    const p = latest;
    if (!p || panel.hidden || !panel.open) return;
    $("plots-error").textContent = p.error || "";
    if (p.error) return;
    const off = !p.profiles_on ? "profiles are not computed (STAT:ComputeProfiles)" : null;
    if (p.x_axis) {
      drawLines($("plot-x"), [
        { values: p.profile_x, colour: "--trace-1" },
        { values: p.cursor_x, colour: "--trace-2" },
      ], { x: axisX(p.x_axis), empty: off });
      drawLines($("plot-y"), [
        { values: p.profile_y, colour: "--trace-1" },
        { values: p.cursor_y, colour: "--trace-2" },
      ], { x: axisX(p.y_axis), empty: off });
    }
    const n = p.histogram.length || 1;
    const lo = p.hist_min || 0, step = ((p.hist_max || n) - lo) / n;
    drawLines($("plot-hist"), [{ values: p.histogram, colour: "--trace-3" }], {
      x: (i) => lo + (i + 0.5) * step,
      log: $("hist-log").checked,
      empty: p.histogram_on ? "no histogram" : "histogram is not computed (STAT:ComputeHistogram)",
    });
    const t = p.history.map((r) => r[0]);
    const now = t.length ? t[t.length - 1] : 0;
    const col = (k) => p.history.map((r) => r[1 + p.history_keys.indexOf(k)]);
    const ago = (s) => (s - now >= -0.5 ? "now" : Math.round(s - now) + " s");
    drawLines($("plot-centroid"), [
      { values: col("cx"), colour: "--trace-1" },
      { values: col("cy"), colour: "--trace-2" },
    ], { x: (i) => t[i], fmtX: ago, empty: "no history yet" });
    drawLines($("plot-mean"), [{ values: col("mean"), colour: "--trace-4" }],
      { x: (i) => t[i], fmtX: ago, empty: "no history yet" });
  }

  V.hooks.snapshot.push((snap) => {
    if (!snap.plots) return;
    latest = snap.plots;
    render();
    V.redraw();  // the cursor marker
  });

  // ---- the cursor: click the image to place it, a marker shows it ----------
  V.hooks.draw.push((ctx) => {
    if (!latest || !latest.cursor || !panel.open) return;
    const [vx, vy] = V.toView(latest.cursor[0] + 0.5, latest.cursor[1] + 0.5);
    ctx.save();
    ctx.strokeStyle = token("--trace-2");
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(vx - 10, vy); ctx.lineTo(vx + 10, vy);
    ctx.moveTo(vx, vy - 10); ctx.lineTo(vx, vy + 10);
    ctx.stroke();
    ctx.restore();
  });

  V.hooks.click.push((ix, iy) => {
    if (!writable || !panel.open) return;
    V.write.post(cursorUrl, { x: ix, y: iy })
      .catch((e) => V.write.sayError("cursor: " + e.message));
  });

  // ---- open/closed: the arrays travel only while the plots are shown ------
  function shown() {
    V.setPlots(panel.open);
    render();
  }
  panel.addEventListener("toggle", shown);
  $("hist-log").addEventListener("change", render);
  new ResizeObserver(render).observe(panel);
  shown();
})();
