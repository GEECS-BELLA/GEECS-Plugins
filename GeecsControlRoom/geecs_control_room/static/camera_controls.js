// The camera page's controls, DOM only: the rail (the family's panel: acquisition, crosshairs, the analysis ROI)
// is device_rail.js's, the device pages' one control set, written through camera.js's writes (one list at a time
// for the page, shared with the image's edits); the camera ROI (one draft of four fields, applied behind its
// dialog: it changes what scans save) and the lineouts and histogram (the on-request arrays, polled only while
// the plots are open) are the camera's own. Each control is a family role: filled from Hare, enabled on the
// module's word (camera_rules.js's control), written once through the module's put route, its outcome shown.
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsCameraRules, V = window.GeecsCameraView;
  if (!P || !V) return;
  const $ = (id) => document.getElementById(id);
  const focused = (el) => document.activeElement === el;
  let roiDraft = false;
  // camera.js's write runs one list at a time for the page (the rail and the image's edits); null: refused, one runs
  const send = (pairs, confirmed = false) => V.write(pairs, confirmed);
  // filled through camera.js's hooks (one refresh per animation frame, and a write's start and end), never on every PV update
  const rail = window.GeecsDeviceRail.make({ ...P, on() {} }, $("camera"), { write: V.write, busy: V.busy });

  // ---- the camera ROI's four fields: filled unless focused or holding the draft, enabled on the module's word
  const CAM_ROI = ["min_x", "min_y", "size_x", "size_y"];
  const roiFields = CAM_ROI.map((role) => $(`c-${role}`));
  function fill() {
    for (const el of roiFields) {
      const role = el.dataset.roiRole, c = R.control(P.view(), P.entry(role), V.busy()), v = V.value(role);
      el.disabled = !c.enabled;
      el.title = c.why;
      if (!focused(el) && !roiDraft && typeof v === "number") el.value = String(v);
    }
    const roiOk = (role) => R.control(P.view(), P.entry(role), V.busy()).enabled;
    $("cam-roi-apply").disabled = !roiOk("min_x");
    $("cam-roi-full").disabled = !roiOk("min_x") || !(V.value("max_x") > 0);
    $("roi-full").disabled = !roiOk("roi_minx") || !V.frameSize()[0]; // no full frame while the geometry is unknown
  }
  V.hooks.push(fill, rail.fill); // a write's start and end: the controls disabled, then enabled
  for (const el of roiFields) el.addEventListener("input", () => { roiDraft = true; $("cam-roi-revert").hidden = false; });

  // ---- the camera ROI: one draft of four fields, applied behind the dialog (it changes what scans save)
  const revert = () => { roiDraft = false; $("cam-roi-revert").hidden = true; fill(); };
  $("cam-roi-revert").addEventListener("click", revert);
  const dialog = $("roi-dialog");
  function ask(r) {
    const pairs = CAM_ROI.map((role, i) => [role, r[i]]);
    const marked = V.hazardsOf(pairs);
    $("roi-dialog-text").textContent = `x ${r[0]}  y ${r[1]}  width ${r[2]}  height ${r[3]}`;
    $("roi-dialog-hazard").textContent = marked.length ? `Hazard: ${marked.join(", ")}.` : "";
    dialog.onclick = async (ev) => {
      if (!ev.target.value) return;
      dialog.close();
      // offsets, then sizes: the last size is checked against the final offset;
      // this dialog named the hazards, so its Apply is their confirm
      if (ev.target.value === "apply" && (await send(pairs, true))) revert();
    };
    dialog.showModal();
  }
  $("cam-roi-apply").addEventListener("click", () => {
    const r = CAM_ROI.map((role) => R.whole($(`c-${role}`).value));
    if (r.some((v) => v === null)) return;
    ask(r);
  });
  $("cam-roi-full").addEventListener("click", () => ask([0, 0, V.value("max_x"), V.value("max_y")]));
  $("roi-full").addEventListener("click", () => {
    const [w, h] = V.frameSize();
    if (w) send([["roi_minx", 0], ["roi_miny", 0], ["roi_sizex", w], ["roi_sizey", h]]);
  });

  // ---- the plots: the arrays, one read in flight, every 0.5 s while open and visible
  const plots = $("plots"), token = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  let latest = null;
  const axis = (k) => ({ min: V.value(`roi_min${k}`) || 0, size: V.value(`roi_size${k}`) || 0, bin: V.value(`roi_bin${k}`) || 1, reverse: !!V.value(`roi_rev${k}`) });

  function lines(canvas, series, opts) {
    const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, w, h);
    const pad = { l: 44, r: 6, t: 6, b: 16 };
    const y = (v) => (opts.log ? Math.log10(v) : v), ok = (v) => v !== null && Number.isFinite(v) && !(opts.log && v <= 0);
    // the ranges as running extremes, never Math.min(...array): a 16-bit raw histogram has 65 536 bins,
    // past the argument count some engines allow a spread
    let n = 0, y0 = Infinity, y1 = -Infinity, x0 = Infinity, x1 = -Infinity;
    for (const s of series) s.values.forEach((v, i) => {
      if (!ok(v)) return;
      const Y = y(v), X = opts.x(i); n++;
      if (Y < y0) y0 = Y; if (Y > y1) y1 = Y; if (X < x0) x0 = X; if (X > x1) x1 = X;
    });
    c.font = "10px " + (token("--ff-mono") || "monospace"); c.fillStyle = token("--muted");
    if (!n) { c.fillText(opts.empty, pad.l, h / 2); return; }
    if (y1 === y0) { y0 -= 1; y1 += 1; }
    const px = (x) => pad.l + ((x - x0) / (x1 - x0 || 1)) * (w - pad.l - pad.r), py = (v) => h - pad.b - ((v - y0) / (y1 - y0)) * (h - pad.t - pad.b);
    c.strokeStyle = token("--rule-soft"); c.lineWidth = 1;
    c.strokeRect(pad.l + 0.5, pad.t + 0.5, w - pad.l - pad.r - 1, h - pad.t - pad.b - 1);
    const label = (v) => R.fmt({ value: opts.log ? Math.pow(10, v) : v }, {});
    c.textAlign = "right"; c.fillText(label(y1), pad.l - 4, pad.t + 8); c.fillText(label(y0), pad.l - 4, h - pad.b);
    c.fillText(String(Math.round(x1)), w - pad.r, h - 3); c.textAlign = "left"; c.fillText(String(Math.round(x0)), pad.l, h - 3);
    for (const s of series) {
      c.strokeStyle = token(s.colour); c.lineWidth = 1.25; c.beginPath();
      let pen = false;
      s.values.forEach((v, i) => { if (!ok(v)) { pen = false; return; } const X = px(opts.x(i)), Y = py(y(v)); if (pen) c.lineTo(X, Y); else c.moveTo(X, Y); pen = true; });
      c.stroke();
    }
  }

  function render() {
    const rawHist = V.rawHistogram();
    if (!plots.open || (!latest && !rawHist)) return;
    const a = (latest && latest.arrays) || {}, ax = axis("x"), ay = axis("y");
    const why = (role) => (P.entry(role) ? (P.word(role) === "live" ? null : `${P.entry(role).label || role}: ${P.word(role)}`) : null);
    // a held array is no live one: STAT's counter is the arrays' freshness (the module's verdict)
    const note = [latest && latest.error, why("stat_counter")].filter(Boolean).join(" · ");
    $("plots-note").textContent = note;
    const n = (k) => V.value(`profile_size_${k}`);
    // a computation switched off keeps its last array in STAT: drawn as nothing, and said
    const profOff = R.notComputed(V.value("profiles_on"), "profiles");
    const prof = (k) => (profOff ? [] : [{ values: R.valid(a[`profile_${k}`], n(k)), colour: "--trace-1" }, { values: R.valid(a[`cursor_profile_${k}`], n(k)), colour: "--trace-2" }]);
    lines($("plot-x"), prof("x"), { x: (i) => R.roiToImage(ax, i), empty: profOff || "no profile" });
    lines($("plot-y"), prof("y"), { x: (i) => R.roiToImage(ay, i), empty: profOff || "no profile" });
    // Raw: the drawn frame's own histogram (0..full); MJPEG: STAT's, over hist_min..hist_max
    const hp = R.histogramPlot(rawHist, a.histogram, V.value("histogram_on"), V.value("hist_min"), V.value("hist_max"));
    lines($("plot-hist"), hp.values.length ? [{ values: hp.values, colour: "--trace-3" }] : [], { x: (i) => hp.lo + (i + 0.5) * hp.step, log: $("hist-log").checked, empty: hp.empty || "no histogram" });
  }

  const ARRAYS_TIMEOUT_MS = 2000; // past this the request is abandoned and the held arrays are said to be old
  let reading = false;
  async function poll() {
    if (reading || !plots.open || document.hidden) return;
    reading = true;
    try {
      const [r, body] = await R.within(ARRAYS_TIMEOUT_MS, async (signal) => {
        const r = await fetch(document.getElementById("camera").dataset.arraysUrl, { cache: "no-store", signal });
        return [r, await r.json()];
      });
      latest = r.ok ? body : { arrays: (latest || {}).arrays, error: body.detail || `answered ${r.status}` };
    } catch (e) {
      latest = { arrays: (latest || {}).arrays, error: e && e.message === "timed out" ? "the Control Room did not answer in time: arrays held" : "the Control Room did not answer" };
    } finally { reading = false; }
    render();
    V.redraw();
  }
  setInterval(poll, 500);
  plots.addEventListener("toggle", () => { poll(); V.redraw(); });
  $("hist-log").addEventListener("change", render);
  V.modes.push(render); // the Raw toggle: the histogram plot changes source at once

  // the lineout cursor: drawn while the plots are open, moved by a click (image pixels → the ROI's own)
  V.layers.push((c) => {
    const x = V.value("cursor_x"), y = V.value("cursor_y");
    if (!plots.open || typeof x !== "number" || typeof y !== "number") return;
    const [vx, vy] = V.toView(R.roiToImage(axis("x"), x) + 0.5, R.roiToImage(axis("y"), y) + 0.5);
    c.save(); c.strokeStyle = token("--trace-2"); c.setLineDash([3, 3]); c.lineWidth = 1; c.beginPath();
    c.moveTo(vx - 10, vy); c.lineTo(vx + 10, vy); c.moveTo(vx, vy - 10); c.lineTo(vx, vy + 10); c.stroke(); c.restore();
  });
  V.clicks.push((ix, iy) => {
    if (!plots.open || !R.control(P.view(), P.entry("cursor_x"), V.busy()).enabled) return;
    send([["cursor_x", Math.round(R.imageToRoi(axis("x"), ix))], ["cursor_y", Math.round(R.imageToRoi(axis("y"), iy))]]);
  });
})();
