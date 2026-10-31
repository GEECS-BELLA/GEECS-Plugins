// The camera page's controls, DOM only: the acquisition controls (exposure,
// period, gain, trigger mode, acquire), the camera ROI behind a confirm
// dialog, the crosshairs and the analysis ROI by value, and the lineouts and
// histogram (the on-request arrays, polled only while the plots are open).
// Each control is a family role: filled from Hare, enabled on the module's
// word (camera_rules.js's control), written through camera.js's write (the
// module's put route, sent once, its outcome shown).
(function () {
  "use strict";
  const P = window.GeecsDevicePage, R = window.GeecsCameraRules, V = window.GeecsCameraView;
  if (!P || !V) return;
  const $ = (id) => document.getElementById(id);
  const focused = (el) => document.activeElement === el;
  let busy = false, roiDraft = false;
  const send = async (pairs) => { busy = true; fill(); const ok = await V.write(pairs); busy = false; fill(); return ok; };

  // ---- every [data-role] control: filled unless focused (or a camera-ROI draft), enabled on the module's word
  const controls = [...document.querySelectorAll("#camera .rail [data-role]")];
  const CAM_ROI = ["min_x", "min_y", "size_x", "size_y"];
  function fill() {
    for (const el of controls) {
      const role = el.dataset.role, c = R.control(P.view(), P.entry(role), busy);
      el.disabled = !c.enabled;
      el.title = c.why;
      const v = V.value(role);
      if (el.tagName === "BUTTON") continue;
      if (el.type === "checkbox") { el.checked = !!v; continue; }
      if (el.tagName === "SELECT") {
        const choices = P.meta(role).enumChoices || [];
        if (el.options.length !== choices.length) el.replaceChildren(...choices.map((t, i) => new Option(t, String(i))));
        if (!focused(el) && typeof v === "number") el.value = String(v);
        continue;
      }
      if (focused(el) || (roiDraft && CAM_ROI.includes(role)) || typeof v !== "number") continue;
      el.value = Number.isInteger(v) ? v : Number(v.toPrecision(6));
    }
    $("acquire-now").textContent = R.fmt(P.frame("acquire"), P.meta("acquire"));
    const roiOk = (role) => R.control(P.view(), P.entry(role), busy).enabled;
    $("cam-roi-apply").disabled = !roiOk("min_x");
    $("cam-roi-full").disabled = !roiOk("min_x") || !(V.value("max_x") > 0);
    $("roi-full").disabled = !roiOk("roi_minx");
  }
  V.hooks.push(fill);

  const DRAFT = (el) => el.closest(".rail") && CAM_ROI.includes(el.dataset.role);
  for (const el of controls) {
    if (el.tagName === "BUTTON") { el.addEventListener("click", () => send([[el.dataset.role, Number(el.dataset.value)]])); continue; }
    if (DRAFT(el)) { el.addEventListener("input", () => { roiDraft = true; $("cam-roi-revert").hidden = false; }); continue; }
    el.addEventListener("change", () => {
      const v = el.type === "checkbox" ? (el.checked ? 1 : 0) : el.tagName === "SELECT" ? Number(el.value) : el.step === "1" ? R.whole(el.value) : R.number(el.value);
      if (v !== null) send([[el.dataset.role, v]]);
    });
  }

  // ---- the camera ROI: one draft of four fields, applied behind the dialog (it changes what scans save)
  const revert = () => { roiDraft = false; $("cam-roi-revert").hidden = true; fill(); };
  $("cam-roi-revert").addEventListener("click", revert);
  const dialog = $("roi-dialog");
  function ask(r) {
    $("roi-dialog-text").textContent = `x ${r[0]}  y ${r[1]}  width ${r[2]}  height ${r[3]}`;
    dialog.onclick = async (ev) => {
      if (!ev.target.value) return;
      dialog.close();
      // offsets, then sizes: the last size is checked against the final offset
      if (ev.target.value === "apply" && (await send(CAM_ROI.map((role, i) => [role, r[i]])))) revert();
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
    const pad = { l: 44, r: 6, t: 6, b: 16 }, ys = [], xs = [];
    const y = (v) => (opts.log ? Math.log10(v) : v), ok = (v) => v !== null && Number.isFinite(v) && !(opts.log && v <= 0);
    for (const s of series) s.values.forEach((v, i) => { if (ok(v)) { ys.push(y(v)); xs.push(opts.x(i)); } });
    c.font = "10px " + (token("--ff-mono") || "monospace"); c.fillStyle = token("--muted");
    if (!ys.length) { c.fillText(opts.empty, pad.l, h / 2); return; }
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (y1 === y0) { y0 -= 1; y1 += 1; }
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
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
    if (!plots.open || !latest) return;
    const a = latest.arrays || {}, ax = axis("x"), ay = axis("y");
    const why = (role) => (P.entry(role) ? (P.word(role) === "live" ? null : `${P.entry(role).label || role}: ${P.word(role)}`) : null);
    // a held array is no live one: STAT's counter is the arrays' freshness (the module's verdict)
    const note = [latest.error, why("stat_counter")].filter(Boolean).join(" · ");
    $("plots-note").textContent = note;
    const n = (k) => V.value(`profile_size_${k}`);
    const prof = (k) => [{ values: R.valid(a[`profile_${k}`], n(k)), colour: "--trace-1" }, { values: R.valid(a[`cursor_profile_${k}`], n(k)), colour: "--trace-2" }];
    lines($("plot-x"), prof("x"), { x: (i) => R.roiToImage(ax, i), empty: "no profile" });
    lines($("plot-y"), prof("y"), { x: (i) => R.roiToImage(ay, i), empty: "no profile" });
    const hist = a.histogram || [], lo = V.value("hist_min") || 0, step = ((V.value("hist_max") || hist.length) - lo) / (hist.length || 1);
    lines($("plot-hist"), [{ values: hist, colour: "--trace-3" }], { x: (i) => lo + (i + 0.5) * step, log: $("hist-log").checked, empty: "no histogram" });
  }

  let reading = false;
  async function poll() {
    if (reading || !plots.open || document.hidden) return;
    reading = true;
    try {
      const r = await fetch(document.getElementById("camera").dataset.arraysUrl, { cache: "no-store" });
      const body = await r.json();
      latest = r.ok ? body : { arrays: (latest || {}).arrays, error: body.detail || `answered ${r.status}` };
    } catch (_) { latest = { arrays: (latest || {}).arrays, error: "the Control Room did not answer" }; }
    reading = false;
    render();
    V.redraw();
  }
  setInterval(poll, 500);
  plots.addEventListener("toggle", () => { poll(); V.redraw(); });
  $("hist-log").addEventListener("change", render);

  // the lineout cursor: drawn while the plots are open, moved by a click (image pixels → the ROI's own)
  V.layers.push((c) => {
    const x = V.value("cursor_x"), y = V.value("cursor_y");
    if (!plots.open || typeof x !== "number" || typeof y !== "number") return;
    const [vx, vy] = V.toView(R.roiToImage(axis("x"), x) + 0.5, R.roiToImage(axis("y"), y) + 0.5);
    c.save(); c.strokeStyle = token("--trace-2"); c.setLineDash([3, 3]); c.lineWidth = 1; c.beginPath();
    c.moveTo(vx - 10, vy); c.lineTo(vx + 10, vy); c.moveTo(vx, vy - 10); c.lineTo(vx, vy + 10); c.stroke(); c.restore();
  });
  V.clicks.push((ix, iy) => {
    if (!plots.open || !R.control(P.view(), P.entry("cursor_x"), busy).enabled) return;
    send([["cursor_x", Math.round(R.imageToRoi(axis("x"), ix))], ["cursor_y", Math.round(R.imageToRoi(axis("y"), iy))]]);
  });
})();
