// The oscilloscope panel: the traces, the settings, the capture state.
//
// It speaks geecs-epics' scope contract only, so
// it fits any scope vendor: every select is filled from its PV's own enum
// choices (the stream's "enums"), channels the model lacks are hidden, and
// each channel shows the chain's statistics (STAT_<CH>) for its window.
//
// One stream (/scope/<name>/events?traces=1) carries the settings in force,
// the status and the decimated traces. Each setting shows its readback
// unless it has focus; changing it posts one allow-listed write, and the
// scope's answer arrives on the stream (a refused setting snaps back, the
// reason in LastError).
//
// The screen draws every enabled channel against its own input range: the
// screen height is its full scale (eight divisions), so channels on
// different ranges share it the way a bench scope's do. The time axis is
// the capture's, zero at the trigger.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const root = $("scope");
  const writable = root.dataset.writable === "true";
  const canvas = $("screen");
  const token = (name) => getComputedStyle(document.body).getPropertyValue(name).trim();
  const CHANNELS = ["A", "B", "C", "D"];
  let snap = null, hoverX = null, source = null;

  // ---- values ---------------------------------------------------------------
  const volts = (range) => {
    const m = /^([\d.]+)\s*(mV|V)$/.exec(range || "");
    return m ? Number(m[1]) * (m[2] === "mV" ? 1e-3 : 1) : null;
  };
  const fmtTime = (s) => {
    const a = Math.abs(s);
    if (a === 0) return "0 s";
    if (a < 1e-6) return (s * 1e9).toPrecision(3) + " ns";
    if (a < 1e-3) return (s * 1e6).toPrecision(3) + " µs";
    if (a < 1) return (s * 1e3).toPrecision(3) + " ms";
    return s.toPrecision(3) + " s";
  };
  const fmtVolts = (v) =>
    v === null || v === undefined ? "—"
      : Math.abs(v) < 1 ? (v * 1e3).toFixed(1) + " mV" : v.toFixed(3) + " V";
  const channel = (name) => (snap ? snap.channels.find((c) => c.name === name) : null);
  const valueOf = (from) => {
    if (!snap) return null;
    const dot = from.indexOf(".");
    if (dot < 0) return snap.scope[from];
    const ch = channel(from.slice(0, dot));
    return ch ? ch[from.slice(dot + 1)] : null;
  };
  const enabled = () =>
    CHANNELS.filter((n) => {
      const c = channel(n);
      return c && c.present === "Present" && c.enable === "On";
    });

  // ---- writes ---------------------------------------------------------------
  function sayError(message) { $("write-error").textContent = message || ""; }

  async function put(pv, value) {
    sayError("");
    try {
      const r = await fetch(root.dataset.putUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pv: pv, value: value }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        sayError(`${pv}: ${body.detail || r.status}`);
      }
    } catch (e) {
      sayError(`${pv}: ${e}`);
    }
  }

  for (const el of root.querySelectorAll("[data-setting]")) {
    el.disabled = !writable;
    el.addEventListener("change", () => {
      let value;
      if (el.type === "checkbox") value = el.checked ? "On" : "Off";
      else if (el.type === "number") {
        value = Number(el.value);
        if (el.value.trim() === "" || !Number.isFinite(value)) return;
      } else value = el.value;
      put(el.dataset.setting, value);
      el.blur();
    });
    if (el.type === "number")
      el.addEventListener("keydown", (ev) => { if (ev.key === "Enter") el.blur(); });
  }
  for (const btn of root.querySelectorAll("[data-put]")) {
    btn.disabled = !writable;
    btn.addEventListener("click", () => put(btn.dataset.put, Number(btn.dataset.value)));
  }

  // ---- render ---------------------------------------------------------------
  function showState() {
    const chip = $("scope-state");
    let state = "unknown", text = "connecting";
    if (snap && !snap.reachable) { state = "failed"; text = "IOC unreachable"; }
    else if (snap) {
      // DetectorState_RBV (areaDetector's) and Acquire_RBV
      const s = snap.scope.state, acq = snap.scope.acquire === "Acquiring";
      if (s === "Disconnected") { state = "failed"; text = "scope disconnected"; }
      else if (s === "Error") { state = "failed"; text = snap.scope.status || "error"; }
      // Armed and reading out alternate at the capture rate: one word for both.
      else if (acq || s === "Acquire" || s === "Readout") { state = "running"; text = "acquiring"; }
      else { state = "queued"; text = "stopped"; }
    }
    chip.dataset.state = state;
    chip.textContent = text;
  }

  // A select's options are its PV's enum choices: the scope's, not a list here.
  function fillChoices() {
    const enums = snap.enums || {};
    for (const el of root.querySelectorAll("select[data-setting]")) {
      const choices = enums[el.dataset.setting];
      if (!choices) continue;
      const have = Array.from(el.options, (o) => o.value);
      if (have.length === choices.length && have.every((v, i) => v === choices[i])) continue;
      el.replaceChildren(...choices.map((c) => new Option(c, c)));
    }
  }

  function showStats(name, c) {
    const el = $("ch-" + name + "-stats");
    const st = c && c.stats;
    if (!st || st.total === null || st.total === undefined) { el.textContent = ""; return; }
    const dt = typeof snap.scope.interval_ns === "number" ? snap.scope.interval_ns * 1e-9 : null;
    const integral = dt === null ? "" : ` · ∫ ${(st.total * dt).toExponential(3)} V·s`;
    el.textContent =
      `mean ${fmtVolts(st.mean)} · σ ${fmtVolts(st.sigma)} · min ${fmtVolts(st.min)} · ` +
      `max ${fmtVolts(st.max)}${integral}`;
  }

  function showSettings() {
    fillChoices();
    for (const el of root.querySelectorAll("[data-setting]")) {
      if (document.activeElement === el) continue;
      const v = valueOf(el.dataset.from);
      if (v === null || v === undefined) continue;
      if (el.type === "checkbox") el.checked = v === "On";
      else if (el.type === "number") el.value = typeof v === "number" ? +v.toPrecision(6) : v;
      else el.value = v;
    }
    for (const name of CHANNELS) {
      const c = channel(name);
      $("ch-" + name).hidden = !(c && c.present === "Present");
      $("ch-" + name + "-over").hidden = !(c && c.overrange === "Over range");
      showStats(name, c);
    }
    const s = snap.scope;
    $("capture-count").textContent = s.count === null ? "" : `${s.count} captures`;
    $("capture-age").textContent =
      snap.capture_age === null ? "no capture yet" : `last capture ${snap.capture_age.toFixed(1)} s ago`;
    if (typeof s.samples === "number" && typeof s.interval_ns === "number")
      $("tb-window").textContent =
        `${fmtTime(s.samples * s.interval_ns * 1e-9)} window · ${fmtTime(s.interval_ns * 1e-9)}/sample`;
    $("trace-legend").textContent = enabled()
      .map((n) => `${n} ±${channel(n).range}`).join(" · ");
    $("identity").textContent = [s.manufacturer, s.model, s.serial].filter(Boolean).join(" ");
    $("identity-detail").textContent =
      `${s.manufacturer || "—"} ${s.model || "—"} · serial ${s.serial || "—"} · ` +
      `firmware ${s.firmware || "—"} · ${s.num_channels || "?"} channels · status ${s.status || "—"}`;
    $("last-error").textContent = s.last_error ? `last error: ${s.last_error}` : "";
  }

  // ---- the screen -----------------------------------------------------------
  function geometry() {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    return { dpr, w, h, pad: { l: 8, r: 8, t: 8, b: 22 } };
  }

  function span(traces) {
    let t0 = Infinity, t1 = -Infinity;
    for (const n of Object.keys(traces)) {
      const t = traces[n].t;
      if (t.length) { t0 = Math.min(t0, t[0]); t1 = Math.max(t1, t[t.length - 1]); }
    }
    return Number.isFinite(t0) && t1 > t0 ? [t0, t1] : null;
  }

  function draw() {
    const { dpr, w, h, pad } = geometry();
    if (!w || !h) return;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;
    const mid = pad.t + ph / 2;

    // Graticule: 10 x 8 divisions, the centre lines stronger.
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const x = Math.round(pad.l + (i * pw) / 10) + 0.5;
      ctx.strokeStyle = token(i === 5 ? "--rule" : "--rule-soft");
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
    }
    for (let j = 0; j <= 8; j++) {
      const y = Math.round(pad.t + (j * ph) / 8) + 0.5;
      ctx.strokeStyle = token(j === 4 ? "--rule" : "--rule-soft");
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    }
    ctx.font = "10px " + (token("--ff-mono") || "monospace");
    ctx.fillStyle = token("--muted");

    const tr = snap && snap.traces;
    if (!tr || tr.error || !Object.keys(tr.traces || {}).length) {
      ctx.fillText(tr && tr.error ? tr.error : "no traces yet", pad.l + 6, mid - 6);
      return;
    }
    const traces = tr.traces;
    const sp = span(traces);
    if (!sp) return;
    const [t0, t1] = sp;
    const px = (t) => pad.l + ((t - t0) / (t1 - t0)) * pw;
    for (let i = 0; i <= 10; i += 2) {
      const t = t0 + ((t1 - t0) * i) / 10;
      const label = fmtTime(t);
      const x = pad.l + (i * pw) / 10 - (i === 10 ? ctx.measureText(label).width : i ? ctx.measureText(label).width / 2 : 0);
      ctx.fillText(label, x, h - 6);
    }

    // The trigger: time zero, and the level on its channel.
    if (t0 <= 0 && t1 >= 0) {
      ctx.strokeStyle = token("--warn");
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(px(0) + 0.5, pad.t); ctx.lineTo(px(0) + 0.5, pad.t + ph); ctx.stroke();
      ctx.setLineDash([]);
    }

    CHANNELS.forEach((name, i) => {
      const trace = traces[name], c = channel(name);
      if (!trace || !c) return;
      const full = volts(c.range);
      if (!full) return;
      const offset = typeof c.offset === "number" ? c.offset : 0;
      const py = (v) => mid - Math.max(-1.02, Math.min(1.02, (v + offset) / full)) * (ph / 2);
      ctx.strokeStyle = token(`--trace-${i + 1}`);
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      let pen = false;
      for (let k = 0; k < trace.t.length; k++) {
        const v = trace.v[k];
        if (v === null) { pen = false; continue; }
        const x = px(trace.t[k]), y = py(v);
        if (pen) ctx.lineTo(x, y); else { ctx.moveTo(x, y); pen = true; }
      }
      ctx.stroke();

      if (snap.scope.trigger_source === name && snap.scope.trigger_mode !== "Free run"
          && typeof snap.scope.trigger_level === "number") {
        const y = Math.round(py(snap.scope.trigger_level)) + 0.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = token(`--trace-${i + 1}`);
        ctx.fillText("T", pad.l + pw - 10, y - 3);
      }
    });

    if (hoverX !== null && hoverX >= pad.l && hoverX <= pad.l + pw) {
      ctx.strokeStyle = token("--muted");
      ctx.beginPath(); ctx.moveTo(hoverX + 0.5, pad.t); ctx.lineTo(hoverX + 0.5, pad.t + ph); ctx.stroke();
      const t = t0 + ((hoverX - pad.l) / pw) * (t1 - t0);
      const parts = [`t ${fmtTime(t)}`];
      for (const name of Object.keys(traces)) parts.push(`${name} ${fmtVolts(at(traces[name], t))}`);
      $("cursor-readout").textContent = parts.join(" · ");
    }
  }

  // The sample nearest *t* (traces are in time order).
  function at(trace, t) {
    let lo = 0, hi = trace.t.length - 1;
    if (hi < 0) return null;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (trace.t[m] < t) lo = m; else hi = m;
    }
    return Math.abs(trace.t[lo] - t) <= Math.abs(trace.t[hi] - t) ? trace.v[lo] : trace.v[hi];
  }

  canvas.addEventListener("pointermove", (ev) => {
    hoverX = ev.clientX - canvas.getBoundingClientRect().left;
    draw();
  });
  canvas.addEventListener("pointerleave", () => {
    hoverX = null;
    $("cursor-readout").textContent = "";
    draw();
  });
  window.addEventListener("resize", draw);

  // ---- the stream -----------------------------------------------------------
  function open() {
    if (source) return;
    source = new EventSource(root.dataset.eventsUrl + "?traces=1");
    source.addEventListener("state", (ev) => {
      snap = JSON.parse(ev.data);
      showState();
      showSettings();
      draw();
    });
    source.onerror = () => {
      const chip = $("scope-state");
      chip.dataset.state = "failed";
      chip.textContent = "panel service unreachable";
    };
  }
  function close() {
    if (source) { source.close(); source = null; }
  }
  // Traces cost the IOC and the VPN: none while the tab is hidden.
  document.addEventListener("visibilitychange", () => (document.hidden ? close() : open()));
  open();
  draw();
})();
