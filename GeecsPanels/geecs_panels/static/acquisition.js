// The camera's own controls: exposure, period, gain, trigger mode,
// start/stop, and the camera (driver) ROI — what scans save.
//
// A scan run holds the IOC's run lock (RUN:Lock) from its first preparation
// until the camera is restored, and the IOC refuses these writes from
// anyone but the scan worker meanwhile. The page shows that ("a scan owns
// the camera", with the run's owner) and disables the controls; the
// service refuses too (423), and a lock that cannot be read counts as held.
// A stuck lock (a killed worker) is released through a dialog that names
// the run and asks who is releasing it; it writes RUN:Lock 0 and nothing else.
(() => {
  "use strict";

  const V = window.GeecsCameraView;
  if (!V) return;
  const $ = (id) => document.getElementById(id);
  const { post, put, putAll, sayError } = V.write;
  const writable = V.root.dataset.writable === "true";
  const box = $("camera-controls");
  const controls = () => box.querySelectorAll("input, select, button:not(#unlock-open)");
  let lock = "unknown", acq = {}, owner = "", capture = "", imageAge = null;

  const focused = (el) => document.activeElement === el;
  const num = (input) => {
    const raw = input.value.trim(), n = Number(raw);
    return raw === "" || !Number.isFinite(n) ? null : n;
  };
  const fill = (id, value) => {
    const el = $(id);
    if (!focused(el) && value !== null && value !== undefined) el.value = value;
  };

  // The controls start disabled (template) and go back to disabled whenever
  // the lock's state is not known to be free: a lost stream included.
  function showLock() {
    $("lock-denied").hidden = lock !== "locked";
    $("lock-unknown").hidden = lock !== "unknown";
    for (const el of controls()) el.disabled = !writable || lock !== "free";
  }
  V.hooks.streamLost.push(() => { lock = "unknown"; showLock(); });

  V.hooks.snapshot.push((snap) => {
    lock = (snap.lock && snap.lock.state) || "unknown";
    owner = (snap.lock && snap.lock.owner) || "";
    capture = (snap.lock && snap.lock.capture) || "";
    imageAge = snap.image ? snap.image.age : null;
    acq = snap.acquisition || {};
    $("lock-owner").textContent = owner || "(no owner recorded)";
    $("unlock-open").disabled = !writable;
    showLock();
    const exposure = snap.readings.find((r) => r.key === "exposure");
    if (exposure && typeof exposure.value === "number") {
      fill("cam-exposure", Number((exposure.value * 1000).toPrecision(6)));
    }
    if (typeof acq.period === "number") fill("cam-period", Number((acq.period * 1000).toPrecision(6)));
    if (typeof acq.gain === "number") fill("cam-gain", Number(acq.gain.toPrecision(4)));
    fill("cam-trigger", acq.trigger_mode);
    $("cam-source").textContent = acq.trigger_source ? "source " + acq.trigger_source : "";
    fill("cam-minx", acq.min_x); fill("cam-miny", acq.min_y);
    fill("cam-sizex", acq.size_x); fill("cam-sizey", acq.size_y);
  });

  // ---- exposure, period, gain, trigger, start/stop ------------------------
  const onNumber = (id, pv, scale) => $(id).addEventListener("change", (ev) => {
    const v = num(ev.target);
    if (v !== null) put(pv, v * scale).catch(() => {});
  });
  onNumber("cam-exposure", "cam1:AcquireTime", 0.001);
  onNumber("cam-period", "cam1:AcquirePeriod", 0.001);
  onNumber("cam-gain", "cam1:Gain", 1);
  $("cam-trigger").addEventListener("change", (ev) => {
    put("cam1:TriggerMode", ev.target.value).catch(() => {});
  });
  $("cam-start").addEventListener("click", () => put("cam1:Acquire", 1).catch(() => {}));
  $("cam-stop").addEventListener("click", () => put("cam1:Acquire", 0).catch(() => {}));

  // ---- the camera ROI: behind a dialog (it changes what scans save) -------
  let pendingRoi = null;
  function askRoi(roi) {
    pendingRoi = roi;
    $("roi-dialog-text").textContent =
      "x " + roi.x + "  y " + roi.y + "  width " + roi.w + "  height " + roi.h;
    $("roi-dialog").showModal();
  }
  $("cam-roi-apply").addEventListener("click", () => {
    const x = num($("cam-minx")), y = num($("cam-miny"));
    const w = num($("cam-sizex")), h = num($("cam-sizey"));
    if ([x, y, w, h].some((v) => v === null)) return sayError("camera ROI: fill in all four");
    askRoi({ x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h) });
  });
  $("cam-roi-full").addEventListener("click", () => {
    if (acq.max_x && acq.max_y) askRoi({ x: 0, y: 0, w: acq.max_x, h: acq.max_y });
  });
  $("roi-confirm").addEventListener("click", () => {
    $("roi-dialog").close();
    const r = pendingRoi;
    if (!r) return;
    // Offsets, then sizes: the last size put is checked against the final
    // offset, so a clamp along the way never sticks.
    putAll([["cam1:MinX", r.x], ["cam1:MinY", r.y], ["cam1:SizeX", r.w], ["cam1:SizeY", r.h]])
      .catch(() => {});
  });

  // ---- the stuck-lock override -------------------------------------------
  $("unlock-open").addEventListener("click", () => {
    const age = imageAge === null ? "unknown" : imageAge.toFixed(1) + " s ago";
    $("unlock-facts").textContent =
      "owner: " + (owner || "none") + " · HDF capture: " + (capture || "?") +
      " · last frame: " + age;
    $("unlock-operator").value = "";
    $("unlock-dialog").showModal();
  });
  $("unlock-confirm").addEventListener("click", () => {
    const operator = $("unlock-operator").value.trim();
    if (!operator) { $("unlock-operator").focus(); return; }
    $("unlock-dialog").close();
    post(V.root.dataset.unlockUrl, { operator: operator })
      .catch((e) => sayError("override: " + e.message));
  });
})();
