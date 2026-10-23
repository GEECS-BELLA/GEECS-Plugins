// The processing controls and the saturation warning.
//
// Background (PROC1): grab one from the next frame — with the beam blocked —
// and subtract it or not; the grabbed image lives in the IOC's memory only
// and is lost when the IOC restarts. Transform (TRANS1): rotate and flip
// the display and the statistics. Neither changes what a scan saves (scans
// take raw frames from the driver). Saturation is judged on the RAW frame
// (STAT0's max against the pixel format's full scale, on the server), so
// background subtraction or the analysis ROI cannot hide it.
(() => {
  "use strict";

  const V = window.GeecsCameraView;
  if (!V) return;
  const $ = (id) => document.getElementById(id);
  const { put } = V.write;
  const writable = V.root.dataset.writable === "true";
  const bgOn = $("bg-on"), transform = $("transform");

  let beeping = false, audio = null;
  function beep() {
    // A short tone once a second while saturated, if the viewer asked for it.
    try {
      audio = audio || new AudioContext();
      const osc = audio.createOscillator(), gain = audio.createGain();
      osc.frequency.value = 880;
      gain.gain.value = 0.05;
      osc.connect(gain).connect(audio.destination);
      osc.start();
      osc.stop(audio.currentTime + 0.12);
    } catch (_) { /* no audio: the chip still says it */ }
  }
  setInterval(() => { if (beeping && $("sat-sound").checked) beep(); }, 1000);

  V.hooks.snapshot.push((snap) => {
    const p = snap.processing || {};
    const valid = p.background_valid === 1;
    if (document.activeElement !== bgOn) bgOn.checked = p.background_on === 1;
    bgOn.disabled = !writable || !valid;
    $("bg-state").textContent = valid ? "a background is held" : "no background grabbed";
    if (p.transform && document.activeElement !== transform) transform.value = p.transform;
    $("transform-warning").hidden = !p.transform || p.transform === "None";

    const s = snap.saturation || {};
    const chip = $("saturation");
    if (s.state === "saturated") {
      chip.dataset.state = "failed";
      chip.textContent = "saturated";
    } else if (s.state === "ok") {
      chip.dataset.state = "ok";
      chip.textContent = "peak " + Math.round(100 * s.fraction) + "%";
    } else {
      chip.dataset.state = "unknown";
      chip.textContent = "peak —";
    }
    beeping = s.state === "saturated";
  });

  $("bg-grab").addEventListener("click", () => {
    put("PROC1:SaveBackground", 1).catch(() => {});
  });
  bgOn.addEventListener("change", () => {
    put("PROC1:EnableBackground", bgOn.checked ? 1 : 0).catch(() => {});
  });
  transform.addEventListener("change", () => {
    put("TRANS1:Type", transform.value).catch(() => {});
  });
  // The sound needs a user gesture before the browser lets it play.
  $("sat-sound").addEventListener("change", () => {
    if ($("sat-sound").checked) beep();
  });
})();
