// The power-supply panel: one card, every PV named by the device's roles.
//
// One stream (/supply/<name>/events) carries the values by key, the bits by
// label and the state word (judged by the service, so the page copies a kit
// word and never invents one). A SETPOINT posts only on Set or Enter in its
// field — never on focus leaving the field, because it energizes a magnet.
// The supply's answer arrives on the stream; a refusal (outside the drive
// limits, not permitted, the supply's #NAK) is said under the card.
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const root = $("supply");
  const writable = root.dataset.writable === "true";
  const card = root.querySelector(".supply");
  let snap = null, source = null;

  const slot = (key) => card.querySelector(`[data-k="${key}"]`);
  const text = (key, value) => { const el = slot(key); if (el) el.textContent = value; };
  const fmt = (v, prec) => {
    if (typeof v !== "number") return "—";
    const p = typeof prec === "number" && prec >= 0 && prec <= 10 ? prec : 3;
    return v.toFixed(p);
  };

  // ---- writes ---------------------------------------------------------------
  function sayError(message) { text("error", message || ""); }

  async function put(action, value) {
    sayError("");
    try {
      const r = await fetch(root.dataset.putUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: action, value: value }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        sayError(`${action}: ${body.detail || r.status}`);
      }
    } catch (e) {
      sayError(`${action}: ${e}`);
    }
  }

  const number = (input) => {
    const raw = input.value.trim(), n = Number(raw);
    return raw === "" || !Number.isFinite(n) ? null : n;
  };

  for (const input of card.querySelectorAll("[data-action]")) {
    input.disabled = !writable;
    // Typed but not sent: the stream must not put the record's value back (a
    // click on Set takes the focus off the field before it fires).
    input.addEventListener("input", () => { input.dataset.dirty = "1"; });
    input.addEventListener("keydown", (ev) => { if (ev.key === "Escape") { delete input.dataset.dirty; input.blur(); } });
    const set = () => {
      const n = number(input);
      if (n !== null) { delete input.dataset.dirty; put(input.dataset.action, n); }
    };
    // One set path: Set, or Enter here. Leaving the field sets nothing.
    input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { set(); input.blur(); } });
    const btn = card.querySelector(`[data-set="${input.dataset.action}"]`);
    if (btn) { btn.disabled = !writable; btn.addEventListener("click", set); }
  }
  for (const btn of card.querySelectorAll("[data-press]")) {
    btn.disabled = !writable;
    btn.addEventListener("click", () => put(btn.dataset.press, Number(btn.dataset.value)));
  }

  // ---- render ---------------------------------------------------------------
  function limits(v, role) {
    const lo = v[`${role}.drvl`], hi = v[`${role}.drvh`], egu = v[`${role}.egu`] || "";
    const prec = v[`${role}.prec`];
    const now = typeof v[role] === "number" ? ` · setpoint ${fmt(v[role], prec)} ${egu}` : "";
    if (typeof lo !== "number" || typeof hi !== "number") return `drive limits not read${now}`;
    if (lo >= hi) return `no drive limits: setting refused${now}`;
    // The IOC refuses at HIHI/LOLO when their severity is INVALID (IVOA).
    const hihi = v[`${role}.hhsv`] === "INVALID" ? ` · IOC refuses ≥ ${fmt(v[`${role}.hihi`], prec)}` : "";
    const lolo = v[`${role}.llsv`] === "INVALID" ? ` · IOC refuses ≤ ${fmt(v[`${role}.lolo`], prec)}` : "";
    return `drive limits ${fmt(lo, prec)} to ${fmt(hi, prec)} ${egu}${hihi}${lolo}${now}`;
  }

  function render() {
    if (!snap) return;
    const v = snap.values || {};
    const top = $("supply-state");
    top.dataset.state = snap.state;
    top.textContent = snap.reachable ? snap.state : "IOC unreachable";
    const chip = slot("state");
    chip.dataset.state = snap.state;
    chip.textContent = snap.status;

    text("current", fmt(v.current, v["current.prec"]));
    text("current-egu", v["current.egu"] || "");
    text("voltage", fmt(v.voltage, v["voltage.prec"]));
    text("voltage-egu", v["voltage.egu"] || "");
    const out = slot("output");
    out.dataset.state = v.output === 1 ? "running" : v.output === 0 ? "ok" : "unknown";
    out.textContent = v.output === 1 ? "on" : v.output === 0 ? "off" : "—";
    text("loop", typeof v.loop_mode === "string" ? `loop: ${v.loop_mode}` : "");

    for (const role of ["current_sp", "voltage_sp"]) {
      const key = role.replace("_sp", "-sp");
      const egu = v[`${role}.egu`] || "";
      text(key, typeof v[role] === "number" ? `${fmt(v[role], v[`${role}.prec`])} ${egu}` : "—");
      text(`${key}-hint`, limits(v, role));
      const input = card.querySelector(`[data-action="${role.replace("_sp", "")}"]`);
      if (input && document.activeElement !== input && !input.dataset.dirty && typeof v[role] === "number") input.value = v[role];
    }

    for (const bit of snap.bits || []) {
      const el = card.querySelector(`[data-bit="${bit.label}"]`);
      if (el) el.dataset.state = bit.value === 1 ? "failed" : bit.value === 0 ? "ok" : "unknown";
    }
    text("status", snap.reachable ? "" : "no PV of this supply has connected");
    const alarms = ["current", "current_sp", "output_sp"]
      .map((k) => `${k} ${v[`${k}.sevr`] || "—"}${v[`${k}.stat`] && v[`${k}.stat`] !== "NO_ALARM" ? " " + v[`${k}.stat`] : ""}`);
    text("detail", `alarms: ${alarms.join(" · ")}` +
      (snap.missing && snap.missing.length ? ` · not connected: ${snap.missing.join(" ")}` : ""));
  }

  // ---- the stream -----------------------------------------------------------
  function open() {
    if (source) return;
    source = new EventSource(root.dataset.eventsUrl);
    source.addEventListener("state", (ev) => { snap = JSON.parse(ev.data); render(); });
    source.onerror = () => {
      const top = $("supply-state");
      top.dataset.state = "failed";
      top.textContent = "panel service unreachable";
    };
  }
  function close() {
    if (source) { source.close(); source = null; }
  }
  document.addEventListener("visibilitychange", () => (document.hidden ? close() : open()));
  open();
})();
