// The positioner panel: one card per axis, every word from the motor record.
//
// One stream (/motor/<name>/events) carries each axis's fields, its MSTA
// flags by name and its state word (judged by the service, so the page
// copies a kit word and never invents one). Each input shows the record's
// value unless it has focus; a change posts one allow-listed write to the
// axis, and the record's answer arrives on the stream (a target outside
// the soft limits is refused by the service first, naming the limits).
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const root = $("motor");
  const writable = root.dataset.writable === "true";
  const cards = Array.from(root.querySelectorAll(".axis"));
  let snap = null, source = null;

  const slot = (card, key) => card.querySelector(`[data-k="${key}"]`);
  const fmt = (v, prec) => {
    if (typeof v !== "number") return "—";
    const p = typeof prec === "number" && prec >= 0 && prec <= 10 ? prec : 3;
    return v.toFixed(p);
  };

  // ---- writes ---------------------------------------------------------------
  function sayError(card, message) { slot(card, "error").textContent = message || ""; }

  async function put(card, axis, field, value) {
    sayError(card, "");
    try {
      const r = await fetch(root.dataset.putUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ axis: axis, field: field, value: value }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        sayError(card, `${axis}${field}: ${body.detail || r.status}`);
      }
    } catch (e) {
      sayError(card, `${axis}${field}: ${e}`);
    }
  }

  const number = (input) => {
    const raw = input.value.trim(), n = Number(raw);
    return raw === "" || !Number.isFinite(n) ? null : n;
  };

  for (const card of cards) {
    const axis = card.dataset.axis;
    for (const el of card.querySelectorAll("[data-field]")) {
      el.disabled = !writable;
      el.addEventListener("change", () => {
        const n = number(el);
        if (n !== null) put(card, axis, el.dataset.field, n);
        el.blur();
      });
      el.addEventListener("keydown", (ev) => { if (ev.key === "Enter") el.blur(); });
    }
    for (const btn of card.querySelectorAll("[data-press]")) {
      btn.disabled = !writable;
      btn.addEventListener("click", () => {
        const field = btn.dataset.press;
        if (field === ".VAL") {
          const n = number(card.querySelector('[data-field=".VAL"]'));
          if (n !== null) put(card, axis, field, n);
        } else {
          put(card, axis, field, 1);
        }
      });
    }
  }

  // ---- render ---------------------------------------------------------------
  function fill(card, key, value) {
    const el = slot(card, key);
    if (el && document.activeElement !== el && value !== null && value !== undefined) el.value = value;
  }

  function showAxis(card, a) {
    const chip = slot(card, "state");
    chip.dataset.state = a.state;
    chip.textContent = a.status;
    slot(card, "desc").textContent = a.desc || "";
    slot(card, "rbv").textContent = fmt(a.rbv, a.prec);
    slot(card, "egu").textContent = a.egu || "";
    slot(card, "motion").textContent = a.state === "running" ? "moving" : (a.dmov === 1 ? "done" : "");
    slot(card, "live").toggleAttribute("data-age", !a.reachable);
    if (!a.reachable) slot(card, "live").dataset.age = "stale";
    slot(card, "status").textContent = a.reachable ? "" : "no field of this record has connected";

    // The meter: where the position sits between the soft limits.
    const meter = slot(card, "meter"), fillEl = slot(card, "fill");
    const span = typeof a.hlm === "number" && typeof a.llm === "number" ? a.hlm - a.llm : 0;
    if (span > 0 && typeof a.rbv === "number") {
      const frac = Math.max(0, Math.min(1, (a.rbv - a.llm) / span));
      fillEl.style.width = (frac * 100).toFixed(1) + "%";
      meter.setAttribute("aria-valuemin", a.llm);
      meter.setAttribute("aria-valuemax", a.hlm);
      meter.setAttribute("aria-valuenow", a.rbv);
      meter.dataset.state = a.state === "degraded" || a.state === "failed" ? "failed" : a.state === "running" ? "running" : "ok";
    } else {
      fillEl.style.width = "0%";
      meter.dataset.state = "unknown";
    }
    slot(card, "llm").textContent = typeof a.llm === "number" ? `low limit ${fmt(a.llm, a.prec)}` : "low limit —";
    slot(card, "hlm").textContent = typeof a.hlm === "number" ? `${fmt(a.hlm, a.prec)} high limit` : "— high limit";

    slot(card, "lls").hidden = !(a.lls || (a.flags && a.flags.minus_ls));
    slot(card, "hls").hidden = !(a.hls || (a.flags && a.flags.plus_ls));
    slot(card, "lvio").hidden = !a.lvio;
    slot(card, "homed").hidden = !(a.flags && a.flags.homed);
    slot(card, "not-homed").hidden = !(a.reachable && a.flags && !a.flags.homed);

    fill(card, "val-hint", null);
    slot(card, "val-hint").textContent = typeof a.val === "number" ? `setpoint ${fmt(a.val, a.prec)} ${a.egu || ""}` : "setpoint —";
    const valInput = card.querySelector('[data-field=".VAL"]');
    if (document.activeElement !== valInput && typeof a.val === "number") valInput.value = +a.val.toFixed(a.prec ?? 3);
    const twv = card.querySelector('[data-field=".TWV"]');
    if (document.activeElement !== twv && typeof a.twv === "number") twv.value = a.twv;
    const velo = card.querySelector('[data-field=".VELO"]');
    if (document.activeElement !== velo && typeof a.velo === "number") velo.value = a.velo;
    const unit = a.egu ? `${a.egu}/s` : "";
    slot(card, "velo-hint").textContent =
      typeof a.vmax === "number" && a.vmax > 0 ? `${unit} · max ${a.vmax}` : `${unit} · no maximum set`;
    slot(card, "accl").textContent = typeof a.accl === "number" ? `${a.accl} s` : "—";
    slot(card, "rdbd").textContent = typeof a.rdbd === "number" ? `${fmt(a.rdbd, a.prec)} ${a.egu || ""}` : "—";

    const on = a.flags ? Object.keys(a.flags).filter((k) => a.flags[k]) : [];
    slot(card, "detail").textContent =
      `alarm ${a.sevr || "—"} ${a.stat && a.stat !== "NO_ALARM" ? a.stat : ""} · ` +
      `offset ${typeof a.off === "number" ? a.off : "—"} · ${a.set || ""} ${a.spmg || ""} · ` +
      `MSTA ${on.length ? on.join(" ") : "—"}` +
      (a.missing && a.missing.length ? ` · not connected: ${a.missing.join(" ")}` : "");
  }

  function render() {
    if (!snap) return;
    const chip = $("motor-state");
    chip.dataset.state = snap.state;
    chip.textContent = snap.reachable ? snap.state : "IOC unreachable";
    for (const a of snap.axes) {
      const card = root.querySelector(`.axis[data-axis="${a.name}"]`);
      if (card) showAxis(card, a);
    }
  }

  // ---- the stream -----------------------------------------------------------
  function open() {
    if (source) return;
    source = new EventSource(root.dataset.eventsUrl);
    source.addEventListener("state", (ev) => { snap = JSON.parse(ev.data); render(); });
    source.onerror = () => {
      const chip = $("motor-state");
      chip.dataset.state = "failed";
      chip.textContent = "panel service unreachable";
    };
  }
  function close() {
    if (source) { source.close(); source = null; }
  }
  document.addEventListener("visibilitychange", () => (document.hidden ? close() : open()));
  open();
})();
