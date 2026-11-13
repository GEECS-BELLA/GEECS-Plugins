// A device page's writes, DOM only: POST /api/devices/<name>/put {role, value,
// operator[, axis][, confirm]}, each sent once, the module's outcome word and code shown,
// never retried. The operator is the header's name (remembered in this browser);
// a list naming a hazard role (the module's confirm: "hazard") is asked once,
// behind the page's hazard dialog, before any of it is sent; one list at a time
// for the page, and a list stops at its first refusal. Every device page writes
// through it, and so does every screen (one per device). A write is [role, value] or, on a per-axis device,
// [role, value, axis]. `opts.out`: where its outcome is said (a screen widget's own line; else #write-out);
// `opts.note`: a screen's own hazard note, which asks first whatever the module marks.
// window.GeecsDeviceWrites.make(page, changed, refused) → { write(pairs, confirmed, opts), busy(), hazardsOf(pairs) }.
(function () {
  "use strict";
  const C = window.GeecsCameraRules, S = window.GeecsScreenRules, D = window.GeecsDeviceRules;
  const OP_KEY = "geecs-control-room-operator";
  const $ = (id) => document.getElementById(id);

  // page: GeecsDevicePage; changed(): the controls redrawn (disabled while a list runs, enabled after);
  // refused(): after a list stops at a refusal (the camera drops its held image edits)
  // `confirmed`: the caller's own dialog, naming these writes and their hazards, was answered (the camera ROI's).
  function make(page, changed, refused = () => {}) {
    const opInput = $("operator"), dialog = $("hazard-dialog");
    try { opInput.value = localStorage.getItem(OP_KEY) || ""; } catch (_) { /* no storage */ }
    opInput.addEventListener("change", () => { try { localStorage.setItem(OP_KEY, opInput.value.trim()); } catch (_) { /* no storage */ } });
    const sayIn = (out) => (word, text, detail) => { if (!out) return; out.dataset.state = S.outcomeState(word); out.textContent = text; out.title = detail || ""; };
    const hazardsOf = (pairs) => C.hazards(pairs, Object.fromEntries(pairs.map(([role, , axis]) => [role, page.entry(role, axis)])));
    const named = (role, axis) => (axis ? `${axis} · ${role}` : role);
    function ask(pairs, roles, note) {
      $("hazard-dialog-text").textContent = pairs.map(([role, v, axis]) => `${page.name} · ${named(role, axis)} ← ${v}`).join("\n");
      $("hazard-dialog-roles").textContent = roles.length ? `Hazard: ${roles.join(", ")}.` : "";
      $("hazard-dialog-title").textContent = roles.length ? "Write a hazard role?" : "Write?"; // a screen's note alone
      $("hazard-dialog-note").textContent = note || "";
      return new Promise((done) => {
        dialog.onclick = (ev) => { if (ev.target.value) { dialog.close(); done(ev.target.value === "write"); } };
        dialog.onclose = () => done(false); // however it closed: not confirmed; a button's answer came first
        dialog.showModal();
      });
    }
    const run = C.single(async (pairs, confirmed = false, opts = {}) => {
      const say = sayIn(opts.out || $("write-out"));
      const operator = opInput.value.trim();
      if (!operator) { opInput.focus(); say("unknown", "unsent · name yourself in the header first"); return false; }
      changed();
      try {
        const marked = hazardsOf(pairs);
        const asking = (marked.length > 0 || !!opts.note) && !confirmed;
        if (asking && !(await ask(pairs, marked, opts.note))) { say("unknown", "unsent · not confirmed"); return false; }
        const confirm = confirmed || asking;
        for (const [role, v, axis] of pairs) {
          const what = named(role, axis);
          say("sending", `${what} ← ${v} · sending`);
          let a;
          try {
            const body = { role, value: v, operator };
            if (axis) body.axis = axis;
            if (confirm) body.confirm = true;
            const r = await fetch(`${page.root}/api/devices/${encodeURIComponent(page.name)}/put`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
            a = await r.json();
          } catch (_) { a = { outcome: "unknown", code: "no_answer", detail: "read the device and the journal before repeating it" }; }
          const word = a.outcome || "unknown";
          say(word, `${what} ← ${v} · ${word}${a.code ? " · " + a.code : ""}`, a.detail);
          if (!D.accepted(a)) { refused(); return false; }
        }
        return true;
      } finally { setTimeout(changed); } // after single() has let go
    });
    const write = async (pairs, confirmed = false, opts = {}) => {
      const ok = await run(pairs, confirmed, opts);
      if (ok === null) sayIn(opts.out || $("write-out"))("unknown", "unsent · another write is in flight");
      return ok;
    };
    return { write, busy: run.busy, hazardsOf };
  }
  window.GeecsDeviceWrites = { make };
})();
