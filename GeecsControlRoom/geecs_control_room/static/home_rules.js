// The home's rules, pure: what a list says from /api/home/devices' body (a
// list the module could not give reads "unknown", never "none"; the alarm
// list is "none" only when every device's roles were read, `unread` empty),
// a preset's row, the strip's baseline chip, a diff row's state, a restore's plan.
// window.GeecsHomeRules in the page, module.exports under node.
(function (rules) {
  "use strict";

  function listView(key, body) {
    const rows = body.ok ? body[key] : null;
    const unread = rows && key === "in_alarm" ? (body.unread || []).map((d) => d.name) : [];
    const count = !rows ? "?" : unread.length ? `${rows.length}+?` : String(rows.length);
    const none = !!rows && !rows.length && !unread.length;
    const busy = rows && rows.length;
    const plural = unread.length > 1 ? "s" : "";
    return {
      rows: rows || [],
      count,
      none,
      chipState: busy ? (key === "in_alarm" ? "failed" : "degraded") : none ? "ok" : "unknown",
      chipText: busy ? count : none ? "none" : "unknown",
      title: unread.length ? `no roles read for ${unread.length}: ${unread.join(", ")}` : "",
      note: !rows ? body.detail || "" : unread.length ? `the module gave no roles for ${unread.length} device${plural}: ${unread.join(", ")}` : "",
    };
  }


  // A preset's row: when, by whom, the scope in words, the golden mark.
  function presetRow(p) {
    const kinds = Object.entries(p.scope || {}).filter(([, names]) => names && names.length);
    const scope = kinds.length ? kinds.map(([k, names]) => `${k}: ${names.join(", ")}`).join(" · ") : "everything";
    return { when: new Date(p.time).toLocaleString(), by: p.operator || "—", scope, golden: !!p.golden, count: p.pvCount };
  }

  // The strip's "differs from baseline": the golden snapshots' differing rows, summed; none without a golden.
  // A row that could not be compared (unread, unsaved, retired, unsupported) makes the verdict incomplete:
  // "as baseline" is said only when every row was.
  function baselineChip(body) {
    if (!body || !body.ok) return { hidden: false, text: "unknown", state: "unknown", title: (body && body.detail) || "" };
    if (!body.golden) return { hidden: true, text: "", state: "unknown", title: "" };
    const n = body.differs, k = body.incomplete || 0;
    const title = (body.presets || []).map((p) => `${p.name}: ${p.differs} differ${p.incomplete ? `, ${p.incomplete} not compared` : ""}`).join(", ");
    if (k) return { hidden: false, text: `${n ? `${n} differ · ` : ""}${k} not compared`, state: "unknown", title };
    return { hidden: false, text: n ? `${n} differ${n === 1 ? "s" : ""}` : "as baseline", state: n ? "degraded" : "ok", title };
  }

  // The diff drawer's selection: opening it for a preset starts a new generation; a diff that answers for an
  // older generation (opened for another preset meanwhile) or for a drawer no longer open is dropped, never shown.
  // The drawer's own openness is the fourth guard: the kit closes it on Esc, the scrim or Close, none of which the page sees.
  const openSelection = (state, preset) => ({ generation: (state ? state.generation : 0) + 1, preset, rows: [] });
  // Whether an answer still belongs to the drawer: it is open and on the generation the request was made for.
  // Every answer the drawer awaits (a diff of any status, a restore's) goes through this one predicate.
  const stillCurrent = (state, generation, open) => !!open && !!state && generation === state.generation;
  const takeDiff = (state, generation, rows, open) => (stillCurrent(state, generation, open) ? { ...state, rows } : null);

  // The drawer's summary of a diff: a row was compared only when its word is same or differs, as the baseline counts.
  function diffSummary(rows) {
    const differs = rows.filter((r) => r.word === "differs").length;
    const notCompared = rows.filter((r) => r.word !== "same" && r.word !== "differs").length;
    return `${rows.length} rows · ${differs} differ · ${notCompared} not compared; tick the devices to restore`;
  }

  // A diff row's word as a kit state; "unsaved: unreachable at save time" explains a null.
  const DIFF = { same: "ok", differs: "degraded", unsaved: "unknown", retired: "unknown", unread: "unknown", unsupported: "unknown" };
  const diffState = (word) => DIFF[word] || "unknown";
  const diffNote = (word) => (word === "unsaved" ? "unsaved: unreachable at save time" : word);

  // What a restore would write: the differing rows of the picked devices; nothing picked is nothing to restore
  // (the drawer ticks the differing devices itself, so an empty pick is the operator's clearing).
  function restorePlan(rows, picked) {
    const chosen = picked.length ? rows.filter((r) => r.word === "differs" && picked.includes(r.device)) : [];
    const devices = [...new Set(chosen.map((r) => r.device))];
    const text = chosen.length ? `${chosen.length} row${chosen.length === 1 ? "" : "s"} on ${devices.join(", ")}` : picked.length ? "nothing differs on the picked devices" : "no device picked";
    return { rows: chosen, devices, text };
  }

  Object.assign(rules, { listView, presetRow, baselineChip, openSelection, stillCurrent, takeDiff, diffSummary, diffState, diffNote, restorePlan });
})(typeof module === "object" ? module.exports : (window.GeecsHomeRules = {}));
