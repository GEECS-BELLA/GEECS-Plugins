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
  function baselineChip(body) {
    if (!body || !body.ok) return { hidden: false, text: "unknown", state: "unknown", title: (body && body.detail) || "" };
    if (!body.golden) return { hidden: true, text: "", state: "unknown", title: "" };
    const n = body.differs;
    return { hidden: false, text: n ? `${n} differ${n === 1 ? "s" : ""}` : "as baseline", state: n ? "degraded" : "ok", title: (body.presets || []).map((p) => `${p.name}: ${p.differs}`).join(", ") };
  }

  // A diff row's word as a kit state; "unsaved: unreachable at save time" explains a null.
  const DIFF = { same: "ok", differs: "degraded", unsaved: "unknown", retired: "unknown", unread: "unknown", unsupported: "unknown" };
  const diffState = (word) => DIFF[word] || "unknown";
  const diffNote = (word) => (word === "unsaved" ? "unsaved: unreachable at save time" : word);

  // What a restore would write: the differing rows among the picked devices (every device when none is picked).
  function restorePlan(rows, picked) {
    const chosen = rows.filter((r) => r.word === "differs" && (!picked.length || picked.includes(r.device)));
    const devices = [...new Set(chosen.map((r) => r.device))];
    return { rows: chosen, devices, text: chosen.length ? `${chosen.length} row${chosen.length === 1 ? "" : "s"} on ${devices.join(", ")}` : "nothing differs" };
  }

  Object.assign(rules, { listView, presetRow, baselineChip, diffState, diffNote, restorePlan });
})(typeof module === "object" ? module.exports : (window.GeecsHomeRules = {}));
