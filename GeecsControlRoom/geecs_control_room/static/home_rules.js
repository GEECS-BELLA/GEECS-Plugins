// The home's list rule, pure: what a list says from /api/home/devices' body.
// A list the module could not give reads "unknown", never "none"; the alarm
// list is "none" only when every device's roles were read (`unread` empty).
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

  rules.listView = listView;
})(typeof module === "object" ? module.exports : (window.GeecsHomeRules = {}));
