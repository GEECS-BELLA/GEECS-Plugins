// The device directory's rules, pure: which rows a filter keeps (every term a
// substring of the name or the family, any case), which groups it hides, a
// row's two words from /api/home/devices' `words` (the module's state and the
// IOC heartbeat; "unknown" whenever the module is silent, never a guess), and
// the panel chip (how many of the listed devices are live).
// window.GeecsDirectoryRules in the page, module.exports under node.
(function (rules) {
  "use strict";

  const terms = (query) => String(query || "").toLowerCase().split(/\s+/).filter(Boolean);

  // Every term of the query is in the name or the family; an empty query keeps every row.
  function matches(query, name, family) {
    const hay = `${name || ""} ${family || ""}`.toLowerCase();
    return terms(query).every((t) => hay.includes(t));
  }

  // groups: [{group, rows: [{name, family}]}] as the page draws them. The rows each group keeps,
  // the groups left with none (hidden), and whether nothing at all matched.
  function filterView(groups, query) {
    const shown = {}, hidden = [];
    let matched = 0, total = 0;
    for (const g of groups) {
      const keep = g.rows.filter((r) => matches(query, r.name, r.family)).map((r) => r.name);
      shown[g.group] = keep;
      if (!keep.length) hidden.push(g.group);
      matched += keep.length;
      total += g.rows.length;
    }
    return { shown, hidden, matched, total, none: total > 0 && matched === 0 };
  }

  const UNKNOWN = { state: "unknown", kit: "unknown", heartbeat: "unknown", heartbeatKit: "unknown" };

  // A row's words: the module's, as /api/home/devices answered them. A silent module, or a name the answer
  // does not carry (the registry changed since the page was drawn), is unknown; a null heartbeat is no IOC.
  function cellWords(body, name) {
    const w = body && body.ok && body.words ? body.words[name] : null;
    if (!w) return UNKNOWN;
    const beat = w.heartbeat === null || w.heartbeat === undefined ? "no IOC" : w.heartbeat;
    return { state: w.state, kit: w.kit || "unknown", heartbeat: beat, heartbeatKit: w.heartbeatKit || "unknown" };
  }

  // The panel's chip: "N/M live" over the listed names; unknown while the module is silent.
  function chip(body, names) {
    if (!body || !body.ok) return { text: "unknown", state: "unknown", title: (body && body.detail) || "" };
    const live = names.filter((n) => cellWords(body, n).state === "live").length;
    const text = `${live}/${names.length} live`;
    return { text, state: !names.length ? "unknown" : live === names.length ? "ok" : "degraded", title: "" };
  }

  Object.assign(rules, { matches, filterView, cellWords, chip });
})(typeof module === "object" ? module.exports : (window.GeecsDirectoryRules = {}));
