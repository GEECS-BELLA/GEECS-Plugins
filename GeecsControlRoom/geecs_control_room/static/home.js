// The Control Room's live half: the strip from /api/home/strip (the scanner's
// words), the two lists from /api/home/devices (the module's words, sorted
// there), the device pickers from the same roster. Nothing here judges a
// device: a list the module could not give reads "unknown", never "none".
// The list rule is home_rules.js's (tested under node), the polling poller.js's.
(function () {
  "use strict";
  const POLL_MS = 2000, MAX_POINTS = 600;
  const root = document.body.dataset.root || "";
  const $ = (id) => document.getElementById(id);
  const deviceHref = (name) => `${root}/api/devices/${encodeURIComponent(name)}`;

  function cell(text) {
    const td = document.createElement("td");
    td.textContent = text === null || text === undefined || text === "" ? "—" : text;
    return td;
  }

  function lastCommand(row) {
    if (!row) return "—";
    const axis = row.axis ? `[${row.axis}]` : "", code = row.canopy_code ? ` · ${row.canopy_code}` : "";
    const when = new Date(row.time).toLocaleTimeString();
    return `${row.outcome}${code} · ${row.role}${axis} ← ${JSON.stringify(row.value)} · ${row.operator} · ${when}`;
  }

  function list(key, body) {
    const journal = body.journal !== false; // a journal the module could not give is unknown, not "no command"
    const panel = document.querySelector(`[data-list="${key}"]`), v = window.GeecsHomeRules.listView(key, body);
    const chip = panel.querySelector("header .chip"), pane = panel.querySelector(".state"), scroll = panel.querySelector(".tscroll");
    $(`n-${key}`).textContent = v.count;
    chip.dataset.state = v.chipState;
    chip.textContent = v.chipText;
    chip.title = v.title;
    scroll.hidden = !v.rows.length;
    pane.hidden = !scroll.hidden;
    pane.dataset.state = v.none ? "empty" : "error";
    pane.querySelector(".t").textContent = v.none ? "none" : "unknown";
    pane.querySelector(".m").textContent = v.note;
    panel.querySelector("tbody").replaceChildren(...v.rows.map((d) => {
      const tr = document.createElement("tr"), name = cell(""), link = document.createElement("a");
      link.href = deviceHref(d.name);
      link.textContent = d.name;
      name.className = "id";
      name.replaceChildren(link);
      const state = cell(""), dot = document.createElement("span");
      dot.className = "dot";
      dot.dataset.state = d.kit;
      state.replaceChildren(dot, ` ${d.state}`);
      tr.append(name, cell(d.family), state, cell(d.cause), cell(d.ioc), cell(journal ? lastCommand(d.last) : "unknown"));
      return tr;
    }));
  }

  let filled = "";
  function pickers(devices) {
    const signature = JSON.stringify(devices);
    if (signature === filled) return;
    filled = signature;
    for (const select of document.querySelectorAll("select[data-families]")) {
      if (select.dataset.reason) continue; // disabled by the launcher, with its reason
      const families = select.dataset.families.split(" ");
      const names = devices.filter((d) => families.includes(d.family)).map((d) => d.name);
      select.replaceChildren(
        new Option(names.length ? "choose a device…" : "no such device", ""),
        ...names.map((n) => new Option(n, n)),
      );
      select.disabled = !names.length;
    }
  }
  document.addEventListener("change", (ev) => {
    const select = ev.target.closest("select[data-families]");
    if (select && select.value) window.location.href = deviceHref(select.value);
  });

  let points = [], scan = null;
  function strip(s) {
    const run = $("run");
    run.dataset.state = s.kit;
    run.textContent = s.run;
    $("scanner").textContent = s.scanner === "ok" ? "scanner" : `scanner ${s.scanner}`;
    const known = s.scan_number !== null && s.scan_number !== undefined;
    $("scan").textContent = known ? String(s.scan_number) : "—";
    $("shot").textContent = known ? `${s.shots_done ?? 0}${s.planned_total ? ` / ${s.planned_total}` : ""}` : "—";
    if (s.scan_number !== scan) { scan = s.scan_number; points = []; }
    // the chip is the manager's word; a scan is open only while its documents say so (not an mv)
    const open = s.state === "running" || s.state === "paused";
    const active = open && (s.run === "running" || s.run === "paused");
    if (active) points = [...points, [Date.now(), s.shots_done || 0]].slice(-MAX_POINTS);
    $("plot-fig").hidden = !active || points.length < 2;
    if ($("plot-fig").hidden) return;
    const t0 = points[0][0], span = points[points.length - 1][0] - t0 || 1;
    const top = Math.max(1, s.planned_total || 0, ...points.map((p) => p[1]));
    $("plot").setAttribute("points", points
      .map(([t, y]) => `${(((t - t0) / span) * 240).toFixed(1)},${(46 - (y / top) * 44).toFixed(1)}`)
      .join(" "));
  }

  // a read that fails or times out backs off; while it does, the page says the module is slow
  const DEVICES = "/api/home/devices", STRIP = "/api/home/strip";
  const draw = {
    [DEVICES]: (body) => {
      list("not_live", body);
      list("in_alarm", body);
      if (body.ok) pickers(body.devices);
    },
    [STRIP]: strip,
  };
  const silent = {
    [DEVICES]: (why) => ({ ok: false, detail: `module slow: the Control Room gave no list (${why}); asking again shortly` }),
    [STRIP]: () => ({ scanner: "unreachable", run: "unknown", kit: "unknown" }),
  };
  const poller = window.GeecsPoller.makePoller({
    fetch: async (path, signal) => (await fetch(root + path, { signal })).json(),
    pollMs: POLL_MS,
    onView: (path, got) => draw[path](got.ok ? got.value : silent[path](got.error)),
  });
  const tick = () => poller.tick([DEVICES, STRIP]);
  tick();
  setInterval(tick, POLL_MS);
})();
