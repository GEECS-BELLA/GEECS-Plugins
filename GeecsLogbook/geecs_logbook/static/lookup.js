/* The day page's "Look up a variable" panel.
 *
 * Folded until a person opens it; the first open fetches
 * /api/day/{day}/lookup, which reads two lines of each scan's s-file. The
 * table's columns are the day's scan-log lists (the scanner's shared list)
 * unless the URL names others (?col=…, kept in sync so a link to the table
 * can be pasted into a note); "Add columns…" opens the kit's drawer picker
 * (GeecsKit.picker — the scanner's device picker) over every column any
 * scan recorded. A value that differs from the scan above it is marked. Reads
 * its facts off <main id="logbook">; sets no colours (the theme guard).
 */
(() => {
  "use strict";
  const host = document.getElementById("logbook");
  const panel = document.querySelector("details[data-lookup]");
  if (!host || !panel) return;
  const API = host.dataset.api || "";
  const DAY = host.dataset.day || "";
  const $ = (id) => document.getElementById(id);
  const MAX_SHOWN = 80;
  // Why a scan has no values (ScanSummary.values_state), as a cell's tooltip.
  const NOTE = {
    recording: "No s-file yet: the scan has not closed",
    no_sfile: "This scan wrote no s-file",
    no_shots: "The scan ended before its first shot",
  };

  const L = { data: null, cols: null, loaded: false, seq: 0 };

  const urlCols = () => new URLSearchParams(location.search).getAll("col");

  function syncUrl() {
    const params = new URLSearchParams(location.search);
    params.delete("col");
    (L.cols || []).forEach((c) => params.append("col", c));
    const q = params.toString();
    history.replaceState(null, "", location.pathname + (q ? "?" + q : "") + location.hash);
  }

  async function load() {
    const seq = ++L.seq;
    const params = new URLSearchParams();
    if (L.cols !== null) {
      if (!L.cols.length) params.append("col", "");
      L.cols.forEach((c) => params.append("col", c));
    }
    $("lk-note").hidden = false;
    $("lk-note").textContent = "Reading the day's s-files…";
    try {
      const res = await fetch(`${API}/day/${DAY}/lookup${params.toString() ? "?" + params : ""}`);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      if (seq !== L.seq) return;
      L.data = data;
      L.cols = data.shown.filter((c) => c);
      render();
      renderPicker();  // a drawer opened while the day was being read fills now
    } catch (err) {
      if (seq !== L.seq) return;
      // A first read that failed (a share blip, a 503) must not stick:
      // reopening the panel tries again rather than needing a page reload.
      if (!L.data) L.loaded = false;
      renderPicker();
      $("lk-note").textContent = `Could not read the day: ${err.message}` +
        (L.data ? "" : " — close and reopen this panel to try again.");
    }
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function split(name) {
    const i = name.indexOf(" ");
    return i > 0 ? [name.slice(0, i), name.slice(i + 1)] : ["", name];
  }

  function render() {
    const data = L.data;
    const listed = new Set(data.default_columns);
    const scanQ = $("lk-scan").value.trim();
    const note = $("lk-note");
    if (!L.cols.length) {
      note.hidden = false;
      note.textContent = data.columns.length
        ? "No columns yet: no scan today carries the scanner's list. Add one above."
        : "No scan today has an s-file yet — they are written when a scan closes.";
    } else {
      note.hidden = true;
    }
    const table = $("lk-table");
    table.textContent = "";
    if (!L.cols.length) return;
    const thead = el("thead");
    const hr = el("tr");
    ["Scan", "Start", "Scan variable"].forEach((t) => hr.appendChild(el("th", "", t)));
    L.cols.forEach((name) => {
      const [dev, variable] = split(name);
      const th = el("th", listed.has(name) ? "v listed" : "v");
      th.appendChild(el("span", "dev", dev));
      const line = el("span", "var", variable);
      const x = el("button", "x", "×");
      x.type = "button";
      x.setAttribute("aria-label", `Remove column ${name}`);
      x.addEventListener("click", () => { L.cols = L.cols.filter((c) => c !== name); syncUrl(); load(); });
      line.appendChild(x);
      th.appendChild(line);
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el("tbody");
    const prev = {};
    data.rows.forEach((row) => {
      const cells = L.cols.map((name) => {
        const v = row.values[name];
        const changed = v != null && prev[name] != null && prev[name] !== v;
        if (v != null) prev[name] = v;
        return { v, changed };
      });
      if (scanQ && Number(scanQ) !== row.number) return;
      const tr = el("tr");
      const id = el("td", "id");
      const a = el("a", "", String(row.number).padStart(3, "0"));
      a.href = `#${row.label}`;
      id.appendChild(a);
      const dot = el("span", "dot");
      dot.dataset.state = row.state;  // a kit word, mapped server-side
      dot.setAttribute("aria-hidden", "true");
      id.appendChild(dot);
      tr.appendChild(id);
      tr.appendChild(el("td", "t", row.started || "—"));
      const sv = el("td", "sv", row.parameter || "—");
      sv.title = row.parameter || "";
      tr.appendChild(sv);
      cells.forEach(({ v, changed }) => {
        const td = el("td", "v" + (v == null ? " na" : "") + (changed ? " chg" : ""), v == null ? "—" : v);
        if (v == null && NOTE[row.values_state]) td.title = NOTE[row.values_state];
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
  }

  /* ---- the column drawer: GeecsKit.picker, the scanner's device picker ---- */
  const picker = window.GeecsKit && window.GeecsKit.picker({
    drawer: $("drw-lookup"),
    opener: $("lk-open"),
    search: $("lkq"),
    list: $("lk-list"),
    add: $("lk-add"),
    all: $("lk-all"),
    none: $("lk-none"),
    noun: ["column", "columns"],
    maxShown: MAX_SHOWN,
    rows: (query) => {
      const needle = query.trim().toLowerCase();
      const total = L.data ? L.data.rows.length : 0;
      return (L.data ? L.data.columns : [])
        .filter((c) => !needle || c.name.toLowerCase().includes(needle))
        .map((c) => {
          const have = (L.cols || []).includes(c.name);
          return {
            name: c.name,
            pickable: !have,
            sub: have ? "a column" : c.listed ? "on the list" : `in ${c.scans} of ${total} scans`,
          };
        });
    },
    onRender: (info) => {
      $("lkq-hint").textContent = L.data
        ? `${info.matches} of ${L.data.columns.length} columns` + (info.matches > info.shown ? " · type to narrow" : "")
        : L.loaded ? "Reading the day's s-files…" : "The day could not be read — close the panel and reopen it to try again.";
    },
    onAdd: (names) => {
      const fresh = names.filter((n) => !(L.cols || []).includes(n));
      if (!fresh.length) return;
      L.cols = (L.cols || []).concat(fresh);
      syncUrl();
      load();
    },
  });
  /** Refill an open drawer when the day's read lands (or fails). */
  function renderPicker() { if (picker) picker.refresh(); }

  $("lk-scan").addEventListener("input", () => { if (L.data) render(); });
  $("lk-scan").addEventListener("keydown", (ev) => ev.stopPropagation());

  panel.addEventListener("toggle", () => {
    if (!panel.open || L.loaded) return;
    L.loaded = true;
    const asked = urlCols();
    L.cols = asked.length ? asked : null;
    load();
  });
  // A link that names columns opens the panel on arrival.
  if (urlCols().length) panel.open = true;
})();
