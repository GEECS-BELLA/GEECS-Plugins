// The Plot page's tools from the portal's old Plot tab: the shot filters,
// the bin settings, the display settings, and copying or sending the plot
// to the scan's logbook entry. Each setting is a hidden field of the page
// form (filters, bincfg, display, view); Apply writes it and submits the
// form, so it saves into the viewer's own set like every other control.
// plotter.js calls PlotTools.config() and PlotTools.patch() when it draws;
// map.js calls config() for its two maps, which copy and send the same way.
"use strict";

window.PlotTools = (function () {
  const form = document.getElementById("pick");
  // The XY plot, the 2-D map's box (its two maps are the plots), or the
  // Images page's big view (a trace device's shot is a plot).
  const host = document.getElementById("plot") || document.getElementById("map")
    || document.getElementById("bigview");
  // The server's "$tok:--name" colour placeholders, resolved against the
  // live theme tokens (one walk: traces, ticks, grids, titles alike).
  const token = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  function themed(value) {
    if (typeof value === "string") return value.startsWith("$tok:") ? (token(value.slice(5)) || value) : value;
    if (Array.isArray(value)) return value.map(themed);
    if (value && typeof value === "object") for (const k of Object.keys(value)) value[k] = themed(value[k]);
    return value;
  }
  if (!form || !host) return { config: () => ({}), patch: (layout) => layout, counted: () => {}, themed };
  const data = host.dataset;
  const EXPORT_SCALE = 2;

  const el = (tag, attrs, ...children) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (key === "on") for (const [kind, fn] of Object.entries(value)) node.addEventListener(kind, fn);
      else if (value !== undefined && value !== null) node.setAttribute(key, value);
    }
    for (const child of children) node.append(child);
    return node;
  };
  const parsed = (text, fallback) => { try { return text ? JSON.parse(text) : fallback; } catch (_) { return fallback; } };
  const toast = (text) => { if (window.GeecsKit) window.GeecsKit.toast(text); };
  const drawer = (id, opener) => { if (window.GeecsKit) window.GeecsKit.drawer(document.getElementById(id)).open(opener); };
  const close = () => { if (window.GeecsKit) window.GeecsKit.closeDrawer(); };
  // Apply: the field into the form, the form to the server (into the set).
  const save = (name, value) => { form.elements[name].value = value; form.requestSubmit(); };

  // ---- the shot filters: OR of groups, AND of conditions in a group ----
  const FILTERS = parsed(form.elements.filters.value, { groups: [], nan_policy: "exclude" });
  const passText = document.getElementById("passcount");
  let draft = null;

  function filterJson(model) {
    const active = (model.groups || []).length;
    return active || model.nan_policy !== "exclude" ? JSON.stringify(model) : "";
  }
  function completed(model) {
    return {
      groups: model.groups
        .map((g) => ({
          name: g.name || "",
          enabled: g.enabled !== false,
          conditions: (g.conditions || []).filter((c) => c.column && Number.isFinite(c.low) && Number.isFinite(c.high)),
        }))
        .filter((g) => g.conditions.length),
      nan_policy: model.nan_policy || "exclude",
    };
  }
  function countUrl(filters) {
    const url = new URL("api/filter-count", form.action);
    url.search = new URLSearchParams({ day: data.day, scan: data.scan, filters }).toString();
    return url;
  }
  function draftCount() {
    const model = completed(draft);
    const out = document.getElementById("draftcount");
    if (!model.groups.length) { out.textContent = "no complete conditions: every shot would pass"; return; }
    fetch(countUrl(JSON.stringify(model)))
      .then((r) => r.json().then((b) => (r.ok ? b : Promise.reject(new Error(b.detail || r.statusText)))))
      .then((c) => { out.textContent = `${c.pass} / ${c.total} shots would pass`; })
      .catch((error) => { out.textContent = `Not counted: ${error.message}`; });
  }
  function renderFilters() {
    document.getElementById("nanpolicy").value = draft.nan_policy || "exclude";
    const groups = document.getElementById("fgroups");
    groups.replaceChildren();
    draft.groups.forEach((g, gi) => {
      if (gi) groups.append(el("div", { class: "orjoin" }, "OR"));
      const rows = (g.conditions || []).map((c, ci) => {
        const number = (key) => el("input", {
          class: "val", value: Number.isFinite(c[key]) ? c[key] : "", placeholder: key === "low" ? "low" : "high",
          "aria-label": key, on: { change: (e) => { c[key] = parseFloat(e.target.value); draftCount(); } },
        });
        const mode = el("select", { "aria-label": "within or outside", on: { change: (e) => { c.mode = e.target.value; draftCount(); } } },
          el("option", { value: "within" }, "within"), el("option", { value: "outside" }, "outside"));
        mode.value = c.mode === "outside" ? "outside" : "within";
        return el("div", { class: "frow" },
          el("input", {
            class: "col", list: "filter-columns", value: c.column || "", placeholder: "column", "aria-label": "column",
            on: { change: (e) => { c.column = e.target.value; draftCount(); } },
          }),
          mode, number("low"), el("span", { class: "hint" }, "to"), number("high"),
          el("button", { type: "button", class: "x", "aria-label": "remove condition",
            on: { click: () => { g.conditions.splice(ci, 1); renderFilters(); } } }, "✕"));
      });
      groups.append(el("div", { class: "fgroup" },
        el("div", { class: "ghead" },
          el("input", { value: g.name || "", placeholder: `filter ${gi + 1}`, "aria-label": "group name",
            on: { change: (e) => { g.name = e.target.value; } } }),
          el("span", { class: "spacer" }),
          el("button", { type: "button", class: "x", on: { click: () => { draft.groups.splice(gi, 1); renderFilters(); } } }, "✕ group")),
        ...rows,
        el("div", {}, el("button", { type: "button", class: "btn sm",
          on: { click: () => { (g.conditions = g.conditions || []).push({ column: "", mode: "within", low: 0, high: 1 }); renderFilters(); } } },
          "+ AND condition"))));
    });
    draftCount();
  }
  document.getElementById("filters-open")?.addEventListener("click", (event) => {
    draft = JSON.parse(JSON.stringify(FILTERS));
    draft.groups = draft.groups || [];
    if (!draft.groups.length) draft.groups.push({ name: "", enabled: true, conditions: [] });
    renderFilters();
    drawer("filters-drawer", event.currentTarget);
  });
  document.getElementById("filters-add-group")?.addEventListener("click", () => {
    draft.groups.push({ name: "", enabled: true, conditions: [] });
    renderFilters();
  });
  document.getElementById("nanpolicy")?.addEventListener("change", (e) => { draft.nan_policy = e.target.value; draftCount(); });
  document.getElementById("filters-apply")?.addEventListener("click", () => save("filters", filterJson(completed(draft))));
  // The rail's chips: a group on or off, or gone, saves at once.
  for (const box of document.querySelectorAll("[data-filter-group]")) {
    const i = Number(box.dataset.filterGroup);
    box.addEventListener("change", () => { FILTERS.groups[i].enabled = box.checked; save("filters", filterJson(FILTERS)); });
  }
  for (const button of document.querySelectorAll("[data-filter-remove]")) {
    button.addEventListener("click", () => {
      FILTERS.groups.splice(Number(button.dataset.filterRemove), 1);
      save("filters", filterJson(FILTERS));
    });
  }

  // ---- the bin settings ----
  document.getElementById("bins-open")?.addEventListener("click", (event) => {
    const cfg = parsed(form.elements.bincfg.value, {});
    const set = (id, value) => { document.getElementById(id).value = value; };
    set("bs-col", cfg.bin_col || "Bin #");
    set("bs-width", cfg.bin_width ?? "");
    set("bs-agg", cfg.agg || "median");
    set("bs-err", cfg.err || "iqr");
    set("bs-plo", cfg.percentiles ? cfg.percentiles[0] : "0.25");
    set("bs-phi", cfg.percentiles ? cfg.percentiles[1] : "0.75");
    set("bs-min", cfg.min_count ?? "1");
    drawer("bins-drawer", event.currentTarget);
  });
  document.getElementById("bins-apply")?.addEventListener("click", () => {
    const value = (id) => document.getElementById(id).value;
    const cfg = {};
    const col = value("bs-col").trim();
    if (col && col !== "Bin #") cfg.bin_col = col;
    const width = parseFloat(value("bs-width"));
    if (Number.isFinite(width) && width > 0) cfg.bin_width = width;
    if (value("bs-agg") !== "median") cfg.agg = value("bs-agg");
    const err = value("bs-err");
    if (err !== "iqr") cfg.err = err;
    const plo = parseFloat(value("bs-plo")), phi = parseFloat(value("bs-phi"));
    if ((err === "iqr" || err === "percentile") && Number.isFinite(plo) && Number.isFinite(phi)
        && (plo !== 0.25 || phi !== 0.75)) cfg.percentiles = [plo, phi];
    const least = parseInt(value("bs-min"), 10);
    if (Number.isFinite(least) && least > 1) cfg.min_count = least;
    form.elements.bincfg.value = Object.keys(cfg).length ? JSON.stringify(cfg) : "";
    save("view", "bin");  // new bins are for the binned view: show it
  });

  // ---- the display settings ----
  const DISPLAY = parsed(form.elements.display.value, {});
  const slots = (data.slots || "").split(",").filter(Boolean).map(Number);
  const tokenColour = (i) => getComputedStyle(document.documentElement).getPropertyValue(`--trace-${i + 1}`).trim();
  const slotColour = (i) => {
    const custom = Array.isArray(DISPLAY.colors) ? DISPLAY.colors[i] : null;
    return typeof custom === "string" && /^#[0-9a-fA-F]{6}$/.test(custom) ? custom : tokenColour(i);
  };
  document.getElementById("display-open")?.addEventListener("click", (event) => {
    const d = DISPLAY;
    document.getElementById("ds-logx").checked = !!d.logx;
    document.getElementById("ds-logy").checked = !!d.logy;
    for (const key of ["xmin", "xmax", "ymin", "ymax", "msize", "width", "height"])
      document.getElementById(`ds-${key}`).value = Number.isFinite(d[key]) ? d[key] : "";
    document.getElementById("ds-layout").value = d.layout ? JSON.stringify(d.layout) : "";
    document.getElementById("ds-error").hidden = true;
    document.getElementById("ds-colors").replaceChildren(...slots.map((i) =>
      el("input", { type: "color", id: `ds-color-${i}`, value: slotColour(i), "aria-label": `Y${i + 1} colour` })));
    drawer("display-drawer", event.currentTarget);
  });
  document.getElementById("display-apply")?.addEventListener("click", () => {
    const d = {};
    // The image keys belong to the image strip; a plot apply keeps them.
    for (const key of ["cmap", "plo", "phi", "mode"]) if (key in DISPLAY) d[key] = DISPLAY[key];
    if (document.getElementById("ds-logx").checked) d.logx = true;
    if (document.getElementById("ds-logy").checked) d.logy = true;
    for (const key of ["xmin", "xmax", "ymin", "ymax"]) {
      const v = parseFloat(document.getElementById(`ds-${key}`).value);
      if (Number.isFinite(v)) d[key] = v;
    }
    for (const key of ["msize", "width", "height"]) {
      const v = parseFloat(document.getElementById(`ds-${key}`).value);
      if (Number.isFinite(v) && v > 0) d[key] = v;
    }
    const colours = [0, 1, 2].map((i) => {
      const input = document.getElementById(`ds-color-${i}`);
      return input ? input.value : slotColour(i);
    });
    if (colours.some((c, i) => c.toLowerCase() !== tokenColour(i).toLowerCase())) d.colors = colours;
    const raw = document.getElementById("ds-layout").value.trim();
    if (raw) {
      try {
        const layout = JSON.parse(raw);
        if (!layout || typeof layout !== "object" || Array.isArray(layout)) throw new Error("expected a JSON object");
        d.layout = layout;
      } catch (error) {
        const box = document.getElementById("ds-error");
        box.textContent = `Layout JSON: ${error.message}. Nothing applied.`;
        box.hidden = false;
        return;
      }
    }
    save("display", Object.keys(d).length ? JSON.stringify(d) : "");
  });
  // The raw layout patch merges last, in the browser only (it never runs
  // on the server). Prototype keys are refused: the patch is a saved,
  // shared set's text.
  function deepMerge(target, patch) {
    for (const key of Object.keys(patch)) {
      if (key === "__proto__" || key === "constructor" || key === "prototype") continue;
      const value = patch[key];
      if (value && typeof value === "object" && !Array.isArray(value)
          && target[key] && typeof target[key] === "object" && !Array.isArray(target[key])) {
        deepMerge(target[key], value);
      } else {
        target[key] = value;
      }
    }
    return target;
  }

  // ---- copy the plot, or send it to the scan's logbook entry ----
  const CLIPBOARD_OK = !!(navigator.clipboard && window.ClipboardItem);
  const sendable = data.logbookSend === "true";
  const AUTHOR_KEY = "geecs-portal:author";
  // Per scan (day and number), as the scan page keys it, so plots sent
  // from either page join one entry.
  const ENTRY_KEY = `geecs-portal:logentry:${data.day}:${data.scan}`;
  const remembered = (key) => { try { return localStorage.getItem(key) || ""; } catch (_) { return ""; } };
  const keep = (key, value) => { try { localStorage.setItem(key, value); } catch (_) { /* private window */ } };
  const download = (gd) => Plotly.downloadImage(gd, { format: "png", scale: EXPORT_SCALE, filename: "geecs-plot" });
  async function copy(gd) {
    // A bitmap reaches the clipboard only on a secure (https or localhost)
    // page; elsewhere the plot downloads, and the toast says why.
    if (!CLIPBOARD_OK) { download(gd); toast("Clipboard needs an https or localhost page: downloaded the PNG instead"); return; }
    try {
      const png = Plotly.toImage(gd, { format: "png", scale: EXPORT_SCALE }).then((url) => fetch(url)).then((r) => r.blob());
      await navigator.clipboard.write([new ClipboardItem({ "image/png": png })]);
      toast("Copied to the clipboard");
    } catch (error) {
      download(gd);
      toast(`Copy failed (${(error && error.name) || "error"}): downloaded the PNG instead`);
    }
  }
  let sending = host;  // the plot whose modebar opened the drawer
  function openSend(gd) {
    sending = gd || host;
    document.getElementById("sl-form").hidden = false;
    document.getElementById("sl-done").hidden = true;
    document.getElementById("sl-author").value = remembered(AUTHOR_KEY);
    document.getElementById("sl-caption").value = sending.dataset.caption || data.caption || "plot";
    const row = document.getElementById("sl-targetrow");
    row.hidden = !remembered(ENTRY_KEY);
    if (!row.hidden) document.getElementById("sl-target").value = "same";
    drawer("send-drawer");
    document.getElementById(remembered(AUTHOR_KEY) ? "sl-send" : "sl-author").focus();
  }
  document.getElementById("sl-send")?.addEventListener("click", async () => {
    const author = document.getElementById("sl-author").value.trim();
    if (!author) { document.getElementById("sl-author").focus(); return; }
    const button = document.getElementById("sl-send");
    const target = document.getElementById("sl-targetrow").hidden ? "new" : document.getElementById("sl-target").value;
    button.disabled = true;
    button.textContent = "Sending…";
    try {
      const d = DISPLAY;
      // A fixed size, so the note's text reads the same in any window; an
      // explicit plot size wins, and a map keeps its plotting area square.
      const m = sending.layout && sending.layout.margin;
      const size = sending.dataset.square && m
        ? { width: 720, height: 720 - m.l - m.r + m.t + m.b }
        : { width: d.width > 0 ? d.width : 720, height: d.height > 0 ? d.height : 460 };
      const image = await Plotly.toImage(sending, { format: "png", scale: EXPORT_SCALE, ...size });
      const url = new URL(`${data.portalRoot}/api/run/${encodeURIComponent(data.uid)}/logbook`, location.origin);
      url.searchParams.set("day", data.day);
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          author, image, caption: document.getElementById("sl-caption").value,
          source_url: location.href, entry: target === "same" ? remembered(ENTRY_KEY) : "",
        }),
      });
      const answer = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(answer.detail || response.statusText);
      keep(AUTHOR_KEY, author);
      keep(ENTRY_KEY, answer.entry_id);
      document.getElementById("sl-form").hidden = true;
      document.getElementById("sl-done").hidden = false;
      document.getElementById("sl-open").href = answer.url;
      document.getElementById("sl-doneword").textContent =
        answer.appended ? "Added to this scan's log entry." : "New log entry created for this scan.";
      toast(answer.appended ? "Added to the scan log entry" : "Scan log entry created");
    } catch (error) {
      toast(`Not sent: ${error.message}`);
    } finally {
      button.disabled = false;
      button.textContent = "Send";
    }
  });

  const ICONS = {
    copy: { width: 24, height: 24, path: "M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" },
    log: { width: 24, height: 24, path: "M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" },
  };

  return {
    /** The plot's Plotly config: wheel zoom, the drawing tools, copy and send. */
    config() {
      return {
        responsive: true, displaylogo: false, scrollZoom: true,
        toImageButtonOptions: { format: "png", scale: EXPORT_SCALE },
        edits: { legendText: true, annotationText: true, shapePosition: true },
        modeBarButtonsToAdd: [
          "togglespikelines", "hoverclosest", "hovercompare",
          "drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape",
          { name: "copyimage", title: CLIPBOARD_OK ? "Copy plot to clipboard (PNG)" : "Download plot PNG", icon: ICONS.copy, click: copy },
        ].concat(sendable ? [{ name: "sendtolog", title: "Send plot to this scan's log entry", icon: ICONS.log, click: openSend }] : []),
      };
    },
    /** The display's raw layout patch over the server's layout. */
    patch(layout) {
      return DISPLAY.layout ? deepMerge(layout, DISPLAY.layout) : layout;
    },
    /** A server figure's colour placeholders resolved against the theme. */
    themed,
    /** The figure answer's pass count, for the rail. */
    counted(body) {
      if (passText && form.elements.filters.value) passText.textContent = `${body.pass} / ${body.total} shots pass`;
    },
    close,
  };
})();
