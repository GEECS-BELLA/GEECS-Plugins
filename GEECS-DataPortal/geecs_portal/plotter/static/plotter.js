// The plotter page's behaviour: filter the pick lists, submit on a pick
// (the server saves the controls into the open settings set), and draw the
// server-authored figure. The figure carries "$tok:--name"
// sentinels wherever a colour goes (figures.THEMED_PALETTE); one walk
// resolves them against the live theme tokens, and a theme change redraws.
"use strict";

(function () {
  // The viewer's own settings set: their name, asked once and remembered
  // by the browser in the form the server stores (settings.set_name). A
  // first visit borrows the logbook send's author name when it fits; the
  // author box's own key is only read, never written or cleared. A page
  // whose script cannot store the name asks again.
  const OWN_KEY = "geecs-portal:set";
  const AUTHOR_KEY = "geecs-portal:author";
  const folded = (name) => name.trim().split(/\s+/).join(" ").toLowerCase();
  const SET_NAME = /^[a-z0-9](?:[a-z0-9 ._-]{0,62}[a-z0-9_-])?$/;
  function ownName() {
    try {
      for (const key of [OWN_KEY, AUTHOR_KEY]) {
        const name = folded(localStorage.getItem(key) || "");
        if (SET_NAME.test(name)) return name;
      }
    } catch (_) { /* no storage: asked */ }
    return "";
  }
  function remember(name) {
    try { localStorage.setItem(OWN_KEY, name); } catch (_) { /* asked again next time */ }
  }
  const BAD_NAME = "Use letters, digits, spaces, '-', '_' and '.', starting with a letter or digit.";

  // A URL with no set asks the name; a remembered one answers at once.
  const ask = document.getElementById("ask-name");
  if (ask) {
    const box = document.getElementById("own-set");
    const known = ownName();
    if (known) { box.value = known; ask.submit(); return; }
    ask.addEventListener("submit", (event) => {
      const name = folded(box.value);
      if (!SET_NAME.test(name)) {
        event.preventDefault();
        box.setAttribute("aria-invalid", "true");
        document.getElementById("own-set-err").textContent = BAD_NAME;
        return;
      }
      box.value = name;
      remember(name);
    });
    return;
  }

  const form = document.getElementById("pick");
  let owner = ownName();
  if (form && form.elements.owner) form.elements.owner.value = owner;
  const you = document.getElementById("you");
  const showYou = () => {
    if (!you || !owner) return;
    document.getElementById("you-name").textContent = owner;
    you.hidden = false;
  };
  showYou();
  // "Change" forgets the name (this page's own key), so the page without
  // a set asks again.
  document.getElementById("you-change")?.addEventListener("click", () => {
    try { localStorage.removeItem(OWN_KEY); } catch (_) { /* nothing kept */ }
  });
  // No name known (a shared link opened first): the first change waits for
  // it. Save as names its own target, so it goes ahead.
  const who = document.getElementById("who");
  const whoName = document.getElementById("who-name");
  let waiting = null;  // the submitter of the change that waits for a name
  if (form && who && !owner) {
    form.addEventListener("submit", (event) => {
      if (owner) return;
      const submitter = event.submitter;
      if (submitter && (submitter.getAttribute("formaction") || "").endsWith("/save-as")) return;
      event.preventDefault();
      waiting = submitter || null;
      who.hidden = false;
      whoName.focus();
    });
    whoName.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      const name = folded(whoName.value);
      if (!SET_NAME.test(name)) {
        whoName.setAttribute("aria-invalid", "true");
        document.getElementById("who-err").textContent = BAD_NAME;
        return;
      }
      owner = name;
      remember(name);
      form.elements.owner.value = name;
      who.hidden = true;
      showYou();
      form.requestSubmit(waiting);
    });
  }
  const DOUBLE_CLICK_MS = 500;
  const AUTO_OPEN_MS = 500;
  // A hand-picked X turns "X = scan variable" off, or the next page would
  // put the scan variable straight back.
  function pick(select) {
    if (select.id === "x" && form.elements.xauto) form.elements.xauto.value = "false";
    form.requestSubmit();
  }

  // A list box fires "change" on every arrow key, so the pick is a click or
  // Enter: arrowing through a filtered list must not reload the page. The
  // click waits out the OS double-click window (500 ms is Windows' default)
  // so a double-click can claim it instead.
  for (const select of document.querySelectorAll("[data-pick]")) {
    const picked = select.value;
    let pending = null;
    select.addEventListener("click", () => {
      clearTimeout(pending);
      pending = setTimeout(() => { if (select.value !== picked) pick(select); }, DOUBLE_CLICK_MS);
    });
    select.addEventListener("dblclick", (event) => {
      clearTimeout(pending);
      const option = event.target.closest("option") || select.selectedOptions[0];
      const filter = document.querySelector(`[data-filter-for="${select.id}"]`);
      if (option && filter) {
        filter.value = option.text;
        filter.dispatchEvent(new Event("input"));
      }
    });
    select.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); pick(select); }
    });
  }

  // A stepped number box (the scan, a polynomial's order) submits after a
  // pause, so typing "12" does not open scan 1 first and a held arrow key
  // does not reload on every step. The next page puts the cursor back in
  // the box it came from, so the arrow keys keep stepping; the marker
  // (that box's id) is per tab and best-effort (storage may be off).
  const STEPPED = "geecs-plotter:stepped";
  try {
    const back = sessionStorage.getItem(STEPPED);
    sessionStorage.removeItem(STEPPED);
    if (back) document.getElementById(back)?.focus();
  } catch (_) { /* no storage: the stepper still works, one key at a time */ }
  function stepper(box, ready) {
    let pending = null;
    box.addEventListener("input", () => {
      clearTimeout(pending);
      pending = setTimeout(() => {
        if (box.value && box.checkValidity() && ready()) {
          try { sessionStorage.setItem(STEPPED, box.id); } catch (_) { /* see above */ }
          form.requestSubmit();
        }
      }, AUTO_OPEN_MS);
    });
  }
  // Stepping the scan number opens that scan (GEECSplotter's scan stepper).
  const scanBox = document.querySelector("[data-auto-open]");
  if (scanBox) stepper(scanBox, () => Boolean(form.elements.day.value));

  // The two-way toggles (name | alias, X = scan variable): set the hidden
  // field and resubmit; the filters ride along in the form.
  for (const button of document.querySelectorAll("[data-toggle-for]")) {
    button.addEventListener("click", () => {
      form.elements[button.dataset.toggleFor].value = button.dataset.value;
      form.requestSubmit();
    });
  }

  // The Fit menus save and redraw on a choice; the polynomial order is a
  // stepper, so stepping 2 → 5 reloads once.
  for (const control of document.querySelectorAll("[data-submit-on-change]")) {
    if (control.type === "number") stepper(control, () => true);
    else control.addEventListener("change", () => form.requestSubmit());
  }

  // The XY plot's other scans: a chip's × drops its scan, and Add scan…
  // (the kit's drawer picker) lists a day's scans and adds the ticked
  // ones; either saves the list into the set and reloads.
  const scansField = form && form.elements.scans;
  const scanList = () => (scansField.value ? scansField.value.split(",").filter(Boolean) : []);
  const saveScans = (list) => { scansField.value = list.join(","); form.requestSubmit(); };
  for (const button of document.querySelectorAll("[data-scan-remove]")) {
    button.addEventListener("click", () => saveScans(scanList().filter((s) => s !== button.dataset.scanRemove)));
  }
  const scansOpen = document.getElementById("scans-open");
  // kit.js loads after this script, so the picker is built on first click.
  let scansPicker = null;
  scansOpen?.addEventListener("click", () => {
    if (!window.GeecsKit) return;
    scansPicker = scansPicker || scanPicker();
    scansPicker();
  });
  // The drawer's picker; the function it returns opens it on the day shown.
  function scanPicker() {
    const dayBox = document.getElementById("scans-day");
    const hint = document.getElementById("scans-hint");
    const main = `${form.elements.day.value}:${Number(form.elements.scan.value)}`;
    const overlay = form.elements.multi.value === "overlay";
    const MAX_OVERLAY = Number(scansOpen.dataset.maxOverlay) || 8;  // settings.MAX_OVERLAY_SCANS
    let listed = { day: "", scans: [], error: "" };
    const key = (n) => `${listed.day}:${n}`;
    const name = (n) => `Scan ${String(n).padStart(3, "0")}`;
    const picker = window.GeecsKit.picker({
      drawer: document.getElementById("scans-drawer"),
      search: document.getElementById("scans-q"),
      list: document.getElementById("scans-list"),
      add: document.getElementById("scans-add"),
      all: document.getElementById("scans-all"),
      none: document.getElementById("scans-none"),
      noun: ["scan", "scans"],
      maxShown: 200,
      // The overlay draws at most eight scans, the main one included.
      room: () => (overlay ? MAX_OVERLAY - 1 - scanList().length : Infinity),
      shownOnly: () => true,
      rows: (query) => {
        const needle = query.trim().toLowerCase();
        const have = new Set(scanList());
        return listed.scans
          .filter((s) => !needle || name(s.scan).toLowerCase().includes(needle)
            || String(s.scan) === needle || (s.description || "").toLowerCase().includes(needle))
          .map((s) => {
            const here = key(s.scan) === main ? "the main scan" : have.has(key(s.scan)) ? "plotted" : "";
            const shots = s.shots === null ? "" : `${s.shots} shots`;
            return { name: name(s.scan), pickable: !here, sub: [here, s.description, shots].filter(Boolean).join(" · ") };
          });
      },
      onRender: (info) => {
        hint.textContent = listed.error ? `No scans listed: ${listed.error}`
          : !listed.day ? "pick a day"
          : !listed.scans.length ? `no scans on ${listed.day}`
          : `${info.matches} of ${listed.scans.length} scans on ${listed.day}`
            + (info.full ? ` · the overlay draws at most ${MAX_OVERLAY} scans (merge for more)` : "");
      },
      onAdd: (names) => saveScans([...scanList(), ...names.map((n) => key(Number(n.replace(/\D/g, ""))))]),
    });
    const load = () => {
      const day = dayBox.value;
      if (!day) { listed = { day: "", scans: [], error: "" }; picker.open(); return; }
      listed = { day, scans: [], error: "" };
      picker.open();
      hint.textContent = "listing…";
      const url = new URL("api/day", form.action);
      url.search = new URLSearchParams({ day }).toString();
      fetch(url)
        .then((r) => r.json().then((b) => (r.ok ? b : Promise.reject(new Error(b.detail || r.statusText)))))
        // A new day starts the picks afresh (open() clears them): a row is
        // named by its number alone, so a pick kept across days would add
        // that number on the new day, not the scan ticked.
        .then((body) => { listed = { day, scans: body.scans, error: "" }; picker.open(); })
        .catch((error) => { listed = { day, scans: [], error: error.message }; picker.open(); });
    };
    dayBox.addEventListener("change", load);
    return () => { picker.open(); if (listed.day !== dayBox.value) load(); };
  }

  // Switching settings sets opens the other set; nothing is saved.
  for (const choice of document.querySelectorAll("[data-switch-set]")) {
    choice.addEventListener("change", () => choice.form.submit());
  }
  // Enter in the Save-as name saves as, rather than taking the form's
  // first button (Open).
  // An empty name does nothing but point at the box.
  const saveAs = document.getElementById("save-as");
  const saveGo = document.getElementById("save-as-go");
  if (saveAs && saveGo) {
    saveAs.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      saveGo.click();
    });
    // A name already saved (compared as the server stores it) asks before
    // replacing; the dialog's Replace button posts overwrite=true.
    const saved = new Set(Array.from(document.querySelectorAll("#saved-sets option"), (o) => o.value));
    const replace = document.getElementById("replace-set");
    saveGo.addEventListener("click", (event) => {
      const name = folded(saveAs.value);
      if (!name) { event.preventDefault(); saveAs.focus(); return; }
      if (saved.has(name) && replace && window.GeecsKit) {
        event.preventDefault();
        replace.querySelector("h4").textContent = `Replace the saved set “${name}”?`;
        window.GeecsKit.confirm(replace).open();
      }
    });
  }

  // A filter is saved into the set as it is typed (after a pause), without
  // a reload, so the text is there next time even with no pick after it.
  const FILTER_SAVE_MS = 500;
  function saveField(name, value) {
    const set = form.elements.set.value;
    // Under someone else's set (or with no name yet) a change is the
    // viewer's: the whole form posts, and the server copies the set into
    // theirs and switches to it; with no name, the form asks for it first.
    if (owner !== set) { form.requestSubmit(); return Promise.resolve(); }
    const url = new URL(`api/settings/${encodeURIComponent(set)}`, form.action);
    return fetch(url, { method: "POST", body: new URLSearchParams({ [name]: value }) })
      .catch(() => { /* the next pick posts the filter anyway */ });
  }

  // GEECSplotter's substring filter, one per axis. The list is rebuilt from
  // the full set on each keystroke rather than hiding options (Safari
  // ignores `hidden` on <option>). The picked column and "shot #" stay.
  for (const filter of document.querySelectorAll("[data-filter-for]")) {
    const select = document.getElementById(filter.dataset.filterFor);
    const all = Array.from(select.options, (o) => ({ value: o.value, text: o.text }));
    const picked = select.value;
    const matches = () => {
      const needle = filter.value.trim().toLowerCase();
      // The shown name (an alias, with the toggle on) or the column itself.
      return all.filter((o) => !needle || o.text.toLowerCase().includes(needle)
        || o.value.toLowerCase().includes(needle));
    };
    const apply = () => {
      const shown = new Set(matches().map((o) => o.value));
      select.replaceChildren(...all
        .filter((o) => shown.has(o.value) || o.value === "" || o.value === picked)
        .map((o) => new Option(o.text, o.value, false, o.value === picked)));
    };
    let saving = null;
    filter.addEventListener("input", () => {
      apply();
      clearTimeout(saving);
      saving = setTimeout(() => saveField(filter.name, filter.value), FILTER_SAVE_MS);
    });
    if (filter.value) apply();  // a filter saved in the set
    // Enter takes the one match left (the kit picker's rule); otherwise it
    // stays in the box instead of submitting the form.
    filter.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      const left = matches().filter((o) => o.value !== "");
      if (left.length === 1) {
        select.value = left[0].value;
        pick(select);
      }
    });
  }

  // The image strip. Tiles across, the colour map and the levels redraw in
  // place (no reload) and save into the set as they change; a tile with no
  // image says why, from the server's answer. The display half of a tile's
  // URL is built here as pages.display_query builds it.
  const strip = document.getElementById("strip");
  if (strip) {
    const tiles = Array.from(strip.querySelectorAll("img[data-tile]"));
    const value = (name) => (form.elements[name] ? form.elements[name].value : "");
    const zscale = strip.dataset.zscale;
    let shared = null;  // [black, white] for the shared z scale, once read
    const display = () => {
      const query = new URLSearchParams({ cmap: value("cmap") });
      if (strip.dataset.roi) query.set("roi", strip.dataset.roi);
      const levels = zscale === "fixed" ? [value("zmin"), value("zmax")]
        : zscale === "shared" && shared ? shared.map(String) : null;
      if (levels && levels.every((v) => v !== "")) {
        query.set("zmin", levels[0]);
        query.set("zmax", levels[1]);
      }
      return query;
    };
    const drawTiles = () => {
      for (const img of tiles) {
        img.closest(".tile").classList.remove("missing");
        img.src = `${img.dataset.tile}&${display()}`;
      }
    };
    // The shared z scale reads every tile first (the server's
    // /api/images/levels), then draws them all on that one scale.
    const sharedNote = document.getElementById("shared-levels");
    const readShared = () => {
      const url = new URL(strip.dataset.levelsUrl, location.href);
      if (sharedNote) sharedNote.textContent = "reading every tile…";
      return fetch(url)
        .then((response) => response.json().then((body) => ({ response, body })))
        .then(({ response, body }) => {
          if (!response.ok) throw new Error(String(body.detail || response.statusText));
          shared = [body.zmin, body.zmax];
          if (sharedNote) sharedNote.textContent =
            `${body.zmin.toPrecision(4)} to ${body.zmax.toPrecision(4)}, over ${body.tiles} tiles`;
          drawTiles();
        })
        .catch((error) => {
          shared = null;
          if (sharedNote) sharedNote.textContent = `no shared scale: ${error.message}`;
          drawTiles();  // each tile its own window, rather than nothing
        });
    };
    if (zscale === "shared") readShared();

    // Tiles across, as the template writes it: equal columns in a grid; in
    // one row, that many tiles spread evenly over the whole scan (the
    // first, the last and evenly between), the rest hidden — the big view
    // still steps through every one.
    const across = (n) => {
      if (strip.classList.contains("row")) {
        strip.style.gridAutoColumns = `calc((100% - ${n - 1} * var(--gap)) / ${n})`;
        const figures = tiles.map((img) => img.closest(".tile"));
        const total = figures.length;
        const shown = new Set(total <= n ? figures.keys()
          : Array.from({ length: n }, (_, i) => Math.round((i * (total - 1)) / Math.max(n - 1, 1))));
        figures.forEach((figure, i) => { figure.hidden = !shown.has(i); });
      } else {
        strip.style.gridTemplateColumns = `repeat(${n}, minmax(0, 1fr))`;
      }
    };
    across(Number(value("cols")) || 1);
    let redraw = null;
    for (const control of document.querySelectorAll("[data-live]")) {
      control.addEventListener(control.tagName === "SELECT" ? "change" : "input", () => {
        // The warning the page draws for white not above black, kept true
        // as the fixed levels are typed (or one is cleared).
        const warning = document.getElementById("level-warning");
        if (warning && (control.name === "zmin" || control.name === "zmax")) {
          const [lo, hi] = [value("zmin"), value("zmax")];
          warning.hidden = lo === "" || hi === "" || Number(hi) > Number(lo);
        }
        if (!control.checkValidity() || control.value === "") return;
        clearTimeout(redraw);
        redraw = setTimeout(() => {
          if (control.name === "cols") across(Number(control.value));
          else drawTiles();
          saveField(control.name, control.value);
        }, AUTO_OPEN_MS);
      });
    }
    for (const img of tiles) {
      img.addEventListener("error", () => {
        const tile = img.closest(".tile");
        const caption = tile.querySelector("figcaption");
        tile.classList.add("missing");
        fetch(img.src)
          .then((response) => response.json())
          .then((body) => { caption.dataset.reason = String(body.detail || "no image"); })
          .catch(() => { caption.dataset.reason = "no image"; });
      });
    }

    // The ROI, kept per camera: saved with the camera's name, then the page
    // reloads so every tile (and a shared scale) is taken over it.
    const saveRoi = (roi) => {
      const shown = form.elements.set.value;
      // Under someone else's set the ROI goes into the viewer's own, copied
      // from the shown one, and the page then opens the viewer's set.
      if (!owner) { form.requestSubmit(); return Promise.resolve(); }  // asks the name
      const set = owner;
      const url = new URL(`api/settings/${encodeURIComponent(set)}`, form.action);
      const body = new URLSearchParams({ device: strip.dataset.device, roi });
      if (set !== shown) body.set("copy_from", shown);
      const reopen = () => {
        if (set === shown) { location.reload(); return; }
        const next = new URL(location.href);
        next.searchParams.set("set", set);
        location.assign(next);
      };
      const problem = document.getElementById("roi-error");
      return fetch(url, { method: "POST", body })
        .then((response) => response.ok ? reopen()
          : response.json().then((answer) => { throw new Error(String(answer.detail || response.statusText)); }))
        .catch((error) => {
          // Shown in the drawer, which stays open with the rectangle drawn.
          if (problem) { problem.textContent = `Not saved: ${error.message}`; problem.hidden = false; }
        });
    };
    document.querySelector("[data-roi-clear]")?.addEventListener("click", () => saveRoi(""));
    const drawerEl = document.getElementById("roi-drawer");
    const opener = document.querySelector("[data-roi-open]");
    // kit.js loads after this script, so the drawer is asked for on click.
    if (drawerEl && opener) roiDrawer(drawerEl, opener, saveRoi);
  }

  // The ROI drawer: the camera's first tile, whole, with a rectangle drawn
  // by dragging (or typed in the four boxes), in the image's own pixels.
  function roiDrawer(drawerEl, opener, saveRoi) {
    const image = document.getElementById("roi-image");
    const box = document.getElementById("roi-box");
    const stage = document.getElementById("roi-stage");
    const boxes = ["x0", "y0", "x1", "y1"].map((k) => document.getElementById(`roi-${k}`));
    const size = document.getElementById("roi-size");
    let roi = null;  // [x0, y0, x1, y1] in image pixels
    const scale = () => image.clientWidth / (image.naturalWidth || 1);
    const show = () => {
      box.hidden = !roi;
      if (!roi) { size.textContent = "whole image"; return; }
      const k = scale();
      Object.assign(box.style, {
        left: `${roi[0] * k}px`, top: `${roi[1] * k}px`,
        width: `${(roi[2] - roi[0]) * k}px`, height: `${(roi[3] - roi[1]) * k}px`,
      });
      boxes.forEach((input, i) => { input.value = roi[i]; });
      size.textContent = `${roi[2] - roi[0]} × ${roi[3] - roi[1]} px`;
    };
    const pixel = (event) => {
      const rect = image.getBoundingClientRect();
      const k = scale();
      const clamp = (v, n) => Math.max(0, Math.min(n, Math.round(v / k)));
      return [clamp(event.clientX - rect.left, image.naturalWidth), clamp(event.clientY - rect.top, image.naturalHeight)];
    };
    let start = null;
    stage.addEventListener("pointerdown", (event) => {
      if (!image.naturalWidth) return;
      event.preventDefault();
      stage.setPointerCapture(event.pointerId);
      start = pixel(event);
    });
    stage.addEventListener("pointermove", (event) => {
      if (!start) return;
      const [x, y] = pixel(event);
      roi = [Math.min(start[0], x), Math.min(start[1], y), Math.max(start[0], x), Math.max(start[1], y)];
      show();
    });
    stage.addEventListener("pointerup", () => {
      start = null;
      if (roi && (roi[2] - roi[0] < 2 || roi[3] - roi[1] < 2)) roi = null;  // a click, not a drag
      show();
    });
    for (const input of boxes) {
      input.addEventListener("input", () => {
        const v = boxes.map((b) => Number(b.value));
        if (boxes.every((b) => b.value !== "") && v[2] > v[0] && v[3] > v[1]) { roi = v; show(); }
      });
    }
    image.addEventListener("load", show);
    window.addEventListener("resize", show);
    opener.addEventListener("click", () => {
      const saved = strip.dataset.roi;
      roi = saved ? saved.split(",").map(Number) : null;
      // The whole image, in the strip's colour map and each tile's window.
      const query = new URLSearchParams({ cmap: form.elements.cmap.value });
      image.src = `${strip.dataset.fullUrl}&${query}`;
      if (window.GeecsKit) window.GeecsKit.drawer(drawerEl).open(opener);
      show();
    });
    document.getElementById("roi-apply").addEventListener("click", () => saveRoi(roi ? roi.join(",") : ""));
  }

  const host = document.getElementById("plot");
  if (!host) return;
  const state = document.getElementById("plot-state");
  // The modebar's copy and send, the drawing tools (plot_tools.js).
  const tools = window.PlotTools || { config: () => ({ responsive: true }), patch: (l) => l, counted: () => {}, themed: (v) => v };
  const PLOT_CONFIG = tools.config();
  let figure = null;

  function draw() {
    const fig = tools.themed(JSON.parse(JSON.stringify(figure)));
    Plotly.react(host, fig.data, tools.patch(fig.layout), PLOT_CONFIG);
  }
  // The fits' numbers under the plot: one table per fitted Y list, keyed
  // by its swatch, each number with its standard error.
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function showFits(fits) {
    const box = document.getElementById("fits");
    if (!box) return;
    box.replaceChildren(...fits.map((fit) => {
      const block = element("section", "fit");
      const head = element("div", "row fit-head");
      // The overlay's fits are one per scan, each in its scan's colour.
      const whose = fit.scan ? ` · ${fit.scan}` : "";
      head.append(element("span", `swatch t${fit.trace || fit.slot}`), element("strong", "", `Y${fit.slot === 1 ? "" : fit.slot} ${fit.label} fit${whose}`));
      if (fit.formula) head.append(element("span", "dim mono", fit.formula));
      block.append(head);
      if (fit.error) {
        block.append(element("p", "fit-error", fit.error));
        return block;
      }
      const table = element("table");
      const body = element("tbody");
      const row = (name, text) => {
        const tr = element("tr");
        tr.append(element("td", "id", name), element("td", "num", text));
        body.append(tr);
      };
      for (const term of fit.terms) row(term.name, term.text);
      row("r²", fit.r2 === null ? "undefined" : fit.r2.toFixed(4));
      row("points", String(fit.points));
      table.append(body);
      block.append(table);
      for (const note of fit.notes) block.append(element("p", "hint", note));
      return block;
    }));
  }
  function fail(message) {
    state.dataset.state = "error";
    state.querySelector(".t").textContent = message;
  }

  fetch(host.dataset.figureUrl)
    .then((response) => response.json().then((body) => ({ response, body })))
    .then(({ response, body }) => {
      if (!response.ok) return fail(String(body.detail || response.statusText));
      figure = body.figure;
      state.hidden = true;
      document.getElementById("code").textContent = body.code;
      showFits(body.fits || []);
      tools.counted(body);
      draw();
    })
    .catch((error) => fail(String(error)));

  window.addEventListener("geecs:theme", () => { if (figure) draw(); });
})();
