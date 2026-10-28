// The plotter page's behaviour: filter the pick lists, submit on a pick
// (the server saves the controls into the open settings set), and draw the
// server-authored figure. The figure carries "$tok:--name"
// sentinels wherever a colour goes (figures.THEMED_PALETTE); one walk
// resolves them against the live theme tokens, and a theme change redraws.
"use strict";

(function () {
  const form = document.getElementById("pick");
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
    const folded = (name) => name.trim().split(/\s+/).join(" ").toLowerCase();
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

  const host = document.getElementById("plot");
  if (!host) return;
  const state = document.getElementById("plot-state");
  const PLOT_CONFIG = { responsive: true, displaylogo: false, scrollZoom: true };
  let figure = null;

  function tokenColour(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }
  function resolveThemeTokens(value) {
    if (typeof value === "string")
      return value.startsWith("$tok:") ? (tokenColour(value.slice(5)) || value) : value;
    if (Array.isArray(value)) return value.map(resolveThemeTokens);
    if (value && typeof value === "object") {
      for (const k of Object.keys(value)) value[k] = resolveThemeTokens(value[k]);
    }
    return value;
  }
  function draw() {
    const fig = resolveThemeTokens(JSON.parse(JSON.stringify(figure)));
    Plotly.react(host, fig.data, fig.layout, PLOT_CONFIG);
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
      head.append(element("span", `swatch t${fit.slot}`), element("strong", "", `Y${fit.slot === 1 ? "" : fit.slot} ${fit.label} fit`));
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
      draw();
    })
    .catch((error) => fail(String(error)));

  window.addEventListener("geecs:theme", () => { if (figure) draw(); });
})();
