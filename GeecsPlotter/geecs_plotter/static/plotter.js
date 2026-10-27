// The plotter page's behaviour: filter the pick lists, submit on a pick,
// and draw the server-authored figure. The figure carries "$tok:--name"
// sentinels wherever a colour goes (figures.THEMED_PALETTE); one walk
// resolves them against the live theme tokens, and a theme change redraws.
"use strict";

(function () {
  const form = document.getElementById("pick");
  const DOUBLE_CLICK_MS = 500;
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

  // The two-way toggles (name | alias, X = scan variable): set the hidden
  // field and resubmit; the filters ride along in the form.
  for (const button of document.querySelectorAll("[data-toggle-for]")) {
    button.addEventListener("click", () => {
      form.elements[button.dataset.toggleFor].value = button.dataset.value;
      form.requestSubmit();
    });
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
    filter.addEventListener("input", apply);
    if (filter.value) apply();  // a filter carried in from the URL
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
      draw();
    })
    .catch((error) => fail(String(error)));

  window.addEventListener("geecs:theme", () => { if (figure) draw(); });
})();
