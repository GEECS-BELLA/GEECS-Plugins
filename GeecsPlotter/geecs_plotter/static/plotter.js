// The plotter page's behaviour: filter the pick lists, submit on a pick,
// and draw the server-authored figure. The figure carries "$tok:--name"
// sentinels wherever a colour goes (figures.THEMED_PALETTE); one walk
// resolves them against the live theme tokens, and a theme change redraws.
"use strict";

(function () {
  const form = document.getElementById("pick");
  if (form) {
    for (const select of form.querySelectorAll("[data-autosubmit]")) {
      select.addEventListener("change", () => form.requestSubmit());
    }
  }

  // GEECSplotter's substring filter, one per axis. The list is rebuilt from
  // the full set on each keystroke rather than hiding options (Safari
  // ignores `hidden` on <option>). The picked column and "shot #" stay.
  for (const filter of document.querySelectorAll("[data-filter-for]")) {
    const select = document.getElementById(filter.dataset.filterFor);
    const all = Array.from(select.options, (o) => ({ value: o.value, text: o.text }));
    const picked = select.value;
    filter.addEventListener("input", () => {
      const needle = filter.value.trim().toLowerCase();
      select.replaceChildren(...all
        .filter((o) => !needle || o.value === "" || o.value === picked
          || o.text.toLowerCase().includes(needle))
        .map((o) => new Option(o.text, o.value, false, o.value === picked)));
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
