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

  // GEECSplotter's substring filter: hide the options that do not match,
  // never the one picked.
  const filter = document.querySelector("[data-column-filter]");
  if (filter) {
    filter.addEventListener("input", () => {
      const needle = filter.value.trim().toLowerCase();
      for (const option of document.querySelectorAll("#x option, #y option")) {
        const keep = !needle || option.selected || option.value === ""
          || option.value.toLowerCase().includes(needle);
        option.hidden = !keep;
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
