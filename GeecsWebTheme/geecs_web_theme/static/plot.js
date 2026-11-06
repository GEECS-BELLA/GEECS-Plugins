/* GEECS web theme — the one Plotly template every line plot is drawn through.
 *
 * Line plots on every GEECS surface (traces, lineouts, histograms, scalar
 * plots) are Plotly, drawn with this layout and config: the theme's tokens
 * read at run time, never a colour written here. Load the vendored bundle
 * beside it (plotly-cartesian-<version>.min.js, same folder), then:
 *
 *   Plotly.react(host, data, GeecsPlot.layout({ ...the page's keys }), GeecsPlot.config());
 *   GeecsPlot.watch(host);   // repainted on the theme's `geecs:theme` event
 *
 * Trace i takes --trace-((i mod n) + 1) through the layout's colorway (n the
 * --trace-k tokens the palette defines), so a page sets no line colour of its
 * own and a theme change recolours every trace. Images are each surface's own.
 *
 * window.GeecsPlot in a page, module.exports under node (tests/test_plot_js.py).
 */
(function (api) {
  "use strict";
  var MAX_TRACE_TOKENS = 12;

  // The token for trace i of n: --trace-1..n, cycling.
  function traceToken(i, n) {
    return "--trace-" + ((i % n) + 1);
  }

  // How many --trace-k tokens `read` defines (k = 1, 2, … until one is empty); at least 1.
  function traceCount(read) {
    var n = 0;
    while (n < MAX_TRACE_TOKENS && read("--trace-" + (n + 1))) n++;
    return Math.max(1, n);
  }

  // One axis: the rules and ticks in the palette's quiet inks, numbers in the mono face.
  function axis(t) {
    return {
      gridcolor: t("--rule-soft"), linecolor: t("--rule"), zerolinecolor: t("--rule"), tickcolor: t("--rule"),
      showline: true, ticks: "outside", ticklen: 4, automargin: true,
      tickfont: { family: t("--ff-mono"), size: 11, color: t("--muted") },
      title: { font: { family: t("--ff-ui"), size: 12, color: t("--muted") }, standoff: 8 },
    };
  }

  // The layout and config, every colour and face read through `read(tokenName)`.
  function template(read) {
    var t = function (name) { return String(read(name) || "").trim(); };
    var n = traceCount(t), colorway = [];
    for (var i = 0; i < n; i++) colorway.push(t(traceToken(i, n)));
    return {
      layout: {
        font: { family: t("--ff-ui"), size: 12, color: t("--ink") },
        paper_bgcolor: t("--surface"), plot_bgcolor: t("--surface"),
        colorway: colorway,
        margin: { l: 56, r: 16, t: 8, b: 40 },
        xaxis: axis(t), yaxis: axis(t),
        legend: { orientation: "h", x: 0, y: 1, xanchor: "left", yanchor: "bottom", bgcolor: t("--surface"), font: { size: 11, color: t("--muted") } },
        hoverlabel: { bgcolor: t("--surface-2"), bordercolor: t("--rule"), font: { family: t("--ff-mono"), size: 11, color: t("--ink") } },
        modebar: { bgcolor: t("--surface"), color: t("--muted"), activecolor: t("--accent") },
        hovermode: "x unified", dragmode: "zoom",
      },
      config: {
        displaylogo: false, displayModeBar: true, responsive: true, doubleClick: "reset", scrollZoom: false, showTips: false,
        // the modebar left: PNG export, zoom, pan, zoom in/out, autoscale, reset. Never `modeBarButtons` by name:
        // Plotly 3.1.1 hangs in its config comparison on the next Plotly.react with it (a live plot reacts at 2 Hz)
        modeBarButtonsToRemove: ["select2d", "lasso2d", "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian"],
        toImageButtonOptions: { format: "png", scale: 2 },
      },
    };
  }

  // A deep merge of plain objects (arrays and values replace): the page's keys over the template's.
  function merge(base, over) {
    var out = {}, k;
    for (k in base) out[k] = base[k];
    for (k in over || {}) {
      var a = out[k], b = over[k];
      out[k] = a && b && typeof a === "object" && typeof b === "object" && !Array.isArray(a) && !Array.isArray(b) ? merge(a, b) : b;
    }
    return out;
  }

  // The template's themed leaves as Plotly.relayout paths ("xaxis.tickfont.color": …): every colour and face,
  // nothing a page sets (a range, a title's text), so a repaint never undoes the viewer's zoom.
  function themeKeys(layout) {
    var out = {};
    (function walk(obj, path) {
      for (var k in obj) {
        var v = obj[k], p = path ? path + "." + k : k;
        if (k === "colorway" || /color$/.test(k) || k === "family") out[p] = v;
        else if (v && typeof v === "object" && !Array.isArray(v)) walk(v, p);
      }
    })(layout, "");
    return out;
  }

  // ---- in a page: the live tokens, and the repaint on the theme's event
  function tokens() {
    var style = getComputedStyle(document.documentElement);
    return function (name) { return style.getPropertyValue(name); };
  }
  var hosts = [], listening = false;
  function repaint() {
    var keys = themeKeys(template(tokens()).layout);
    hosts = hosts.filter(function (h) { return h.isConnected; });
    for (var i = 0; i < hosts.length; i++) if (hosts[i]._fullLayout) window.Plotly.relayout(hosts[i], keys);
  }
  // Repaint `host` on every `geecs:theme` (theme.js dispatches it on each palette or mode change).
  function watch(host) {
    if (hosts.indexOf(host) < 0) hosts.push(host);
    if (!listening) { window.addEventListener("geecs:theme", repaint); listening = true; }
  }

  api.traceToken = traceToken;
  api.traceCount = traceCount;
  api.template = template;
  api.merge = merge;
  api.themeKeys = themeKeys;
  api.layout = function (over) { return merge(template(tokens()).layout, over); };
  api.config = function (over) { return merge(template(tokens()).config, over); };
  api.watch = watch;
})(typeof module === "object" && module.exports ? module.exports : (window.GeecsPlot = {}));
