/* Display-only trajectory renderer. Coordinates arrive from the Python service. */
(function () {
  "use strict";
  function el(tag, text) { var n = document.createElement(tag); if (text != null) n.textContent = text; return n; }
  function svg(tag, attrs) { var n = document.createElementNS("http://www.w3.org/2000/svg", tag); Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); }); return n; }
  function fmt(v) { return Number(v.toPrecision(5)).toString(); }
  function chart(title, x, y, xlabel, ylabel) {
    var figure = el("figure"); figure.className = "trajectory-chart well";
    figure.appendChild(el("figcaption", title));
    var image = svg("svg", {viewBox: "0 0 500 195", role: "img", "aria-label": title + "; " + xlabel + " against " + ylabel});
    var xmin = Math.min.apply(null, x), xmax = Math.max.apply(null, x), ymin = Math.min.apply(null, y), ymax = Math.max.apply(null, y);
    function fraction(v, lo, hi) { if (lo === hi) return .5; var scale = Math.max(Math.abs(lo), Math.abs(hi), 1); return (v / scale - lo / scale) / (hi / scale - lo / scale); }
    function px(v) { return 64 + fraction(v, xmin, xmax) * 418; }
    function py(v) { return 153 - fraction(v, ymin, ymax) * 123; }
    for (var i = 0; i <= 4; i++) {
      var yy = 153 - i * 123 / 4;
      image.appendChild(svg("line", {x1: 64, y1: yy, x2: 482, y2: yy, class: "trajectory-grid"}));
      var label = svg("text", {x: 58, y: yy + 4, "text-anchor": "end"}); label.textContent = fmt(ymin * (1 - i / 4) + ymax * i / 4); image.appendChild(label);
    }
    var path = x.map(function (v, j) { return (j ? "L" : "M") + px(v) + " " + py(y[j]); }).join(" ");
    image.appendChild(svg("path", {d: path, class: "trajectory-path"}));
    x.forEach(function (v, j) {
      if (x.length > 200 && j !== 0 && j !== x.length - 1) return;
      var dot = svg("circle", {cx: px(v), cy: py(y[j]), r: j === 0 ? 4 : 2, class: "trajectory-dot"});
      var tip = svg("title", {}); tip.textContent = xlabel + ": " + fmt(v) + "; " + ylabel + ": " + fmt(y[j]); dot.appendChild(tip); image.appendChild(dot);
    });
    [[64, fmt(xmin), "start"], [482, fmt(xmax), "end"], [273, xlabel, "middle"]].forEach(function (item) { var text = svg("text", {x: item[0], y: item[2] === "middle" ? 188 : 173, "text-anchor": item[2]}); text.textContent = item[1]; image.appendChild(text); });
    figure.appendChild(image); return figure;
  }
  function render(plots, tableRoot, data) {
    plots.replaceChildren();
    var steps = data.indices.map(function (i) { return i + 1; });
    data.axes.forEach(function (a) { plots.appendChild(chart(a.axis + (a.relative ? " · offsets" : " · absolute"), steps, a.positions, "Step", a.axis)); });
    if (data.axes.length === 2) { var x = data.axes[0], y = data.axes[1]; plots.appendChild(chart("X–Y trajectory · " + x.axis + " / " + y.axis, x.positions, y.positions, x.axis, y.axis)); }
    tableRoot.replaceChildren();
    var disclosure = tableRoot.closest("details"), built = false;
    function buildTable() {
    if (!disclosure.open || built) return;
    built = true;
    var table = el("table"), head = el("thead"), row = el("tr"); row.appendChild(el("th", "Step"));
    data.axes.forEach(function (a) { row.appendChild(el("th", a.axis + (a.relative ? " (offset)" : ""))); }); head.appendChild(row); table.appendChild(head);
    var body = el("tbody"); data.indices.forEach(function (step, i) { var r = el("tr"); r.appendChild(el("td", String(step + 1))); data.axes.forEach(function (a) { r.appendChild(el("td", String(a.positions[i]))); }); body.appendChild(r); });
    table.appendChild(body); tableRoot.appendChild(table);
    }
    disclosure.ontoggle = buildTable; buildTable();
  }
  window.GEECS_TRAJECTORY = {render: render};
}());
