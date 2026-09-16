/* Sweep editor. Builds typed inputs; all trajectory expansion stays in Python. */
(function () {
  "use strict";
  function node(tag, attrs, text) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text != null) n.textContent = text;
    return n;
  }
  function numeric(value, label, integer) {
    if (String(value).trim() === "" || !Number.isFinite(Number(value))) throw new Error(label + ": enter a finite number.");
    var n = Number(value);
    if (integer && (!Number.isInteger(n) || n < 1)) throw new Error(label + ": enter a positive whole number.");
    return n;
  }
  function parseList(text) {
    text = String(text).trim();
    if (!text || /(^,|,$|,\s*,)/.test(text)) throw new Error("Positions: supply numbers separated by commas, tabs, spaces or newlines; no empty entries.");
    return text.split(/[,\s]+/).map(function (token) {
      if (!/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(token) || !Number.isFinite(Number(token))) {
        throw new Error("Invalid position: “" + token.slice(0, 40) + "”. Use numeric values, not expressions.");
      }
      return Number(token);
    });
  }
  function create(root, api, changed) {
    var axes = [{axis: "", kind: "range", relative: false, start: 0, stop: 1, num: 11}],
        tab = "axes", combine = "zip", snake = false, kind = "spiral",
        pair = [{axis: "", relative: false}, {axis: "", relative: false}],
        params = {}, result = null, current = null, generation = 0, timer, controller;
    var $ = function (id) { return root.querySelector("#" + id); };
    var patterns = {
      spiral: [["x_center", "X center", 0], ["y_center", "Y center", 0], ["x_range", "X width", 10], ["y_range", "Y width", 10], ["dr", "Radial step", 1], ["nth", "First-ring points", 16, true], ["dr_y", "Y radial step (optional)", ""], ["tilt", "Tilt (radians)", 0]],
      spiral_fermat: [["x_center", "X center", 0], ["y_center", "Y center", 0], ["x_range", "X width", 10], ["y_range", "Y width", 10], ["dr", "Radial step", 1], ["factor", "Angular factor", 1], ["dr_y", "Y radial step (optional)", ""], ["tilt", "Tilt (radians)", 0]],
      spiral_square: [["x_center", "X center", 0], ["y_center", "Y center", 0], ["x_range", "X width", 10], ["y_range", "Y width", 10], ["x_num", "X points", 5, true], ["y_num", "Y points", 5, true]],
      x2x: [["start", "X start offset", -1], ["stop", "X stop offset", 1], ["num", "Points", 11, true]]
    };
    function field(label, key, value, type) {
      var wrap = node("label", {class: "field"}); wrap.appendChild(node("span", {}, label));
      var input = node(type === "textarea" ? "textarea" : "input", {"data-field": key});
      if (type !== "textarea") { input.type = type || "number"; if (input.type === "number") input.step = "any"; }
      input.value = value == null ? "" : value;
      wrap.appendChild(input); return wrap;
    }
    function select(label, key, value, choices) {
      var wrap = node("label", {class: "field"}); wrap.appendChild(node("span", {}, label));
      var sel = node("select", {"data-field": key});
      choices.forEach(function (c) { sel.appendChild(node("option", {value: c[0]}, c[1])); });
      sel.value = value; wrap.appendChild(sel); return wrap;
    }
    function axisRow(a, i, pattern) {
      var row = node("div", {class: "sweep-axis group", "data-axis": i, "data-pattern": String(!!pattern)});
      var head = node("div", {class: "row"}); head.appendChild(node("span", {class: "eyebrow"}, pattern ? (i ? "Y axis" : "X axis") : "Axis " + (i + 1)));
      head.appendChild(node("span", {class: "spacer"}));
      var label = node("label", {class: "note"}), rel = node("input", {type: "checkbox", "data-field": "relative"});
      rel.checked = pattern && kind === "x2x" ? true : !!a.relative; rel.disabled = pattern && kind === "x2x";
      label.appendChild(rel); label.appendChild(document.createTextNode(" Relative · return to start")); head.appendChild(label);
      if (!pattern) { var rm = node("button", {class: "btn sm ghost", type: "button", "data-remove": i, "aria-label": "Remove axis " + (i + 1)}, "Remove"); rm.disabled = axes.length === 1; head.appendChild(rm); }
      row.appendChild(head);
      var fields = node("div", {class: "sweep-axis-fields"}), variable = field("Variable", "axis", a.axis, "text");
      variable.querySelector("input").setAttribute("list", "sweep-variables"); variable.querySelector("input").placeholder = "Choose or type Device:Variable";
      fields.appendChild(variable);
      if (!pattern) {
        fields.appendChild(select("Spacing", "kind", a.kind, [["range", "Linear range"], ["list", "Position list"], ["log", "Logarithmic"]]));
        if (a.kind === "list") {
          fields.appendChild(field("Positions · comma, tab, space or newline", "positions", Array.isArray(a.positions) ? a.positions.join(", ") : a.positions || "", "textarea"));
        } else {
          fields.appendChild(field(a.kind === "log" ? "Start exponent (10ⁿ)" : "Start", a.kind === "log" ? "start_exp" : "start", a.kind === "log" ? a.start_exp : a.start));
          fields.appendChild(field(a.kind === "log" ? "Stop exponent (10ⁿ)" : "Stop", a.kind === "log" ? "stop_exp" : "stop", a.kind === "log" ? a.stop_exp : a.stop));
          fields.appendChild(field("Points", "num", a.num));
        }
      }
      row.appendChild(fields);
      if (!pattern && a.kind === "list") { var count = node("span", {class: "hint", "data-list-count": "true"}); row.appendChild(count); listCount(row, a.positions); }
      return row;
    }
    function listCount(row, positions) {
      var hint = row.querySelector("[data-list-count]"); if (!hint) return;
      try { hint.textContent = parseList(positions || "").length + " positions"; }
      catch (e) { hint.textContent = e.message; }
    }
    function render() {
      $("sweep-axis-editor").hidden = tab !== "axes"; $("sweep-pattern-editor").hidden = tab === "axes";
      root.querySelectorAll("[data-sweep-tab]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.sweepTab === tab)); });
      $("sweep-combine").value = combine; $("sweep-snake").checked = snake; $("sweep-snake").disabled = combine !== "product";
      $("sweep-axes").replaceChildren(); axes.forEach(function (a, i) { $("sweep-axes").appendChild(axisRow(a, i, false)); });
      $("sweep-pattern").value = kind; $("sweep-pattern-fields").replaceChildren();
      pair.forEach(function (a, i) { $("sweep-pattern-fields").appendChild(axisRow(a, i, true)); });
      var group = node("div", {class: "fields", "data-params": "true"});
      patterns[kind].forEach(function (p) { group.appendChild(field(p[1], p[0], params[p[0]] == null ? p[2] : params[p[0]])); });
      $("sweep-pattern-fields").appendChild(group);
    }
    function axisRef(a, i, relative) {
      if (!a.axis.trim()) throw new Error("Choose a variable for axis " + (i + 1) + ".");
      return {axis: a.axis.trim(), relative: relative == null ? !!a.relative : relative};
    }
    function build() {
      var t;
      if (tab === "axes") {
        t = {kind: "axes", combine: combine, snake: snake, axes: axes.map(function (a, i) {
          var v = axisRef(a, i); v.kind = a.kind;
          if (a.kind === "list") v.positions = parseList(a.positions || "");
          else { var start = a.kind === "log" ? "start_exp" : "start", stop = a.kind === "log" ? "stop_exp" : "stop";
            v[start] = numeric(a[start], "Axis " + (i + 1) + " start"); v[stop] = numeric(a[stop], "Axis " + (i + 1) + " stop"); v.num = numeric(a.num, "Points", true); }
          return v;
        })};
      } else {
        t = {kind: kind, x: axisRef(pair[0], 0, kind === "x2x" ? true : null), y: axisRef(pair[1], 1, kind === "x2x" ? true : null)};
        patterns[kind].forEach(function (p) { var v = params[p[0]] == null ? p[2] : params[p[0]]; if (p[0] === "dr_y" && v === "") return; t[p[0]] = numeric(v, p[1], p[3]); });
      }
      return {trajectory: t};
    }
    function invalidate() {
      ++generation; clearTimeout(timer); if (controller) controller.abort(); result = null; current = null;
      $("sweep-plots").replaceChildren(); $("sweep-table").replaceChildren();
      $("sweep-table").closest("details").ontoggle = null;
      $("sweep-payload").textContent = "";
    }
    function reset() {
      invalidate(); tab = "axes"; combine = "zip"; snake = false; kind = "spiral"; params = {};
      axes = [{axis: "", kind: "range", relative: false, start: 0, stop: 1, num: 11}];
      pair = [{axis: "", relative: false}, {axis: "", relative: false}]; render();
      $("sweep-message").textContent = "Choose a variable to compose a trajectory.";
      $("sweep-preview-note").textContent = "No preview available.";
    }
    function update() {
      invalidate(); var request = generation, retries = 0;
      $("sweep-preview-note").textContent = "Preview is out of date.";
      try { current = build(); $("sweep-payload").textContent = JSON.stringify(current, null, 2); }
      catch (e) { $("sweep-message").textContent = e.message; changed(); return; }
      $("sweep-message").textContent = "Calculating trajectory…"; changed();
      var sent = current;
      function requestPreview() {
        if (request !== generation) return;
        controller = new AbortController();
        api(sent, controller.signal).then(function (data) {
          if (request !== generation) return;
          result = data; $("sweep-message").textContent = data.total_steps + " positions · " + data.axes.length + (data.axes.length === 1 ? " axis" : " axes");
          $("sweep-preview-note").textContent = (data.sampled ? "Sampled preview: " + data.indices.length + " of " + data.total_steps + " positions. " : "All positions shown. ") + "Relative axes show offsets; the starting readback is captured when the run begins.";
          window.GEECS_TRAJECTORY.render($("sweep-plots"), $("sweep-table"), data);
          changed();
        }).catch(function (e) {
          if (request !== generation || e.name === "AbortError") return;
          if (e.status === 409 && retries < 2) {
            retries++; $("sweep-message").textContent = "Preview queue busy; retrying (" + retries + "/2)…";
            timer = setTimeout(requestPreview, 500); return;
          }
          $("sweep-message").textContent = e.status === 409 ? "Preview queue busy. Use Refresh preview to try again." : e.message;
          $("sweep-preview-note").textContent = "Preview unavailable. Start still runs the normal preflight validation."; changed();
        });
      }
      timer = setTimeout(requestPreview, 300);
    }
    function edit(e) {
      var input = e.target, key = input.dataset.field;
      if (!key) return;
      var row = input.closest("[data-axis]"), a = row ? (row.dataset.pattern === "true" ? pair : axes)[Number(row.dataset.axis)] : params;
      a[key] = input.type === "checkbox" ? input.checked : input.value;
      if (key === "positions") listCount(row, a.positions);
      if (key === "kind") { if (a.num == null) a.num = 11; render(); }
      update();
    }
    root.addEventListener("input", function (e) { if (e.target.tagName !== "SELECT" && e.target.type !== "checkbox") edit(e); });
    root.addEventListener("change", function (e) { if (e.target.tagName === "SELECT" || e.target.type === "checkbox") edit(e); });
    root.addEventListener("click", function (e) {
      var b = e.target.closest("button"); if (!b) return;
      if (b.dataset.sweepTab) tab = b.dataset.sweepTab;
      else if (b.hasAttribute("data-remove")) axes.splice(Number(b.dataset.remove), 1);
      else if (b.id === "sweep-refresh") { update(); return; }
      else if (b.id === "sweep-add-axis") axes.push({axis: "", kind: "range", relative: false, start: 0, stop: 1, num: 11});
      else return;
      render(); update();
    });
    $("sweep-combine").addEventListener("change", function () { combine = this.value; if (combine !== "product") snake = false; render(); update(); });
    $("sweep-snake").addEventListener("change", function () { snake = this.checked; update(); });
    $("sweep-pattern").addEventListener("change", function () { kind = this.value; params = {}; render(); update(); });
    render();
    return {
      value: build,
      result: function () { return result; },
      refresh: update,
      reset: reset,
      variables: function (items) { var list = $("sweep-variables"); list.replaceChildren(); items.forEach(function (v) { list.appendChild(node("option", {value: v.name}, v.alias || v.target || "")); }); },
      load: function (payload) {
        reset();
        try {
          if (!payload || !payload.trajectory) throw new Error("Missing Sweep trajectory.");
          var t = JSON.parse(JSON.stringify(payload.trajectory));
          function reference(a) { return a && typeof a.axis === "string"; }
          if (t.kind === "axes") {
            if (!Array.isArray(t.axes) || !t.axes.length || !t.axes.every(function (a) { return reference(a) && ["range", "list", "log"].indexOf(a.kind) >= 0 && (a.kind !== "list" || Array.isArray(a.positions)); })) throw new Error("Invalid Sweep axes or position list.");
            axes = t.axes; axes.forEach(function (a) { if (a.kind === "list") a.positions = a.positions.join(", "); }); combine = t.combine || "zip"; snake = !!t.snake;
          } else {
            if (!Object.prototype.hasOwnProperty.call(patterns, t.kind) || !reference(t.x) || !reference(t.y)) throw new Error("Unsupported or incomplete Sweep pattern.");
            tab = "patterns"; kind = t.kind; pair = [t.x, t.y]; params = t;
          }
          render(); update();
        } catch (e) { reset(); throw e; }
      }
    };
  }
  window.GEECS_SWEEP = {create: create, parseList: parseList};
}());
