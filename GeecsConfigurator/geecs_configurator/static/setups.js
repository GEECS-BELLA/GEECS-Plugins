/* The setups surface on the configurator's page: capture, the diff against live, the apply.
 *
 * - Capture reads live readbacks through the service (POST /api/setups/capture: the scanner's readback
 *   path, nothing written to a device) into a draft, then saves it through the one save route as a new
 *   document (PUT with create): the operator is the author, the tier and ownership are the save's.
 * - The diff and the apply are the scanner's, relayed by the service, on the setup as committed: an
 *   unsaved edit in the form is never what is applied, so Apply waits for it to be saved or discarded.
 * - The apply's questions come back from the scanner (409, needs_acknowledgement); every one is shown in
 *   the acknowledgement dialog and the apply is sent again only with the ones ticked. The page never
 *   answers a question itself, and the scanner checks every answer against the setup's digest.
 */
(function () {
  "use strict";
  var body = document.body, root = body.dataset.root || "", id = body.dataset.id || "";
  var query = "?experiment=" + encodeURIComponent(body.dataset.experiment || "");
  var page = window.GeecsConfiguratorPage;
  var $ = function (sel) { return document.querySelector(sel); };
  var name = id.replace(/\.(ya?ml)$/i, "");

  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function fill(host, kids) { host.textContent = ""; kids.forEach(function (k) { if (k) host.appendChild(k); }); }
  // A JSON answer with its status, whatever the status: the scanner's {"error": {...}} or the service's {"detail"}.
  function call(method, path, payload) {
    var init = { method: method, headers: { "Content-Type": "application/json" } };
    if (payload !== undefined) init.body = JSON.stringify(payload);
    return fetch(root + path, init).then(function (r) {
      return r.text().then(function (t) { var b = {}; try { b = t ? JSON.parse(t) : {}; } catch (e) { b = { detail: t }; } return { status: r.status, body: b }; });
    });
  }
  function why(b) {
    if (b && b.error) return b.error.message || b.error.kind;
    var d = b && b.detail;
    if (Array.isArray(d)) return d.map(function (e) { return (Array.isArray(e.loc) ? e.loc.join(".") : e.loc) + ": " + e.msg; }).join("; ");
    return typeof d === "string" ? d : JSON.stringify(d);
  }
  function show(v) { return typeof v === "string" ? JSON.stringify(v) : String(v); }
  function chip(state, word) { return el("span", { class: "chip", "data-state": state }, word); }

  // ---------------------------------------------------------------- capture (the kind's list)
  var capVars = $("#cap-vars"), capSave = $("#cap-save"), captured = null, settablesRead = false;
  if (capVars) {
    $("#capture-open").addEventListener("click", function () {
      if (!settablesRead) {
        settablesRead = true;
        call("GET", "/api/settables" + query).then(function (r) {
          var list = $("#cap-settables");
          ((r.body && r.body.items) || []).forEach(function (s) { list.appendChild(el("option", { value: s.name }, s.alias || s.units || "")); });
          if (r.status !== 200) $("#cap-pick-hint").textContent = "The settables did not load (" + why(r.body) + "); type Device:Variable names.";
        });
      }
      if (!capVars.value && id) {  // the open setup's variables, a starting point for a re-capture
        call("GET", "/api/documents/setup/" + encodeURIComponent(id) + query).then(function (r) {
          if (r.status !== 200 || capVars.value) return;
          capVars.value = (r.body.document.steps || []).filter(function (s) { return s.do === "set"; }).map(function (s) { return s.device + ":" + s.variable; }).join("\n");
        });
      }
    });
    function stale() { captured = null; capSave.disabled = true; }
    ["#cap-name", "#cap-desc", "#cap-vars"].forEach(function (s) { $(s).addEventListener("input", stale); });
    function addPicked() {
      var pick = $("#cap-pick"), v = pick.value.trim();
      if (!v) return;
      var lines = capVars.value.split("\n").map(function (x) { return x.trim(); }).filter(Boolean);
      if (lines.indexOf(v) < 0) lines.push(v);
      capVars.value = lines.join("\n"); pick.value = ""; stale();
    }
    $("#cap-pick").addEventListener("change", addPicked);
    $("#cap-pick").addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); addPicked(); } });

    $("#cap-read").addEventListener("click", function () {
      var who = page.operator(), result = $("#cap-result");
      if (!who) { fill(result, [el("div", { class: "banner", "data-state": "error" }, "Type your name (top right) first: you are the setup's capturer and its author.")]); return; }
      var variables = capVars.value.split("\n").map(function (x) { return x.trim(); }).filter(Boolean);
      fill(result, [el("div", { class: "state", "data-state": "loading" }, "Reading " + variables.length + " readback(s)…")]);
      stale();
      call("POST", "/api/setups/capture" + query, { name: $("#cap-name").value.trim(), description: $("#cap-desc").value, operator: who, variables: variables }).then(function (r) {
        if (r.status !== 200) { fill(result, [el("div", { class: "banner", "data-state": "error" }, "Nothing captured: " + why(r.body))]); return; }
        captured = r.body;
        var rows = el("tbody");
        captured.document.steps.forEach(function (s, i) {
          var tr = el("tr"); tr.appendChild(el("td", {}, String(i + 1))); tr.appendChild(el("td", { class: "mono" }, s.device + ":" + s.variable)); tr.appendChild(el("td", { class: "mono" }, show(s.value))); rows.appendChild(tr);
        });
        var table = el("table"); var head = el("thead"); var hr = el("tr");
        ["#", "Set", "To (read now)"].forEach(function (h) { hr.appendChild(el("th", {}, h)); });
        head.appendChild(hr); table.appendChild(head); table.appendChild(rows);
        var kids = [el("p", {}, captured.document.steps.length + " set step(s) from the readbacks, as " + captured.id + ":"), el("div", { class: "tscroll" }, undefined)];
        kids[1].appendChild(table);
        if (captured.unread.length) {
          var ul = el("ul");
          captured.unread.forEach(function (u) { ul.appendChild(el("li", {}, u.variable + ": " + u.detail)); });
          kids.push(el("div", { class: "banner", "data-state": "stale" }, captured.unread.length + " not read, left out of the setup:"), ul);
        }
        fill(result, kids);
        capSave.disabled = false;
      }).catch(function (e) { fill(result, [el("div", { class: "banner", "data-state": "error" }, String(e.message || e))]); });
    });

    capSave.addEventListener("click", function () {
      if (!captured) return;
      var who = page.operator(), result = $("#cap-result");
      capSave.disabled = true;
      call("PUT", "/api/documents/setup/" + encodeURIComponent(captured.id) + query, { document: captured.document, etag: "", operator: who, create: true }).then(function (r) {
        if (r.status === 200) {
          location.href = root + "/" + query + "&kind=setup&id=" + encodeURIComponent(captured.id);
          return;
        }
        var head = r.status === 409 ? "A setup named " + captured.document.name + " exists: choose another name, or open it and edit it." : "Not saved: " + why(r.body);
        fill(result, [el("div", { class: "banner", "data-state": "error" }, head)].concat(Array.prototype.slice.call(result.childNodes)));
        capSave.disabled = r.status === 409;
      }).catch(function (e) { capSave.disabled = false; fill(result, [el("div", { class: "banner", "data-state": "error" }, String(e.message || e))]); });
    });
  }

  // ---------------------------------------------------------------- the diff and the apply (an open setup)
  var live = $("#live");
  if (!live || live.dataset.scanner !== "on" || !name) return;
  var setupPath = "/api/setups/" + encodeURIComponent(name);

  $("#live-diff").addEventListener("click", function () {
    var host = $("#live-body");
    fill(host, [el("div", { class: "state", "data-state": "loading" }, "Reading the readbacks…")]);
    call("GET", setupPath + "/diff" + query).then(function (r) {
      if (r.status !== 200) { fill(host, [el("div", { class: "state", "data-state": "error" }, "No diff: " + why(r.body))]); return; }
      var d = r.body, tbody = el("tbody");
      d.rows.forEach(function (row) {
        var tr = el("tr", { "data-will-change": String(row.will_change) });
        tr.appendChild(el("td", {}, String(row.index)));
        tr.appendChild(el("td", { class: "mono" }, row.device + ":" + row.variable));
        tr.appendChild(el("td", { class: "mono" }, show(row.value)));
        tr.appendChild(el("td", { class: "mono" }, row.ok ? show(row.reading) : "—"));
        var st = el("td");
        st.appendChild(!row.ok ? chip("unknown", "unread") : row.will_change ? chip("degraded", "will change") : chip("ok", "already"));
        tr.appendChild(st);
        tr.appendChild(el("td", { class: "mono" }, row.difference == null ? (row.ok ? "" : row.detail || "") : (row.difference > 0 ? "+" : "") + row.difference.toPrecision(4)));
        tbody.appendChild(tr);
      });
      var table = el("table"), thead = el("thead"), hr = el("tr");
      ["Step", "Set", "To", "Reads now", "", "Reading − value"].forEach(function (h) { hr.appendChild(el("th", {}, h)); });
      thead.appendChild(hr); table.appendChild(thead); table.appendChild(tbody);
      var wrap = el("div", { class: "tscroll" }); wrap.appendChild(table);
      fill(host, [el("p", {}, d.changes + " of " + d.rows.length + " write(s) would change a reading" + (d.unread ? "; " + d.unread + " not read" : "") + ". A noisy readback differs from its value: read the last column."), wrap]);
    }).catch(function (e) { fill(host, [el("div", { class: "state", "data-state": "error" }, String(e.message || e))]); });
  });

  function says(t) { $("#apply-says").textContent = t; }
  function apply(acknowledged) {
    var who = page.operator();
    if (!who) { says("Type your name (top right): the apply is queued as yours."); return; }
    if (!page.ready()) { says("The setup did not load here: reload the page before applying it."); return; }
    if (page.dirty()) { says("Save or discard your edit first: the scanner applies the setup as committed."); return; }
    $("#live-apply").disabled = true; says("Asking the scanner…");
    call("POST", setupPath + "/apply" + query, { operator: who, acknowledged: acknowledged }).then(function (r) {
      $("#live-apply").disabled = false;
      var err = r.body && r.body.error;
      if (r.status === 200) { says("Queued on the worker: " + (r.body.summary || r.body.plan) + (r.body.item_uid ? " (item " + r.body.item_uid.slice(0, 8) + ")" : "") + ". The scanner's queue shows how it went."); return; }
      if (r.status === 409 && err && err.needs_acknowledgement) { says(""); openAck(err.needs_acknowledgement); return; }
      says("Not applied: " + why(r.body));
    }).catch(function (e) { $("#live-apply").disabled = false; says("Not applied: " + (e.message || e)); });
  }
  // Every question the scanner asked, each ticked by the operator; Apply stays off until all are.
  function openAck(questions) {
    var list = $("#ack-list"); list.textContent = "";
    questions.forEach(function (q) {
      var li = el("li"), label = el("label"), box = el("input", { type: "checkbox", class: "ackbox", "data-check": q.check });
      var text = el("span", {}, q.title); text.appendChild(el("span", { class: "m" }, q.message));
      label.appendChild(box); label.appendChild(text); li.appendChild(label); list.appendChild(li);
    });
    $("#dlg-ack-title").textContent = "Setup " + name + ": " + questions.length + " thing" + (questions.length === 1 ? "" : "s") + " to acknowledge";
    $("#do-apply").disabled = true;
    window.GeecsKit.confirm($("#dlg-ack")).open();
  }
  $("#ack-list").addEventListener("change", function () {
    $("#do-apply").disabled = !Array.prototype.every.call(document.querySelectorAll("#ack-list .ackbox"), function (c) { return c.checked; });
  });
  $("#do-apply").addEventListener("click", function () {
    var acked = Array.prototype.filter.call(document.querySelectorAll("#ack-list .ackbox"), function (c) { return c.checked; }).map(function (c) { return c.getAttribute("data-check"); });
    window.GeecsKit.confirm($("#dlg-ack")).close();
    apply(acked);
  });
  $("#live-apply").addEventListener("click", function () { apply([]); });
})();
