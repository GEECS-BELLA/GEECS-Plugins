/* The configurator's home: the open document drawn by the theme's form.js, with its tier,
 * its readers, validate / diff / YAML on the unsaved edit, and Save when the commit path is on:
 * the edit goes with the etag it was made on and the operator's name; a 409 shows the document
 * as it now is and what the edit would change in it, a 428 asks to convert a legacy file, and
 * the edit is kept in the page on every refusal. */
(function () {
  "use strict";
  var body = document.body, root = body.dataset.root || "", kind = body.dataset.kind, id = body.dataset.id;
  var query = "?experiment=" + encodeURIComponent(body.dataset.experiment || "");
  var OPERATOR_KEY = "geecs.configurator.operator", operator = document.getElementById("operator");
  // The operator, typed once and remembered in this browser; a host passes its own (data-operator).
  if (operator) {
    try { operator.value = body.dataset.operator || localStorage.getItem(OPERATOR_KEY) || ""; } catch (e) { /* no storage: type it */ }
    operator.addEventListener("change", function () { try { localStorage.setItem(OPERATOR_KEY, operator.value.trim()); } catch (e) { /* not remembered */ } });
  }
  //: What the setups surface (setups.js) asks of this page: the operator, and whether the form holds an unsaved edit.
  window.GeecsConfiguratorPage = { operator: function () { return operator ? operator.value.trim() : ""; }, ready: function () { return false; }, dirty: function () { return false; } };
  if (!kind || !id) return;
  //: What saving a change of each tier will do (the commit path arrives in a later slice).
  var ACTION = {
    operator: "commits to the configs repo's main",
    maintainer: "opens a pull request for the maintainer's review",
    parameter: "commits to the IOC instances' repo (a later arc)",
    sudo: "hands the maintainer a command to run",
    code: "nothing here: it changes by a pull request to GEECS-Plugins",
  };
  var $ = function (sel) { return document.querySelector(sel); };
  var form, schema, stored, info, etag, readOnly;
  window.GeecsConfiguratorPage.ready = function () { return !!form; };
  window.GeecsConfiguratorPage.dirty = function () { return !!form && GeecsForm.diff(stored, form.value(), schema).changes.length > 0; };

  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function fill(host, kids) { host.textContent = ""; kids.forEach(function (k) { if (k) host.appendChild(k); }); }
  function tierMark(t) { return t ? el("span", { class: "tier", "data-tier": t }, t) : el("span", { class: "dim" }, "none"); }
  function line(label, value) { var p = el("p"); p.appendChild(el("span", { class: "dim" }, label + " ")); p.appendChild(value); return p; }
  function api(path, payload) {
    var init = payload === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
    return fetch(root + path, init).then(function (r) {
      if (!r.ok) return r.text().then(function (t) { throw new Error(r.status + " " + t); });
      return (r.headers.get("content-type") || "").indexOf("json") >= 0 ? r.json() : r.text();
    });
  }
  function fail(err) { fill($("#result"), [el("div", { class: "banner", "data-state": "error" }, String(err.message || err))]); }

  // The tier panel: the document's tier, its stricter fields, and what saving the edit as it stands would do.
  function drawTier() {
    var stricter = [];
    (function walk(n) { if (n.tier && n.path.length) stricter.push(n); (n.children || []).concat(n.items || []).forEach(walk); })(GeecsForm.tree(schema, form.value()));
    var d = GeecsForm.diff(stored, form.value(), schema), kids = [line("Document", tierMark(schema.tier))];
    if (stricter.length) {
      var ul = el("ul", { class: "dim" });
      stricter.forEach(function (n) { var li = el("li", {}, n.path.join(".") + " "); li.appendChild(tierMark(n.tier)); ul.appendChild(li); });
      kids.push(el("p", { class: "dim" }, "Stricter fields:"), ul);
    }
    kids.push(line(d.changes.length + " field" + (d.changes.length === 1 ? "" : "s") + " edited; strictest", tierMark(d.tier)));
    if (d.tier) kids.push(el("p", {}, "Saving would: " + ACTION[d.tier] + "."));
    fill($("#tier"), kids);
  }
  function drawChanges(d) {
    var list = el("div", { class: "changes" });
    d.changes.forEach(function (c) {
      var row = el("div", {}, c.path.join(".") + " ");
      if (c.op !== "add") row.appendChild(el("span", { class: "was" }, JSON.stringify(c.before)));
      row.appendChild(document.createTextNode(c.op === "remove" ? " (removed) " : " → " + JSON.stringify(c.after) + " "));
      row.appendChild(tierMark(c.tier));
      list.appendChild(row);
    });
    fill($("#result"), [el("p", {}, d.changes.length ? "Changes against the stored document (the service's diff):" : "No changes against the stored document."), list,
      d.tier ? line("Strictest tier", tierMark(d.tier)) : null]);
  }

  var docPath = "/api/documents/" + kind + "/" + id.split("/").map(encodeURIComponent).join("/") + query;
  // The document with its etag, and the service's word when it is serving an older main (X-Configs-Stale).
  function load() {
    return fetch(root + docPath).then(function (r) {
      if (!r.ok) return r.text().then(function (t) { throw new Error(r.status + " " + t); });
      var stale = r.headers.get("X-Configs-Stale"), box = $("#stale");
      box.hidden = !stale;
      if (stale) fill(box, [el("div", { class: "banner", "data-state": "stale" }, "Showing the last accepted main: " + stale + ". A save checks the base again.")]);
      return r.json();
    });
  }
  Promise.all([api("/api/schema/" + kind), load(), api("/api/kinds")])
    .then(function (got) {
      schema = got[0]; stored = got[1].document; etag = got[1].etag; readOnly = got[1].read_only;
      // a read-only kind, or a document another facility's box owns: no save row
      var saveRow = $(".save-row"); if (saveRow && readOnly) saveRow.hidden = true;
      $("#doc-path").textContent = got[1].path;
      form = GeecsForm.mount($("#form"), { schema: schema, value: stored, readOnly: got[1].read_only, onChange: drawTier });
      if (got[1].errors.length) form.setErrors(got[1].errors);
      info = got[2].filter(function (k) { return k.kind === kind; })[0];
      if (got[1].note) fill($("#result"), [el("div", { class: "banner" }, "This file is in the " + got[1].note + ": the form shows the converted profile; the file is unchanged.")]);
      fill($("#consumers"), info.consumers.map(function (c) { return line(c.name + ":", el("span", {}, "reads " + c.reads + "; apply: " + c.apply)); })
        .concat([line("Dry run:", el("span", {}, info.dry_run))]));
      drawTier();
    }).catch(function (err) { fill($("#form"), [el("div", { class: "state", "data-state": "error" }, String(err.message || err))]); });

  $("#validate").addEventListener("click", function () {
    api("/api/validate/" + kind, { document: form.value(), id: id }).then(function (r) {
      var placed = form.setErrors(r.errors), list = el("ul");
      r.errors.forEach(function (e) { list.appendChild(el("li", {}, (Array.isArray(e.loc) ? e.loc.join(".") : e.loc) + ": " + e.msg)); });
      fill($("#result"), [el("div", r.ok ? { class: "banner" } : { class: "banner", "data-state": "error" }, r.ok ? (info.dry_run === "none" ? "Schema-valid (this kind has no consumer check)." : "Schema-valid (the consumer's check: " + info.dry_run + ", not run yet).") : r.errors.length + " error(s); " + Object.keys(placed).length + " drawn at their fields."), r.ok ? null : list]);
    }).catch(fail);
  });
  $("#diff").addEventListener("click", function () {
    api("/api/diff/" + kind + "/" + id.split("/").map(encodeURIComponent).join("/") + query, { document: form.value() }).then(drawChanges).catch(fail);
  });
  $("#yaml").addEventListener("click", function () {
    api("/api/yaml", { document: form.value() }).then(function (text) { fill($("#result"), [el("pre", { class: "well" }, text)]); }).catch(fail);
  });
  $("#reset").addEventListener("click", function () { form.reset(); drawTier(); fill($("#result"), [el("p", { class: "dim" }, "Edits discarded.")]); });

  // ---- Save (the commit path on): the embed's one save flow (embed.js), into this page's result panel.
  var saveBtn = $("#save");
  if (!saveBtn) return;
  function remount(doc) {
    stored = doc.document; etag = doc.etag;
    form = GeecsForm.mount($("#form"), { schema: schema, value: stored, readOnly: doc.read_only, onChange: drawTier });
    drawTier();
  }
  var save = GeecsConfigurator.saveFlow({
    call: GeecsConfigurator.caller(root), kind: kind, query: function () { return query; }, docId: function () { return id; },
    form: function () { return form; }, value: function () { return form.value(); }, etag: function () { return etag; },
    operator: function () {
      var name = operator.value.trim();
      if (name) { try { localStorage.setItem(OPERATOR_KEY, name); } catch (e) { /* not remembered */ } }
      return name;
    },
    noOperator: function () { operator.focus(); },
    dryRun: function () { return info.dry_run; }, button: saveBtn, status: $("#result"),
    // a proposal leaves main, and so the baseline, as it was
    saved: function (b, doc) { etag = b.etag; if (b.outcome === "committed") stored = doc; drawTier(); },
    reload: remount,
  });
  saveBtn.addEventListener("click", function () { save(false, false); });
})();
