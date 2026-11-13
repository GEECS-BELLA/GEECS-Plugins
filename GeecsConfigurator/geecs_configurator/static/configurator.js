/* The configurator's home: the open document drawn by the theme's form.js, with its tier,
 * its readers, validate / diff / YAML on the unsaved edit, and Save when the commit path is on:
 * the edit goes with the etag it was made on and the operator's name; a 409 shows the document
 * as it now is and what the edit would change in it, a 428 asks to convert a legacy file, and
 * the edit is kept in the page on every refusal. */
(function () {
  "use strict";
  var body = document.body, root = body.dataset.root || "", kind = body.dataset.kind, id = body.dataset.id;
  var query = "?experiment=" + encodeURIComponent(body.dataset.experiment || "");
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
  var OPERATOR_KEY = "geecs.configurator.operator";

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

  // ---- Save (the commit path on): the operator passed in by a host, else typed here and remembered.
  var operator = $("#operator"), saveBtn = $("#save");
  if (!saveBtn) return;
  try { operator.value = body.dataset.operator || localStorage.getItem(OPERATOR_KEY) || ""; } catch (e) { /* no storage: type it */ }
  function says(text) { $("#save-says").textContent = text; }
  function remount(doc) {
    stored = doc.document; etag = doc.etag;
    form = GeecsForm.mount($("#form"), { schema: schema, value: stored, readOnly: doc.read_only, onChange: drawTier });
    drawTier();
  }
  function changeList(changes) {
    var list = el("div", { class: "changes" });
    changes.forEach(function (c) {
      var row = el("div", {}, (c.path.join(".") || "(the whole file)") + " " + c.op + " ");
      if (c.op === "change") { row.appendChild(el("span", { class: "was" }, JSON.stringify(c.before))); row.appendChild(document.createTextNode(" → " + JSON.stringify(c.after) + " ")); }
      row.appendChild(tierMark(c.tier)); list.appendChild(row);
    });
    return list;
  }
  function save(convert) {
    var name = operator.value.trim();
    if (!name) { operator.focus(); says("Your name is the git author: type it first."); return; }
    try { localStorage.setItem(OPERATOR_KEY, name); } catch (e) { /* not remembered */ }
    saveBtn.disabled = true; says("Saving…");
    fetch(root + docPath, { method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ document: form.value(), etag: etag, operator: name, convert_legacy: !!convert }) })
      .then(function (r) { return r.json().then(function (b) { return { status: r.status, body: b }; }); })
      .then(function (r) {
        var b = r.body, d = b.detail;
        if (r.status === 200) {
          etag = b.etag; if (b.outcome === "committed") stored = form.value(); drawTier();  // a proposal leaves main, and so the baseline, as it was
          var head = b.outcome === "committed" ? "Committed " + (b.commit || "").slice(0, 10) + " to main as " + b.author + "."
            : b.outcome === "proposed" ? "Proposed for the maintainer's review on " + b.branch + "." : "Nothing to do: the stored document already says this.";
          var kids = [el("div", { class: "banner" }, head)];
          if (b.pull_request) { var a = el("a", { href: b.pull_request }, b.pull_request); var p = el("p", {}, "Pull request: "); p.appendChild(a); kids.push(p); }
          (b.notes || []).forEach(function (n) { kids.push(el("p", { class: "dim" }, n)); });
          kids.push(el("p", { class: "dim" }, "Whether it took effect: the log."));
          fill($("#result"), kids); says("");
        } else if (r.status === 409) {
          var keep = el("button", { class: "btn", type: "button" }, "Load the current document (drops your edit)");
          keep.addEventListener("click", function () { remount(d.current); fill($("#result"), [el("p", { class: "dim" }, "Loaded the current document: re-apply your edit.")]); });
          fill($("#result"), [el("div", { class: "banner", "data-state": "error" }, "Changed since you opened it: " + d.reason + ". Nothing was saved; your edit is still in the form."),
            el("p", {}, "What your edit would change in the document as it is now:"), changeList(d.diff), keep]);
          says("Not saved.");
        } else if (r.status === 428) {
          var go = el("button", { class: "btn primary", type: "button" }, "Convert and save");
          go.addEventListener("click", function () { save(true); });
          fill($("#result"), [el("div", { class: "banner", "data-state": "stale" }, "This file is in the legacy dialect. Saving rewrites the whole file in the current schema, as the form shows it; the scanner reads either."), go]);
          says("Confirm the conversion.");
        } else if (r.status === 422 && d && d.errors) {
          form.setErrors(d.errors);
          fill($("#result"), [el("div", { class: "banner", "data-state": "error" }, d.errors.length + " error(s); nothing was saved.")]);
          says("Not saved.");
        } else {
          fill($("#result"), [el("div", { class: "banner", "data-state": "error" }, (typeof d === "string" ? d : JSON.stringify(d)) + " Your edit is still in the form.")]);
          says("Not saved.");
        }
      })
      .catch(fail)
      .then(function () { saveBtn.disabled = false; });
  }
  saveBtn.addEventListener("click", function () { save(false); });
})();
