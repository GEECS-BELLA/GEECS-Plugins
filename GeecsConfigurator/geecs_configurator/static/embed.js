/* The embedded editor: one configs-repo document edited in a kit drawer on another surface's page,
 * and the one save flow every page of the configurator saves through.
 *
 * The mount contract. A host page loads, in order, its own /theme/kit.css, /theme/kit.js and
 * /theme/form.js and this script from the configurator (<configurator>/static/embed.js; the host
 * never imports the package), and the configurator lists the host's origin in
 * GEECS_CONFIGURATOR_CORS_ORIGINS. Then:
 *
 *   GeecsConfigurator.open({
 *     api,         // the configurator's URL as the browser reaches it (a site value of the host)
 *     kind,        // a file kind, e.g. "preset", "analysis_recipe"
 *     experiment,  // the experiment folder ("" for a kind outside the experiment tree)
 *     id,          // the document id ("name.yaml", or "HTU/name.yaml" in a nested tree); a bare name is
 *                  //   the kind's one document with that stem, in any folder (two: refused, named)
 *     fields,      // optional: the top-level fields drawn; every other field is kept exactly as stored
 *     operator,    // the operator's name, or a function returning it: the host's own identity, required to save
 *     preview,     // optional: function (document) — the host's Preview button. Returning a promise of an
 *                  //   Element (or text) shows it in the drawer under the buttons; a rejection shows its message
 *     toolbar,     // optional: an Element of the host's, drawn beside the buttons (a shot number, a name field)
 *     onSaved,     // optional: function (answer, id) — after a save, copy or delete the configurator accepted
 *     draft,       // optional: a new document's draft (from GeecsConfigurator.create): nothing is read at `id`, and
 *                  //   the first Save is the save route's create there
 *   })  ->  { close(), value() }
 *
 *   GeecsConfigurator.create({ api, kind, experiment, operator, preview, toolbar, onSaved })
 *     A new document of a kind the catalogue says starts blank (/api/kinds `blank`): the drawer asks its name,
 *     the folder (this box's, /api/folders/<kind>) and, for a recipe, its data folder and frame kind (the
 *     schema's input kinds, and what each requires beyond its kind), then opens the configurator's draft
 *     (/api/blank/<kind>/<folder>/<name>.yaml: the portal's, which already binds) with open({..., draft}).
 *
 * What a save does is the configurator's, exactly as on its own page (saveFlow below is that page's save
 * too): the kind's dry run first (a refusal stops it there, its errors drawn at their fields), then
 * PUT /api/documents/<kind>/<id> with the etag the document was served with and the operator, who is the
 * git author; the tier decides commit or proposal; a moved base is a 409 that keeps the edit; a document
 * this box does not own, or a read-only service, is drawn read-only. No identity is stored here: the
 * operator is the host's, asked for on every save. On a page that is not on the kit (no `kit` class on
 * <body>), the drawer is drawn inside a `.kit` wrapper of its own, with its own scrim.
 *
 * window.GeecsConfigurator in a page, module.exports under node (tests/test_embed_js.py).
 */
(function (out) {
  "use strict";
  var drawerEl = null, titleEl = null, bodyEl = null, current = null;  // current: {key, handle}, the document in the drawer

  function el(tag, attrs, text) {
    var n = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function fill(host, kids) { host.textContent = ""; kids.forEach(function (k) { if (k) host.appendChild(k); }); }
  function banner(text, state) { var b = el("div", state ? { class: "banner", "data-state": state } : { class: "banner" }, text); b.style && (b.style.overflowWrap = "anywhere"); return b; }
  function button(text, cls, onclick) { var b = el("button", { class: "btn " + (cls || ""), type: "button" }, text); b.addEventListener("click", onclick); return b; }
  function why(b) {
    var d = b && b.detail;
    if (d && d.reason) return d.reason;
    return typeof d === "string" ? d : JSON.stringify(d);
  }
  function tierMark(t) { return t ? el("span", { class: "tier", "data-tier": t }, t) : el("span", { class: "dim" }, "none"); }

  // One request to the configurator: {status, body}, the body parsed when it is JSON.
  function caller(base) {
    return function (method, path, payload) {
      var init = { method: method, headers: { "Content-Type": "application/json" } };
      if (payload !== undefined) init.body = JSON.stringify(payload);
      return fetch(base + path, init).then(function (r) {
        return r.text().then(function (t) { var b = {}; try { b = t ? JSON.parse(t) : {}; } catch (e) { b = { detail: t }; } return { status: r.status, body: b }; });
      });
    };
  }

  // ---- the one save flow -------------------------------------------------------------------------
  //
  // saveFlow(o) -> save(convert, checked). The configurator's page and the embedded drawer both save
  // through it; each passes its own elements and keeps its own document state:
  //   o.call(method, path, payload)  the configurator's API (caller() above)
  //   o.kind, o.query()              the kind, and "?experiment=…" (the dry run adds &id=)
  //   o.docId()                      the document id the PUT and the dry run name
  //   o.form()                       the mounted GeecsForm (value, invalid, setErrors)
  //   o.value()                      the whole document to save
  //   o.etag()                       the etag it was served with
  //   o.operator()                   the git author (the host's), "" when not given
  //   o.dryRun()                     the kind's dry-run word from /api/kinds ("none": no consumer check)
  //   o.button, o.status             the Save button (disabled while asking) and where the answers go
  //   o.saved(answer)                a 200: the page takes the new etag (and, committed, the new baseline)
  //   o.noOperator()                 optional: the page points at its operator field
  //   o.reload(current)              optional: a 409's "Load the current document" (drops the edit)
  //   o.create()                     optional: true while the document is not yet a file (the PUT creates it)
  function saveFlow(o) {
    function show(kids) { fill(o.status, kids); }
    function outcomes(d) {
      if (!d.outcomes || !d.outcomes.length) return null;
      var ul = el("ul", { class: "dim" });
      d.outcomes.forEach(function (x) { ul.appendChild(el("li", {}, (x.check || "") + ": " + (x.result || "") + (x.detail ? " (" + x.detail + ")" : ""))); });
      return ul;
    }
    function errorsAt(errors) { if (errors && errors.length) o.form().setErrors(errors); }
    function changeList(changes) {
      var list = el("div", { class: "changes" });
      (changes || []).forEach(function (c) {
        var row = el("div", {}, (c.path.join(".") || "(the whole file)") + " " + c.op + " ");
        if (c.op === "change") { row.appendChild(el("span", { class: "was" }, JSON.stringify(c.before))); row.appendChild(document.createTextNode(" → " + JSON.stringify(c.after) + " ")); }
        row.appendChild(tierMark(c.tier)); list.appendChild(row);
      });
      return list;
    }
    function refused(d) {
      errorsAt(d.errors);
      show([banner("The " + d.by + "'s dry run refused your edit: " + d.refusal + ". Nothing was saved; your edit is still in the form.", "error"), outcomes(d)]);
    }
    // A dry run answers for the draft it was asked about: an answer to an older ask is dropped, and its
    // Save goes on only while the form still holds that draft (an edit since asks again).
    var asks = 0;
    function dryRunFirst(convert) {
      o.button.disabled = true;
      var path = "/api/dry-run/" + o.kind + o.query() + "&id=" + encodeURIComponent(o.docId());
      var mine = ++asks, asked = JSON.stringify(o.value());
      o.call("POST", path, { document: JSON.parse(asked) }).then(function (r) {
        if (mine !== asks) return;  // a later ask is the one shown
        o.button.disabled = false;
        var d = r.body || {};
        if (JSON.stringify(o.value()) !== asked) { dryRunFirst(convert); return; }  // edited while it ran
        if (r.status !== 200) { show([banner("The dry run did not answer: " + why(d), "error")]); return; }
        if (d.refusal) { refused(d); return; }
        if (d.by === "none") { save(convert, true); return; }  // no consumer check here: nothing to show
        var kids = [banner(d.ran ? "The " + d.by + "'s dry run of your edit: " + (d.detail || "no objection") + "." : "The " + d.by + "'s dry run did not run: " + d.detail + ".", d.ran ? null : "stale"), outcomes(d)];
        (d.questions || []).forEach(function (q) { kids.push(el("p", { class: "dim" }, "It asks: " + (q.title || q.check) + (q.message ? " (" + q.message + ")" : ""))); });
        if (d.output) { var det = el("details", { class: "disc" }); det.appendChild(el("summary", {}, "Its output")); det.appendChild(el("pre", { class: "well" }, d.output)); kids.push(det); }
        kids.push(button(d.ran ? "Save" : "Save without it", "primary", function () {
          if (JSON.stringify(o.value()) !== asked) { dryRunFirst(convert); return; }  // this answer is for an earlier draft
          save(convert, true);
        }));
        show(kids);
      }).catch(function (e) { if (mine === asks) { o.button.disabled = false; show([banner(String(e.message || e), "error")]); } });
    }
    function save(convert, checked) {
      var who = o.operator();
      if (!who) { show([banner("Type your name first (the operator field): it is the git author.", "error")]); if (o.noOperator) o.noOperator(); return; }
      if (o.form().invalid().length) { show([banner("A field does not parse: " + o.form().invalid().join(", "), "error")]); return; }
      if (!checked && o.dryRun() !== "none") { dryRunFirst(convert); return; }
      o.button.disabled = true;
      show([el("p", { class: "dim" }, "Saving…")]);
      var doc = o.value();
      o.call("PUT", "/api/documents/" + o.kind + "/" + o.docId().split("/").map(encodeURIComponent).join("/") + o.query(),
        { document: doc, etag: o.etag(), operator: who, convert_legacy: !!convert, create: !!(o.create && o.create()) }).then(function (r) {
        o.button.disabled = false;
        var b = r.body || {}, d = b.detail;
        if (r.status === 200) {
          o.saved(b, doc);
          var head = b.outcome === "committed" ? "Committed " + (b.commit || "").slice(0, 10) + " to main as " + b.author + "."
            : b.outcome === "proposed" ? "Proposed for the maintainer's review on " + b.branch + "." : "Nothing to do: the stored document already says this.";
          var kids = [banner(head)];
          if (b.pull_request) { var p = el("p", {}, "Pull request: "); p.appendChild(el("a", { href: b.pull_request, target: "_blank", rel: "noopener" }, b.pull_request)); kids.push(p); }
          (b.notes || []).forEach(function (n) { kids.push(el("p", { class: "dim" }, n)); });
          kids.push(el("p", { class: "dim" }, "Whether it took effect: the configurator's log."));
          show(kids);
        } else if (r.status === 409) {
          var kids409 = [banner("Changed since it was opened: " + why(b) + ". Nothing was saved; your edit is still in the form.", "error")];
          if (d && d.diff) kids409.push(el("p", {}, "What your edit would change in the document as it is now:"), changeList(d.diff));
          if (o.reload && d && d.current) kids409.push(button("Load the current document (drops your edit)", "", function () { o.reload(d.current); show([el("p", { class: "dim" }, "Loaded the current document: re-apply your edit.")]); }));
          show(kids409);
        } else if (r.status === 428) {
          show([banner("This file is in the legacy dialect. Saving rewrites the whole file in the current schema, as the form shows it; its reader reads either.", "stale"),
            button("Convert and save", "primary", function () { if (JSON.stringify(o.value()) === JSON.stringify(doc)) save(true, true); else dryRunFirst(true); })]);  // its dry run was shown for this draft; an edit since asks again
        } else if (r.status === 422 && d && d.dry_run && d.dry_run.refusal) {
          refused(d.dry_run);
        } else if (r.status === 422 && d && d.errors) {
          errorsAt(d.errors);
          show([banner(d.errors.length + " error(s); nothing was saved.", "error")]);
        } else {
          show([banner("Not saved (" + r.status + "): " + why(b) + ". Your edit is still in the form.", "error")]);
        }
      }).catch(function (e) { o.button.disabled = false; show([banner(String(e.message || e), "error")]); });
    }
    return save;
  }

  // ---- delete and copy: the save's mechanics, the same callers --------------------------------------
  //
  // deleteFlow(o) -> start(). Asks the delete's dry run (/api/delete-check: the file and every document
  // that names it; a named document is refused, its referrers listed, and no dialog opens), then the kit's
  // <dialog> (rung 3), whose Delete waits for the document's name typed, and says the file stays in git's
  // history; then DELETE with the etag served and the operator, as its tier says (a commit or a proposal).
  // o: call, kind, query(), docId(), etag(), operator(), noOperator?, status, host (the element the dialog
  // is drawn in: a .kit one), deleted(answer).
  function deleteFlow(o) {
    function show(kids) { fill(o.status, kids); }
    function refusedBy(names, reason) {
      var ul = el("ul", { class: "mono", style: "margin:0;padding-left:1.2em;overflow-wrap:anywhere" });
      names.forEach(function (n) { ul.appendChild(el("li", {}, n)); });
      show([banner("Not deleted: " + reason, "error"), names.length ? el("p", {}, "Named by:") : null, names.length ? ul : null]);
    }
    function ask(check) {
      var name = o.docId().split("/").pop().replace(/\.ya?ml$/i, "");
      var d = el("dialog", { "aria-labelledby": "geecs-configurator-delete-title" }), body = d.appendChild(el("div", { class: "body" }));
      body.appendChild(el("h4", { id: "geecs-configurator-delete-title" }, "Delete " + name + "?"));
      body.appendChild(el("p", {}, check.path + " is removed from main as " + o.operator() + ", as its tier says. Nothing names it. The file stays in git's history; a commit can bring it back."));
      var label = body.appendChild(el("label", { class: "field" }));
      label.appendChild(el("span", { class: "lbl" }, "Type " + name + " to confirm"));
      var typed = label.appendChild(el("input", { class: "btn", autocomplete: "off", "aria-label": "the document's name" }));
      var row = body.appendChild(el("div", { class: "row end" }));
      var keep = row.appendChild(button("Keep it", "", function () { close(); }));
      var go = row.appendChild(button("Delete " + name, "danger", function () { if (typed.value.trim() === name) { close(); send(); } }));
      keep.setAttribute("autofocus", "");
      go.disabled = true;
      typed.addEventListener("input", function () { go.disabled = typed.value.trim() !== name; });
      function close() { if (d.close) d.close(); if (d.parentNode) d.parentNode.removeChild(d); }
      d.addEventListener("close", function () { if (d.parentNode) d.parentNode.removeChild(d); });  // Esc too: never a second dialog behind this one
      o.host.appendChild(d);
      var kit = typeof window !== "undefined" && window.GeecsKit;
      if (kit && kit.confirm) kit.confirm(d).open(); else if (d.showModal) d.showModal(); else d.setAttribute("open", "");
      return d;
    }
    function send() {
      show([el("p", { class: "dim" }, "Deleting…")]);
      o.call("DELETE", "/api/documents/" + o.kind + "/" + o.docId().split("/").map(encodeURIComponent).join("/") + o.query(), { etag: o.etag(), operator: o.operator() }).then(function (r) {
        var b = r.body || {}, d = b.detail;
        if (r.status === 200) {
          show([banner(b.outcome === "committed" ? "Deleted " + b.path + " from main as " + b.author + " (" + (b.commit || "").slice(0, 10) + "). It stays in git's history." : b.outcome === "proposed" ? "The delete is proposed for the maintainer's review on " + b.branch + "." : "Nothing to do: " + ((b.notes || [])[0] || "the file was already gone"))]);
          if (o.deleted) o.deleted(b);
        } else if (r.status === 422 && d && d.referrers) {
          refusedBy(d.referrers, d.reason);
        } else {
          show([banner("Not deleted (" + r.status + "): " + why(b), "error")]);
        }
      }).catch(function (e) { show([banner(String(e.message || e), "error")]); });
    }
    return function start() {
      if (!o.operator()) { show([banner("Type your name first (the operator field): it is the git author.", "error")]); if (o.noOperator) o.noOperator(); return null; }
      return o.call("POST", "/api/delete-check/" + o.kind + "/" + o.docId().split("/").map(encodeURIComponent).join("/") + o.query()).then(function (r) {
        var c = r.body || {};
        if (r.status !== 200) { show([banner("The delete's check did not answer (" + r.status + "): " + why(c), "error")]); return null; }
        if (!c.deletable) { refusedBy(c.referrers || [], c.reason); return null; }
        return ask(c);
      });
    };
  }

  // copyFlow(o) -> copy(name). Saves the document as it stands in the form under a new name in the same
  // folder: a new file, never over one (the save route's create: its dry run, its tier, its stem rule). A
  // document whose `name` is its stem gets the new one; a recipe gets the new name as its `output_name` too,
  // so the copy writes its own columns and folders. o: call, kind, query(), docId(), value(), operator(),
  // noOperator?, status, copied(newId, answer).
  function copyFlow(o) {
    function show(kids) { fill(o.status, kids); }
    return function copy(name) {
      name = String(name || "").trim().replace(/\.ya?ml$/i, "");
      var who = o.operator();
      if (!who) { show([banner("Type your name first (the operator field): it is the git author.", "error")]); if (o.noOperator) o.noOperator(); return null; }
      if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(name)) { show([banner("A new name is letters, digits, _ . -", "error")]); return null; }
      var parts = o.docId().split("/"), stem = parts.pop().replace(/\.ya?ml$/i, ""), newId = parts.concat([name + ".yaml"]).join("/");
      var doc = JSON.parse(JSON.stringify(o.value()));
      if (doc.name === stem) doc.name = name;
      if (o.kind === "analysis_recipe" || o.kind === "himg_job") doc.output_name = name;  // its own columns and folders (a job: its label)
      show([el("p", { class: "dim" }, "Saving a copy as " + newId + "…")]);
      return o.call("PUT", "/api/documents/" + o.kind + "/" + newId.split("/").map(encodeURIComponent).join("/") + o.query(), { document: doc, etag: "", operator: who, create: true }).then(function (r) {
        var b = r.body || {}, d = b.detail;
        if (r.status === 200) {
          show([banner(b.outcome === "committed" ? "Saved a copy as " + newId + " (" + b.author + ")." : "The copy is proposed for the maintainer's review on " + b.branch + ".")]);
          if (o.copied) o.copied(newId, b);
        } else if (r.status === 422 && d && d.dry_run && d.dry_run.refusal) {
          show([banner("Not copied: the " + d.dry_run.by + "'s dry run refused it: " + d.dry_run.refusal, "error")]);
        } else if (r.status === 422 && d && d.errors) {
          show([banner("Not copied: " + d.errors.map(function (e) { return e.msg; }).join("; "), "error")]);
        } else {
          show([banner("Not copied (" + r.status + "): " + why(b), "error")]);
        }
      });
    };
  }
  //: Whether a kind's documents can be copied and deleted here: the service's word (/api/kinds `deletable`).
  function named(info) { return !!(info && info.deletable); }

  // ---- the drawer --------------------------------------------------------------------------------

  // The drawer, made once and kept: closing it (Esc, the scrim) never loses the edit in it.
  function drawer() {
    if (drawerEl) return drawerEl;
    var body = document.body, onKit = !(body.classList && !body.classList.contains("kit"));
    var host = body;
    drawerEl = el("aside", { class: "drawer", id: "geecs-configurator-drawer", role: "dialog", "aria-modal": "false", "aria-labelledby": "geecs-configurator-title", "aria-hidden": "true", "data-open": "false" });
    if (!onKit) {  // a host page not on the kit: the kit's rules reach the drawer through a .kit wrapper of its own
      host = body.appendChild(el("div", { class: "kit", id: "geecs-configurator-kit" }));
      host.appendChild(el("div", { class: "scrim", id: "geecs-configurator-scrim", "data-open": "false" }));
      drawerEl.setAttribute("data-scrim", "geecs-configurator-scrim");
    }
    var head = el("header");
    titleEl = head.appendChild(el("span", { class: "eyebrow", id: "geecs-configurator-title" }, "configurator"));
    head.appendChild(el("button", { class: "btn sm ghost", type: "button", "data-drawer-close": "" }, "Close"));
    drawerEl.appendChild(head);
    bodyEl = drawerEl.appendChild(el("div", { class: "body" }));
    host.appendChild(drawerEl);
    return drawerEl;
  }
  function show(d) {
    var kit = typeof window !== "undefined" && window.GeecsKit;
    if (kit) kit.drawer(d).open(); else { d.setAttribute("data-open", "true"); d.setAttribute("aria-hidden", "false"); }
  }

  // The fields drawn: the schema's own, limited to `fields`; the stored document keeps the rest.
  function limit(schema, fields) {
    if (!fields || !fields.length) return schema;
    var props = {}, s = JSON.parse(JSON.stringify(schema));
    fields.forEach(function (f) { if (s.properties && f in s.properties) props[f] = s.properties[f]; });
    s.properties = props;
    s.required = (s.required || []).filter(function (f) { return f in props; });
    return s;
  }
  function pick(doc, fields) {
    if (!fields || !fields.length) return doc;
    var o = {};
    fields.forEach(function (f) { if (f in doc) o[f] = doc[f]; });
    return o;
  }
  // The whole document to save: the stored one, each drawn field as edited (a cleared optional one removed).
  function merge(stored, edited, fields) {
    if (!fields || !fields.length) return edited;
    var o = JSON.parse(JSON.stringify(stored));
    fields.forEach(function (f) { if (f in edited) o[f] = edited[f]; else delete o[f]; });
    return o;
  }
  // A bare name's document among the kind's ids: the one whose file stem it is, in any folder.
  function byStem(ids, name) {
    var hits = ids.filter(function (i) { var leaf = i.split("/").pop(); return leaf === name + ".yaml" || leaf === name + ".yml"; });
    if (hits.length > 1) throw new Error("two documents are named " + name + ": " + hits.join(", ") + " (stems must be unique)");
    return hits[0];
  }

  function open(opts) {
    if (!opts || opts.api == null || !opts.kind || !opts.id) throw new Error("GeecsConfigurator.open needs api, kind and id");
    var base = String(opts.api).replace(/\/$/, ""), query = "?experiment=" + encodeURIComponent(opts.experiment || "");
    var d = drawer(), host = bodyEl, key = JSON.stringify([base, opts.kind, opts.experiment || "", opts.id, opts.fields || null, opts.draft ? "new" : ""]);
    // the same document again: the drawer as it was left, the edit in it kept
    if (current && current.key === key) { show(d); return current.handle; }
    var call = caller(base);
    // This opening of the drawer. Every answer that arrives later (the load, a copy, a delete) touches the
    // drawer only while it is still the one open: the drawer is shared, and another document may be in it now.
    var self = null;
    function open_() { return current !== null && current === self; }
    var form = null, schema = null, stored = null, etag = "", info = null, docId = opts.id, fields = opts.fields || null;
    var isNew = !!opts.draft;  // not yet a file: the first save creates it
    var status = el("div"), previewBox = el("div", { class: "group" });
    // a host that previews wants room for the picture beside the form
    d.style && (d.style.width = opts.preview ? "min(980px, 96vw)" : "");

    function docPath() { return "/api/documents/" + opts.kind + "/" + docId.split("/").map(encodeURIComponent).join("/") + query; }
    function operator() { var o = typeof opts.operator === "function" ? opts.operator() : opts.operator; return String(o || "").trim(); }
    function value() { return form ? merge(stored, form.value(), fields) : null; }
    // A preview shows the draft it was drawn from, or says it does not: an older request's answer is
    // dropped (the latest Preview wins, whatever order they resolve in), and an edit after it marks it stale.
    var previews = 0, drawn = null;  // drawn: the draft the shown preview is of
    function markStale() {
      if (drawn === null || previewBox.getAttribute("data-stale") === "true") return;
      previewBox.setAttribute("data-stale", "true");
      previewBox.insertBefore(banner("This preview is of an earlier edit: press Preview again.", "stale"), previewBox.firstChild);
    }
    function edited() { if (drawn !== null && JSON.stringify(value()) !== drawn) markStale(); }
    function preview() {
      var draft = value(), of = JSON.stringify(draft);
      var got = opts.preview(draft);
      if (!got || typeof got.then !== "function") return;  // a host that draws by itself: nothing to wait for or drop
      var mine = ++previews;
      drawn = null;
      previewBox.setAttribute("data-stale", "false");
      fill(previewBox, [el("div", { class: "state", "data-state": "loading" }, "Rendering the preview…")]);
      got.then(function (node) {
        if (mine !== previews) return;  // a later Preview is the one shown
        drawn = of;
        fill(previewBox, [typeof node === "string" ? el("p", {}, node) : node]);
        edited();  // edited while it rendered
      }, function (e) {
        if (mine !== previews) return;
        fill(previewBox, [banner("No preview: " + (e && e.message || e), "error")]);
      });
    }

    titleEl.textContent = opts.kind + " · " + opts.id + (isNew ? " (new)" : "");
    fill(host, [el("div", { class: "state", "data-state": "loading" }, "Loading " + opts.id + "…")]);
    show(d);
    // a new document: its draft as the stored one, no etag; nothing is read at its id
    var served = isNew ? function () { return Promise.resolve({ status: 200, body: { document: opts.draft, etag: "", read_only: false } }); } : function () { return call("GET", docPath()); };

    // a bare name: the kind's document whose stem it is
    var found = (isNew || /\.ya?ml$/i.test(docId)) ? Promise.resolve(docId) : call("GET", "/api/documents/" + opts.kind + query).then(function (r) {
      var hit = byStem((Array.isArray(r.body) ? r.body : []).map(function (e) { return e.id; }), docId);
      if (!hit) throw new Error("no " + opts.kind + " named " + docId + " in " + (opts.experiment || "the configs repo"));
      return hit;
    });
    found.then(function (id) {
      docId = id;
      return Promise.all([call("GET", "/api/schema/" + opts.kind), served(), call("GET", "/api/kinds")]);
    }).then(function (got) {
      if (!open_()) return;  // another document was opened meanwhile: the drawer is its
      if (got[1].status !== 200) throw new Error(got[1].status + " " + why(got[1].body));
      schema = got[0].body; stored = got[1].body.document; etag = got[1].body.etag;
      info = (got[2].body || []).filter(function (k) { return k.kind === opts.kind; })[0] || { dry_run: "none" };
      titleEl.textContent = opts.kind + " · " + docId + (isNew ? " (new)" : "");
      var formHost = el("div", { class: "group" }), readOnly = !!got[1].body.read_only;
      form = GeecsForm.mount(formHost, { schema: limit(schema, fields), value: pick(stored, fields), readOnly: readOnly, onChange: edited });
      var tools = el("div", { class: "row end" }), saveBtn = null;
      if (opts.toolbar) tools.appendChild(opts.toolbar);
      if (!isNew) tools.appendChild(el("a", { class: "btn ghost", href: base + "/" + query + "&kind=" + encodeURIComponent(opts.kind) + "&id=" + encodeURIComponent(docId), target: "_blank", rel: "noopener" }, "Open in the configurator"));
      if (opts.preview) tools.appendChild(button("Preview", "", preview));
      if (!readOnly) {
        saveBtn = button("Save", "primary", function () { save(false, false); });
        tools.appendChild(saveBtn);
        var save = saveFlow({
          call: call, kind: opts.kind, query: function () { return query; }, docId: function () { return docId; },
          form: function () { return form; }, value: value, etag: function () { return etag; }, operator: operator,
          dryRun: function () { return info.dry_run; }, button: saveBtn, status: status, create: function () { return isNew; },
          saved: function (b, doc) {
            etag = b.etag;
            if (b.outcome === "committed") stored = doc;
            if (opts.onSaved) opts.onSaved(b, docId);
            // created: the drawer opens the file as stored (its copy and delete), while it still shows this draft
            if (isNew && b.outcome === "committed" && open_()) open(Object.assign({}, opts, { draft: null, note: "Created " + docId + " as " + b.author + "." }));
          },
        });
      }
      var more = null;
      if (!readOnly && !isNew && named(info)) {  // new, copy or delete: rung 0, closed until asked
        more = el("details", { class: "disc" });
        more.appendChild(el("summary", {}, info.blank ? "New, copy or delete" : "Copy or delete"));
        var row = more.appendChild(el("div", { class: "row" })), newName = el("input", { class: "btn sm", placeholder: "new name", "aria-label": "the copy's name" });
        var ctx = { call: call, kind: opts.kind, query: function () { return query; }, docId: function () { return docId; }, value: value, etag: function () { return etag; }, operator: operator, status: status };  // prettier-ignore
        // the host hears of every accepted copy and delete; the drawer follows only while it still shows this document
        var copy = copyFlow(Object.assign({}, ctx, { copied: function (newId, b) {
          if (opts.onSaved) opts.onSaved(b, newId);
          if (open_()) open(Object.assign({}, opts, { id: newId, note: "Saved as a copy of " + docId + " by " + b.author + "." }));
        } }));
        var remove = deleteFlow(Object.assign({}, ctx, { host: d.parentNode || document.body, deleted: function (b) {
          if (opts.onSaved) opts.onSaved(b, docId);
          if (!open_()) return;
          current = null; form = null;  // the document is gone: opened again, it asks again
          fill(host, [status]);
        } }));
        row.appendChild(newName);
        row.appendChild(button("Save as a copy", "", function () { copy(newName.value); }));
        row.appendChild(el("span", { class: "spacer" }));
        row.appendChild(button("Delete…", "danger", function () { remove(); }));
        if (info.blank) more.appendChild(el("div", { class: "row" })).appendChild(button(newLabel(opts.kind), "", function () { create(Object.assign({}, opts, { id: null, draft: null, note: null })); }));
      }
      var fresh = isNew ? banner("New: nothing is written until Save, which creates " + docId + " (its dry run, its tier, its name unique)." + (opts.source && opts.source !== "portal" ? " Drafted from " + opts.source + "." : ""), opts.source && opts.source !== "portal" ? "stale" : null) : null;
      fill(host, [fresh, readOnly ? banner("Read-only here: " + (got[1].body.path || docId) + " is not saved from this configurator.", "denied") : null,
        fields ? el("p", { class: "dim" }, "Showing " + fields.join(", ") + "; the rest of the document is kept as it is.") : null, tools, more, status, previewBox, formHost]);
      if (opts.note) fill(status, [banner(opts.note)]);
    }).catch(function (e) {
      if (!open_()) return;  // another document was opened meanwhile
      current = null;  // opened again, it asks again
      fill(host, [el("div", { class: "state", "data-state": "error" }, "The configurator did not serve " + opts.id + ": " + (e.message || e))]);
    });

    var handle = { close: function () { var kit = typeof window !== "undefined" && window.GeecsKit; if (kit) kit.closeDrawer(); }, value: value };
    current = self = { key: key, handle: handle };
    return handle;
  }

  // ---- a new document --------------------------------------------------------------------------
  function newLabel(kind) { return "New " + kind.replace(/^analysis_/, "") + "…"; }

  // What a recipe's start form asks beyond its name, read off the kind's schema: each input kind (the
  // discriminator's mapping), and for each the enumerated fields it requires beyond `kind` (a trace's
  // loading.data_type). [{kind, fields: [{path: [...], label, options, description}]}]
  function startChoices(schema) {
    var input = ((schema || {}).properties || {}).input || {}, map = (input.discriminator || {}).mapping || {}, defs = (schema || {}).$defs || {};
    function def(ref) { return ref ? defs[String(ref).split("/").pop()] || null : null; }
    return Object.keys(map).map(function (k) {
      var d = def(map[k]) || {}, fields = [];
      (d.required || []).forEach(function (f) {
        if (f === "kind") return;
        var p = (d.properties || {})[f] || {}, sub = def(p.$ref);
        (sub && sub.required || []).forEach(function (g) {
          var q = (sub.properties || {})[g] || {}, e = def(q.$ref) || q;
          if (e.enum) fields.push({ path: [f, g], label: f + "." + g, options: e.enum, description: q.description || e.description || "" });
        });
      });
      return { kind: k, fields: fields };
    });
  }
  function nest(path, value, into) { var o = into; path.slice(0, -1).forEach(function (p) { o = o[p] = o[p] || {}; }); o[path[path.length - 1]] = value; return into; }

  function create(opts) {
    if (!opts || opts.api == null || !opts.kind) throw new Error("GeecsConfigurator.create needs api and kind");
    var base = String(opts.api).replace(/\/$/, ""), call = caller(base), d = drawer(), host = bodyEl;
    var self = { key: JSON.stringify(["new", base, opts.kind, opts.experiment || ""]), handle: { close: function () { var kit = typeof window !== "undefined" && window.GeecsKit; if (kit) kit.closeDrawer(); }, value: function () { return null; } } };
    function open_() { return current === self; }
    current = self;
    titleEl.textContent = opts.kind + " · new";
    fill(host, [el("div", { class: "state", "data-state": "loading" }, "Asking where a new " + opts.kind + " may go…")]);
    show(d);
    function field(label, input, hint) {
      var f = el("label", { class: "field" });
      f.appendChild(el("span", { class: "lbl" }, label)); f.appendChild(input);
      if (hint) f.appendChild(el("span", { class: "hint" }, hint));
      return f;
    }
    function select(options, aria) {
      var s = el("select", { class: "btn sm", "aria-label": aria });
      options.forEach(function (o, i) { var opt = el("option", i === 0 ? { value: o, selected: "" } : { value: o }, o); s.appendChild(opt); });
      if (options.length) s.value = options[0];
      return s;
    }
    Promise.all([call("GET", "/api/folders/" + opts.kind), call("GET", "/api/schema/" + opts.kind)]).then(function (got) {
      if (!open_()) return;
      if (got[0].status !== 200) throw new Error(got[0].status + " " + why(got[0].body));
      var folders = Array.isArray(got[0].body) ? got[0].body : [], status = el("div");
      if (!folders.length) { fill(host, [el("div", { class: "state", "data-state": "denied" }, "Nothing new is saved here: this configurator owns no folder of the " + opts.kind + " tree (or its commit path is off).")]); return; }
      var name = el("input", { class: "btn sm", autocomplete: "off", placeholder: "UC_Name", "aria-label": "the new document's name" });
      var folder = select(folders, "the folder");
      var kids = [el("p", { class: "dim" }, "A new " + opts.kind.replace(/^analysis_/, "") + ": its name is its file's stem, unique across the tree a run loads it from. Nothing is written until you save the draft."),
        field("name", name, "letters, digits, _ . -"), field("folder", folder, "this box's folders")];
      var device = null, input = null, extra = el("div", { class: "group" }), extras = [], choices = [];
      if (opts.kind === "analysis_recipe") {
        choices = startChoices(got[1].body);
        device = el("input", { class: "btn sm", autocomplete: "off", placeholder: "the name", "aria-label": "the data folder" });
        input = select(choices.map(function (c) { return c.kind; }), "the frame kind");
        var drawExtra = function () {
          var c = choices.filter(function (x) { return x.kind === input.value; })[0] || { fields: [] };
          extras = c.fields.map(function (f) { return { path: f.path, el: select(f.options, f.label) }; });
          fill(extra, extras.map(function (x, i) { return field(c.fields[i].label, x.el, c.fields[i].description); }));
        };
        input.addEventListener("change", drawExtra);
        drawExtra();
        kids.push(field("device", device, "the data folder its frames are read from (the name when empty)"), field("frames", input, "what the input reads: the default measure for those frames is chosen"), extra);
      }
      var go = button("Open the draft", "primary", function () {
        var stem = String(name.value || "").trim().replace(/\.ya?ml$/i, "");
        if (!/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(stem)) { fill(status, [banner("A new name is letters, digits, _ . -", "error")]); return; }
        var id = folder.value + "/" + stem + ".yaml", q = [];
        if (input) {
          var fields = {};
          extras.forEach(function (x) { nest(x.path, x.el.value, fields); });
          q.push("input=" + encodeURIComponent(input.value), "device=" + encodeURIComponent(String(device.value || "").trim()));
          if (extras.length) q.push("fields=" + encodeURIComponent(JSON.stringify(fields)));
        }
        go.disabled = true;
        fill(status, [el("p", { class: "dim" }, "Drafting " + id + "…")]);
        call("GET", "/api/blank/" + opts.kind + "/" + id.split("/").map(encodeURIComponent).join("/") + (q.length ? "?" + q.join("&") : "")).then(function (r) {
          go.disabled = false;
          if (!open_()) return;  // the drawer moved on while it was drafted
          if (r.status !== 200) { fill(status, [banner("No draft (" + r.status + "): " + why(r.body), "error")]); return; }
          open(Object.assign({}, opts, { id: id, draft: r.body.document, source: r.body.source }));
        }).catch(function (e) { go.disabled = false; if (open_()) fill(status, [banner(String(e.message || e), "error")]); });
      });
      kids.push(el("div", { class: "row end" }).appendChild(go).parentNode, status);
      fill(host, kids);
    }).catch(function (e) {
      if (!open_()) return;
      current = null;
      fill(host, [el("div", { class: "state", "data-state": "error" }, "The configurator did not answer: " + (e.message || e))]);
    });
    return self.handle;
  }

  out.open = open;
  out.create = create;
  out.startChoices = startChoices;
  out.saveFlow = saveFlow;
  out.deleteFlow = deleteFlow;
  out.copyFlow = copyFlow;
  out.named = named;
  out.caller = caller;
  out.limit = limit;
  out.merge = merge;
  out.byStem = byStem;
})(typeof module === "object" && module.exports ? module.exports : (window.GeecsConfigurator = {}));
