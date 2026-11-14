/* The embedded editor: one configs-repo document edited in a kit drawer on another surface's page.
 *
 * The mount contract. A host page loads, in order, its own /theme/kit.js and /theme/form.js and this
 * script from the configurator (<configurator>/static/embed.js; the host never imports the package),
 * and the configurator lists the host's origin in GEECS_CONFIGURATOR_CORS_ORIGINS. Then:
 *
 *   GeecsConfigurator.open({
 *     api,         // the configurator's URL as the browser reaches it (a site value of the host)
 *     kind,        // a file kind, e.g. "preset"
 *     experiment,  // the experiment folder
 *     id,          // the document id ("name.yaml"); a bare name is looked up among the kind's documents
 *     fields,      // optional: the top-level fields drawn; every other field is kept exactly as stored
 *     operator,    // the operator's name, or a function returning it: the host's own identity, required to save
 *     preview,     // optional: function (document) — the host's Preview button
 *     onSaved,     // optional: function (answer) — after a save the configurator accepted
 *   })  ->  { close(), value() }
 *
 * What a save does is the configurator's, exactly as on its own page: the kind's dry run first (a refusal
 * stops it there), then PUT /api/documents/<kind>/<id> with the etag the document was served with and the
 * operator, who is the git author; the tier decides commit or proposal; a moved base is a 409 that keeps
 * the edit; a document this box does not own, or a read-only service, is drawn read-only. No identity is
 * stored here: the operator is the host's, asked for on every save.
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
  function banner(text, state) { return el("div", state ? { class: "banner", "data-state": state } : { class: "banner" }, text); }
  function button(text, cls, onclick) { var b = el("button", { class: "btn " + (cls || ""), type: "button" }, text); b.addEventListener("click", onclick); return b; }
  function why(b) {
    var d = b && b.detail;
    if (d && d.reason) return d.reason;
    return typeof d === "string" ? d : JSON.stringify(d);
  }

  // The drawer, made once and kept: closing it (Esc, the scrim) never loses the edit in it.
  function drawer() {
    if (drawerEl) return drawerEl;
    drawerEl = el("aside", { class: "drawer", id: "geecs-configurator-drawer", role: "dialog", "aria-modal": "false", "aria-labelledby": "geecs-configurator-title", "aria-hidden": "true", "data-open": "false" });
    var head = el("header");
    titleEl = head.appendChild(el("span", { class: "eyebrow", id: "geecs-configurator-title" }, "configurator"));
    head.appendChild(el("button", { class: "btn sm ghost", type: "button", "data-drawer-close": "" }, "Close"));
    drawerEl.appendChild(head);
    bodyEl = drawerEl.appendChild(el("div", { class: "body" }));
    document.body.appendChild(drawerEl);
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

  function open(opts) {
    if (!opts || !opts.api || !opts.kind || !opts.id) throw new Error("GeecsConfigurator.open needs api, kind and id");
    var base = String(opts.api).replace(/\/$/, ""), query = "?experiment=" + encodeURIComponent(opts.experiment || "");
    var d = drawer(), host = bodyEl, key = JSON.stringify([base, opts.kind, opts.experiment || "", opts.id, opts.fields || null]);
    // the same document again: the drawer as it was left, the edit in it kept
    if (current && current.key === key) { show(d); return current.handle; }
    var form = null, schema = null, stored = null, etag = "", info = null, docId = opts.id, fields = opts.fields || null;
    var status = el("div"), saveBtn = null;

    function call(method, path, payload) {
      var init = { method: method, headers: { "Content-Type": "application/json" } };
      if (payload !== undefined) init.body = JSON.stringify(payload);
      return fetch(base + path, init).then(function (r) {
        return r.text().then(function (t) { var b = {}; try { b = t ? JSON.parse(t) : {}; } catch (e) { b = { detail: t }; } return { status: r.status, body: b }; });
      });
    }
    function docPath() { return "/api/documents/" + opts.kind + "/" + docId.split("/").map(encodeURIComponent).join("/") + query; }
    function operator() { var o = typeof opts.operator === "function" ? opts.operator() : opts.operator; return String(o || "").trim(); }
    function value() { return form ? merge(stored, form.value(), fields) : null; }

    titleEl.textContent = opts.kind + " · " + opts.id;
    fill(host, [el("div", { class: "state", "data-state": "loading" }, "Loading " + opts.id + "…")]);
    show(d);

    // a bare name: the kind's document whose stem it is
    var found = /\.ya?ml$/i.test(docId) ? Promise.resolve(docId) : call("GET", "/api/documents/" + opts.kind + query).then(function (r) {
      var ids = (Array.isArray(r.body) ? r.body : []).map(function (e) { return e.id; });
      var hit = ids.filter(function (i) { return i === docId + ".yaml" || i === docId + ".yml"; })[0];
      if (!hit) throw new Error("no " + opts.kind + " named " + docId + " in " + (opts.experiment || "the configs repo"));
      return hit;
    });
    found.then(function (id) {
      docId = id;
      return Promise.all([call("GET", "/api/schema/" + opts.kind), call("GET", docPath()), call("GET", "/api/kinds")]);
    }).then(function (got) {
      if (got[1].status !== 200) throw new Error(got[1].status + " " + why(got[1].body));
      schema = got[0].body; stored = got[1].body.document; etag = got[1].body.etag;
      info = (got[2].body || []).filter(function (k) { return k.kind === opts.kind; })[0] || { dry_run: "none" };
      var formHost = el("div", { class: "group" }), readOnly = !!got[1].body.read_only;
      form = GeecsForm.mount(formHost, { schema: limit(schema, fields), value: pick(stored, fields), readOnly: readOnly });
      var tools = el("div", { class: "row end" });
      tools.appendChild(el("a", { class: "btn ghost", href: base + "/" + query + "&kind=" + encodeURIComponent(opts.kind) + "&id=" + encodeURIComponent(docId), target: "_blank", rel: "noopener" }, "Open in the configurator"));
      if (opts.preview) tools.appendChild(button("Preview", "", function () { opts.preview(value()); }));
      if (!readOnly) { saveBtn = button("Save", "primary", function () { save(false, false); }); tools.appendChild(saveBtn); }
      fill(host, [readOnly ? banner("Read-only here: " + (got[1].body.path || docId) + " is not saved from this configurator.", "denied") : null,
        fields ? el("p", { class: "dim" }, "Showing " + fields.join(", ") + "; the rest of the document is kept as it is.") : null, tools, status, formHost]);
    }).catch(function (e) {
      if (current && current.key === key) current = null;  // opened again, it asks again
      fill(host, [el("div", { class: "state", "data-state": "error" }, "The configurator did not serve " + opts.id + ": " + (e.message || e))]);
    });

    function save(convert, checked) {
      var who = operator();
      if (!who) { fill(status, [banner("Type your name first (the host's operator field): it is the git author.", "error")]); return; }
      if (form.invalid().length) { fill(status, [banner("A field does not parse: " + form.invalid().join(", "), "error")]); return; }
      saveBtn.disabled = true;
      if (!checked && info.dry_run !== "none") {
        call("POST", "/api/dry-run/" + opts.kind + query, { document: value() }).then(function (r) {
          saveBtn.disabled = false;
          var dr = r.body || {};
          if (r.status !== 200) { fill(status, [banner("The dry run did not answer: " + why(dr), "error")]); return; }
          if (dr.refusal) { fill(status, [banner("The " + dr.by + "'s dry run refused your edit: " + dr.refusal + ". Nothing was saved.", "error")]); return; }
          if (dr.by === "none") { save(convert, true); return; }
          fill(status, [banner(dr.ran ? "The " + dr.by + "'s dry run: " + (dr.detail || "no objection") + "." : "The " + dr.by + "'s dry run did not run: " + dr.detail + ".", dr.ran ? null : "stale"),
            button(dr.ran ? "Save" : "Save without it", "primary", function () { save(convert, true); })]);
        }).catch(function (e) { saveBtn.disabled = false; fill(status, [banner(String(e.message || e), "error")]); });
        return;
      }
      call("PUT", docPath(), { document: value(), etag: etag, operator: who, convert_legacy: !!convert }).then(function (r) {
        saveBtn.disabled = false;
        var b = r.body;
        if (r.status === 200) {
          etag = b.etag; if (b.outcome === "committed") stored = value();
          fill(status, [banner(b.outcome === "committed" ? "Committed to main as " + b.author + "." : b.outcome === "proposed" ? "Proposed for the maintainer's review on " + b.branch + "." : "Nothing to do: the stored document already says this.")]);
          if (opts.onSaved) opts.onSaved(b);
        } else if (r.status === 409) {
          fill(status, [banner("Changed since it was opened: " + why(b) + ". Nothing was saved; your edit is still here. Open it in the configurator to re-apply it.", "error")]);
        } else if (r.status === 428) {
          fill(status, [banner("This file is in the legacy dialect: saving rewrites it whole in the current schema.", "stale"), button("Convert and save", "primary", function () { save(true, true); })]);
        } else if (r.status === 422 && b.detail && b.detail.errors) {
          form.setErrors(b.detail.errors);
          fill(status, [banner(b.detail.errors.length + " error(s); nothing was saved.", "error")]);
        } else {
          fill(status, [banner("Not saved (" + r.status + "): " + why(b), "error")]);
        }
      }).catch(function (e) { saveBtn.disabled = false; fill(status, [banner(String(e.message || e), "error")]); });
    }

    var handle = { close: function () { var kit = typeof window !== "undefined" && window.GeecsKit; if (kit) kit.closeDrawer(); }, value: value };
    current = { key: key, handle: handle };
    return handle;
  }

  out.open = open;
  out.limit = limit;
  out.merge = merge;
})(typeof module === "object" && module.exports ? module.exports : (window.GeecsConfigurator = {}));
