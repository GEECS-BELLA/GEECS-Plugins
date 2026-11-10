/* GEECS web theme — the one schema-driven form every GEECS editor draws with.
 *
 * A JSON Schema (pydantic's model_json_schema(), with GEECS-Schemas' `tier`
 * annotations) in, kit controls out, the edited document and its diff back:
 *
 *   var form = GeecsForm.mount(host, { schema, value, onChange: function (doc) { … } });
 *   form.value();                      // the document as edited
 *   form.invalid();                    // paths whose JSON text does not parse (the document keeps the last good value)
 *   form.setErrors(errors);            // [{loc, msg}] from the service's validate, drawn at the fields
 *   GeecsForm.diff(stored, form.value(), schema)   // {changes: [{path, op, before, after, tier}], tier}
 *
 * The rules, each pinned in tests/test_form_js.py under node:
 * - $ref / $defs resolve, allOf merges; `anyOf [X, null]` is an optional X.
 * - A union of objects is a choice, then the chosen variant's fields: by its discriminator
 *   (`discriminator.propertyName`, else a property every variant fixes with `const`), else structurally (the
 *   first variant whose required keys, closed key set and const / enum values the value fits). A union with a
 *   structured member that is not a union of objects is a JSON text area: it parses, or it is marked invalid
 *   and the document keeps its last good value. A union of scalars is a text field read back as its kind.
 * - Objects are folds; arrays and maps (additionalProperties, or one patternProperties) are item panels with
 *   add / remove / reorder, and adding an existing key is refused; keys an open object does not declare are
 *   drawn as key / value rows, never hidden. Enums are selects, booleans a true/false select; a field's
 *   description is its help, `required` marks it, `schema_version` is never drawn (tools own it).
 * - Untouched text is the stored value: a field whose text is put back to what it showed holds the stored
 *   value exactly (a "4.0" word never becomes the number 4); an emptied optional or nullable field returns to
 *   the stored state (null stays null, absent stays absent).
 * - A field stricter than its document carries `tier` in its property schema; the form marks it. A change's
 *   tier is the strictest declared on its path, and when a whole subtree is added, removed, set, cleared or
 *   switched to another variant, the strictest declared anywhere in that subtree's schema too. A tier this
 *   script does not know ranks strictest (fail closed).
 * - "Changed" is any difference at a field's path, a reordered list included (a reorder changes every
 *   index it moved); the diff's tier is the strictest tier among its changes, which decides what a save does.
 * - An error's location is walked down the drawn tree to the nearest node (a variant's name in the location
 *   is skipped); the document's own errors show in a banner at the top.
 *
 * window.GeecsForm in a page, module.exports under node. No dependency, no colour: kit classes only.
 */
(function (api) {
  "use strict";
  var TIERS = ["operator", "maintainer", "parameter", "sudo", "code"];
  var HIDDEN = ["schema_version"];
  var SCALAR_TYPES = ["string", "number", "integer", "boolean", "null"];

  function isObj(v) { return v !== null && typeof v === "object" && !Array.isArray(v); }
  function clone(v) { return v === undefined ? undefined : JSON.parse(JSON.stringify(v)); }
  // Deep equality: key order is not a difference, list order is.
  function equal(a, b) {
    if (a === b) return true;
    if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every(function (x, i) { return equal(x, b[i]); });
    if (!isObj(a) || !isObj(b)) return false;
    var ka = Object.keys(a), kb = Object.keys(b);
    return ka.length === kb.length && ka.every(function (k) { return k in b && equal(a[k], b[k]); });
  }
  // A tier's rank: none is -1, an unknown one is the strictest (fail closed).
  function rank(t) { if (!t) return -1; var i = TIERS.indexOf(t); return i < 0 ? TIERS.length : i; }
  function strictest(a, b) { return rank(a) >= rank(b) ? (a || b || null) : b; }
  function humanize(k) { var s = String(k).replace(/_/g, " "); return s.charAt(0).toUpperCase() + s.slice(1); }
  function pathKey(path) { return path.join("."); }

  // ---- schema reading ----
  // $ref (against the root's $defs) and allOf, the referring node's own keys (description, default, tier) winning.
  function deref(s, root) {
    if (!isObj(s)) return {};
    var out = s;
    if (s.$ref) {
      var name = s.$ref.split("/").pop(), defs = root.$defs || root.definitions || {};
      out = Object.assign({ title: name }, deref(defs[name] || {}, root), s);
      delete out.$ref;
    }
    if (out.allOf) {
      var merged = Object.assign({}, out); delete merged.allOf;
      out.allOf.forEach(function (part) {
        var p = deref(part, root);
        merged = Object.assign({}, p, merged, { properties: Object.assign({}, p.properties, merged.properties), required: (p.required || []).concat(merged.required || []) });
      });
      out = merged;
    }
    return out;
  }
  // {schema, nullable}: `anyOf [X, null]` (or `type: [X, "null"]`) is an optional X, the outer keys kept.
  function unwrap(s, root) {
    s = deref(s, root);
    if (Array.isArray(s.type) && s.type.indexOf("null") >= 0) {
      var t = s.type.filter(function (x) { return x !== "null"; });
      return { schema: Object.assign({}, s, { type: t.length === 1 ? t[0] : t }), nullable: true };
    }
    var list = s.anyOf || s.oneOf;
    if (!list) return { schema: s, nullable: false };
    var rest = list.filter(function (x) { return deref(x, root).type !== "null"; });
    if (rest.length === list.length) return { schema: s, nullable: false };
    var outer = {}; ["title", "description", "default", "tier"].forEach(function (k) { if (k in s) outer[k] = s[k]; });
    if (rest.length === 1) return { schema: Object.assign({}, deref(rest[0], root), outer), nullable: true };
    var u = Object.assign({}, s, { anyOf: rest }); delete u.oneOf;
    return { schema: u, nullable: true };
  }
  // The schema of an object's undeclared keys: additionalProperties, one patternProperties, {} when the object
  // is open (no `additionalProperties: false`), null when closed.
  function extraSchema(s) {
    if (isObj(s.additionalProperties)) return s.additionalProperties;
    var pp = isObj(s.patternProperties) ? Object.keys(s.patternProperties) : [];
    if (pp.length === 1) return s.patternProperties[pp[0]];
    return s.additionalProperties === false || pp.length > 1 ? null : {};
  }
  // A union of objects: {prop (null: chosen structurally), list: [{tag, schema}]}, or null when a member is not an object.
  function variants(s, root) {
    var list = s.oneOf || s.anyOf;
    if (!list) return null;
    var vs = list.map(function (x) { return deref(x, root); });
    if (!vs.every(function (v) { return v.type === "object" || v.properties; })) return null;
    var prop = s.discriminator && s.discriminator.propertyName;
    if (!prop) {
      var keys = Object.keys(vs[0].properties || {});
      prop = keys.filter(function (k) { return vs.every(function (v) { return v.properties[k] && deref(v.properties[k], root).const !== undefined; }); })[0] || null;
    }
    return { prop: prop, list: vs.map(function (v, i) { return { tag: prop ? deref(v.properties[prop], root).const : v.title || "option " + (i + 1), schema: v }; }) };
  }
  // Whether a value fits a variant: its required keys present, no key a closed variant lacks, every const / enum met.
  function fits(v, value, root) {
    if (!isObj(value)) return false;
    var props = v.properties || {};
    if ((v.required || []).some(function (k) { return !(k in value); })) return false;
    if (v.additionalProperties === false && Object.keys(value).some(function (k) { return !(k in props); })) return false;
    return Object.keys(props).every(function (k) {
      var p = deref(props[k], root);
      if (!(k in value)) return true;
      if (p.const !== undefined) return value[k] === p.const;
      return p.enum ? p.enum.indexOf(value[k]) >= 0 : true;
    });
  }
  function pick(vs, value, root) {
    if (vs.prop) {
      var tag = isObj(value) ? value[vs.prop] : undefined;
      return vs.list.filter(function (v) { return v.tag === tag; })[0] || vs.list[0];
    }
    return vs.list.filter(function (v) { return fits(v.schema, value, root); })[0] || vs.list[0];
  }
  // What a schema is drawn as.
  function kindOf(s, root) {
    if (s.const !== undefined) return "const";
    if (s.enum) return "enum";
    if (variants(s, root)) return "union";
    var list = s.anyOf || s.oneOf;
    if (list) return list.every(function (x) { var d = deref(x, root); return SCALAR_TYPES.indexOf(d.type) >= 0 || d.const !== undefined || d.enum; }) ? "scalar" : "json";
    if (Array.isArray(s.type)) return s.type.every(function (t) { return SCALAR_TYPES.indexOf(t) >= 0; }) ? "scalar" : "json";
    var t = s.type;
    if (t === "boolean" || t === "integer" || t === "number" || t === "string") return t;
    if (t === "array") return "array";
    if (t === "object" || s.properties || s.additionalProperties !== undefined || s.patternProperties) {
      if (s.properties && Object.keys(s.properties).length) return "object";
      return extraSchema(s) ? "map" : "object";
    }
    return "any";  // untyped: a number, true / false, null, [list] or {object} as JSON, anything else as a word
  }
  // The strictest tier declared anywhere in a schema (every variant, item and entry), or null.
  function tierIn(s, root, seen) {
    seen = seen || [];
    if (!isObj(s)) return null;
    if (s.$ref) { if (seen.indexOf(s.$ref) >= 0) return s.tier || null; seen = seen.concat([s.$ref]); }
    var d = deref(s, root), t = d.tier || null;
    var kids = [].concat(d.anyOf || [], d.oneOf || [], d.items ? [d.items] : [], isObj(d.additionalProperties) ? [d.additionalProperties] : [],
      Object.keys(d.properties || {}).map(function (k) { return d.properties[k]; }),
      Object.keys(d.patternProperties || {}).map(function (k) { return d.patternProperties[k]; }));
    kids.forEach(function (k) { t = strictest(t, tierIn(k, root, seen)); });
    return t;
  }
  // The value a new item, a newly set optional or a variant switch starts from.
  function defaultFor(s, root) {
    var u = unwrap(s, root); s = u.schema;
    if (s.default !== undefined) return clone(s.default);
    if (u.nullable) return null;
    if (s.const !== undefined) return s.const;
    var vs = variants(s, root);
    if (vs) { var v0 = defaultFor(vs.list[0].schema, root); if (vs.prop) v0[vs.prop] = vs.list[0].tag; return v0; }
    var k = kindOf(s, root);
    if (k === "object") {
      var o = {};
      Object.keys(s.properties || {}).forEach(function (key) {
        var p = deref(s.properties[key], root);
        if ((s.required || []).indexOf(key) >= 0 || p.const !== undefined) o[key] = defaultFor(p, root);
      });
      return o;
    }
    return { enum: s.enum && s.enum[0], array: [], map: {}, boolean: false, integer: 0, number: 0, string: "" }[k] ?? null;
  }

  // ---- the tree a form is drawn from ----
  // node: {path, key, label, help, required, tier (declared here), kind, value, nullable, extra (an undeclared key),
  //        options, choice, children, items, itemSchema}
  function node(s, value, path, ctx) {
    var u = unwrap(s, ctx.root), sc = u.schema, kind = kindOf(sc, ctx.root), last = path[path.length - 1];
    var n = {
      path: path, key: last, schema: sc, kind: kind, value: value, nullable: u.nullable, extra: !!ctx.extra,
      label: typeof last === "number" ? "#" + (last + 1) : ctx.extra ? String(last) : path.length ? humanize(last) : sc.title || "Document",
      help: sc.description || "", required: !!ctx.required, tier: sc.tier || null,
    };
    if (kind === "enum") n.options = sc.enum;
    if (kind === "const") n.constValue = sc.const;
    if ((value === null || value === undefined) && ["object", "union", "array", "map"].indexOf(kind) >= 0) { n.unset = true; return n; }
    if (kind === "union") {
      var vs = variants(sc, ctx.root), chosen = pick(vs, value, ctx.root);
      n.choice = { prop: vs.prop, tags: vs.list.map(function (v) { return v.tag; }), tag: chosen.tag };
      n.children = fields(chosen.schema, value, path, ctx, vs.prop);
      n.itemSchema = extraSchema(chosen.schema);
    } else if (kind === "object") {
      n.children = fields(sc, value, path, ctx, null);
      n.itemSchema = extraSchema(sc);
    } else if (kind === "array") {
      n.itemSchema = sc.items || {};
      n.items = (Array.isArray(value) ? value : []).map(function (v, i) { return node(n.itemSchema, v, path.concat([i]), { root: ctx.root }); });
    } else if (kind === "map") {
      n.itemSchema = extraSchema(sc);
      n.items = Object.keys(isObj(value) ? value : {}).map(function (k) { return node(n.itemSchema, value[k], path.concat([k]), { root: ctx.root }); });
    }
    return n;
  }
  // An object's declared fields, then the keys it holds that it does not declare (drawn, never hidden).
  function fields(s, value, path, ctx, skip) {
    var props = s.properties || {}, obj = isObj(value) ? value : {}, extra = extraSchema(s) || {};
    var declared = Object.keys(props).filter(function (k) { return k !== skip && HIDDEN.indexOf(k) < 0; }).map(function (k) {
      return node(props[k], obj[k], path.concat([k]), { root: ctx.root, required: (s.required || []).indexOf(k) >= 0 });
    });
    var undeclared = Object.keys(obj).filter(function (k) { return !(k in props) && k !== skip && HIDDEN.indexOf(k) < 0; }).map(function (k) {
      return node(extra, obj[k], path.concat([k]), { root: ctx.root, extra: true });
    });
    return declared.concat(undeclared);
  }
  function tree(schema, value) { return node(schema, value, [], { root: schema }); }

  // ---- the document's edits (each returns a new document) ----
  function getAt(doc, path) { return path.reduce(function (o, k) { return o == null ? undefined : o[k]; }, doc); }
  function setAt(doc, path, v) {
    if (!path.length) return clone(v);
    var out = clone(doc) || {}, o = out;
    for (var i = 0; i < path.length - 1; i++) { if (o[path[i]] == null) o[path[i]] = typeof path[i + 1] === "number" ? [] : {}; o = o[path[i]]; }
    var last = path[path.length - 1];
    if (v === undefined) { if (Array.isArray(o)) o.splice(last, 1); else delete o[last]; } else o[last] = clone(v);
    return out;
  }
  function removeAt(doc, path) { return setAt(doc, path, undefined); }
  // Append an item, or add a map entry by key; an existing key is refused (the document comes back unchanged).
  function addItem(doc, path, itemSchema, root, key) {
    var list = getAt(doc, path);
    if (key !== undefined) return isObj(list) && key in list ? doc : setAt(doc, path.concat([key]), defaultFor(itemSchema, root));
    return setAt(doc, path, (Array.isArray(list) ? list : []).concat([defaultFor(itemSchema, root)]));
  }
  function moveItem(doc, path, delta) {
    var list = clone(getAt(doc, path.slice(0, -1))), i = path[path.length - 1], j = i + delta;
    if (!Array.isArray(list) || j < 0 || j >= list.length) return doc;
    var t = list[i]; list[i] = list[j]; list[j] = t;
    return setAt(doc, path.slice(0, -1), list);
  }
  // Switch a union to the variant `tag`: its defaults, keeping the fields the two variants share.
  function choose(doc, path, unionSchema, tag, root) {
    var vs = variants(unwrap(unionSchema, root).schema, root), v = vs.list.filter(function (x) { return x.tag === tag; })[0];
    var next = defaultFor(v.schema, root), old = getAt(doc, path) || {};
    Object.keys(v.schema.properties || {}).forEach(function (k) { if (k !== vs.prop && old[k] !== undefined) next[k] = old[k]; });
    if (vs.prop) next[vs.prop] = tag;
    return setAt(doc, path, next);
  }
  // An input's text as the value its kind holds; an empty optional is absent. Throws on JSON that does not parse.
  function parse(kind, text) {
    if (kind === "boolean") return text === "" ? undefined : text === "true";
    if (text === "") return kind === "string" ? "" : undefined;
    if (kind === "json") return JSON.parse(text);
    if (kind === "any") { try { return JSON.parse(text); } catch (e) { return text; } }
    if (kind === "integer" || kind === "number" || kind === "scalar") {
      var n = Number(text);
      return isFinite(n) && /^\s*-?(\d|\.\d)/.test(text) ? n : text;  // a word stays a word; the service says if it is wrong
    }
    return text;
  }
  // The text a field shows for a value.
  function display(kind, v) {
    if (v === undefined || v === null) return "";
    if (kind === "json") return JSON.stringify(v, null, 1);
    return typeof v === "object" || (kind === "any" && typeof v !== "string") ? JSON.stringify(v) : String(v);
  }
  // A field's new value from its text, measured against the stored value: the text it showed is the stored value
  // exactly, and an emptied optional or nullable field is the stored state.
  function readBack(n, text, stored) {
    if (text === display(n.kind, stored)) return stored;
    if (text === "" && (!n.required || n.nullable) && n.kind !== "string") return stored === null || stored === undefined ? stored : n.nullable ? null : undefined;
    if (text === "" && n.kind === "string" && (!n.required || n.nullable) && (stored === null || stored === undefined)) return stored;
    return parse(n.kind, text);
  }

  // ---- the diff ----
  // Every difference between two documents at a field's path, each with the strictest tier on that path, and on a
  // whole subtree's add, remove, set, clear or variant switch, the strictest declared anywhere in its schema.
  function diff(before, after, schema) {
    var root = schema, changes = [];
    function push(sc, b, a, path, tier) {
      if (isObj(b) || Array.isArray(b) || isObj(a) || Array.isArray(a)) tier = strictest(tier, tierIn(sc, root));
      changes.push({ path: path, op: b === undefined ? "add" : a === undefined ? "remove" : "change", before: b, after: a, tier: tier });
    }
    (function walk(s, b, a, path, tier) {
      var u = unwrap(s, root), sc = u.schema;
      tier = strictest(tier, sc.tier || null);
      if (equal(b, a)) return;
      var vs = variants(sc, root);
      if (vs) {
        var vb = isObj(b) && pick(vs, b, root), va = isObj(a) && pick(vs, a, root);
        if (!vb || !va || vb.tag !== va.tag) return push(sc, b, a, path, tier);
        sc = va.schema;
      }
      if (isObj(b) && isObj(a)) {
        var extra = extraSchema(sc) || {};
        var keys = Object.keys(b).concat(Object.keys(a).filter(function (k) { return !(k in b); }));
        keys.forEach(function (k) { walk((sc.properties || {})[k] || extra, b[k], a[k], path.concat([k]), tier); });
      } else if (Array.isArray(b) && Array.isArray(a)) {
        for (var i = 0; i < Math.max(b.length, a.length); i++) walk(sc.items || {}, b[i], a[i], path.concat([i]), tier);
      } else {
        push(sc, b, a, path, tier);
      }
    })(schema, before, after, [], null);
    return { changes: changes, tier: changes.reduce(function (t, c) { return strictest(t, c.tier); }, null) };
  }

  // Validation errors ({loc, msg}, pydantic's locations) by the drawn node they belong to: each location is walked
  // down the tree as far as it goes, a segment that is not a field of a chosen variant (its tag or class name) being
  // skipped, so every error lands at its nearest drawn node ("" is the document itself).
  function placeErrors(t, errors) {
    var out = {};
    (errors || []).forEach(function (e) {
      var loc = Array.isArray(e.loc) ? e.loc : String(e.loc || "").split(".").filter(Boolean), n = t;
      for (var i = 0; i < loc.length; i++) {
        var kids = (n.children || []).concat(n.items || []);
        var next = kids.filter(function (c) { return String(c.key) === String(loc[i]); })[0];
        if (next) n = next; else if (!n.choice) break;
      }
      var key = pathKey(n.path);
      out[key] = out[key] ? out[key] + "; " + e.msg : e.msg;
    });
    return out;
  }

  // ---- in a page: kit controls ----
  var ids = 0;
  function h(tag, attrs, kids) {
    var el = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      if (attrs[k] === undefined || attrs[k] === false) return;
      if (k === "text") el.textContent = attrs[k]; else if (k.slice(0, 2) === "on") el.addEventListener(k.slice(2), attrs[k]); else el.setAttribute(k, attrs[k] === true ? "" : String(attrs[k]));
    });
    (kids || []).forEach(function (c) { if (c) el.appendChild(c); });
    return el;
  }
  // An absent field reads as its default, which is what a consumer will use.
  function unsetText(n) { return n.schema.default !== undefined && n.schema.default !== null && typeof n.schema.default !== "object" ? "(default: " + n.schema.default + ")" : "(not set)"; }
  function tierMark(n) { return n.tier ? h("span", { class: "tier", "data-tier": n.tier, title: "changing this needs the " + n.tier + " tier", text: n.tier }) : null; }
  function banner(msg) { return msg ? h("div", { class: "banner", "data-state": "error", text: msg }) : null; }

  function mount(host, opts) {
    var schema = opts.schema, root = schema, original = clone(opts.original !== undefined ? opts.original : opts.value);
    var doc = clone(opts.value), errors = {}, bad = {}, ro = !!opts.readOnly;
    function changed(path) { return !equal(getAt(original, path), getAt(doc, path)); }
    function update(next, redraw) { doc = next; if (redraw) draw(); if (opts.onChange) opts.onChange(clone(doc)); }
    function btn(label, onclick, title) { return ro ? null : h("button", { type: "button", class: "btn sm ghost", title: title, onclick: onclick, text: label }); }

    function control(n, id, err) {
      var dis = ro || n.kind === "const";
      var onIn = function (ev) {
        var c = ev.target, key = pathKey(n.path), next;
        try { next = readBack(n, c.value, getAt(original, n.path)); delete bad[key]; } catch (e) { bad[key] = "not JSON: " + e.message; }
        if (bad[key]) { c.setAttribute("aria-invalid", "true"); err.textContent = bad[key]; return; }
        c.setAttribute("aria-invalid", "false"); err.textContent = "";
        update(setAt(doc, n.path, next));
        if (c.parentNode && c.parentNode.setAttribute) c.parentNode.setAttribute("data-changed", String(changed(n.path)));
      };
      if (n.kind === "enum" || n.kind === "boolean") {
        var choices = n.kind === "boolean" ? ["true", "false"] : n.options.map(String);
        if (!n.required || n.value === undefined) choices = [""].concat(choices);
        return h("select", { id: id, disabled: dis, onchange: onIn }, choices.map(function (o) { return h("option", { value: o, selected: String(n.value ?? "") === o, text: o === "" ? unsetText(n) : o }); }));
      }
      var text = n.kind === "const" ? String(n.constValue) : display(n.kind, n.value);
      if (n.kind === "json" || (n.kind === "string" && (text.length > 60 || /description|expression/.test(String(n.key))))) {
        return h("textarea", { id: id, rows: n.kind === "json" ? Math.min(8, text.split("\n").length + 1) : 2, spellcheck: "false", disabled: dis, oninput: onIn, text: text });
      }
      var input = h("input", { id: id, type: "text", inputmode: n.kind === "integer" || n.kind === "number" ? "decimal" : undefined, placeholder: n.value === undefined ? unsetText(n) : undefined, disabled: dis, oninput: onIn });
      input.value = text;
      return input;
    }
    function scalar(n, tools) {
      var id = "gf-" + (++ids), key = pathKey(n.path), msg = bad[key] || errors[key], err = h("span", { class: "err", text: msg || "" }), c = control(n, id, err);
      if (msg) c.setAttribute("aria-invalid", "true");
      return h("div", { class: "field", "data-path": key, "data-changed": String(changed(n.path)) }, [
        h("label", { for: id }, [document.createTextNode(n.label), n.required ? h("span", { class: "req", text: "*" }) : null, tierMark(n)]),
        c, err, n.help ? h("span", { class: "hint", text: n.help }) : null, tools && tools.length ? h("div", { class: "row" }, tools) : null,
      ]);
    }
    function fold(n, body, tools) {
      var key = pathKey(n.path);
      return h("details", { class: "fold", open: true, "data-path": key, "data-changed": String(changed(n.path)) }, [
        h("summary", { class: "eyebrow" }, [document.createTextNode(n.label + (n.items ? " (" + n.items.length + ")" : "")), tierMark(n)]),
        n.help ? h("p", { class: "hint dim", text: n.help }) : null, banner(errors[key]),
        tools && tools.length ? h("div", { class: "row" }, tools.filter(Boolean)) : null, body,
      ]);
    }
    function group(kids) {
      var simple = kids.filter(function (k) { return !k.complex; }), complex = kids.filter(function (k) { return k.complex; });
      return h("div", { class: "group" }, [simple.length ? h("div", { class: "grid tight" }, simple.map(function (k) { return k.el; })) : null].concat(complex.map(function (k) { return k.el; })));
    }
    // "new key" + Add: a map's entries, or an open object's undeclared keys; an existing key is refused.
    function keyAdder(n) {
      if (ro || !n.itemSchema) return [];
      var kid = "gf-" + (++ids), keyIn = h("input", { id: kid, placeholder: "new key", "aria-label": "new key" });
      return [h("span", { class: "field" }, [keyIn]), btn("Add", function () {
        if (!keyIn.value) return;
        var next = addItem(doc, n.path, n.itemSchema, root, keyIn.value);
        if (next === doc) { keyIn.setAttribute("aria-invalid", "true"); keyIn.setAttribute("title", "that key exists"); return; }
        update(next, true);
      })];
    }
    function draw1(n, inPanel) {
      var complex = ["object", "union", "array", "map"].indexOf(n.kind) >= 0;
      var remove = n.extra ? btn("Remove", function () { update(removeAt(doc, n.path), true); }) : null;
      if (!complex) return { el: scalar(n, [remove]) };
      if (n.unset) return { complex: true, el: fold(n, h("p", { class: "dim", text: "not set" }), [btn("Set", function () { update(setAt(doc, n.path, defaultFor(Object.assign({}, n.schema, { default: undefined }), root)), true); })]) };
      var clear = n.nullable || (!n.required && !n.extra && n.path.length) ? btn("Clear", function () { update(n.nullable ? setAt(doc, n.path, null) : removeAt(doc, n.path), true); }) : null;
      if (n.kind === "object" || n.kind === "union") {
        var kids = n.children.map(function (c) { return draw1(c); });
        if (n.choice) {
          var cid = "gf-" + (++ids);
          kids.unshift({ el: h("div", { class: "field" }, [h("label", { for: cid, text: n.choice.prop ? humanize(n.choice.prop) : "Variant" }),
            h("select", { id: cid, disabled: ro, onchange: function (ev) { update(choose(doc, n.path, n.schema, ev.target.value, root), true); } },
              n.choice.tags.map(function (t) { return h("option", { value: t, selected: t === n.choice.tag, text: t }); }))]) });
        }
        var body = group(kids), tools = [clear, remove].concat(n.itemSchema && Object.keys(n.itemSchema).length ? keyAdder(n) : []);
        if (!n.path.length || inPanel) return { complex: true, el: h("div", { class: "group" }, [tools.filter(Boolean).length ? h("div", { class: "row" }, tools.filter(Boolean)) : null, body]) };
        return { complex: true, el: fold(n, body, tools) };
      }
      // an array or a map: one panel per item, its tools in the panel's header
      var items = n.items.map(function (it, i) {
        var last = n.items.length - 1, isList = n.kind === "array";
        var tools = [isList && i > 0 ? btn("Up", function () { update(moveItem(doc, it.path, -1), true); }, "move up") : null,
          isList && i < last ? btn("Down", function () { update(moveItem(doc, it.path, +1), true); }, "move down") : null,
          btn("Remove", function () { update(removeAt(doc, it.path), true); })];
        var title = isList ? "#" + (i + 1) + (it.choice ? " · " + it.choice.tag : "") : String(it.key);
        var inner = draw1(it, true), complexItem = ["object", "union", "array", "map"].indexOf(it.kind) >= 0;
        return h("div", { class: "panel", "data-path": pathKey(it.path), "data-changed": String(changed(it.path)) }, [
          h("header", {}, [h("span", { class: "eyebrow", text: title })].concat(tools)),
          h("div", { class: "body" }, [complexItem ? banner(errors[pathKey(it.path)]) : null, inner.el]),
        ]);
      });
      var addTools = n.kind === "array" ? [ro ? null : btn("Add", function () { update(addItem(doc, n.path, n.itemSchema, root), true); })] : keyAdder(n);
      return { complex: true, el: fold(n, h("div", { class: "group" }, items), addTools.concat([clear, remove])) };
    }
    function draw() {
      while (host.firstChild) host.removeChild(host.firstChild);
      var top = banner(errors[""]);
      if (top) host.appendChild(top);
      host.appendChild(draw1(tree(schema, doc)).el);
    }
    draw();
    return {
      value: function () { return clone(doc); },
      invalid: function () { return Object.keys(bad); },
      setErrors: function (errs) { errors = placeErrors(tree(schema, doc), errs); draw(); return errors; },
      reset: function () { doc = clone(original); errors = {}; bad = {}; draw(); },
    };
  }

  api.TIERS = TIERS;
  api.equal = equal;
  api.strictest = strictest;
  api.deref = deref;
  api.unwrap = unwrap;
  api.variants = variants;
  api.kindOf = kindOf;
  api.tierIn = tierIn;
  api.defaultFor = defaultFor;
  api.tree = tree;
  api.getAt = getAt;
  api.setAt = setAt;
  api.removeAt = removeAt;
  api.addItem = addItem;
  api.moveItem = moveItem;
  api.choose = choose;
  api.parse = parse;
  api.readBack = readBack;
  api.diff = diff;
  api.placeErrors = placeErrors;
  api.mount = mount;
})(typeof module === "object" && module.exports ? module.exports : (window.GeecsForm = {}));
