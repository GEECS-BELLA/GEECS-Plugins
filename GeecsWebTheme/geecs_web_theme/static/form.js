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
 *   is skipped); the document's own errors show in a banner at the top, and every fold holding one opens.
 * - A tuple (`prefixItems`) is one field, its members side by side; a list or map item that is one value is a
 *   field with its tools, not a panel. A list item that is a union folds to its header (its choice, then a
 *   one-line gist), closed until opened; a list of unions adds by choosing one ("Add step…"). What is open
 *   survives a redraw and moves with a reordered item. `minItems` / `maxItems` disable Remove / hide the adder.
 * - A union whose tag it does not know is kept as written, marked "(unknown)", its keys drawn as rows.
 * - Layout hints, read wherever they appear, never needed: `x-sections` on the root ([{title, keys, help, open,
 *   gist, note, advanced: {title, keys, help}}]: closed folds titled with a gist of their keys; `note` a line with
 *   "{a.b|c}" filled from the document); `x-ndim-from` on the root ({path, values, names}: the document's frame
 *   dimensionality, read off a field); `x-ndim` on a variant (the dimensionalities it fits: the others are not
 *   offered, a variant fitting one is named with it); `x-docs` (a reference link) and `x-scalars` (name: meaning)
 *   on a variant, drawn under its choice; `x-axes` on a list ({"<ndim>": [names]}: one named row per axis while
 *   it holds none or one per axis); `x-suggest` on a word (offered as a datalist); `x-form-note` on the root (a
 *   banner: why the form is plainer than it can be).
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
    if (t === "array") return s.prefixItems ? "tuple" : "array";
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
    if (k === "tuple") return s.prefixItems.map(function (p) { return defaultFor(p, root); });
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
      var vs = variants(sc, ctx.root), chosen = pick(vs, value, ctx.root), tag = vs.prop && isObj(value) ? value[vs.prop] : undefined;
      n.variants = vs.list;
      if (vs.prop && tag !== undefined && !vs.list.some(function (v) { return v.tag === tag; })) {
        // a tag the union does not know (a typo in the file) is kept as written, its keys drawn as rows, never
        // silently drawn as the first variant
        n.unknown = true;
        n.choice = { prop: vs.prop, tags: vs.list.map(function (v) { return v.tag; }), tag: tag };
        n.children = fields({}, value, path, ctx, vs.prop);
        n.itemSchema = {};
        return n;
      }
      n.variant = chosen.schema;
      n.choice = { prop: vs.prop, tags: vs.list.map(function (v) { return v.tag; }), tag: chosen.tag };
      n.children = fields(chosen.schema, value, path, ctx, vs.prop);
      n.itemSchema = extraSchema(chosen.schema);
    } else if (kind === "object") {
      n.children = fields(sc, value, path, ctx, null);
      n.itemSchema = extraSchema(sc);
    } else if (kind === "array") {
      n.itemSchema = sc.items || {};
      n.items = (Array.isArray(value) ? value : []).map(function (v, i) { return node(n.itemSchema, v, path.concat([i]), { root: ctx.root }); });
    } else if (kind === "tuple") {
      n.items = sc.prefixItems.map(function (p, i) { return node(p, Array.isArray(value) ? value[i] : undefined, path.concat([i]), { root: ctx.root }); });
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
  // Switch a union to the variant `tag`: its defaults, its fixed values (a const, a one-value enum) set, and the old
  // value's fields the new variant declares kept, unless the new variant fixes them to something the old value fails:
  // the result fits the chosen variant, never a hybrid of the two.
  function choose(doc, path, unionSchema, tag, root) {
    var vs = variants(unwrap(unionSchema, root).schema, root), v = vs.list.filter(function (x) { return x.tag === tag; })[0];
    var next = defaultFor(v.schema, root), old = getAt(doc, path) || {}, props = v.schema.properties || {};
    Object.keys(props).forEach(function (k) {
      var p = deref(props[k], root);
      if (p.const !== undefined) next[k] = p.const;
      else if (p.enum && p.enum.length === 1) next[k] = p.enum[0];
      else if (old[k] !== undefined && k !== vs.prop && !(p.enum && p.enum.indexOf(old[k]) < 0)) next[k] = old[k];
    });
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
    // A whole subtree's change takes the strictest tier inside it: the schemas involved (for a union, the variants
    // the two sides hold, not every variant).
    function push(schemas, b, a, path, tier) {
      if (isObj(b) || Array.isArray(b) || isObj(a) || Array.isArray(a)) schemas.forEach(function (x) { tier = strictest(tier, tierIn(x, root)); });
      changes.push({ path: path, op: b === undefined ? "add" : a === undefined ? "remove" : "change", before: b, after: a, tier: tier });
    }
    (function walk(s, b, a, path, tier) {
      var u = unwrap(s, root), sc = u.schema;
      tier = strictest(tier, sc.tier || null);
      if (equal(b, a)) return;
      var vs = variants(sc, root);
      if (vs) {
        var vb = isObj(b) && pick(vs, b, root), va = isObj(a) && pick(vs, a, root);
        if (!vb || !va || vb.tag !== va.tag) return push([vb, va].filter(Boolean).map(function (v) { return v.schema; }), b, a, path, tier);
        sc = va.schema;
      }
      if (isObj(b) && isObj(a)) {
        var extra = extraSchema(sc) || {};
        var keys = Object.keys(b).concat(Object.keys(a).filter(function (k) { return !(k in b); }));
        keys.forEach(function (k) { walk((sc.properties || {})[k] || extra, b[k], a[k], path.concat([k]), tier); });
      } else if (Array.isArray(b) && Array.isArray(a)) {
        for (var i = 0; i < Math.max(b.length, a.length); i++) walk(sc.items || {}, b[i], a[i], path.concat([i]), tier);
      } else {
        push([sc], b, a, path, tier);
      }
    })(schema, before, after, [], null);
    return { changes: changes, tier: changes.reduce(function (t, c) { return strictest(t, c.tier); }, null) };
  }

  // Validation errors ({loc, msg}, pydantic's locations) by the drawn node they belong to: each location is walked
  // down the tree as far as it goes, a segment that is not a field of a chosen variant (its tag or class name) being
  // skipped, so every error lands at its nearest drawn node ("" is the document itself).
  // An error naming a variant the union does not draw goes to the union's own banner, and is dropped when the drawn
  // variant has errors of its own (pydantic reports every variant's failure; only the drawn one is the operator's).
  function placeErrors(t, errors) {
    var out = {}, other = {}, drawn = {};
    (errors || []).forEach(function (e) {
      var loc = Array.isArray(e.loc) ? e.loc : String(e.loc || "").split(".").filter(Boolean), n = t, elsewhere = false;
      for (var i = 0; i < loc.length; i++) {
        var kids = (n.children || []).concat(n.items || []);
        var next = kids.filter(function (c) { return String(c.key) === String(loc[i]); })[0];
        if (next) { n = next; continue; }
        if (!n.choice) break;
        if (String(loc[i]) !== String(n.choice.tag) && n.choice.tags.map(String).indexOf(String(loc[i])) >= 0) { elsewhere = true; break; }
        if (String(loc[i]) === String(n.choice.tag)) drawn[pathKey(n.path)] = true;
      }
      var key = pathKey(n.path), into = elsewhere ? other : out;
      into[key] = into[key] ? into[key] + "; " + e.msg : e.msg;
    });
    Object.keys(other).forEach(function (key) {
      var own = drawn[key] || Object.keys(out).some(function (k) { return k !== key && k.indexOf(key === "" ? "" : key + ".") === 0; });
      if (!own) out[key] = out[key] ? out[key] + "; " + other[key] : other[key];
    });
    return out;
  }

  // ---- one-line readings of a value, for a folded header: display only, nothing parses them back ----
  var COMPLEX = ["object", "union", "array", "map"];
  function clip(text, n) { return text.length > n ? text.slice(0, n - 1) + "…" : text; }
  function fmt(v) {
    if (v === null || v === undefined) return "·";
    if (Array.isArray(v)) return "[" + v.map(fmt).join(",") + "]";
    return isObj(v) ? "{…}" : String(v);
  }
  // The frame dimensionality a document's frames have, from the root's `x-ndim-from` ({path, values}), or null.
  function ndimOf(schema, doc) {
    var f = schema && schema["x-ndim-from"];
    if (!isObj(f) || !Array.isArray(f.path) || !isObj(f.values)) return null;
    var v = getAt(doc, f.path);
    return v !== undefined && v !== null && String(v) in f.values ? f.values[String(v)] : null;
  }
  // A list's fixed axis names for the document's frames (`x-axes`: {"<ndim>": [names]}), or null.
  function axesOf(n, ndim) {
    var a = n.schema && n.schema["x-axes"];
    return isObj(a) && ndim && Array.isArray(a[String(ndim)]) ? a[String(ndim)] : null;
  }
  // gist(node, ndim, noTag): a union reads as its tag and its set parameters, a list as its items (a list of choices
  // joined by arrows: the order is the point), a list with axis names (`x-axes`) as "y lo–hi x lo–hi".
  function gist(n, ndim, noTag) {
    if (!n || n.unset || n.value === undefined) return "";
    if (n.kind === "tuple") return fmt(n.value);
    if (n.kind === "array") {
      var names = axesOf(n, ndim);
      if (names && n.items.length === names.length) {
        return n.items.map(function (it, i) { return names[i] + " " + (Array.isArray(it.value) ? it.value.map(fmt).join("–") : fmt(it.value)); }).join(" ");
      }
      if (!n.items.length) return "none";
      var arrows = n.items.some(function (it) { return it.kind === "union"; });
      if (!n.items.some(function (it) { return COMPLEX.indexOf(it.kind) >= 0; })) return fmt(n.value);  // a list of values reads as written
      return n.items.map(function (it) { return COMPLEX.indexOf(it.kind) >= 0 ? gist(it, ndim) : fmt(it.value); }).join(arrows ? " → " : ", ");
    }
    if (n.kind === "map") return n.items.length ? n.items.map(function (it) { return String(it.key); }).join(", ") : "none";
    if (n.kind === "object" || n.kind === "union") {
      var parts = n.choice && !noTag ? [String(n.choice.tag)] : [];
      (n.children || []).forEach(function (c) {
        if (c.unset || c.value === undefined || c.value === null) return;
        var names = c.kind === "array" ? axesOf(c, ndim) : null;
        if (names && c.items.length === names.length) { parts.push(gist(c, ndim)); return; }
        parts.push(c.key + "=" + (COMPLEX.indexOf(c.kind) >= 0 || c.kind === "tuple" ? gist(c, ndim) : fmt(c.value)));
      });
      return parts.join(" ");
    }
    return fmt(n.value);
  }
  // A section's gist: its one list or choice read whole, or each set key of several ("defaults" when none is set);
  // `only` limits which keys a gist of several reads.
  function sectionGist(t, keys, ndim, only) {
    var kids = (t.children || []).filter(function (c) { return keys.indexOf(c.key) >= 0; });
    if (kids.length === 1) {
      var g = gist(kids[0], ndim);
      return g || (kids[0].kind === "array" || kids[0].kind === "map" ? "none" : "defaults");
    }
    var set = kids.filter(function (c) {
      return (!only || only.indexOf(c.key) >= 0) && !c.unset && c.value !== undefined && c.value !== null && c.value !== "" && !(isObj(c.value) && !Object.keys(c.value).length);
    });
    return set.length ? set.map(function (c) {
      return COMPLEX.indexOf(c.kind) >= 0 ? c.key + ": " + (gist(c, ndim) || "defaults") : c.key + "=" + fmt(c.value);
    }).join(" · ") : "defaults";
  }
  // A section's note: "{a.b|c}" is the first of those paths set in the document, else nothing.
  function note(template, doc) {
    return String(template).replace(/\{([^}]*)\}/g, function (_, alts) {
      var hit = alts.split("|").map(function (p) { return getAt(doc, p.split(".")); }).filter(function (v) { return v !== undefined && v !== null && v !== ""; })[0];
      return hit === undefined ? "" : String(hit);
    });
  }
  function find(t, path) {
    return path.reduce(function (n, k) { return n && (n.children || []).concat(n.items || []).filter(function (c) { return String(c.key) === String(k); })[0]; }, t);
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
  function banner(msg, state) { return msg ? h("div", { class: "banner", "data-state": state || "error", text: msg }) : null; }
  function plain(text) { return String(text || "").replace(/``([^`]*)``/g, "$1").replace(/\s*\n\s*/g, " ").trim(); }
  // A chosen variant's own help: its description, its card in a published reference (`x-docs`), and the scalars
  // it can write (`x-scalars`, name: meaning) folded on rung 0.
  function variantHelp(v) {
    if (!v) return [];
    var kids = [], text = plain(v.description), docs = v["x-docs"], scalars = isObj(v["x-scalars"]) ? Object.keys(v["x-scalars"]) : [];
    if (text || docs) kids.push(h("p", { class: "hint" }, [text ? document.createTextNode(text + " ") : null, docs ? h("a", { href: docs, target: "_blank", rel: "noopener", text: "reference ↗" }) : null]));
    if (scalars.length) {
      kids.push(h("details", { class: "disc" }, [h("summary", { text: "Can write " + scalars.length + " scalar" + (scalars.length === 1 ? "" : "s") }),
        h("div", { class: "dbody" }, scalars.map(function (k) { return h("p", {}, [h("code", { text: k }), document.createTextNode(" " + v["x-scalars"][k])]); }))]));
    }
    return kids;
  }

  function mount(host, opts) {
    var schema = opts.schema, root = schema, original = clone(opts.original !== undefined ? opts.original : opts.value);
    var doc = clone(opts.value), errors = {}, bad = {}, ro = !!opts.readOnly;
    // Laid out as titled sections (`x-sections` on the root: [{title, keys, help, open, gist, note, advanced: {title, keys,
    // help}}]), folded closed unless `open`; every key no section names follows them.
    var sections = Array.isArray(schema["x-sections"]) ? schema["x-sections"] : null;
    var ndimNames = isObj(schema["x-ndim-from"]) && isObj(schema["x-ndim-from"].names) ? schema["x-ndim-from"].names : null;
    // What is open survives a redraw (a fold the reader closed stays closed); a fold holding an error always opens.
    var opened = {}, gists = [], ndim = null;
    function isOpen(key, dflt) { return key in opened ? opened[key] : dflt; }
    function errUnder(key) { return Object.keys(errors).concat(Object.keys(bad)).some(function (k) { return k === key || k.indexOf(key + ".") === 0; }); }
    function changed(path) { return !equal(getAt(original, path), getAt(doc, path)); }
    function refresh() { var t = tree(schema, doc); gists.forEach(function (g) { g.el.textContent = clip(g.read(t), 140); }); }
    // A structural edit at `at` (a path) redraws, and forgets the unparsed text under it: those fields moved or went.
    function update(next, at) {
      doc = next;
      if (at) {
        var p = pathKey(at);
        Object.keys(bad).forEach(function (k) { if (p === "" || k === p || k.indexOf(p + ".") === 0) delete bad[k]; });
        draw();
      } else refresh();
      if (opts.onChange) opts.onChange(clone(doc));
    }
    // A list's items move or go: what was open moves with them (to(i) is item i's new index, null when it went).
    function reopen(listPath, to) {
      var lp = pathKey(listPath) + ".", moved = {};
      Object.keys(opened).forEach(function (k) {
        if (k.indexOf(lp) !== 0) return;
        var rest = k.slice(lp.length).split("."), i = Number(rest[0]);
        if (rest[0] === "" || isNaN(i)) return;
        var j = to(i);
        if (j !== null) moved[lp + [j].concat(rest.slice(1)).join(".")] = opened[k];
        delete opened[k];
      });
      Object.assign(opened, moved);
    }
    function btn(label, onclick, title) { return ro ? null : h("button", { type: "button", class: "btn sm ghost", title: title, onclick: onclick, text: label }); }
    function fits(v) { var d = v["x-ndim"]; return !ndim || !Array.isArray(d) || d.indexOf(ndim) >= 0; }
    function hint(v) { var d = v["x-ndim"]; return ndimNames && Array.isArray(d) && d.length === 1 && ndimNames[String(d[0])] ? " (" + ndimNames[String(d[0])] + ")" : ""; }
    function withGist(cls, read) { var g = h("span", { class: cls }); gists.push({ el: g, read: read }); return g; }

    // edit(doc, path, value): how this field's value goes into the document (setAt, unless a fixed row or a tuple's
    // member says otherwise); a value the document already holds changes nothing (no edit, no empty parent made)
    function control(n, id, err, edit) {
      var dis = ro || n.kind === "const";
      var onIn = function (ev) {
        var c = ev.target, key = pathKey(n.path), next;
        try { next = readBack(n, c.value, getAt(original, n.path)); delete bad[key]; } catch (e) { bad[key] = { text: c.value, msg: "not JSON: " + e.message }; }
        if (bad[key]) { c.setAttribute("aria-invalid", "true"); err.textContent = bad[key].msg; return; }
        c.setAttribute("aria-invalid", "false"); err.textContent = "";
        if (!equal(getAt(doc, n.path), next)) update((edit || setAt)(doc, n.path, next));
        if (c.parentNode && c.parentNode.setAttribute) c.parentNode.setAttribute("data-changed", String(changed(n.path)));
      };
      if (n.kind === "enum" || n.kind === "boolean") {
        var choices = n.kind === "boolean" ? ["true", "false"] : n.options.map(String);
        if (!n.required || n.value === undefined) choices = [""].concat(choices);
        return h("select", { id: id, disabled: dis, onchange: onIn }, choices.map(function (o) { return h("option", { value: o, selected: String(n.value ?? "") === o, text: o === "" ? unsetText(n) : o }); }));
      }
      // unparsed text survives a redraw, so the field still shows what its error is about
      var text = bad[pathKey(n.path)] ? bad[pathKey(n.path)].text : n.kind === "const" ? String(n.constValue) : display(n.kind, n.value);
      if (n.kind === "json" || (n.kind === "string" && (text.length > 60 || /description|expression/.test(String(n.key))))) {
        return h("textarea", { id: id, rows: n.kind === "json" ? Math.min(8, text.split("\n").length + 1) : 2, spellcheck: "false", disabled: dis, oninput: onIn, text: text });
      }
      var input = h("input", { id: id, type: "text", inputmode: n.kind === "integer" || n.kind === "number" ? "decimal" : undefined, placeholder: n.value === undefined ? unsetText(n) : undefined, disabled: dis, oninput: onIn });
      input.value = text;
      return input;
    }
    // A field; `label` overrides its name (a list item's number, an axis); a string with `x-suggest` offers those words.
    function scalar(n, tools, label, edit) {
      var id = "gf-" + (++ids), key = pathKey(n.path), msg = (bad[key] && bad[key].msg) || errors[key], err = h("span", { class: "err", text: msg || "" }), c = control(n, id, err, edit);
      if (msg) c.setAttribute("aria-invalid", "true");
      var words = null;
      if (Array.isArray(n.schema["x-suggest"]) && c.tagName === "INPUT") {
        c.setAttribute("list", id + "-words");
        words = h("datalist", { id: id + "-words" }, n.schema["x-suggest"].map(function (w) { return h("option", { value: w }); }));
      }
      tools = (tools || []).filter(Boolean);
      return h("div", { class: "field", "data-path": key, "data-changed": String(changed(n.path)) }, [
        h("label", { for: id }, [document.createTextNode(label || n.label), n.required ? h("span", { class: "req", text: "*" }) : null, tierMark(n)]),
        c, words, err, n.help ? h("span", { class: "hint", text: n.help }) : null, tools.length ? h("div", { class: "row" }, tools) : null,
      ]);
    }
    // A fixed-length tuple (a low / high pair): its members side by side in one field. A member's edit fills the
    // tuple to its length first (an emptied member is null, never a shorter tuple); a tuple of its own (not a list's
    // item) emptied of every member goes back to its stored state. prep(doc): the document made ready first (a
    // fixed-row list padded to its rows).
    function tupleField(n, tools, label, prep) {
      var key = pathKey(n.path), first = null, msgs = [], err = h("span", { class: "err" }), size = n.items.length;
      if (errors[key]) msgs.push(errors[key]);
      var edit = function (d, p, v) {
        d = prep ? prep(d) : d;
        var t = getAt(d, n.path), full = Array.isArray(t) ? t.slice() : [];
        while (full.length < size) full.push(null);  // a member not typed yet is open, never a made-up number
        full[p[p.length - 1]] = v === undefined ? null : v;
        if (typeof n.key !== "number" && full.every(function (x) { return x === null || x === undefined; })) {
          var stored = getAt(original, n.path);
          if (!Array.isArray(stored)) return stored === undefined ? removeAt(d, n.path) : setAt(d, n.path, stored);
        }
        return setAt(d, n.path, full);
      };
      var inputs = n.items.map(function (m, j) {
        var id = "gf-" + (++ids), mk = pathKey(m.path), msg = (bad[mk] && bad[mk].msg) || errors[mk], c = control(m, id, err, edit);
        first = first || id;
        c.setAttribute("aria-label", (label || n.label) + " " + (j + 1));
        if (msg) { msgs.push(msg); c.setAttribute("aria-invalid", "true"); }
        return c;
      });
      err.textContent = msgs.join("; ");
      tools = (tools || []).filter(Boolean);
      return h("div", { class: "field", "data-path": key, "data-changed": String(changed(n.path)) }, [
        h("label", { for: first }, [document.createTextNode(label || n.label), n.required ? h("span", { class: "req", text: "*" }) : null, tierMark(n)]),
        h("div", { class: "pair" }, inputs), err, n.help ? h("span", { class: "hint", text: n.help }) : null,
        tools.length ? h("div", { class: "row" }, tools) : null,
      ]);
    }
    function fold(n, body, tools) {
      var key = pathKey(n.path), d = h("details", { class: "fold", open: isOpen(key, !(n.items && !n.items.length && n.kind !== "tuple")) || errUnder(key), "data-path": key, "data-changed": String(changed(n.path)) }, [
        h("summary", { class: "eyebrow" }, [document.createTextNode(n.label + (n.items && n.kind !== "tuple" ? " (" + n.items.length + ")" : "")), tierMark(n)]),
        n.help ? h("p", { class: "hint dim", text: n.help }) : null, banner(errors[key]),
        tools && tools.filter(Boolean).length ? h("div", { class: "row" }, tools.filter(Boolean)) : null, body,
      ]);
      d.addEventListener("toggle", function () { opened[key] = !!d.open; });
      return d;
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
        update(next, n.path);
      })];
    }
    // A union's choice: the variants that fit the document's frames (`x-ndim`; the chosen one always), each named
    // with the frames it is for when it fits one kind only; a tag the union does not know is kept, marked unknown.
    function chooser(n, id, aria) {
      var options = n.variants.filter(function (v) { return v.tag === n.choice.tag || fits(v.schema); }).map(function (v) {
        return h("option", { value: v.tag, selected: v.tag === n.choice.tag, title: plain(v.schema.description) || undefined, text: v.tag + hint(v.schema) });
      });
      if (n.unknown) options.push(h("option", { value: n.choice.tag, selected: true, text: n.choice.tag + " (unknown)" }));
      return h("select", { id: id, class: aria ? "btn sm" : undefined, disabled: ro, "aria-label": aria, onchange: function (ev) { update(choose(doc, n.path, n.schema, ev.target.value, root), n.path); } }, options);
    }
    function unknownBanner(n) { return n.unknown ? banner("Unknown " + (n.choice.prop || "variant") + " \"" + n.choice.tag + "\": pick a registered one (its keys are kept below as written).") : null; }
    // mode: undefined (a fold of its own), "bare" (its contents only: a section holds it), "item" (a list item's
    // panel: a union's choice goes to the panel's header, returned as `chooser`).
    function draw1(n, mode) {
      var complex = COMPLEX.indexOf(n.kind) >= 0;
      var remove = n.extra ? btn("Remove", function () { update(removeAt(doc, n.path), n.path); }) : null;
      if (n.kind === "tuple") return { el: tupleField(n, [remove]) };
      if (!complex) return { el: scalar(n, [remove]) };
      // an absent list or map that is not nullable is drawn empty, its adder ready (adding writes it)
      if (n.unset && !n.nullable && n.value === undefined && (n.kind === "array" || n.kind === "map")) {
        return { complex: true, el: drawList(Object.assign({}, n, { unset: false, items: [], itemSchema: n.kind === "array" ? n.schema.items || {} : extraSchema(n.schema) }), mode, [remove]) };
      }
      // an unset sub-document is one line, its Set beside its name
      if (n.unset) {
        var set = btn("Set", function () { update(setAt(doc, n.path, defaultFor(Object.assign({}, n.schema, { default: undefined }), root)), n.path); });
        return { el: h("div", { class: "field", "data-path": pathKey(n.path), "data-changed": String(changed(n.path)) }, [
          h("label", {}, [document.createTextNode(n.label), tierMark(n)]), h("div", { class: "row" }, [set || h("span", { class: "dim", text: "(not set)" }), remove]),
          banner(errors[pathKey(n.path)]), n.help ? h("span", { class: "hint", text: n.help }) : null]) };
      }
      // a list's or a map's item goes by its Remove, never a Clear; a section's own field keeps its default
      var item = mode === "item" || mode === "bare" || typeof n.key === "number";
      var clear = !item && n.value !== undefined && (n.nullable || (!n.required && !n.extra && n.path.length)) ? btn("Clear", function () { update(n.nullable ? setAt(doc, n.path, null) : removeAt(doc, n.path), n.path); }) : null;
      if (n.kind === "object" || n.kind === "union") {
        if (!n.path.length && sections) return { complex: true, el: drawSections(n) };
        var kids = n.children.map(function (c) { return draw1(c); }), sel = null, help = [];
        if (n.choice) {
          var cid = "gf-" + (++ids);
          sel = chooser(n, cid, mode === "item" ? humanize(n.choice.prop || "variant") : undefined);
          help = [unknownBanner(n)].concat(variantHelp(n.variant)).filter(Boolean);
          if (mode !== "item") kids.unshift({ el: h("div", { class: "field" }, [h("label", { for: cid, text: n.choice.prop ? humanize(n.choice.prop) : "Variant" }), sel]) });
        }
        var tools = [clear, remove].concat(n.itemSchema && Object.keys(n.itemSchema).length ? keyAdder(n) : []).filter(Boolean);
        var toolRow = tools.length ? h("div", { class: "row" }, tools) : null;
        if (mode === "item") return { complex: true, chooser: sel, el: h("div", { class: "group" }, help.concat([toolRow, group(kids)])) };
        // a choice's own help sits under the choice
        var body = h("div", { class: "group" }, [n.choice ? group(kids.slice(0, 1)) : null].concat(n.choice ? help : [], [group(n.choice ? kids.slice(1) : kids)]));
        if (!n.path.length || mode === "bare") return { complex: true, el: h("div", { class: "group" }, [mode === "bare" ? banner(errors[pathKey(n.path)]) : null, toolRow, body]) };
        return { complex: true, el: fold(n, body, tools) };
      }
      return { complex: true, el: drawList(n, mode, [clear, remove]) };
    }
    // An array or a map: one panel per item, its tools in the panel's header; a union item folds to its gist, its
    // choice in the header. An item that is one value (a number, a word, a pair) is a field, not a panel. A list
    // with axis names for the document's frames (`x-axes`) is one fixed row per axis while it holds none or one
    // per axis.
    function drawList(n, mode, outer) {
      var isList = n.kind === "array", key = pathKey(n.path), names = isList ? axesOf(n, ndim) : null, len = (n.items || []).length;
      var rows, addTools = [];
      if (names && (len === 0 || len === names.length)) {
        var pad = function (d) {
          var list = getAt(d, n.path), full = Array.isArray(list) ? list.slice() : [];
          if (full.length >= names.length) return d;
          while (full.length < names.length) full.push(defaultFor(n.itemSchema, root));
          return setAt(d, n.path, full);
        };
        rows = names.map(function (nm, i) {
          var it = n.items[i] || node(n.itemSchema, undefined, n.path.concat([i]), { root: root });
          return it.kind === "tuple" ? tupleField(it, [], nm, pad) : scalar(it, [], nm, function (d, p, v) { return setAt(pad(d), p, v); });
        });
      } else {
        var min = n.schema.minItems, max = n.schema.maxItems;
        rows = n.items.map(function (it, i) {
          var last = n.items.length - 1, ik = pathKey(it.path);
          var tools = [isList && i > 0 ? btn("Up", function () { reopen(n.path, function (j) { return j === i ? i - 1 : j === i - 1 ? i : j; }); update(moveItem(doc, it.path, -1), n.path); }, "move up") : null,
            isList && i < last ? btn("Down", function () { reopen(n.path, function (j) { return j === i ? i + 1 : j === i + 1 ? i : j; }); update(moveItem(doc, it.path, +1), n.path); }, "move down") : null,
            btn("Remove", function () { if (isList) reopen(n.path, function (j) { return j === i ? null : j > i ? j - 1 : j; }); update(removeAt(doc, it.path), n.path); })];
          if (isList && min !== undefined && n.items.length <= min && tools[2]) tools[2].setAttribute("disabled", "");
          var title = isList ? "#" + (i + 1) : String(it.key);
          if (COMPLEX.indexOf(it.kind) < 0) return it.kind === "tuple" ? tupleField(it, tools, title) : scalar(it, tools, title);
          var foldable = it.kind === "union" && !it.unset, open = !foldable || isOpen(ik, false) || errUnder(ik);
          var inner = draw1(it, "item"), body = h("div", { class: "body", hidden: !open }, [banner(errors[ik]), inner.el]), head;
          if (foldable) {
            var toggle = h("button", { type: "button", class: "btn sm ghost", "aria-expanded": String(open), title: "show or hide its fields", text: open ? "▾" : "▸" });
            var shut = !open, flip = function () {
              shut = !shut; opened[ik] = !shut;
              if (shut) body.setAttribute("hidden", ""); else body.removeAttribute("hidden");
              toggle.textContent = shut ? "▸" : "▾"; toggle.setAttribute("aria-expanded", String(!shut));
            };
            toggle.addEventListener("click", flip);
            var g = withGist("gist", function (t) { return gist(find(t, it.path), ndim, true); });
            g.addEventListener("click", flip);
            head = [toggle, h("span", { class: "eyebrow", text: title }), inner.chooser, g];
          } else head = [h("span", { class: "eyebrow", text: title + (it.choice ? " · " + it.choice.tag : "") }), inner.chooser || null];
          return h("div", { class: "panel", "data-path": ik, "data-changed": String(changed(it.path)) }, [h("header", { class: foldable ? "item" : undefined }, head.concat(tools)), body]);
        });
        if (isList && !ro && !(max !== undefined && len >= max)) {
          var vs = variants(unwrap(n.itemSchema, root).schema, root);
          if (vs) {
            // a list of choices adds by choosing one
            var noun = String(n.key).replace(/ies$/, "y").replace(/s$/, "");
            var add = h("select", { class: "btn sm", "aria-label": "add a " + noun, onchange: function (ev) {
              var v = vs.list.filter(function (x) { return String(x.tag) === ev.target.value; })[0];
              if (!v) return;
              var next = addItem(doc, n.path, n.itemSchema, root), at = n.path.concat([(getAt(next, n.path) || []).length - 1]);
              opened[pathKey(at)] = true;
              update(choose(next, at, n.itemSchema, v.tag, root), n.path);
            } }, [h("option", { value: "", selected: true, text: "Add " + noun + "…" })].concat(vs.list.filter(function (v) { return fits(v.schema); }).map(function (v) {
              return h("option", { value: v.tag, title: plain(v.schema.description) || undefined, text: v.tag + hint(v.schema) });
            })));
            addTools = [add];
          } else addTools = [btn("Add", function () { opened[pathKey(n.path.concat([len]))] = true; update(addItem(doc, n.path, n.itemSchema, root), n.path); })];
        }
        if (!isList) addTools = keyAdder(n);
      }
      var tools = addTools.concat(outer).filter(Boolean);
      if (mode === "bare") return h("div", { class: "group" }, [banner(errors[key]), h("div", { class: "group" }, rows), tools.length ? h("div", { class: "row" }, tools) : null]);
      return fold(n, h("div", { class: "group" }, rows), tools);
    }
    // The root laid out as sections: each a fold titled with a gist of what is in it, closed until asked for.
    function drawSections(n) {
      var byKey = {}, used = {}, out = [];
      n.children.forEach(function (c) { byKey[c.key] = c; });
      sections.forEach(function (sec) {
        var keys = (sec.keys || []).filter(function (k) { return k in byKey && !used[k]; });
        if (!keys.length) return;
        keys.forEach(function (k) { used[k] = true; });
        var adv = sec.advanced ? (sec.advanced.keys || []).filter(function (k) { return keys.indexOf(k) >= 0; }) : [];
        var key = "§" + sec.title, one = keys.length === 1 ? byKey[keys[0]] : null;
        var body = [sec.help ? h("p", { class: "hint dim", text: sec.help }) : null];
        if (one && COMPLEX.indexOf(one.kind) >= 0 && !one.unset) body.push(draw1(one, "bare").el);
        else {
          body.push(group(keys.filter(function (k) { return adv.indexOf(k) < 0; }).map(function (k) { return draw1(byKey[k]); })));
          if (sec.note) body.push(withGist("dim mono", function () { return note(sec.note, doc); }));
          if (adv.length) {
            var ak = key + ".advanced", set = adv.some(function (k) { var v = byKey[k].value; return v !== undefined && v !== null && v !== ""; });
            var ad = h("details", { class: "fold", open: isOpen(ak, set) || adv.some(errUnder) }, [h("summary", { class: "eyebrow", text: sec.advanced.title }),
              sec.advanced.help ? h("p", { class: "hint dim", text: sec.advanced.help }) : null, group(adv.map(function (k) { return draw1(byKey[k]); }))]);
            ad.addEventListener("toggle", function () { opened[ak] = !!ad.open; });
            body.push(ad);
          }
        }
        var d = h("details", { class: "fold section", "data-section": sec.title, open: isOpen(key, !!sec.open) || keys.some(errUnder) }, [
          h("summary", {}, [h("span", { class: "eyebrow", text: sec.title }), withGist("gist", function (t) { return sectionGist(t, keys, ndim, Array.isArray(sec.gist) ? sec.gist : null); })]),
        ].concat(body));
        d.addEventListener("toggle", function () { opened[key] = !!d.open; });
        out.push(d);
      });
      var rest = n.children.filter(function (c) { return !used[c.key]; }).map(function (c) { return draw1(c); });
      if (rest.length) out.push(group(rest));
      return h("div", { class: "group" }, out);
    }
    function draw() {
      while (host.firstChild) host.removeChild(host.firstChild);
      gists = []; ndim = ndimOf(schema, doc);
      [banner(schema["x-form-note"], "stale"), banner(errors[""])].forEach(function (b) { if (b) host.appendChild(b); });
      host.appendChild(draw1(tree(schema, doc)).el);
      refresh();
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
  api.gist = gist;
  api.sectionGist = sectionGist;
  api.ndimOf = ndimOf;
  api.note = note;
  api.mount = mount;
})(typeof module === "object" && module.exports ? module.exports : (window.GeecsForm = {}));
