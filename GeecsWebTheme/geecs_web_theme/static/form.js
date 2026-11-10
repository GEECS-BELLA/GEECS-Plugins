/* GEECS web theme — the one schema-driven form every GEECS editor draws with.
 *
 * A JSON Schema (pydantic's model_json_schema(), with GEECS-Schemas' `tier`
 * annotations) in, kit controls out, the edited document and its diff back:
 *
 *   var form = GeecsForm.mount(host, { schema, value, onChange: function (doc) { … } });
 *   form.value();                      // the document as edited
 *   form.setErrors(errors);            // [{loc, msg}] from the service's validate, drawn at the fields
 *   GeecsForm.diff(stored, form.value(), schema)   // {changes: [{path, op, before, after, tier}], tier}
 *
 * The rules, each pinned in tests/test_form_js.py under node:
 * - $ref / $defs resolve, allOf merges; `anyOf [X, null]` is an optional X.
 * - A union of objects is drawn as a choice of its discriminator (`discriminator.propertyName`, else the one
 *   property every variant fixes with `const`), then the chosen variant's fields.
 * - Objects are folds, arrays and maps are item panels with add / remove / reorder, enums are selects,
 *   booleans a true/false select; a field's description is its help, `required` marks it, `schema_version`
 *   is never drawn (tools own it).
 * - A field stricter than its document carries `tier` in its property schema; the form marks it. A field's
 *   tier is the strictest declared on its path, the document's own at the root.
 * - "Changed" is any difference at a field's path, a reordered list included (a reorder changes every
 *   index it moved); the diff's tier is the strictest tier among its changes, which decides what a save does.
 *
 * window.GeecsForm in a page, module.exports under node. No dependency, no colour: kit classes only.
 */
(function (api) {
  "use strict";
  var TIERS = ["operator", "maintainer", "parameter", "sudo", "code"];
  var HIDDEN = ["schema_version"];

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
  function strictest(a, b) { return TIERS.indexOf(a) >= TIERS.indexOf(b) ? (a || b || null) : b; }
  function humanize(k) { var s = String(k).replace(/_/g, " "); return s.charAt(0).toUpperCase() + s.slice(1); }
  function pathKey(path) { return path.join("."); }

  // ---- schema reading ----
  // $ref (against the root's $defs) and allOf, the referring node's own keys (description, default, tier) winning.
  function deref(s, root) {
    if (!isObj(s)) return {};
    var out = s;
    if (s.$ref) {
      var name = s.$ref.split("/").pop(), defs = root.$defs || root.definitions || {};
      out = Object.assign({}, deref(defs[name] || {}, root), s);
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
  // A union of objects: {prop, list: [{tag, schema}]}, or null when the members are not all objects.
  function variants(s, root) {
    var list = s.oneOf || s.anyOf;
    if (!list) return null;
    var vs = list.map(function (x) { return deref(x, root); });
    if (!vs.every(function (v) { return v.type === "object" || v.properties; })) return null;
    var prop = s.discriminator && s.discriminator.propertyName;
    if (!prop) {
      var keys = Object.keys(vs[0].properties || {});
      prop = keys.filter(function (k) { return vs.every(function (v) { return v.properties[k] && v.properties[k].const !== undefined; }); })[0];
    }
    if (!prop) return null;
    return { prop: prop, list: vs.map(function (v) { return { tag: v.properties[prop].const, schema: v }; }) };
  }
  function pick(vs, value) {
    var tag = isObj(value) ? value[vs.prop] : undefined;
    return vs.list.filter(function (v) { return v.tag === tag; })[0] || vs.list[0];
  }
  // What a schema is drawn as.
  function kindOf(s, root) {
    if (s.const !== undefined) return "const";
    if (s.enum) return "enum";
    if (variants(s, root)) return "union";
    var t = Array.isArray(s.type) ? null : s.type;
    if (t === "boolean" || t === "integer" || t === "number" || t === "string") return t;
    if (t === "array") return "array";
    if (t === "object" || s.properties || s.additionalProperties) return s.properties && Object.keys(s.properties).length ? "object" : (isObj(s.additionalProperties) ? "map" : "any");
    return "scalar";  // a union of scalars (a number or a word), or an untyped value
  }
  // The value a new item, a newly set optional or a variant switch starts from.
  function defaultFor(s, root) {
    var u = unwrap(s, root); s = u.schema;
    if (s.default !== undefined) return clone(s.default);
    if (u.nullable) return null;
    if (s.const !== undefined) return s.const;
    var vs = variants(s, root);
    if (vs) return defaultFor(vs.list[0].schema, root);
    var k = kindOf(s, root);
    if (k === "object") {
      var o = {};
      Object.keys(s.properties).forEach(function (key) {
        var p = deref(s.properties[key], root);
        if ((s.required || []).indexOf(key) >= 0 || p.const !== undefined) o[key] = defaultFor(p, root);
      });
      return o;
    }
    return { enum: s.enum && s.enum[0], array: [], map: {}, boolean: false, integer: 0, number: 0, string: "" }[k] ?? null;
  }

  // ---- the tree a form is drawn from ----
  // node: {path, key, label, help, required, tier (declared here), kind, value, nullable, options, choice, children, items}
  function node(s, value, path, ctx) {
    var u = unwrap(s, ctx.root), sc = u.schema, kind = kindOf(sc, ctx.root);
    var n = {
      path: path, key: path[path.length - 1],
      label: typeof path[path.length - 1] === "number" ? "#" + (path[path.length - 1] + 1) : sc.title || humanize(path.length ? path[path.length - 1] : "document"),
      help: sc.description || "", required: !!ctx.required, tier: sc.tier || null, kind: kind, value: value,
      nullable: u.nullable, schema: sc,
    };
    if (kind === "enum") n.options = sc.enum;
    if (kind === "const") n.constValue = sc.const;
    if (value === null || value === undefined) { if (kind === "object" || kind === "union" || kind === "array" || kind === "map") n.unset = true; }
    if (n.unset) return n;
    if (kind === "union") {
      var vs = variants(sc, ctx.root), chosen = pick(vs, value);
      n.choice = { prop: vs.prop, tags: vs.list.map(function (v) { return v.tag; }), tag: chosen.tag };
      n.children = fields(chosen.schema, value, path, ctx, vs.prop);
    } else if (kind === "object") {
      n.children = fields(sc, value, path, ctx, null);
    } else if (kind === "array") {
      n.itemSchema = sc.items || {};
      n.items = (Array.isArray(value) ? value : []).map(function (v, i) { return node(n.itemSchema, v, path.concat([i]), { root: ctx.root }); });
    } else if (kind === "map") {
      n.itemSchema = sc.additionalProperties;
      n.items = Object.keys(isObj(value) ? value : {}).map(function (k) { return node(n.itemSchema, value[k], path.concat([k]), { root: ctx.root }); });
    }
    return n;
  }
  function fields(s, value, path, ctx, skip) {
    var props = s.properties || {}, obj = isObj(value) ? value : {};
    return Object.keys(props).filter(function (k) { return k !== skip && HIDDEN.indexOf(k) < 0; }).map(function (k) {
      return node(props[k], obj[k], path.concat([k]), { root: ctx.root, required: (s.required || []).indexOf(k) >= 0 });
    });
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
  function addItem(doc, path, itemSchema, root, key) {
    var list = getAt(doc, path);
    if (key !== undefined) return setAt(doc, path.concat([key]), defaultFor(itemSchema, root));
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
    Object.keys(v.schema.properties).forEach(function (k) { if (k !== vs.prop && old[k] !== undefined) next[k] = old[k]; });
    next[vs.prop] = tag;
    return setAt(doc, path, next);
  }
  // An input's text as the value its kind holds; an empty optional is absent.
  function parse(kind, text) {
    if (kind === "boolean") return text === "" ? undefined : text === "true";
    if (text === "") return kind === "string" ? "" : undefined;
    if (kind === "integer" || kind === "number" || kind === "scalar") {
      var n = Number(text);
      return isFinite(n) && /^\s*-?(\d|\.\d)/.test(text) ? n : text;  // a word stays a word; the service says if it is wrong
    }
    return text;
  }

  // ---- the diff ----
  // Every difference between two documents at a field's path, each with the strictest tier on that path.
  function diff(before, after, schema) {
    var root = schema, changes = [];
    (function walk(s, b, a, path, tier) {
      var u = unwrap(s, root), sc = u.schema;
      tier = strictest(tier, sc.tier || null);
      if (equal(b, a)) return;
      var vs = variants(sc, root);
      if (vs && isObj(b) && isObj(a) && b[vs.prop] === a[vs.prop]) sc = pick(vs, a).schema;
      else if (vs) { changes.push({ path: path, op: b === undefined ? "add" : a === undefined ? "remove" : "change", before: b, after: a, tier: tier }); return; }
      if (isObj(b) && isObj(a)) {
        var keys = Object.keys(b).concat(Object.keys(a).filter(function (k) { return !(k in b); }));
        keys.forEach(function (k) { walk((sc.properties || {})[k] || (isObj(sc.additionalProperties) ? sc.additionalProperties : {}), b[k], a[k], path.concat([k]), tier); });
      } else if (Array.isArray(b) && Array.isArray(a)) {
        for (var i = 0; i < Math.max(b.length, a.length); i++) walk(sc.items || {}, b[i], a[i], path.concat([i]), tier);
      } else {
        changes.push({ path: path, op: b === undefined ? "add" : a === undefined ? "remove" : "change", before: b, after: a, tier: tier });
      }
    })(schema, before, after, [], null);
    return { changes: changes, tier: changes.reduce(function (t, c) { return strictest(t, c.tier); }, null) };
  }

  // Validation errors ({loc, msg}, pydantic's locations) by the form's field paths: a discriminated union's
  // location carries the variant's tag (steps.0.set.value), which the form's path does not (steps.0.value).
  function placeErrors(t, errors) {
    var tags = [];
    (function walk(n) {
      if (n.choice) tags.push([pathKey(n.path.concat([n.choice.tag])), pathKey(n.path)]);
      (n.children || []).concat(n.items || []).forEach(walk);
    })(t);
    var out = {};
    (errors || []).forEach(function (e) {
      var key = Array.isArray(e.loc) ? pathKey(e.loc) : String(e.loc || "");
      tags.forEach(function (p) { if (key === p[0] || key.indexOf(p[0] + ".") === 0) key = p[1] + key.slice(p[0].length); });
      out[key.replace(/^\./, "")] = e.msg;
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
  function unsetText(n) { return n.schema.default !== undefined && n.schema.default !== null ? "(default: " + n.schema.default + ")" : "(not set)"; }
  function tierMark(n) { return n.tier ? h("span", { class: "tier", "data-tier": n.tier, title: "changing this needs the " + n.tier + " tier", text: n.tier }) : null; }

  function mount(host, opts) {
    var schema = opts.schema, root = schema, original = clone(opts.original !== undefined ? opts.original : opts.value);
    var doc = clone(opts.value), errors = {}, ro = !!opts.readOnly;
    function changed(path) { return !equal(getAt(original, path), getAt(doc, path)); }
    function update(next, redraw) { doc = next; if (redraw) draw(); if (opts.onChange) opts.onChange(clone(doc)); }
    function btn(label, onclick, title) { return ro ? null : h("button", { type: "button", class: "btn sm ghost", title: title, onclick: onclick, text: label }); }

    function control(n, id) {
      var dis = ro || n.kind === "const";
      var onIn = function (ev) {
        update(setAt(doc, n.path, parse(n.kind, ev.target.value)));
        var f = ev.target.parentNode; if (f && f.setAttribute) f.setAttribute("data-changed", String(changed(n.path)));
      };
      if (n.kind === "enum" || n.kind === "boolean") {
        var opts2 = n.kind === "boolean" ? ["true", "false"] : n.options.map(String);
        if (!n.required || n.value === undefined) opts2 = [""].concat(opts2);
        return h("select", { id: id, disabled: dis, onchange: onIn }, opts2.map(function (o) { return h("option", { value: o, selected: String(n.value ?? "") === o, text: o === "" ? unsetText(n) : o }); }));
      }
      var text = n.kind === "const" ? n.constValue : n.value;
      text = text === undefined || text === null ? "" : (typeof text === "object" ? JSON.stringify(text) : String(text));
      if (n.kind === "string" && (text.length > 60 || /description|expression/.test(String(n.key)))) return h("textarea", { id: id, rows: 2, disabled: dis, oninput: onIn, text: text });
      var input = h("input", { id: id, type: "text", inputmode: n.kind === "integer" || n.kind === "number" ? "decimal" : undefined, placeholder: n.value === undefined ? unsetText(n) : undefined, disabled: dis, oninput: onIn });
      input.value = text;
      return input;
    }
    function scalar(n) {
      var id = "gf-" + (++ids), key = pathKey(n.path), err = errors[key], c = control(n, id);
      if (err) c.setAttribute("aria-invalid", "true");
      return h("div", { class: "field", "data-path": key, "data-changed": String(changed(n.path)) }, [
        h("label", { for: id }, [document.createTextNode(n.label), n.required ? h("span", { class: "req", text: "*" }) : null, tierMark(n)]),
        c, h("span", { class: "err", text: err || "" }), n.help ? h("span", { class: "hint", text: n.help }) : null,
      ]);
    }
    function fold(n, body, tools) {
      var key = pathKey(n.path), err = errors[key];
      return h("details", { class: "fold", open: true, "data-path": key, "data-changed": String(changed(n.path)) }, [
        h("summary", { class: "eyebrow" }, [document.createTextNode(n.label + (n.items ? " (" + n.items.length + ")" : "")), tierMark(n)]),
        n.help ? h("p", { class: "hint dim", text: n.help }) : null, err ? h("div", { class: "banner", "data-state": "error", text: err }) : null,
        tools && tools.length ? h("div", { class: "row" }, tools) : null, body,
      ]);
    }
    function group(kids) {
      var simple = kids.filter(function (k) { return !k.complex; }), complex = kids.filter(function (k) { return k.complex; });
      return h("div", { class: "group" }, [simple.length ? h("div", { class: "grid tight" }, simple.map(function (k) { return k.el; })) : null].concat(complex.map(function (k) { return k.el; })));
    }
    function draw1(n, s) {
      var complex = ["object", "union", "array", "map", "any"].indexOf(n.kind) >= 0;
      if (!complex) return { el: scalar(n) };
      if (n.unset) return { complex: true, el: fold(n, h("p", { class: "dim", text: "not set" }), [btn("Set", function () { update(setAt(doc, n.path, defaultFor(Object.assign({}, n.schema, { default: undefined }), root)), true); })]) };
      var clear = n.nullable || !n.required ? btn("Clear", function () { update(n.nullable ? setAt(doc, n.path, null) : removeAt(doc, n.path), true); }) : null;
      if (n.kind === "any") return { complex: true, el: fold(n, h("pre", { class: "well", text: JSON.stringify(n.value, null, 2) })) };
      if (n.kind === "object" || n.kind === "union") {
        var kids = n.children.map(function (c) { return draw1(c); });
        if (n.choice) {
          var cid = "gf-" + (++ids);
          kids.unshift({ el: h("div", { class: "field" }, [h("label", { for: cid, text: humanize(n.choice.prop) }),
            h("select", { id: cid, disabled: ro, onchange: function (ev) { update(choose(doc, n.path, n.schema, ev.target.value, root), true); } },
              n.choice.tags.map(function (t) { return h("option", { value: t, selected: t === n.choice.tag, text: t }); }))]) });
        }
        return { complex: true, el: n.path.length && !s ? fold(n, group(kids), [clear]) : group(kids) };
      }
      // an array or a map: one panel per item, its tools in the panel's header
      var items = n.items.map(function (it, i) {
        var last = n.items.length - 1, isList = n.kind === "array";
        var tools = [isList ? btn("Up", function () { update(moveItem(doc, it.path, -1), true); }, "move up") : null,
          isList ? btn("Down", function () { update(moveItem(doc, it.path, +1), true); }, "move down") : null,
          btn("Remove", function () { update(removeAt(doc, it.path), true); })];
        if (isList && i === 0) tools[0] = null;
        if (isList && i === last) tools[1] = null;
        var title = isList ? "#" + (i + 1) + (it.choice ? " · " + it.choice.tag : "") : String(it.key);
        var inner = draw1(it, true);
        return h("div", { class: "panel", "data-path": pathKey(it.path), "data-changed": String(changed(it.path)) }, [
          h("header", {}, [h("span", { class: "eyebrow", text: title })].concat(tools)), h("div", { class: "body" }, [inner.el]),
        ]);
      });
      var addTools = [];
      if (!ro && n.kind === "array") addTools.push(btn("Add", function () { update(addItem(doc, n.path, n.itemSchema, root), true); }));
      if (!ro && n.kind === "map") {
        var kid = "gf-" + (++ids), keyIn = h("input", { id: kid, placeholder: "new key", "aria-label": "new key" });
        addTools.push(h("span", { class: "field" }, [keyIn]), btn("Add", function () { if (keyIn.value) update(addItem(doc, n.path, n.itemSchema, root, keyIn.value), true); }));
      }
      return { complex: true, el: fold(n, h("div", { class: "group" }, items), addTools.concat([clear])) };
    }
    function draw() {
      while (host.firstChild) host.removeChild(host.firstChild);
      host.appendChild(draw1(tree(schema, doc)).el);
    }
    draw();
    return {
      value: function () { return clone(doc); },
      setErrors: function (errs) { errors = placeErrors(tree(schema, doc), errs); draw(); return errors; },
      reset: function () { doc = clone(original); errors = {}; draw(); },
    };
  }

  api.TIERS = TIERS;
  api.equal = equal;
  api.strictest = strictest;
  api.deref = deref;
  api.unwrap = unwrap;
  api.variants = variants;
  api.kindOf = kindOf;
  api.defaultFor = defaultFor;
  api.tree = tree;
  api.getAt = getAt;
  api.setAt = setAt;
  api.removeAt = removeAt;
  api.addItem = addItem;
  api.moveItem = moveItem;
  api.choose = choose;
  api.parse = parse;
  api.diff = diff;
  api.placeErrors = placeErrors;
  api.mount = mount;
})(typeof module === "object" && module.exports ? module.exports : (window.GeecsForm = {}));
