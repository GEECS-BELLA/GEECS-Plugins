/* GEECS analysis config editor.
 *
 * A schema-driven form over the JSON Schema the store serves: the analysis
 * recipe (format 3: input, ordered steps and a measure from the analysis
 * core's registry, the figure, the summaries, the scan runtime) and the
 * group, plus the list, YAML preview, save and (when the host provides one)
 * a live preview of the document under edit drawn as a scan run draws it.
 * A format 2 diagnostic (a kind the core has not ported) is shown read-only.
 *
 *   ConfigEditor.mount(container, {
 *     base,                 // URL of the editor router mount ("" | "/configs" | proxied)
 *     preview,              // null, or {params: () => ({uid, device, shot, day}), label}
 *     readOnly,             // hide Save / New / Delete
 *     layout,               // "page" (sidebar + form + right pane) | "drawer" (form + right pane)
 *     initial,              // {kind: "analyzer"|"group", id} to open first
 *     onSaved,              // callback(saved) after a successful write
 *   })
 *
 * Renders exactly the JSON Schema shapes pydantic v2 emits for these
 * models: objects (with $ref / $defs), optionals (anyOf [T, null]),
 * discriminated unions (oneOf + discriminator, as ordered cards in a list),
 * enums / const, arrays of enums / scalars / objects, tuples (prefixItems),
 * keyed mappings (additionalProperties: object -> named cards) and keyword
 * mappings (additionalProperties: any -> key / value rows).  No build
 * chain, no library.
 */
(function () {
  "use strict";

  // ---------------------------------------------------------------- utils
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const el = (tag, attrs, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children) if (c !== null && c !== undefined) node.append(c);
    return node;
  };
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  async function api(base, path, opts) {
    const r = await fetch(base + "/api" + path, Object.assign({ headers: { "Content-Type": "application/json" } }, opts || {}));
    if (r.status === 204) return null;
    const ct = r.headers.get("content-type") || "";
    const body = ct.includes("json") ? await r.json() : await r.text();
    if (!r.ok) { const e = new Error((body && body.detail) || `HTTP ${r.status}`); e.status = r.status; e.body = body; throw e; }
    return body;
  }

  // -------------------------------------------------------------- schema
  class Schema {
    constructor(root) { this.root = root; this.defs = root.$defs || {}; }
    resolve(node) {
      let n = node;
      while (n && n.$ref) {
        const name = n.$ref.replace("#/$defs/", "");
        const target = this.defs[name];
        if (!target) throw new Error("unresolved $ref " + n.$ref);
        n = Object.assign({}, target, Object.fromEntries(Object.entries(n).filter(([k]) => k !== "$ref")));
        n.__name = name;
      }
      return n;
    }
    // anyOf [T, null] -> {inner, optional: true}
    unwrapOptional(node) {
      const n = this.resolve(node);
      const alts = n.anyOf;
      if (alts && alts.some((a) => a.type === "null")) {
        // Optional[T]: the non-null alternative; Optional[Union[str, Path]]
        // arrives as several string alternatives — the first one is the form
        const others = alts.filter((a) => a.type !== "null");
        const inner = others.length === 1 ? others[0] : (others.find((a) => a.type === "string") || others[0]);
        return { inner: this.resolve(Object.assign({}, inner, { description: n.description, title: n.title })), optional: true, meta: n };
      }
      if (alts && alts.every((a) => a.type === "string")) return { inner: Object.assign({}, n, { type: "string", anyOf: undefined }), optional: false, meta: n };
      return { inner: n, optional: false, meta: n };
    }
    kindOf(n) {
      if (n.discriminator && n.oneOf) return "union";
      if (n.enum) return "enum";
      if (n.const !== undefined) return "const";
      if (n.type === "object" || n.properties) return n.additionalProperties && !n.properties ? "map" : "object";
      if (n.type === "array") return n.prefixItems ? "tuple" : "array";
      if (n.type === "boolean") return "bool";
      if (n.type === "integer" || n.type === "number") return "number";
      if (n.type === "string") return "string";
      if (n.anyOf) return "any";
      return "json";
    }
    defaultFor(node) {
      const { inner, optional } = this.unwrapOptional(node);
      if (inner.default !== undefined) return inner.default;
      if (optional) return null;
      switch (this.kindOf(inner)) {
        case "object": {
          const out = {};
          for (const [k, sub] of Object.entries(inner.properties || {})) {
            const d = this.defaultFor(sub);
            if (d !== undefined && d !== null) out[k] = d;
          }
          return out;
        }
        case "union": { const first = this.resolve(inner.oneOf[0]); return this.defaultFor(first); }
        case "const": return inner.const;
        case "enum": return inner.enum[0];
        case "array": return [];
        case "tuple": return inner.prefixItems.map((p) => this.defaultFor(p));
        case "map": return {};
        case "bool": return false;
        case "number": return 0;
        case "string": return "";
        default: return null;
      }
    }
  }

  // ---------------------------------------------------------------- form
  // Each renderer returns {node, get()}; get() returns the JSON value or
  // undefined (omit the key).
  class Form {
    constructor(schema, onChange) { this.schema = schema; this.onChange = onChange; }

    render(node, value, path) {
      const { inner, optional, meta } = this.schema.unwrapOptional(node);
      const kind = this.schema.kindOf(inner);
      if (kind === "const") return { node: null, get: () => inner.const };
      if (value === undefined && !optional && inner.default !== undefined) value = inner.default;
      if (optional && ["object", "union", "map"].includes(kind)) return this.renderOptionalSection(inner, meta, value, path);
      return this[kind](inner, value, path, optional);
    }

    label(meta, path) {
      const name = path[path.length - 1];
      const title = typeof name === "number" ? `#${name + 1}` : String(name);
      return el("label", { title: meta.description || "" }, title);
    }
    static plain(text) { return String(text || "").replace(/``([^`]*)``/g, "$1"); }
    help(meta) { return meta.description ? el("div", { class: "help" }, Form.plain(meta.description)) : null; }
    field(meta, path, control, extra) {
      return el("div", { class: "field", "data-path": path.join(".") }, this.label(meta, path), control, this.help(meta), extra || null);
    }

    // --- scalars
    string(n, value, path, optional) {
      const input = el("input", { type: "text", value: value ?? "", placeholder: optional ? "(unset)" : "", oninput: () => this.onChange() });
      return { node: this.field(n, path, input), get: () => (input.value === "" ? (optional ? undefined : "") : input.value) };
    }
    number(n, value, path, optional) {
      const input = el("input", { type: "number", step: n.type === "integer" ? "1" : "any", value: value ?? "", placeholder: optional ? "(unset)" : "", oninput: () => this.onChange() });
      if (n.minimum !== undefined) input.min = n.minimum;
      if (n.maximum !== undefined) input.max = n.maximum;
      return { node: this.field(n, path, input), get: () => { if (input.value === "") return undefined; const v = Number(input.value); return n.type === "integer" ? Math.trunc(v) : v; } };
    }
    bool(n, value, path, optional) {
      const ctl = optional
        ? el("select", { onchange: () => this.onChange() }, el("option", { value: "" }, "(unset)"), el("option", { value: "true" }, "true"), el("option", { value: "false" }, "false"))
        : el("input", { type: "checkbox", onchange: () => this.onChange() });
      if (optional) ctl.value = value === null || value === undefined ? "" : String(value); else ctl.checked = !!value;
      return { node: this.field(n, path, ctl), get: () => (optional ? (ctl.value === "" ? undefined : ctl.value === "true") : ctl.checked) };
    }
    enum(n, value, path, optional) {
      const sel = el("select", { onchange: () => this.onChange() });
      if (optional) sel.append(el("option", { value: " " }, "(unset)"));
      for (const v of n.enum) sel.append(el("option", { value: String(v) }, String(v)));
      sel.value = value === null || value === undefined ? (optional ? " " : String(n.enum[0])) : String(value);
      return { node: this.field(n, path, sel), get: () => (sel.value === " " ? undefined : n.enum.find((v) => String(v) === sel.value)) };
    }
    any(n, value, path) { return this.json(n, value, path); }
    json(n, value, path) {
      const ta = el("textarea", { rows: 3, spellcheck: "false", oninput: () => this.onChange() });
      ta.value = value === undefined || value === null ? "" : JSON.stringify(value, null, 1);
      const err = el("div", { class: "ferr" });
      // Unparseable text is an error state, never "unset": get() throws, and
      // the editor refuses to validate, preview or save until it is fixed —
      // otherwise a typo in `metadata` would silently drop the mapping on Save.
      return { node: this.field(n, path, ta, err), get: () => { err.textContent = ""; if (!ta.value.trim()) return undefined; try { return JSON.parse(ta.value); } catch (e) { err.textContent = "invalid JSON: " + e.message; throw new FormParseError(path.join("."), e.message); } } };
    }
    map(n, value, path) {
      // Dict[str, T]: named cards when T is an object (frame inputs, a style
      // per overlay id), key / value rows when T is anything (a matplotlib
      // keyword group, free-form metadata)
      const item = n.additionalProperties;
      if (item && typeof item === "object" && (item.$ref || item.properties || item.type === "object")) return this.keyedList(n, this.schema.resolve(item), value, path);
      return this.kvRows(n, value, path);
    }
    // key / value rows: numbers, true / false, null, [lists] and {objects}
    // parse as JSON; anything else is text (quote text that looks like a
    // number: "1")
    static parseValue(text) { const s = text.trim(); if (s === "") return ""; try { return JSON.parse(s); } catch (_) { return text; } }
    static showValue(v) { return typeof v === "string" ? v : JSON.stringify(v); }
    kvRows(n, value, path, opts) {
      const box = el("div", { class: "kv" });
      const rows = [];
      const addBtn = el("button", { type: "button", class: "add", onclick: () => { add("", undefined); rows[rows.length - 1].key.focus(); } }, "+ keyword");
      box.append(addBtn);
      const add = (k, v) => {
        const key = el("input", { type: "text", class: "k", placeholder: "keyword", value: k ?? "", spellcheck: "false", oninput: () => this.onChange() });
        const val = el("input", { type: "text", class: "v", placeholder: "value", value: v === undefined ? "" : Form.showValue(v), spellcheck: "false", oninput: () => this.onChange() });
        const entry = { key, val, row: null };
        entry.row = el("div", { class: "row" }, key, val, el("button", { type: "button", title: "remove", onclick: () => { rows.splice(rows.indexOf(entry), 1); entry.row.remove(); this.onChange(); } }, "x"));
        rows.push(entry); box.insertBefore(entry.row, addBtn);
      };
      for (const [k, v] of Object.entries(value || {})) add(k, v);
      const get = () => { const out = {}; for (const r of rows) { const k = r.key.value.trim(); if (k) out[k] = Form.parseValue(r.val.value); } return out; };
      if (opts && opts.bare) return { node: box, get };
      return { node: this.field(n, path, box), get };
    }
    // named cards: a name input in the header, the item's fields below
    keyedList(n, item, value, path) {
      const list = el("div", { class: "list" });
      const rows = [];
      const kind = this.schema.kindOf(item);
      const addBtn = el("button", { type: "button", class: "add", onclick: () => { add("", this.schema.defaultFor(item)); rows[rows.length - 1].key.focus(); this.onChange(); } }, "+ add");
      list.append(addBtn);
      const add = (k, v) => {
        const key = el("input", { type: "text", class: "key", placeholder: "name", value: k ?? "", spellcheck: "false", oninput: () => this.onChange() });
        const sub = path.concat(k || "?");
        const r = kind === "map" ? this.kvRows(item, v, sub, { bare: true }) : this.object(item, v, sub, false);
        const body = el("div", { class: "body" }, r.node.tagName === "FIELDSET" ? (r.node.querySelector(".obj") || r.node) : r.node);
        const entry = { key, r, row: null };
        entry.row = el("div", { class: "item card" },
          el("div", { class: "head" }, key, el("span", { class: "spacer" }), el("button", { type: "button", title: "remove", onclick: () => { rows.splice(rows.indexOf(entry), 1); entry.row.remove(); this.onChange(); } }, "x")),
          body);
        rows.push(entry); list.insertBefore(entry.row, addBtn);
      };
      for (const [k, v] of Object.entries(value || {})) add(k, v);
      const legend = el("legend", { title: n.description || "" }, String(path[path.length - 1]));
      return { node: el("fieldset", {}, legend, this.help(n), list), get: () => { const out = {}; for (const e of rows) { const k = e.key.value.trim(); if (k) out[k] = e.r.get(); } return out; } };
    }

    // --- tuples: fixed inputs side by side
    tuple(n, value, path) {
      const items = n.prefixItems.map((p, i) => this.render(p, Array.isArray(value) ? value[i] : undefined, path.concat(i)));
      const row = el("div", { class: "tuple" }, ...items.map((it) => it.node.querySelector("input,select") || it.node));
      return { node: this.field(n, path, row), get: () => { const vals = items.map((it) => it.get()); return vals.every((v) => v === undefined) ? undefined : vals; } };
    }

    // --- arrays
    array(n, value, path, optional) {
      const item = this.schema.resolve(n.items || {});
      const itemKind = this.schema.kindOf(item);
      const values = Array.isArray(value) ? value.slice() : [];
      if (itemKind === "enum") return this.enumList(n, item, values, path, optional);
      if (itemKind === "string" || itemKind === "number") return this.scalarList(n, item, values, path, optional);
      return this.objectList(n, item, values, path, optional);
    }
    enumList(n, item, values, path, optional) {
      // ordered list of enum values with add / remove / reorder (pipelines are order-sensitive)
      const list = el("div", { class: "list" });
      const state = values.slice();
      const redraw = () => {
        list.innerHTML = "";
        state.forEach((v, i) => list.append(el("div", { class: "item" },
          el("span", { class: "mono" }, String(v)),
          el("button", { type: "button", title: "up", disabled: i === 0, onclick: () => { [state[i - 1], state[i]] = [state[i], state[i - 1]]; redraw(); this.onChange(); } }, "up"),
          el("button", { type: "button", title: "down", disabled: i === state.length - 1, onclick: () => { [state[i + 1], state[i]] = [state[i], state[i + 1]]; redraw(); this.onChange(); } }, "down"),
          el("button", { type: "button", title: "remove", onclick: () => { state.splice(i, 1); redraw(); this.onChange(); } }, "x"))));
        const sel = el("select");
        sel.append(el("option", { value: "" }, "add step..."));
        for (const v of item.enum) sel.append(el("option", { value: String(v) }, String(v)));
        sel.addEventListener("change", () => { if (sel.value) { state.push(item.enum.find((v) => String(v) === sel.value)); redraw(); this.onChange(); } });
        list.append(sel);
      };
      redraw();
      return { node: this.field(n, path, list), get: () => (state.length === 0 && optional ? undefined : state.slice()) };
    }
    scalarList(n, item, values, path, optional) {
      const input = el("input", { type: "text", value: values.join(", "), placeholder: optional ? "(unset) - comma separated" : "comma separated", oninput: () => this.onChange() });
      return { node: this.field(n, path, input), get: () => { const parts = input.value.split(",").map((s) => s.trim()).filter(Boolean); if (!parts.length) return optional ? undefined : []; return item.type === "string" ? parts : parts.map(Number); } };
    }
    objectList(n, item, values, path, optional) {
      // Ordered cards (a step, a summary): the header is the kind select for
      // a discriminated union, else the index, plus up / down / remove; the
      // fields sit below.  Reorder and remove rebuild the list from the
      // current values, so every field path (steps.2.bounds) stays true.
      const isUnion = this.schema.kindOf(item) === "union";
      // a pair of numbers or a scalar is a row with its buttons, not a card
      const light = !isUnion && ["tuple", "number", "string", "bool", "enum"].includes(this.schema.kindOf(item));
      const list = el("div", { class: "list" });
      const items = [];
      const current = () => items.map((it) => { try { return it.get(); } catch (e) { if (e instanceof FormParseError) return it.last; throw e; } });
      let adder;
      const rebuild = (vals) => {
        items.length = 0; list.innerHTML = "";
        vals.forEach((v, i) => {
          const r = isUnion ? this.union(item, v, path.concat(i), { header: true }) : this.render(item, v, path.concat(i));
          r.last = v;
          const buttons = [
            el("button", { type: "button", title: "move up", disabled: i === 0, onclick: () => { const c = current(); [c[i - 1], c[i]] = [c[i], c[i - 1]]; rebuild(c); this.onChange(); } }, "\u2191"),
            el("button", { type: "button", title: "move down", disabled: i === vals.length - 1, onclick: () => { const c = current(); [c[i + 1], c[i]] = [c[i], c[i + 1]]; rebuild(c); this.onChange(); } }, "\u2193"),
            el("button", { type: "button", title: "remove", onclick: () => { const c = current(); c.splice(i, 1); rebuild(c); this.onChange(); } }, "x"),
          ];
          if (light) { list.append(el("div", { class: "item light" }, r.node, ...buttons)); items.push(r); return; }
          const head = el("div", { class: "head" }, isUnion ? r.head : el("span", { class: "idx" }, `#${i + 1}`), el("span", { class: "spacer" }), ...buttons);
          const body = el("div", { class: "body" }, isUnion ? r.node : (r.node.tagName === "FIELDSET" ? (r.node.querySelector(".obj") || r.node) : r.node));
          list.append(el("div", { class: "item card" }, head, body));
          items.push(r);
        });
        list.append(adder);
      };
      const noun = String(path[path.length - 1]).replace(/ies$/, "y").replace(/s$/, "");
      if (isUnion) {
        adder = el("select", { class: "add", onchange: () => { if (adder.value) { rebuild(current().concat([this.schema.defaultFor(this.unionVariant(item, adder.value))])); this.onChange(); } } });
        adder.append(el("option", { value: "" }, `add ${noun}...`));
        for (const v of this.unionVariants(item)) adder.append(el("option", { value: String(v.tag), title: v.schema.description || "" }, v.label));
      } else {
        adder = el("button", { type: "button", class: "add", onclick: () => { rebuild(current().concat([this.schema.defaultFor(item)])); this.onChange(); } }, `+ add ${noun}`);
      }
      rebuild(values);
      return { node: el("fieldset", {}, el("legend", { title: n.description || "" }, String(path[path.length - 1])), this.help(n), list), get: () => { const vals = items.map((it) => it.get()).filter((v) => v !== undefined); return vals.length === 0 && optional ? undefined : vals; } };
    }

    // --- objects
    // opts: skip (keys not rendered nor written), hidden (keys not rendered,
    // written as loaded or defaulted: the format version), sections (the
    // root laid out as titled groups of keys, in that order)
    object(n, value, path, optional, opts) {
      const props = n.properties || {};
      const children = [];
      const body = el("div", { class: "obj" });
      const skip = (opts && opts.skip) || new Set();
      const hidden = (opts && opts.hidden) || new Set();
      const sections = (opts && opts.sections) || null;
      if (sections) for (const s of sections) {
        s.body = el("fieldset", { class: "ce-section" + (s.keys.length === 1 ? " ce-flat" : "") }, el("legend", {}, s.title), s.help ? el("div", { class: "ce-section-help" }, s.help) : null);
        body.append(s.body);
      }
      for (const [key, sub] of Object.entries(props)) {
        if (skip.has(key)) continue;
        // Retired config fields remain round-trippable without offering controls.
        if (sub.deprecated) {
          children.push([key, { get: () => value == null ? undefined : value[key] }]);
          continue;
        }
        if (hidden.has(key)) {
          children.push([key, { hidden: true, get: () => (value != null && value[key] !== undefined ? value[key] : this.schema.resolve(sub).default) }]);
          continue;
        }
        const r = this.render(sub, value && value[key] !== undefined ? value[key] : undefined, path.concat(key));
        if (r.node) { const sec = sections && sections.find((s) => s.keys.includes(key)); (sec ? sec.body : body).append(r.node); }
        children.push([key, r]);
      }
      const get = () => {
        const out = {};
        for (const [key, r] of children) {
          const v = r.get();
          if (v === undefined) continue;
          if (r.hidden) { out[key] = v; continue; }
          // a scalar equal to its schema default is left unwritten — the
          // file keeps only what the author set (canonical-form doctrine)
          const sub = this.schema.unwrapOptional(props[key]);
          const required = (n.required || []).includes(key);
          const isScalar = ["bool", "number", "string", "enum"].includes(this.schema.kindOf(sub.inner));
          if (isScalar && sub.inner.default !== undefined && JSON.stringify(v) === JSON.stringify(sub.inner.default) && !required) continue;
          if (!required && v !== null && typeof v === "object" && !Array.isArray(v) && Object.keys(v).length === 0) continue;
          out[key] = v;
        }
        return out;
      };
      if (path.length === 0) return { node: body, get };
      const legend = el("legend", { title: n.description || "" }, String(path[path.length - 1]));
      return { node: el("fieldset", {}, legend, body), get };
    }
    renderOptionalSection(inner, meta, value, path) {
      // a nullable object / union / map: a checkbox in the legend toggles presence
      const present = value !== null && value !== undefined;
      const check = el("input", { type: "checkbox" });
      check.checked = present;
      const kind = this.schema.kindOf(inner);
      const r = kind === "union" ? this.union(inner, present ? value : this.schema.defaultFor(inner), path)
        : kind === "map" ? this.map(inner, present ? value : undefined, path)
        : this.object(inner, present ? value : this.schema.defaultFor(inner), path, false);
      const fs = el("fieldset", { class: present ? "" : "off" });
      const legend = el("legend", { title: meta.description || inner.description || "" }, el("label", {}, check, " ", String(path[path.length - 1])));
      fs.append(legend);
      const help = kind === "map" ? null : this.help(meta); if (help) fs.append(help);
      fs.append(r.node.tagName === "FIELDSET" ? (r.node.querySelector(".obj") || r.node) : r.node);
      check.addEventListener("change", () => { fs.classList.toggle("off", !check.checked); this.onChange(); });
      return { node: fs, get: () => (check.checked ? r.get() : undefined) };
    }
    unionVariants(n) {
      const disc = n.discriminator.propertyName;
      return n.oneOf.map((v) => {
        const schema = this.schema.resolve(v);
        const p = schema.properties && schema.properties[disc];
        const tag = p ? (p.const !== undefined ? p.const : (p.enum || [])[0]) : schema.__name;
        return { tag, schema, label: String(tag) + Form.ndimHint(schema) };
      });
    }
    unionVariant(n, tag) { const v = this.unionVariants(n).find((x) => String(x.tag) === String(tag)); return v ? v.schema : null; }
    // "(images)" / "(traces)" after a kind the registry says fits one frame shape
    static ndimHint(v) { const d = v["x-ndim"]; return Array.isArray(d) && d.length === 1 ? (d[0] === 2 ? " (images)" : " (traces)") : ""; }
    // opts.header: return the kind select separately (a card header) instead
    // of a labelled field above the variant's fields
    union(n, value, path, opts) {
      const disc = n.discriminator.propertyName;
      const variants = this.unionVariants(n);
      const sel = el("select", { class: "kindsel" });
      for (const v of variants) sel.append(el("option", { value: String(v.tag), title: v.schema.description || "" }, v.label));
      const currentTag = value && value[disc] !== undefined ? value[disc] : variants[0].tag;
      // a name the registry does not know (a typo in the file) is kept as
      // written and shown as such, never silently swapped for the first kind
      if (!variants.some((v) => String(v.tag) === String(currentTag))) sel.append(el("option", { value: String(currentTag) }, `${currentTag} (unknown)`));
      sel.value = String(currentTag);
      const holder = el("div", {});
      let current = null;
      const build = (tag, v) => {
        const variant = variants.find((x) => String(x.tag) === String(tag));
        holder.innerHTML = "";
        if (!variant) {
          holder.append(el("div", { class: "ferr" }, `unknown ${disc} "${tag}" - pick a registered one`));
          current = { tag, get: () => Object.fromEntries(Object.entries(v || {}).filter(([k]) => k !== disc)) };
          return;
        }
        current = this.object(variant.schema, v, path, false, { skip: new Set([disc]) });
        holder.append(current.node.querySelector(".obj") || current.node);
        if (variant.schema.description) holder.prepend(el("div", { class: "help variant" }, Form.plain(variant.schema.description)));
        current.tag = variant.tag;
      };
      build(currentTag, value);
      sel.addEventListener("change", () => { build(sel.value, this.schema.defaultFor(this.unionVariant(n, sel.value) || {})); this.onChange(); });
      const get = () => Object.assign({ [disc]: current.tag }, current.get());
      if (opts && opts.header) return { node: holder, get, head: sel };
      const top = el("div", { class: "field" }, el("label", { title: n.description || "" }, disc), sel);
      const node = path.length === 0 ? el("div", {}, top, holder) : el("fieldset", {}, el("legend", { title: n.description || "" }, String(path[path.length - 1])), top, holder);
      return { node, get };
    }

    // --- server-side error display
    static showErrors(root, errors) {
      root.querySelectorAll(".field.err").forEach((f) => { f.classList.remove("err"); const e = f.querySelector(".ferr.srv"); if (e) e.remove(); });
      for (const err of errors || []) {
        // pydantic locations name the union variant ("analyzer.beam.compute_slopes"): try with and without it
        const parts = err.loc.split(".");
        let target = null;
        for (let n = parts.length; n > 0 && !target; n--) {
          const cand = parts.slice(0, n);
          target = root.querySelector(`.field[data-path="${cand.join(".")}"]`)
            || root.querySelector(`.field[data-path="${cand.filter((_, i) => i !== 1).join(".")}"]`);
        }
        if (target) { target.classList.add("err"); target.append(el("div", { class: "ferr srv" }, err.msg)); }
      }
    }
  }

  class FormParseError extends Error {
    constructor(loc, detail) { super(`${loc}: invalid JSON (${detail}) - fix it before validating or saving`); this.loc = loc; }
  }

  // -------------------------------------------------------------- editor
  function mount(container, opts) {
    const base = opts.base || "";
    const readOnly = !!opts.readOnly;
    const hasPreview = !!opts.preview;
    const layout = opts.layout || "page";

    const state = { kind: null, id: null, namespace: null, etag: null, listing: null, schemas: {}, form: null, dirty: false, get: null, formRoot: null, dirtyEl: null, saveBtn: null, loadError: null, loadYaml: null, readOnlyDoc: null };

    // The recipe form, in reading order: what is read, how each frame is
    // processed, what is measured, how it is drawn, the scan-level figures,
    // how the run behaves.
    const RECIPE_SECTIONS = () => [
      { title: "Source", keys: ["device", "output_name", "scalar_suffix", "description", "input", "inputs"], help: "The device whose folder is read, how one frame is read, and any frame loaded before the run (a background image) for a step to use by name." },
      { title: "Steps", keys: ["steps"], help: "Processing in order, top to bottom; a step may repeat. A step marked (images) or (traces) fits that input kind only." },
      { title: "Measure", keys: ["measure"], help: "What is measured on every processed frame; its scalars become s-file columns." },
      { title: "Figure", keys: ["figure"], help: "The per-frame draw, reused by every product image and summary panel: matplotlib keywords by call (imshow, plot, colorbar, axes, fig) and a style per overlay id (hidden, scale, or plot keywords). Numbers, true / false and [lists] are typed; other text is a string." },
      { title: "Summaries", keys: ["summaries"], help: "Scan-level figures, each a fixed kind with its own options; an empty list draws none." },
      { title: "Scan", keys: ["scan", "metadata"], help: "How the recipe runs over a scan, and free-form notes nothing reads." },
    ];

    container.innerHTML = "";
    const root = el("div", { class: "ce" + (layout === "drawer" ? " ce-drawer" : hasPreview ? "" : " ce-nopreview") });
    const side = layout === "drawer" ? null : el("div", { class: "ce-side" });
    const main = el("div", { class: "ce-main" });
    const right = el("div", { class: "ce-side-right" });
    if (side) root.append(side);
    root.append(main, right);
    container.append(root);

    const yamlBox = el("pre", { class: "ce-yaml" }, "");
    const errBox = el("div", { class: "ce-errors" });
    const okBox = el("div", { class: "ce-ok" });
    const previewBox = el("div", { class: "ce-preview" });
    // The preview renders the edited (unsaved) document on the host's shot.
    // On demand by default - one render per click - or after every edit
    // with "auto" on; the choice is remembered per browser.
    let autoPreview = false, previewStale = false;
    try { autoPreview = localStorage.getItem("ce.autoPreview") === "1"; } catch (_) { /* storage blocked */ }
    const previewBtn = el("button", { type: "button", title: "render the current shot through the document as edited (not saved)", onclick: () => { const d = currentDoc(); if (d && state.kind === "analyzer") preview(d); } }, "preview");
    const autoBox = el("input", { type: "checkbox", title: "re-render after every edit (one request per change)" });
    autoBox.checked = autoPreview;
    autoBox.addEventListener("change", () => {
      autoPreview = autoBox.checked;
      try { localStorage.setItem("ce.autoPreview", autoPreview ? "1" : "0"); } catch (_) { /* storage blocked */ }
      if (autoPreview && previewStale) previewBtn.click();
    });
    if (hasPreview) right.append(el("div", { class: "ce-preview-head" }, el("h4", {}, opts.preview.label || "preview"), previewBtn, el("label", { class: "ce-auto" }, autoBox, " auto")), previewBox);
    right.append(el("h4", {}, "yaml"), yamlBox, okBox, errBox);

    // ----- listing
    async function loadListing() {
      state.listing = await api(base, "/list");
      if (side) renderSide();
      return state.listing;
    }
    // The sidebar's collapse state, per browser.  It is a convenience, not
    // state anything depends on: a private window or blocked storage just
    // means the tree forgets between visits.
    const OPEN_KEY = "ce-side-open";
    // Tri-state, deliberately: a key is open, closed, or unsaid.  A set of
    // open keys cannot express "I closed this one", so the node holding the
    // open document — which auto-expands — would spring back open on the next
    // render, and Save renders.
    function openState() {
      try {
        const raw = window.localStorage.getItem(OPEN_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
      } catch (e) { return {}; }
    }
    function rememberOpen(key, isOpen) {
      try {
        const state = openState();
        state[key] = isOpen;
        window.localStorage.setItem(OPEN_KEY, JSON.stringify(state));
      } catch (e) { /* the tree still works; it just forgets */ }
    }
    // What the user said, and only failing that, whether this node holds the
    // document on screen.
    function wantOpen(remembered, key, holds) {
      const said = remembered[key];
      return said === undefined ? holds : said;
    }
    // Only a click (or Enter/Space, which fires one) on the summary is the
    // user's own choice — an auto-expansion is never written back, or every
    // document opened would leave its section permanently expanded.  The
    // timeout reads the state the browser has just applied.
    function trackOpen(details, key) {
      details.querySelector("summary").addEventListener("click", () => {
        setTimeout(() => rememberOpen(key, details.open), 0);
      });
    }

    function renderSide() {
      const L = state.listing; side.innerHTML = "";
      const remembered = openState();
      // Everything is collapsed until asked for: the corpus is dozens of
      // documents over two kinds and several namespaces, and one flat
      // expanded list is what buries the New rows.  A node opens when the
      // user opened it before, or when it holds what is on screen now.
      const section = (title, kind, entries, newLabel) => {
        const byNs = {};
        for (const e of entries) (byNs[e.namespace] = byNs[e.namespace] || []).push(e);
        const holdsOpen = state.kind === kind;
        const sec = el("details", { class: "ce-sec", open: wantOpen(remembered, kind, holdsOpen) });
        sec.append(el("summary", {}, el("span", { class: "ce-sec-t" }, title), el("span", { class: "ce-count" }, String(entries.length))));
        trackOpen(sec, kind);
        if (!readOnly) {
          // Named, and first in the body: an unlabelled "new" at the end of a
          // long list is how a diagnostic gets created as a group by mistake.
          const ns = el("input", { placeholder: "namespace", list: `ce-ns-${kind}` });
          const dl = el("datalist", { id: `ce-ns-${kind}` });
          for (const name of L.namespaces[kind]) dl.append(el("option", { value: name }));
          const id = el("input", { placeholder: "new id" });
          sec.append(el("div", { class: "ce-new" }, ns, dl, id, el("button", { type: "button", onclick: () => { if (ns.value && id.value) create(kind, ns.value.trim(), id.value.trim()); } }, newLabel)));
        }
        for (const ns of Object.keys(byNs).sort()) {
          const key = `${kind}/${ns}`;
          const bad = byNs[ns].filter((e) => !e.valid).length;
          const grp = el("details", { class: "ce-ns", open: wantOpen(remembered, key, holdsOpen && state.namespace === ns) });
          // a collapsed namespace must still admit it is hiding a broken file
          grp.append(el("summary", { class: bad ? "bad" : "", title: bad ? `${bad} file${bad === 1 ? "" : "s"} here do not validate` : "" },
            el("span", {}, ns),
            el("span", { class: "ce-count" }, bad ? `${byNs[ns].length} · ${bad} bad` : String(byNs[ns].length))));
          trackOpen(grp, key);
          for (const e of byNs[ns]) grp.append(el("a", { href: "#", class: (state.kind === kind && state.id === e.id ? "sel" : "") + (e.valid ? "" : " bad"), title: e.error || (e.analyzer_kind ? `${e.analyzer_kind} / ${e.device}` : ""), onclick: (ev) => { ev.preventDefault(); open(kind, e.id); } }, e.id,
            kind === "analyzer" && e.valid && e.schema_version !== 3 ? el("span", { class: "tag", title: "format 2 diagnostic: read-only until its kind is ported" }, "v2") : null));
          sec.append(grp);
        }
        side.append(sec);
      };
      section("recipes", "analyzer", L.analyzers, "new recipe");
      section("groups", "group", L.groups, "new group");
      if (L.pending && L.pending.length) side.append(el("div", { class: "pending" }, `${L.pending.length} uncommitted change${L.pending.length === 1 ? "" : "s"} in the configs checkout`));
    }

    async function schemaFor(kind) {
      if (!state.schemas[kind]) state.schemas[kind] = new Schema(await api(base, `/schema/${kind}`));
      return state.schemas[kind];
    }

    // ----- documents
    async function open(kind, id) {
      const loaded = await api(base, `/${kind}s/${encodeURIComponent(id)}`);
      state.kind = kind; state.id = loaded.id; state.namespace = loaded.namespace; state.etag = loaded.etag; state.dirty = false;
      // A file that does not validate on disk (a stray v1 file, a typo) is
      // shown as it is: the form is only a schema-shaped reconstruction that
      // drops unknown keys, so Save stays off until the user edits on purpose.
      state.loadError = loaded.valid ? null : (loaded.errors || []).map((e) => (e.loc ? `${e.loc}: ` : "") + e.msg).join("\n");
      state.loadYaml = loaded.valid ? null : loaded.yaml;
      // The form is the recipe's (format 3). A format 2 diagnostic - a kind
      // the analysis core has not ported - is shown as the file it is, read
      // only; it converts to a recipe when its kind is ported.
      state.readOnlyDoc = kind === "analyzer" && !(loaded.document && loaded.document.schema_version === 3) ? loaded : null;
      await buildForm(kind, loaded.document, loaded.errors);
      if (side) renderSide();
      if (layout === "page") location.hash = `#/${kind}s/${encodeURIComponent(id)}`;
    }
    async function create(kind, namespace, id) {
      state.readOnlyDoc = null;
      const schema = await schemaFor(kind);
      // a new recipe: a camera read as the device, measured as a beam, the
      // grid and the average as summaries - the corpus's common shape
      const doc = kind === "analyzer"
        ? { schema_version: 3, device: id, input: { kind: "camera" }, steps: [], measure: { kind: "beam" }, summaries: [{ kind: "image_grid" }, { kind: "average" }] }
        : Object.assign(schema.defaultFor(schema.root), { name: id });
      state.kind = kind; state.id = id; state.namespace = namespace; state.etag = null; state.loadError = null; state.loadYaml = null;
      await buildForm(kind, doc, []);
      if (side) renderSide();
    }
    async function buildForm(kind, document, errors) {
      const schema = await schemaFor(kind);
      main.innerHTML = "";
      // The kind, always: a new document is not in the listing yet, so the
      // sidebar highlight cannot say whether this is a diagnostic or a group.
      const ro = state.readOnlyDoc;
      const title = el("span", { class: "title" },
        el("span", { class: "ce-kind" }, kind === "analyzer" ? (ro ? "diagnostic \u00b7 format 2" : "recipe") : "group"),
        `${state.namespace}/${state.id}`);
      const dirty = el("span", { class: "dirty" });
      const bar = el("div", { class: "ce-bar" }, title, dirty);
      if (ro) {
        const d = ro.document || {};
        const kindName = d.analyzer && d.analyzer.kind ? d.analyzer.kind : "unknown";
        if (!readOnly && state.etag) bar.append(el("button", { type: "button", onclick: remove }, "Delete"));
        main.append(bar);
        main.append(el("p", { class: "ce-readonly-note", html: `This is a format 2 diagnostic (kind <code>${esc(kindName)}</code>). The analysis core does not run this kind yet, so the file is shown as saved; it converts to a recipe when the kind is ported. Edit it in the configs repository.` }));
        if (!ro.valid) main.append(el("div", { class: "ce-errors" }, (ro.errors || []).map((e) => (e.loc ? `${e.loc}: ` : "") + e.msg).join("\n")));
        main.append(el("pre", { class: "ce-yaml" }, ro.yaml || ""));
        state.form = null; state.formRoot = null; state.dirtyEl = dirty; state.saveBtn = null;
        errBox.textContent = ""; okBox.textContent = ""; previewBox.innerHTML = ""; yamlBox.textContent = "";
        // the saved document still previews: the pane shows what its run draws
        state.get = ro.valid ? () => d : null;
        if (state.get) await validate();
        return;
      }
      if (!readOnly) {
        state.saveBtn = el("button", { type: "button", class: "primary", onclick: save, disabled: !!state.loadError, title: state.loadError ? "this file does not validate on disk - edit it first, Save then replaces it" : "" }, state.etag ? "Save" : "Create");
        bar.append(state.saveBtn);
        if (state.etag) bar.append(el("button", { type: "button", onclick: () => open(kind, state.id) }, "Reload"));
        if (state.etag) bar.append(el("button", { type: "button", onclick: remove }, "Delete"));
      }
      main.append(bar);
      const formRoot = el("div", { class: "ce-form" });
      state.form = new Form(schema, onFormChange);
      const rendered = kind === "analyzer"
        ? state.form.object(schema.resolve(schema.root), document, [], false, { sections: RECIPE_SECTIONS(), hidden: new Set(["schema_version"]) })
        : state.form.render(schema.root, document, []);
      formRoot.append(rendered.node);
      main.append(formRoot);
      state.get = rendered.get; state.formRoot = formRoot; state.dirtyEl = dirty;
      if (kind === "group" && state.listing) {
        formRoot.querySelectorAll('.field[data-path$=".ref"] input').forEach((inp) => inp.setAttribute("list", "ce-known-ids"));
        if (!window.document.getElementById("ce-known-ids")) {
          const dl = el("datalist", { id: "ce-known-ids" });
          for (const k of state.listing.known_ids) dl.append(el("option", { value: k }));
          main.append(dl);
        }
      }
      errBox.textContent = ""; okBox.textContent = ""; previewBox.innerHTML = "";
      if (errors && errors.length) { Form.showErrors(formRoot, errors); errBox.textContent = errors.map((e) => `${e.loc}: ${e.msg}`).join("\n"); }
      await validate();
      if (state.etag === null) markDirty();
    }
    // The form's document, or null (with the error shown) when a free-mapping
    // textarea does not parse.
    function currentDoc() {
      if (!state.get) return null;
      try { return state.get(); } catch (e) {
        if (!(e instanceof FormParseError)) throw e;
        errBox.textContent = e.message; okBox.textContent = ""; return null;
      }
    }
    function markDirty() { state.dirty = true; if (state.dirtyEl) state.dirtyEl.textContent = "unsaved"; if (state.saveBtn) { state.saveBtn.disabled = false; state.saveBtn.title = ""; } }
    // The banner an invalid-on-disk file keeps until it is saved over.
    function loadBanner() {
      if (!state.loadError) return "";
      return state.dirty
        ? `Save will REPLACE a file that does not validate on disk:\n${state.loadError}`
        : `This file does not validate on disk:\n${state.loadError}\nThe form is a reconstruction from the schema (unknown keys dropped); the YAML pane shows the file as it is. Edit to enable Save.`;
    }
    const onFormChange = () => { markDirty(); validateDebounced(); };

    async function validate() {
      const doc = currentDoc();
      if (!doc) return null;
      const report = await api(base, `/validate/${state.kind}`, { method: "POST", body: JSON.stringify({ document: doc }) });
      if (state.formRoot) Form.showErrors(state.formRoot, report.errors);  // a read-only document has no form
      const banner = loadBanner();
      const untouchedInvalid = !!state.loadError && !state.dirty;
      if (report.ok) {
        yamlBox.textContent = untouchedInvalid ? state.loadYaml : report.yaml;
        errBox.textContent = banner; okBox.textContent = banner ? "" : "valid";
        if (hasPreview && state.kind === "analyzer" && !untouchedInvalid) {
          // first render on open; afterwards only in auto mode, else flag the image stale
          if (autoPreview || !previewBox.hasChildNodes()) previewDebounced(doc); else markPreviewStale();
        }
      } else {
        yamlBox.textContent = untouchedInvalid ? state.loadYaml : "";
        errBox.textContent = (banner ? banner + "\n\n" : "") + report.errors.map((e) => `${e.loc}: ${e.msg}`).join("\n"); okBox.textContent = "";
      }
      return report;
    }
    const validateDebounced = debounce(validate, 250);

    async function save() {
      const report = await validate();
      if (!report || !report.ok) return;
      try {
        const doc = currentDoc(); if (!doc) return;
        const saved = await api(base, `/${state.kind}s/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.id)}`, { method: "PUT", body: JSON.stringify({ document: doc, etag: state.etag }) });
        state.etag = saved.etag; state.dirty = false; state.loadError = null; state.loadYaml = null; state.dirtyEl.textContent = "saved"; okBox.textContent = "saved"; errBox.textContent = "";
        await loadListing();
        if (opts.onSaved) opts.onSaved(saved);
        const btn = main.querySelector("button.primary"); if (btn) btn.textContent = "Save";
      } catch (e) {
        errBox.textContent = e.status === 409
          ? `conflict: ${e.message} - reload to pick up the on-disk version`
          : (e.body && e.body.errors ? e.body.errors.map((x) => `${x.loc}: ${x.msg}`).join("\n") : e.message);
      }
    }
    async function remove() {
      if (!confirm(`Delete ${state.kind} ${state.id}? This removes the file from the configs tree.`)) return;
      try {
        await api(base, `/${state.kind}s/${encodeURIComponent(state.id)}?etag=${encodeURIComponent(state.etag)}`, { method: "DELETE" });
        state.kind = null; state.id = null; main.innerHTML = '<div class="ce-empty">deleted</div>'; yamlBox.textContent = "";
        await loadListing();
      } catch (e) { errBox.textContent = e.message; }
    }

    // ----- live preview (host-provided)
    let previewSeq = 0;
    async function preview(doc) {
      const params = opts.preview.params();
      if (!params) { previewBox.innerHTML = '<div class="msg">select a device and shot on the Images tab to preview</div>'; return; }
      const seq = ++previewSeq;
      const r = await fetch(base + "/api/preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ document: doc, params }) });
      if (seq !== previewSeq) return;
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try { const b = await r.json(); detail = b.detail + (b.errors ? "\n" + b.errors.map((x) => `${x.loc}: ${x.msg}`).join("\n") : ""); } catch (_) { /* text body */ }
        previewBox.innerHTML = ""; previewBox.append(el("div", { class: "perr" }, detail)); return;
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      previewBox.innerHTML = ""; previewBox.classList.remove("stale"); previewStale = false;
      const img = el("img", { src: url, alt: "preview" }); img.onload = () => URL.revokeObjectURL(url);
      previewBox.append(img, el("div", { class: "msg" }, `${params.device} / shot ${params.shot} - drawn as a scan run of the document above draws it (unsaved)`));
    }
    const previewDebounced = debounce(preview, 300);
    function markPreviewStale() {
      previewStale = true; previewBox.classList.add("stale");
      const m = previewBox.querySelector(".msg");
      if (m) m.textContent = "edited since this render - click preview (or turn on auto)";
    }

    // ----- boot
    const ready = (async () => {
      await loadListing();
      let initial = opts.initial || null;
      if (!initial && layout === "page") { const m = location.hash.match(/^#\/(analyzer|group)s\/(.+)$/); if (m) initial = { kind: m[1], id: decodeURIComponent(m[2]) }; }
      if (initial) { try { await open(initial.kind, initial.id); } catch (e) { main.innerHTML = `<div class="ce-empty">${esc(e.message)}</div>`; } }
      else if (!state.kind) main.innerHTML = '<div class="ce-empty">select a diagnostic or group</div>';
    })();

    // Start a new document from the current form's content (a variant of the
    // open diagnostic for the same device, say); Save then creates it.
    async function duplicate(namespace, id, patch) {
      const cur = currentDoc(); if (!cur || state.readOnlyDoc) return;
      state.readOnlyDoc = null;
      const doc = Object.assign(JSON.parse(JSON.stringify(cur)), patch || {});
      // The copy is a new identity: anything that pins the original's data
      // folder or output location would make the two overwrite each other.
      delete doc.output_name;
      if (doc.input) delete doc.input.folder;
      state.id = id; state.namespace = namespace; state.etag = null; state.loadError = null; state.loadYaml = null;
      await buildForm(state.kind, doc, []);
      if (side) renderSide();
    }

    return {
      open,
      ready,
      duplicate,
      isDirty: () => state.dirty,
      current: () => ({ kind: state.kind, id: state.id, namespace: state.namespace, etag: state.etag }),
      listing: () => state.listing,
      reloadListing: loadListing,
      refreshPreview: () => { const d = currentDoc(); if (d && hasPreview && state.kind === "analyzer") preview(d); },
    };
  }

  window.ConfigEditor = { mount };
})();
