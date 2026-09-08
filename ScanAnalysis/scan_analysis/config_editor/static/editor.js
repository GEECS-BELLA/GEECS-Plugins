/* GEECS analysis config editor.
 *
 * A schema-driven form over the JSON Schema GEECS-Schemas exports for
 * AnalysisDiagnostic / AnalysisGroup, plus the list, YAML preview, save and
 * (when the host provides one) a live preview of the document under edit.
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
 * discriminated unions (oneOf + discriminator), enums / const, arrays of
 * enums / scalars / objects, tuples (prefixItems), and free mappings
 * (additionalProperties -> JSON textarea).  No build chain, no library.
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
    help(meta) { return meta.description ? el("div", { class: "help" }, meta.description) : null; }
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
      return { node: this.field(n, path, ta, err), get: () => { err.textContent = ""; if (!ta.value.trim()) return undefined; try { return JSON.parse(ta.value); } catch (e) { err.textContent = "invalid JSON: " + e.message; return undefined; } } };
    }
    map(n, value, path) { return this.json(Object.assign({}, n, { description: (n.description || "") + " (JSON mapping)" }), value, path); }

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
      const list = el("div", { class: "list" });
      const items = [];
      const add = (v, i) => {
        const r = this.render(item, v, path.concat(i));
        const row = el("div", { class: "item" }, r.node, el("button", { type: "button", title: "remove", onclick: () => { items.splice(items.indexOf(r), 1); row.remove(); this.onChange(); } }, "x"));
        items.push(r); list.append(row);
      };
      values.forEach(add);
      const wrap = el("div", {}, list, el("button", { type: "button", onclick: () => { add(this.schema.defaultFor(item), items.length); this.onChange(); } }, "+ add"));
      return { node: el("fieldset", {}, el("legend", { title: n.description || "" }, String(path[path.length - 1])), wrap), get: () => { const vals = items.map((it) => it.get()).filter((v) => v !== undefined); return vals.length === 0 && optional ? undefined : vals; } };
    }

    // --- objects
    object(n, value, path, optional, opts) {
      const props = n.properties || {};
      const children = [];
      const body = el("div", { class: "obj" });
      const skip = (opts && opts.skip) || new Set();
      for (const [key, sub] of Object.entries(props)) {
        if (skip.has(key)) continue;
        const r = this.render(sub, value && value[key] !== undefined ? value[key] : undefined, path.concat(key));
        if (r.node) body.append(r.node);
        children.push([key, r]);
      }
      const get = () => {
        const out = {};
        for (const [key, r] of children) {
          const v = r.get();
          if (v === undefined) continue;
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
    union(n, value, path) {
      const disc = n.discriminator.propertyName;
      const variants = n.oneOf.map((v) => this.schema.resolve(v));
      const tagOf = (v) => { const p = v.properties && v.properties[disc]; return p ? (p.const !== undefined ? p.const : (p.enum || [])[0]) : v.__name; };
      const sel = el("select", { class: "kindsel" });
      for (const v of variants) sel.append(el("option", { value: String(tagOf(v)), title: v.description || "" }, String(tagOf(v))));
      const currentTag = value && value[disc] !== undefined ? value[disc] : tagOf(variants[0]);
      sel.value = String(currentTag);
      const holder = el("div", {});
      let current = null;
      const build = (tag, v) => {
        const variant = variants.find((x) => String(tagOf(x)) === String(tag)) || variants[0];
        holder.innerHTML = "";
        current = this.object(variant, v, path, false, { skip: new Set([disc]) });
        holder.append(current.node.querySelector(".obj") || current.node);
        if (variant.description) holder.prepend(el("div", { class: "help" }, variant.description));
        current.tag = tagOf(variant);
      };
      build(currentTag, value);
      sel.addEventListener("change", () => { build(sel.value, this.schema.defaultFor(variants.find((x) => String(tagOf(x)) === sel.value))); this.onChange(); });
      const top = el("div", { class: "field" }, el("label", { title: n.description || "" }, disc), sel);
      const node = path.length === 0 ? el("div", {}, top, holder) : el("fieldset", {}, el("legend", { title: n.description || "" }, String(path[path.length - 1])), top, holder);
      return { node, get: () => Object.assign({ [disc]: current.tag }, current.get()) };
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

  // -------------------------------------------------------------- editor
  function mount(container, opts) {
    const base = opts.base || "";
    const readOnly = !!opts.readOnly;
    const hasPreview = !!opts.preview;
    const layout = opts.layout || "page";

    const state = { kind: null, id: null, namespace: null, etag: null, listing: null, schemas: {}, form: null, dirty: false, get: null, formRoot: null, dirtyEl: null, saveBtn: null, loadError: null, loadYaml: null };

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
    const previewBtn = el("button", { type: "button", title: "render the current shot through the document as edited (not saved)", onclick: () => { if (state.get && state.kind === "analyzer") preview(state.get()); } }, "preview");
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
    function renderSide() {
      const L = state.listing; side.innerHTML = "";
      const section = (title, kind, entries) => {
        side.append(el("h4", {}, title));
        const byNs = {};
        for (const e of entries) (byNs[e.namespace] = byNs[e.namespace] || []).push(e);
        for (const ns of Object.keys(byNs).sort()) {
          side.append(el("div", { class: "ns" }, ns));
          for (const e of byNs[ns]) side.append(el("a", { href: "#", class: (state.kind === kind && state.id === e.id ? "sel" : "") + (e.valid ? "" : " bad"), title: e.error || (e.analyzer_kind ? `${e.analyzer_kind} / ${e.device}` : ""), onclick: (ev) => { ev.preventDefault(); open(kind, e.id); } }, e.id));
        }
        if (!readOnly) {
          const ns = el("input", { placeholder: "namespace", list: `ce-ns-${kind}` });
          const dl = el("datalist", { id: `ce-ns-${kind}` });
          for (const name of L.namespaces[kind]) dl.append(el("option", { value: name }));
          const id = el("input", { placeholder: "new id" });
          side.append(el("div", { class: "ce-new" }, ns, dl, id, el("button", { type: "button", onclick: () => { if (ns.value && id.value) create(kind, ns.value.trim(), id.value.trim()); } }, "new")));
        }
      };
      section("analyzers", "analyzer", L.analyzers);
      section("groups", "group", L.groups);
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
      await buildForm(kind, loaded.document, loaded.errors);
      if (side) renderSide();
      if (layout === "page") location.hash = `#/${kind}s/${encodeURIComponent(id)}`;
    }
    async function create(kind, namespace, id) {
      const schema = await schemaFor(kind);
      const doc = schema.defaultFor(schema.root);
      if (kind === "analyzer") { doc.name = id; doc.analyzer = { kind: "beam" }; doc.image = { type: "camera" }; }
      if (kind === "group") doc.name = id;
      state.kind = kind; state.id = id; state.namespace = namespace; state.etag = null; state.loadError = null; state.loadYaml = null;
      await buildForm(kind, doc, []);
      if (side) renderSide();
    }
    async function buildForm(kind, document, errors) {
      const schema = await schemaFor(kind);
      main.innerHTML = "";
      const title = el("span", { class: "title" }, `${state.namespace}/${state.id}`);
      const dirty = el("span", { class: "dirty" });
      const bar = el("div", { class: "ce-bar" }, title, dirty);
      if (!readOnly) {
        state.saveBtn = el("button", { type: "button", class: "primary", onclick: save, disabled: !!state.loadError, title: state.loadError ? "this file does not validate on disk - edit it first, Save then replaces it" : "" }, state.etag ? "Save" : "Create");
        bar.append(state.saveBtn);
        if (state.etag) bar.append(el("button", { type: "button", onclick: () => open(kind, state.id) }, "Reload"));
        if (state.etag) bar.append(el("button", { type: "button", onclick: remove }, "Delete"));
      }
      main.append(bar);
      const formRoot = el("div", { class: "ce-form" });
      state.form = new Form(schema, onFormChange);
      const rendered = state.form.render(schema.root, document, []);
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
      if (!state.get) return null;
      const doc = state.get();
      const report = await api(base, `/validate/${state.kind}`, { method: "POST", body: JSON.stringify({ document: doc }) });
      Form.showErrors(state.formRoot, report.errors);
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
        errBox.textContent = (banner ? banner + "\n\n" : "") + report.errors.map((e) => `${e.loc}: ${e.msg}`).join("\n"); okBox.textContent = "";
      }
      return report;
    }
    const validateDebounced = debounce(validate, 250);

    async function save() {
      const report = await validate();
      if (!report || !report.ok) return;
      try {
        const saved = await api(base, `/${state.kind}s/${encodeURIComponent(state.namespace)}/${encodeURIComponent(state.id)}`, { method: "PUT", body: JSON.stringify({ document: state.get(), etag: state.etag }) });
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
      previewBox.append(img, el("div", { class: "msg" }, `${params.device} / shot ${params.shot} - rendered through the document above (unsaved)`));
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
      if (!state.get) return;
      const doc = Object.assign(JSON.parse(JSON.stringify(state.get())), patch || {});
      // The copy is a new identity: anything that pins the original's data
      // folder or output location would make the two overwrite each other.
      delete doc.output_name;
      if (doc.scan) delete doc.scan.device;
      if (doc.analyzer) delete doc.analyzer.output_label;
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
      refreshPreview: () => { if (state.get && hasPreview && state.kind === "analyzer") preview(state.get()); },
    };
  }

  window.ConfigEditor = { mount };
})();
