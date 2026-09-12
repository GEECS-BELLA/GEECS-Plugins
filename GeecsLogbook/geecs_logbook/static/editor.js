/* GeecsLogbook editor — everything a composer does, for every composer.
 *
 * One implementation serves the day page's per-scan, per-gap and day
 * composers, the in-place edit form, and the month page's composer. The
 * page hands it its facts through <main id="logbook" data-api data-day
 * data-book data-accept> and the type prefills through one JSON block
 * (#logbook-seeds); nothing else is templated into this file. A form
 * with no page-wide day (the month page) carries its own: a .when date
 * input, or data-day.
 *
 * What it adds over a bare textarea:
 *   - a toolbar that writes markdown around the selection (no WYSIWYG —
 *     the body stays plain text the mirror can hold);
 *   - paste or drop a screenshot: the file goes to the upload endpoint and
 *     a relative image link lands at the cursor. A NEW entry has no id
 *     until it is saved, so the first attachment saves it first (an
 *     "autosave") and the Save button then edits it; Discard deletes the
 *     stub;
 *   - paste a spreadsheet range (tab-separated, or an HTML table): it
 *     becomes a markdown table;
 *   - Preview, rendered by the server exactly as the page will show it;
 *   - type buttons: a template's prefill lands in the textarea and the
 *     entry records which template it started from;
 *   - ⌘/Ctrl+Enter saves.
 *
 * Styling is entirely through the page's tokens; this file sets no colours.
 */
(function () {
  "use strict";

  const host = document.getElementById("logbook");
  if (!host) return;
  const API = host.dataset.api;
  const DAY = host.dataset.day;
  const BOOK = host.dataset.book || "scans";
  const ACCEPT = (host.dataset.accept || "image/png,image/jpeg,image/gif,image/webp,application/pdf").split(",");
  const AUTHOR_KEY = "geecs.author";
  const SEEDS = (() => {
    const el = document.getElementById("logbook-seeds");
    try { return el ? JSON.parse(el.textContent) : {}; } catch (e) { return {}; }
  })();

  // ------------------------------------------------------------ plumbing

  const who = () => { try { return localStorage.getItem(AUTHOR_KEY) || ""; } catch (e) { return ""; } };

  /** The day a form writes to: its own date field, its data-day, or the page's. */
  function dayOf(form) {
    const when = form.querySelector(".when");
    return (when && when.value) || form.dataset.day || DAY || "";
  }
  const remember = (name) => { try { localStorage.setItem(AUTHOR_KEY, name); } catch (e) { /* private window */ } };

  async function api(method, path, body) {
    const res = await fetch(API + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 204) return null;
    const data = await res.json().catch(() => ({}));
    const d = data.detail;
    const text = Array.isArray(d) ? d.map((x) => x.msg || JSON.stringify(x)).join("; ")
      : (d && d.message) || d || res.statusText;
    if (!res.ok) throw Object.assign(new Error(text), { status: res.status, data });
    return data;
  }

  function fail(el, err) {
    let msg = el.querySelector(".errmsg");
    if (!msg) { msg = document.createElement("div"); msg.className = "errmsg"; el.appendChild(msg); }
    msg.textContent = err.status === 409
      ? "Someone else saved this first. Reload to see their version, then re-apply your change."
      : (err.message || "Could not save.");
  }
  function clearFail(el) { const m = el.querySelector(".errmsg"); if (m) m.remove(); }

  // ------------------------------------------------------ text helpers

  function replaceSelection(ta, text, selectFrom, selectTo) {
    const s = ta.selectionStart, e = ta.selectionEnd;
    ta.setRangeText(text, s, e, "end");
    if (selectFrom !== undefined) ta.setSelectionRange(s + selectFrom, s + selectTo);
    ta.focus();
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }

  /** Surround the selection (or a placeholder) with before/after. */
  function wrap(ta, before, after, placeholder) {
    const s = ta.selectionStart, e = ta.selectionEnd;
    const inner = s === e ? placeholder : ta.value.slice(s, e);
    replaceSelection(ta, before + inner + after, before.length, before.length + inner.length);
  }

  /** Prefix every selected line (or the current one). */
  function prefixLines(ta, prefix) {
    const s = ta.selectionStart;
    // A selection that ends just after a newline (a triple-clicked line)
    // does not reach into the next line.
    const e = ta.selectionEnd > s && ta.value[ta.selectionEnd - 1] === "\n" ? ta.selectionEnd - 1 : ta.selectionEnd;
    const lineStart = s === 0 ? 0 : ta.value.lastIndexOf("\n", s - 1) + 1;
    let lineEnd = ta.value.indexOf("\n", e);
    if (lineEnd === -1) lineEnd = ta.value.length;
    if (lineEnd < lineStart) lineEnd = lineStart;
    const block = ta.value.slice(lineStart, lineEnd);
    const done = block.split("\n").map((l) => (l.startsWith(prefix) ? l : prefix + l)).join("\n");
    ta.setRangeText(done, lineStart, lineEnd, "select");
    ta.focus();
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }

  /** Insert a block on its own lines at the cursor.
   *
   * ``select`` leaves the block selected — right for a skeleton the author
   * will overtype (a table, a callout), wrong for an attachment link, which
   * the next keystroke or the next upload would otherwise replace.
   */
  function insertBlock(ta, text, select) {
    const s = ta.selectionStart;
    const before = ta.value.slice(0, s);
    const lead = before.length === 0 || before.endsWith("\n\n") ? "" : before.endsWith("\n") ? "\n" : "\n\n";
    if (select) replaceSelection(ta, lead + text + "\n", lead.length, lead.length + text.length);
    else replaceSelection(ta, lead + text + "\n");
  }

  // ------------------------------------------------------- tables

  const cell = (v) => String(v).replace(/\|/g, "\\|").replace(/\s+/g, " ").trim();

  function rowsToMarkdown(rows) {
    const width = Math.max(...rows.map((r) => r.length));
    const pad = (r) => Array.from({ length: width }, (_, i) => cell(r[i] === undefined ? "" : r[i]));
    const [head, ...body] = rows.map(pad);
    const line = (r) => "| " + r.join(" | ") + " |";
    return [line(head), "|" + head.map(() => "---|").join(""), ...body.map(line)].join("\n");
  }

  /** A tab-separated block with two or more lines is a spreadsheet range. */
  function tsvToRows(text) {
    const lines = text.replace(/\r/g, "").split("\n").filter((l) => l.trim() !== "");
    if (lines.length < 2 || !lines.every((l) => l.includes("\t"))) return null;
    const rows = lines.map((l) => l.split("\t"));
    // Tab-indented prose is not a spreadsheet: every row has the same
    // number of columns (2+) and the first cell of the header is not empty.
    const width = rows[0].length;
    if (width < 2 || rows[0][0].trim() === "" || !rows.every((r) => r.length === width)) return null;
    return rows;
  }

  /** Excel/Sheets put an HTML table on the clipboard too; prefer it when present. */
  function htmlTableToRows(html) {
    const doc = new DOMParser().parseFromString(html, "text/html");
    const table = doc.querySelector("table");
    if (!table) return null;
    const rows = [...table.querySelectorAll("tr")].map((tr) =>
      [...tr.querySelectorAll("th,td")].map((td) => td.textContent));
    return rows.length >= 2 ? rows : null;
  }

  // ----------------------------------------------------- attachments

  /** The author's name from the form, or null (with the field focused). */
  function authorName(form) {
    const input = form.querySelector(".who");
    const name = input.value.trim();
    if (name) { remember(name); return name; }
    input.focus();
    fail(form, { message: "Enter your name first." });
    return null;
  }

  /** The entry this form edits, creating it first if it is brand new.
   *
   * One create per form, ever: the pending promise is kept on the form so
   * a second paste, a drop and a Save racing the first create all await
   * the same POST instead of each making an entry.
   */
  function ensureEntry(form) {
    if (form.dataset.entry) return Promise.resolve(form.dataset.entry);
    if (form._creating) return form._creating;
    const name = authorName(form);
    if (!name) return Promise.reject(Object.assign(new Error("Enter your name first."), { silent: true }));
    const day = dayOf(form);
    if (!day) {
      const when = form.querySelector(".when"); if (when) when.focus();
      fail(form, { message: "Pick a day first." });
      return Promise.reject(Object.assign(new Error("Pick a day first."), { silent: true }));
    }
    const ta = form.querySelector(".ta");
    const body = { day, book: BOOK, author: name, body_md: ta.value || "",
      template: form.dataset.template || "blank" };
    if (form.dataset.after !== undefined && form.dataset.after !== "") body.after = Number(form.dataset.after);
    else if (form.dataset.scan !== undefined && form.dataset.scan !== "") body.scan = Number(form.dataset.scan);
    form._creating = api("POST", "/entries", body).then((entry) => {
      form.dataset.entry = entry.entry_id;
      form.dataset.version = String(entry.version);
      form.classList.add("autosaved");
      // The entry now has its day; an edit cannot move it. Lock the
      // picker so a later change is not silently ignored (Discard frees it).
      const when = form.querySelector(".when");
      if (when) { when.disabled = true; when.title = "Saved on this day — Discard to pick another"; }
      // Same for the type: the entry recorded its template at creation and
      // an edit does not carry one, so a later press would show a type the
      // store does not hold.
      form.querySelectorAll(".typebtn").forEach((b) => {
        b.disabled = true; b.title = "Saved as " + (form.dataset.template || "a plain note") + " — Discard to change the type";
      });
      const discard = form.querySelector("[data-discard]");
      if (discard) discard.hidden = false;
      return entry.entry_id;
    }).finally(() => { form._creating = null; });
    return form._creating;
  }

  /** Record the entry's current version on the form and, for an in-place
   *  edit, on the article too, so a Cancel-then-Edit does not start stale. */
  function noteVersion(form, version) {
    form.dataset.version = String(version);
    const art = form.closest("article");
    if (art && art.dataset.entry === form.dataset.entry) art.dataset.version = String(version);
  }

  async function uploadFiles(form, files) {
    const ta = form.querySelector(".ta");
    const all = [...files];
    const list = all.filter((f) => ACCEPT.includes(f.type));
    if (all.length && !list.length) {
      fail(form, { message: `Not an accepted type (${all.map((f) => f.type || "unknown").join(", ")}); accepted: ${ACCEPT.join(", ")}` });
      return;
    }
    if (!list.length) return;
    clearFail(form);
    form.classList.add("busy");
    let id = null;
    let done;
    // Kept on the form so a Save that lands mid-upload waits for the link
    // to be inserted and the version to be re-read, instead of racing it.
    form._uploading = new Promise((resolve) => { done = resolve; });
    try {
      id = await ensureEntry(form);
      for (const file of list) {
        const fd = new FormData();
        fd.append("file", file, file.name || "image.png");
        const res = await fetch(`${API}/entries/${id}/attachments`, { method: "POST", body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw Object.assign(new Error((data.detail && data.detail.message) || data.detail || res.statusText), { status: res.status });
        const alt = (file.name || "image").replace(/\.[^.]+$/, "");
        const md = file.type === "application/pdf" ? `[${alt}](${data.link})` : `![${alt}](${data.link})`;
        insertBlock(ta, md, false);
      }
    } catch (err) {
      if (!err.silent) fail(form, err);
    } finally {
      // Every upload that landed bumped the version — including the ones
      // before a failure — so the eventual Save must carry the current one.
      if (id) {
        try { const fresh = await api("GET", `/entries/${id}`); noteVersion(form, fresh.version); }
        catch (err) { fail(form, err); }
      }
      form.classList.remove("busy");
      form._uploading = null;
      done();
    }
  }

  // --------------------------------------------------------- preview

  async function togglePreview(form) {
    const ta = form.querySelector(".ta");
    let pane = form.querySelector(".preview");
    const btn = form.querySelector("[data-tool=preview]");
    if (pane && !pane.hidden) { pane.hidden = true; ta.hidden = false; btn.classList.remove("is-on"); return; }
    if (!pane) { pane = document.createElement("div"); pane.className = "entry-body preview"; pane.hidden = true; ta.after(pane); }
    try {
      const data = await api("POST", "/preview", { body_md: ta.value });
      pane.innerHTML = data.html;
      pane.hidden = false; ta.hidden = true; btn.classList.add("is-on");
    } catch (err) { fail(form, err); }
  }

  // ---------------------------------------------------- type buttons

  /** A template's prefill: replaces an empty textarea, else lands as a block. */
  function applyType(form, name) {
    if (form.dataset.entry) return; // saved already; its template is fixed
    const ta = form.querySelector(".ta");
    const body = SEEDS[name];
    if (body === undefined) return;
    if (!ta.value.trim()) { ta.value = ""; ta.setSelectionRange(0, 0); }
    insertBlock(ta, body.replace(/\n$/, ""), false);
    // Provenance: which template the entry started from. The last one
    // pressed wins; the body is the author's either way.
    form.dataset.template = name;
    form.querySelectorAll(".typebtn").forEach((b) => b.classList.toggle("is-on", b.dataset.type === name));
  }

  // --------------------------------------------------------- toolbar

  const TOOLS = [
    { key: "bold", label: "B", title: "Bold", run: (ta) => wrap(ta, "**", "**", "bold") },
    { key: "italic", label: "I", title: "Italic", run: (ta) => wrap(ta, "_", "_", "italic") },
    { key: "code", label: "</>", title: "Code", run: (ta) => wrap(ta, "`", "`", "code") },
    { key: "list", label: "• list", title: "Bulleted list", run: (ta) => prefixLines(ta, "- ") },
    { key: "task", label: "☐ task", title: "Task list", run: (ta) => prefixLines(ta, "- [ ] ") },
    { key: "link", label: "link", title: "Link", run: (ta) => wrap(ta, "[", "](https://)", "text") },
    { key: "table", label: "table", title: "Table (or paste a spreadsheet range)",
      run: (ta) => insertBlock(ta, rowsToMarkdown([["Parameter", "Value", "Note"], ["", "", ""]]), true) },
    { key: "callout", label: "callout", title: "Callout — NOTE, TIP, WARNING or CAUTION",
      run: (ta) => insertBlock(ta, "> [!NOTE]\n> ", true) },
    { key: "image", label: "image", title: "Attach an image or PDF (or paste / drop one)", run: null },
    { key: "preview", label: "preview", title: "Show it as the page will", run: null },
  ];

  function buildToolbar(form) {
    if (form.querySelector(".toolbar")) return;
    const ta = form.querySelector(".ta");
    const bar = document.createElement("div");
    bar.className = "toolbar";
    bar.setAttribute("role", "toolbar");
    for (const t of TOOLS) {
      const b = document.createElement("button");
      b.type = "button"; b.className = "tool"; b.dataset.tool = t.key; b.title = t.title; b.textContent = t.label;
      if (t.key === "bold") b.classList.add("tool-b");
      if (t.key === "italic") b.classList.add("tool-i");
      bar.appendChild(b);
    }
    const picker = document.createElement("input");
    picker.type = "file"; picker.accept = ACCEPT.join(","); picker.multiple = true; picker.hidden = true;
    picker.addEventListener("change", () => { uploadFiles(form, picker.files); picker.value = ""; });
    bar.appendChild(picker);
    bar.addEventListener("click", (ev) => {
      const b = ev.target.closest("button.tool"); if (!b) return;
      const tool = TOOLS.find((t) => t.key === b.dataset.tool);
      if (tool.key === "image") picker.click();
      else if (tool.key === "preview") togglePreview(form);
      else tool.run(ta);
    });
    ta.before(bar);
  }

  // ---------------------------------------------------- paste & drop

  function onPaste(form, ev) {
    const cd = ev.clipboardData; if (!cd) return;
    const ta = form.querySelector(".ta");
    // A spreadsheet range often arrives as an HTML table AND a bitmap of
    // the same cells (Excel); the table is what was meant.
    const html = cd.getData("text/html");
    const tableRows = html ? htmlTableToRows(html) : null;
    if (tableRows) { ev.preventDefault(); insertBlock(ta, rowsToMarkdown(tableRows), true); return; }
    // A single row still comes with a <table> (and, from Excel, a bitmap
    // of the cells): let the default text paste through, upload nothing.
    if (html && /<table[\s>]/i.test(html)) return;
    const files = [...cd.items].filter((i) => i.kind === "file").map((i) => i.getAsFile()).filter(Boolean);
    if (files.length) {
      // A file plus text (a rich paste from a page or a document): keep the
      // browser's default text paste and upload the file as well.
      if (!cd.getData("text/plain")) ev.preventDefault();
      uploadFiles(form, files);
      return;
    }
    const rows = tsvToRows(cd.getData("text/plain") || "");
    if (rows) { ev.preventDefault(); insertBlock(ta, rowsToMarkdown(rows), true); }
  }

  function onDrop(form, ev) {
    if (!ev.dataTransfer || !ev.dataTransfer.files.length) return;
    ev.preventDefault(); form.classList.remove("dragover");
    uploadFiles(form, ev.dataTransfer.files);
  }

  // ------------------------------------------------------------ save

  /** Run one save-shaped action per form at a time; the buttons follow. */
  async function exclusive(form, action) {
    if (form.dataset.saving === "1") return;
    form.dataset.saving = "1";
    form.querySelectorAll("button[type=submit],[data-discard]").forEach((b) => { b.disabled = true; });
    try {
      if (form._uploading) await form._uploading; // let a paste finish landing first
      await action();
    }
    finally {
      form.dataset.saving = "";
      form.querySelectorAll("button[type=submit],[data-discard]").forEach((b) => { b.disabled = false; });
    }
  }

  function save(form) {
    return exclusive(form, async () => {
      const ta = form.querySelector(".ta");
      if (!ta.value.trim()) {
        ta.focus();
        if (form.dataset.entry && form.classList.contains("autosaved")) {
          fail(form, { message: "Nothing to save — Discard removes this entry." });
        }
        return;
      }
      const name = authorName(form);
      if (!name) return;
      clearFail(form);
      try {
        if (form.dataset.entry) {
          await api("PATCH", `/entries/${form.dataset.entry}`, {
            body_md: ta.value, editor: name, expected_version: Number(form.dataset.version) });
        } else {
          await ensureEntry(form);
        }
        location.reload();
      } catch (err) { if (!err.silent) fail(form, err); }
    });
  }

  function discard(form) {
    if (!form.dataset.entry) return Promise.resolve();
    return exclusive(form, async () => {
      try { await api("DELETE", `/entries/${form.dataset.entry}`); location.reload(); }
      catch (err) { fail(form, err); }
    });
  }

  // ----------------------------------------------------------- mount

  function mount(form) {
    if (form.dataset.mounted) return;
    form.dataset.mounted = "1";
    const ta = form.querySelector(".ta");
    const whoInput = form.querySelector(".who");
    if (whoInput && !whoInput.value) whoInput.value = who();
    buildToolbar(form);
    if (!form.classList.contains("editing") && !form.querySelector("[data-discard]")) {
      const d = document.createElement("button");
      d.type = "button"; d.className = "btn btn-sm"; d.dataset.discard = "1"; d.hidden = true; d.textContent = "Discard";
      d.title = "This entry was saved when you attached a file; discard deletes it";
      form.querySelector(".editor-bar").appendChild(d);
      d.addEventListener("click", () => discard(form));
    }
    ta.addEventListener("paste", (ev) => onPaste(form, ev));
    form.addEventListener("dragover", (ev) => { ev.preventDefault(); form.classList.add("dragover"); });
    form.addEventListener("dragleave", () => form.classList.remove("dragover"));
    form.addEventListener("drop", (ev) => onDrop(form, ev));
    form.addEventListener("submit", (ev) => { ev.preventDefault(); save(form); });
    form.querySelectorAll(".typebtn").forEach((b) => b.addEventListener("click", () => applyType(form, b.dataset.type)));
    ta.addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") {
        ev.preventDefault();
        if (!ev.repeat) save(form); // a held key must not save N times
      }
    });
  }

  document.querySelectorAll("form.composer").forEach(mount);

  // ------------------------------------------ entry buttons on the page

  document.addEventListener("click", async (ev) => {
    const b = ev.target.closest("button"); if (!b) return;
    if (b.closest("[data-insert]")) {
      const after = b.closest("[data-insert]").dataset.insert;
      const hostEl = document.querySelector(`[data-between-host="${after}"]`);
      if (hostEl) {
        hostEl.hidden = false; hostEl.open = true; b.closest("[data-insert]").hidden = true;
        const ta = hostEl.querySelector(".ta"); if (ta) ta.focus();
      }
      return;
    }
    if (b.dataset.keep) {
      try { await api("POST", `/entries/${b.dataset.keep}/status`, { status: "kept" }); location.reload(); }
      catch (err) { fail(b.closest(".entry-main"), err); }
      return;
    }
    if (b.dataset.del) {
      if (b.dataset.armed !== "1") { b.dataset.armed = "1"; b.textContent = "Really delete"; return; }
      try { await api("DELETE", `/entries/${b.dataset.del}`); location.reload(); }
      catch (err) { fail(b.closest(".entry-main"), err); }
      return;
    }
    if (b.dataset.edit) {
      const art = document.getElementById("entry-" + b.dataset.edit);
      const main = art.querySelector(".entry-main");
      const raw = JSON.parse(art.querySelector(".raw").textContent);
      const bodyEl = main.querySelector(".entry-body");
      if (main.querySelector("form.editing")) return;
      const form = document.createElement("form");
      form.className = "composer editing";
      form.dataset.entry = art.dataset.entry;
      form.dataset.version = art.dataset.version;
      form.innerHTML = `<div class="entry-main"><textarea class="ta" rows="8"></textarea>
        <div class="editor-bar"><input class="who" placeholder="Your name" required aria-label="Your name">
        <button class="btn btn-sm btn-primary" type="submit">Save</button>
        <button class="btn btn-sm" type="button" data-cancel="1">Cancel</button>
        <span class="editor-hint">⌘↩ saves</span></div></div>`;
      form.querySelector(".ta").value = raw;
      bodyEl.hidden = true; bodyEl.after(form);
      mount(form);
      form.querySelector(".ta").focus();
      form.querySelector("[data-cancel]").addEventListener("click", () => { form.remove(); bodyEl.hidden = false; });
    }
  });
})();
