/* GeecsLogbook editor — everything a composer does, for every composer.
 *
 * One implementation serves the day page's per-scan, per-gap and day
 * composers, the in-place edit form, and (next) the month page. The page
 * hands it two facts through <main id="logbook" data-api data-day
 * data-book>; nothing else is templated into this file.
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
  const AUTHOR_KEY = "geecs.author";

  // ------------------------------------------------------------ plumbing

  const who = () => { try { return localStorage.getItem(AUTHOR_KEY) || ""; } catch (e) { return ""; } };
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
    const s = ta.selectionStart, e = ta.selectionEnd;
    const lineStart = ta.value.lastIndexOf("\n", s - 1) + 1;
    let lineEnd = ta.value.indexOf("\n", e);
    if (lineEnd === -1) lineEnd = ta.value.length;
    const block = ta.value.slice(lineStart, lineEnd);
    const done = block.split("\n").map((l) => (l.startsWith(prefix) ? l : prefix + l)).join("\n");
    ta.setRangeText(done, lineStart, lineEnd, "select");
    ta.focus();
    ta.dispatchEvent(new Event("input", { bubbles: true }));
  }

  /** Insert a block on its own lines at the cursor. */
  function insertBlock(ta, text) {
    const s = ta.selectionStart;
    const before = ta.value.slice(0, s);
    const lead = before.length === 0 || before.endsWith("\n\n") ? "" : before.endsWith("\n") ? "\n" : "\n\n";
    replaceSelection(ta, lead + text + "\n", lead.length, lead.length + text.length);
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
    return lines.map((l) => l.split("\t"));
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

  const IMAGE_TYPES = ["image/png", "image/jpeg", "image/gif", "image/webp", "application/pdf"];

  /** The entry this form edits, creating it first if it is brand new. */
  async function ensureEntry(form) {
    if (form.dataset.entry) return form.dataset.entry;
    const ta = form.querySelector(".ta");
    const name = form.querySelector(".who").value.trim() || who() || "unknown";
    const body = { day: DAY, book: BOOK, author: name, body_md: ta.value || "" };
    if (form.dataset.after !== undefined && form.dataset.after !== "") body.after = Number(form.dataset.after);
    else if (form.dataset.scan !== undefined && form.dataset.scan !== "") body.scan = Number(form.dataset.scan);
    const entry = await api("POST", "/entries", body);
    form.dataset.entry = entry.entry_id;
    form.dataset.version = String(entry.version);
    form.classList.add("autosaved");
    const discard = form.querySelector("[data-discard]");
    if (discard) discard.hidden = false;
    return entry.entry_id;
  }

  async function uploadFiles(form, files) {
    const ta = form.querySelector(".ta");
    const list = [...files].filter((f) => IMAGE_TYPES.includes(f.type));
    if (!list.length) return;
    clearFail(form);
    form.classList.add("busy");
    try {
      const id = await ensureEntry(form);
      for (const file of list) {
        const fd = new FormData();
        fd.append("file", file, file.name || "image.png");
        const res = await fetch(`${API}/entries/${id}/attachments`, { method: "POST", body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw Object.assign(new Error((data.detail && data.detail.message) || data.detail || res.statusText), { status: res.status });
        const alt = (file.name || "image").replace(/\.[^.]+$/, "");
        const md = file.type === "application/pdf" ? `[${alt}](${data.link})` : `![${alt}](${data.link})`;
        insertBlock(ta, md);
      }
      // Uploads bump the version; the eventual Save must carry the current one.
      const fresh = await api("GET", `/entries/${id}`);
      form.dataset.version = String(fresh.version);
    } catch (err) {
      fail(form, err);
    } finally {
      form.classList.remove("busy");
    }
  }

  // --------------------------------------------------------- preview

  async function togglePreview(form) {
    const ta = form.querySelector(".ta");
    let pane = form.querySelector(".preview");
    const btn = form.querySelector("[data-tool=preview]");
    if (pane && !pane.hidden) { pane.hidden = true; ta.hidden = false; btn.classList.remove("is-on"); return; }
    if (!pane) { pane = document.createElement("div"); pane.className = "entry-body preview"; ta.after(pane); }
    try {
      const data = await api("POST", "/preview", { body_md: ta.value, entry_id: form.dataset.entry || null });
      pane.innerHTML = data.html;
      pane.hidden = false; ta.hidden = true; btn.classList.add("is-on");
    } catch (err) { fail(form, err); }
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
      run: (ta) => insertBlock(ta, rowsToMarkdown([["Parameter", "Value", "Note"], ["", "", ""]])) },
    { key: "callout", label: "callout", title: "Callout — NOTE, TIP, WARNING or CAUTION",
      run: (ta) => insertBlock(ta, "> [!NOTE]\n> ") },
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
    picker.type = "file"; picker.accept = IMAGE_TYPES.join(","); picker.multiple = true; picker.hidden = true;
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
    const files = [...cd.items].filter((i) => i.kind === "file").map((i) => i.getAsFile()).filter(Boolean);
    if (files.length) { ev.preventDefault(); uploadFiles(form, files); return; }
    const html = cd.getData("text/html");
    const text = cd.getData("text/plain");
    const rows = (html && htmlTableToRows(html)) || (text && tsvToRows(text));
    if (rows) { ev.preventDefault(); insertBlock(form.querySelector(".ta"), rowsToMarkdown(rows)); }
  }

  function onDrop(form, ev) {
    if (!ev.dataTransfer || !ev.dataTransfer.files.length) return;
    ev.preventDefault(); form.classList.remove("dragover");
    uploadFiles(form, ev.dataTransfer.files);
  }

  // ------------------------------------------------------------ save

  async function save(form) {
    const ta = form.querySelector(".ta");
    const name = form.querySelector(".who").value.trim();
    if (!name) { form.querySelector(".who").focus(); return; }
    remember(name);
    clearFail(form);
    try {
      if (form.dataset.entry) {
        await api("PATCH", `/entries/${form.dataset.entry}`, {
          body_md: ta.value, editor: name, expected_version: Number(form.dataset.version) });
      } else {
        if (!ta.value.trim()) return;
        await ensureEntry(form);
      }
      location.reload();
    } catch (err) { fail(form, err); }
  }

  async function discard(form) {
    if (!form.dataset.entry) return;
    try { await api("DELETE", `/entries/${form.dataset.entry}`); location.reload(); }
    catch (err) { fail(form, err); }
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
    ta.addEventListener("keydown", (ev) => {
      if ((ev.metaKey || ev.ctrlKey) && ev.key === "Enter") { ev.preventDefault(); save(form); }
    });
  }

  document.querySelectorAll("form.composer").forEach(mount);

  // ------------------------------------------ entry buttons on the page

  document.addEventListener("click", async (ev) => {
    const b = ev.target.closest("button"); if (!b) return;
    if (b.closest("[data-insert]")) {
      const after = b.closest("[data-insert]").dataset.insert;
      const hostEl = document.querySelector(`[data-between-host="${after}"]`);
      if (hostEl) { hostEl.hidden = false; b.closest("[data-insert]").hidden = true; const ta = hostEl.querySelector(".ta"); if (ta) ta.focus(); }
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
        <span class="editor-hint">&#8984;&#8629; saves</span></div></div>`;
      form.querySelector(".ta").value = raw;
      bodyEl.hidden = true; bodyEl.after(form);
      mount(form);
      form.querySelector(".ta").focus();
      form.querySelector("[data-cancel]").addEventListener("click", () => { form.remove(); bodyEl.hidden = false; });
    }
  });
})();
