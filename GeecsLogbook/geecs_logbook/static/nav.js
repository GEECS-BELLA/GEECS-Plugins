/* GeecsLogbook navigation — the rail's calendar, keyboard stepping, and
 * hover prefetch. One file for both pages: the day page steps by day,
 * the month page by month; the calendar is the same grid on either.
 *
 * The page hands it its facts through <main id="logbook" data-api
 * data-day|data-month data-prev data-next data-today>; nothing is
 * templated into this file. Every URL the calendar builds is derived
 * from data-api, so a mount prefix or a proxy root_path carries through.
 *
 *   - Calendar: a <details class="cal"> in the rail. Opening it fetches
 *     /api/month/{m}/days — the store's per-day counts plus which day
 *     folders exist (one listing of the month folder on the server). The
 *     page itself never asks; only the open calendar does, so a slow
 *     share delays a popup, not a page. Months are cached per page load.
 *   - Keys: ← / → step to the previous / next day (or month), t goes to
 *     today. Ignored while typing in a field.
 *   - Prefetch: hovering a day or month link for a moment adds a
 *     <link rel="prefetch">, so the click that follows is served warm.
 *
 * Styling is entirely through the page's tokens; this file sets no colours.
 */
(function () {
  "use strict";

  const host = document.getElementById("logbook");
  if (!host) return;
  const API = host.dataset.api || "";
  const ROOT = API.replace(/\/api$/, "");
  const PAGE_DAY = host.dataset.day || "";        // day page: YYYY-MM-DD
  const PAGE_MONTH = host.dataset.month || "";    // month page: YYYY-MM
  const OPEN_KEY = "scanlog.calendarOpen";

  const pad = (n) => String(n).padStart(2, "0");
  const iso = (y, m, d) => `${y}-${pad(m)}-${pad(d)}`;
  const todayIso = () => { const t = new Date(); return iso(t.getFullYear(), t.getMonth() + 1, t.getDate()); };

  /** Where a calendar day leads: the day document, or its group on the month page. */
  function dayHref(day) {
    return PAGE_MONTH
      ? `${ROOT}/month/${day.slice(0, 7)}#day-${day}`
      : `${ROOT}/day/${day}`;
  }

  // ------------------------------------------------------------ calendar

  const cal = document.querySelector("details.cal");
  const body = cal && cal.querySelector(".calbody");
  const marks = new Map();  // "YYYY-MM" -> the /days payload (or null: failed)

  async function fetchMarks(month) {
    if (marks.has(month)) return marks.get(month);
    let got = null;
    try {
      const res = await fetch(`${API}/month/${month}/days`);
      if (res.ok) got = await res.json();
    } catch (e) { /* offline: draw the grid without marks */ }
    marks.set(month, got);
    return got;
  }

  function stepMonth(month, by) {
    const [y, m] = month.split("-").map(Number);
    const idx = y * 12 + (m - 1) + by;
    return `${Math.floor(idx / 12)}-${pad((idx % 12) + 1)}`;
  }

  const MONTHS = ["January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December"];

  function title(day, m) {
    const bits = [];
    if (m && m.folder) bits.push("scans on the share");
    if (m && m.notes) bits.push(`${m.notes} scan-log note${m.notes === 1 ? "" : "s"}`);
    if (m && m.ops) bits.push(`${m.ops} ops note${m.ops === 1 ? "" : "s"}`);
    return bits.length ? `${day} — ${bits.join(", ")}` : day;
  }

  async function draw(month) {
    if (!body) return;
    body.dataset.month = month;
    const [y, m] = month.split("-").map(Number);
    const first = new Date(y, m - 1, 1);
    const days = new Date(y, m, 0).getDate();
    const lead = (first.getDay() + 6) % 7;  // Monday first
    const today = todayIso();

    const head = document.createElement("div");
    head.className = "calhead";
    const prev = document.createElement("button");
    prev.type = "button"; prev.className = "stepday"; prev.textContent = "\u2039";
    prev.setAttribute("aria-label", "Previous month");
    const label = document.createElement("b");
    label.textContent = `${MONTHS[m - 1]} ${y}`;
    const next = document.createElement("button");
    next.type = "button"; next.className = "stepday"; next.textContent = "\u203a";
    next.setAttribute("aria-label", "Next month");
    prev.addEventListener("click", () => draw(stepMonth(month, -1)));
    next.addEventListener("click", () => draw(stepMonth(month, 1)));
    head.append(prev, label, next);

    const grid = document.createElement("div");
    grid.className = "calgrid";
    for (const wd of ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]) {
      const el = document.createElement("span"); el.className = "wd"; el.textContent = wd; grid.append(el);
    }
    for (let i = 0; i < lead; i++) {
      const el = document.createElement("span"); el.className = "cell blank"; grid.append(el);
    }
    const cells = new Map();
    for (let d = 1; d <= days; d++) {
      const day = iso(y, m, d);
      const a = document.createElement("a");
      a.className = "cell";
      a.href = dayHref(day);
      a.textContent = String(d);
      a.title = day;
      if (day === today) a.classList.add("is-today");
      if (day > today) a.classList.add("future");
      if (day === PAGE_DAY) a.classList.add("is-active");
      grid.append(a);
      cells.set(day, a);
    }

    const legend = document.createElement("div");
    legend.className = "callegend";
    legend.innerHTML = '<span><i class="k"></i>scans</span><span><i class="kn"></i>notes</span>';
    const note = document.createElement("div");
    note.className = "calnote";
    note.hidden = true;

    body.replaceChildren(head, grid, legend, note);

    const got = await fetchMarks(month);
    if (body.dataset.month !== month) return;  // the viewer moved on
    if (got === null) { note.textContent = "Marks unavailable."; note.hidden = false; return; }
    if (!got.share) { note.textContent = "Share not reachable: scan days unmarked."; note.hidden = false; }
    for (const [day, mk] of Object.entries(got.days || {})) {
      const a = cells.get(day);
      if (!a) continue;
      if (mk.folder) a.classList.add("has-folder");
      if (mk.notes || mk.ops) a.classList.add("has-notes");
      a.title = title(day, mk);
    }
  }

  if (cal && body) {
    const start = PAGE_DAY ? PAGE_DAY.slice(0, 7) : (PAGE_MONTH || todayIso().slice(0, 7));
    cal.addEventListener("toggle", () => {
      try { localStorage.setItem(OPEN_KEY, String(cal.open)); } catch (e) { /* private window */ }
      if (cal.open && !body.dataset.month) draw(start);
    });
    let remembered = null;
    try { remembered = localStorage.getItem(OPEN_KEY); } catch (e) { /* private window */ }
    if (remembered === "true") cal.open = true;  // fires toggle → draw
  }

  // ------------------------------------------------------------ keyboard

  const typing = (el) => !!el && (
    el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);

  document.addEventListener("keydown", (ev) => {
    if (ev.defaultPrevented || ev.metaKey || ev.ctrlKey || ev.altKey || typing(ev.target)) return;
    let url = "";
    if (ev.key === "ArrowLeft") url = host.dataset.prev;
    else if (ev.key === "ArrowRight") url = host.dataset.next;
    else if (ev.key === "t" || ev.key === "T") url = host.dataset.today;
    else if (ev.key === "c" || ev.key === "C") { if (cal) { cal.open = !cal.open; ev.preventDefault(); } return; }
    if (!url) return;
    ev.preventDefault();
    window.location.href = url;
  });

  // ------------------------------------------------------------ prefetch

  const warmed = new Set();
  const PAGES = new RegExp(`^${ROOT.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}/(day|month)/`);

  function warm(a) {
    let url;
    try { url = new URL(a.href, window.location.href); } catch (e) { return; }
    if (url.origin !== window.location.origin || !PAGES.test(url.pathname)) return;
    const key = url.pathname;  // the page, not its fragment
    if (warmed.has(key) || key === window.location.pathname) return;
    warmed.add(key);
    const link = document.createElement("link");
    link.rel = "prefetch";
    link.href = key;
    document.head.append(link);
  }

  let timer = 0;
  document.addEventListener("mouseover", (ev) => {
    const a = ev.target && ev.target.closest && ev.target.closest("a[href]");
    if (!a) return;
    clearTimeout(timer);
    timer = setTimeout(() => warm(a), 80);  // a pass-through hover costs nothing
  });
  document.addEventListener("mouseout", () => clearTimeout(timer));
  document.addEventListener("focusin", (ev) => {
    const a = ev.target && ev.target.closest && ev.target.closest("a[href]");
    if (a) warm(a);
  });
})();
