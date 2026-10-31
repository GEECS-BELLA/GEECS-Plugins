// A device page's live half, DOM only. The picker's states from /api/device/<kind>
// (the module's roster, polled); a chosen device opens its page. On a device
// page, each role row's PV and verdict from the module's device route (polled)
// and its value from Hare; the header's state and heartbeat are the module's.
// On a stale page (the module silent) the device route is polled until it
// answers, and the page reloads to show what it says. Read only. The rules are
// device_rules.js's and screen_rules.js's (tested under node), the polling poller.js's.
(function () {
  "use strict";
  const POLL_MS = 2000, ROSTER_MS = 5000;
  const root = document.body.dataset.root || "";
  const main = document.querySelector("main.device");
  const kind = main.dataset.kind, name = main.dataset.device;
  const KIT = JSON.parse(document.getElementById("kit-state").textContent);
  const D = window.GeecsDeviceRules, S = window.GeecsScreenRules;
  const open = (device) => { window.location.href = `${root}/device/${kind}/${encodeURIComponent(device)}`; };

  // ---- the picker (and the device page's switcher)
  const pick = document.getElementById("pick");
  if (pick) {
    pick.addEventListener("change", () => { if (pick.value) open(pick.value); });
    const roster = async () => {
      try {
        const r = await fetch(`${root}/api/device/${kind}`);
        const body = await r.json();
        pick.replaceChildren(...D.pickerOptions(body.devices || [], pick.dataset.current, kind).map((o) => new Option(o.text, o.value, o.selected, o.selected)));
        document.getElementById("roster-note").textContent = D.pickerNote(body);
        for (const d of body.devices || []) { // the picker page's table: the module's word beside each
          const tr = document.querySelector(`#roster tr[data-name="${CSS.escape(d.name)}"]`);
          if (!tr) continue;
          tr.querySelector(".dot").dataset.state = d.kit;
          tr.querySelector(".word").textContent = d.state;
        }
      } catch (_) { document.getElementById("roster-note").textContent = "the Control Room did not answer"; }
    };
    roster();
    setInterval(roster, ROSTER_MS);
  }
  if (!name) return;

  // a 5xx is the module silent (back off); any other answer is the module's word
  async function readDevice(device, signal) {
    const r = await fetch(`${root}/api/devices/${encodeURIComponent(device)}`, { signal });
    if (r.status >= 500) throw new Error(`answered ${r.status}`);
    return D.deviceView(r.ok, await r.json());
  }

  if (main.dataset.recover) { // stale: ask until the module answers, then show its word
    const poller = window.GeecsPoller.makePoller({ fetch: readDevice, pollMs: POLL_MS, onView: (_, got) => { if (got.ok) window.location.reload(); } });
    poller.tick([name]);
    setInterval(() => poller.tick([name]), POLL_MS);
    return;
  }
  const rows = [...document.querySelectorAll("tr[data-role]")];
  if (!rows.length) return;

  const hare = window.GeecsHare.fromPage();
  const wire = {}, users = {};
  let view = null, hareState = "connecting";
  const entryOf = (tr) => view && view.roles[S.key(tr.dataset.role, tr.dataset.axis)];

  function draw(tr) {
    const entry = entryOf(tr), pv = D.watched(entry) ? entry.pv : null;
    const word = D.rowWord({ hareUp: hareState === "open", frame: pv && wire[pv], view, entry });
    tr.querySelector(".ws .dot").dataset.state = KIT[word] || "unknown";
    tr.querySelector(".ws .word").textContent = word;
    tr.querySelector(".pv").textContent = (entry && entry.pv) || "—";
    const meta = pv ? hare.meta(pv) : {};
    tr.querySelector(".v").textContent = S.show(pv && wire[pv], meta, null);
    tr.querySelector(".u").textContent = D.units(tr.dataset.type, tr.dataset.units, meta);
  }

  function use(pv, tr) {
    if (!pv || (users[pv] && users[pv].has(tr))) return;
    (users[pv] = users[pv] || new Set()).add(tr);
    hare.watch([pv], (pvName, frame) => { wire[pvName] = frame; users[pvName].forEach(draw); });
  }

  function seen(_, got) {
    view = got.ok ? got.value : { ok: false, roles: {} }; // silent: every word is unknown
    const h = D.header(view), chip = document.getElementById("device-state");
    chip.textContent = h.state;
    chip.dataset.state = KIT[h.state] || "unknown";
    document.getElementById("heartbeat").textContent = h.heartbeat;
    for (const tr of rows) {
      const entry = entryOf(tr);
      if (D.watched(entry)) use(entry.pv, tr);
      draw(tr);
    }
  }

  hare.onStatus((s) => {
    hareState = s;
    const line = S.sourceLine(s, []), host = document.getElementById("hare-state");
    host.textContent = line.text;
    document.querySelector(".hare .dot").dataset.state = line.kit;
    rows.forEach(draw);
  });
  const poller = window.GeecsPoller.makePoller({ fetch: readDevice, pollMs: POLL_MS, onView: seen });
  poller.tick([name]);
  setInterval(() => poller.tick([name]), POLL_MS);
})();
