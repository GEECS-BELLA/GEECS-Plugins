// CDP over node's WebSocket: the camera page in raw mode, log scale, auto levels, 1:1, then 16x on the hot pixel.
import fs from "node:fs";
const [port, cdp, outDir] = process.argv.slice(2);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const targets = await (await fetch(`http://127.0.0.1:${cdp}/json/list`)).json();
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0; const pending = new Map();
ws.addEventListener("message", (e) => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
const call = (method, params = {}) => new Promise((r) => { const n = ++id; pending.set(n, r); ws.send(JSON.stringify({ id: n, method, params })); });
const js = async (expr) => { const r = await call("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true }); if (r.result.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails)); return r.result.result.value; };
const shot = async (name) => { const r = await call("Page.captureScreenshot", { format: "png" }); fs.writeFileSync(`${outDir}/${name}.png`, Buffer.from(r.result.data, "base64")); };
await call("Page.enable"); await call("Runtime.enable");
await call("Emulation.setDeviceMetricsOverride", { width: 1500, height: 1180, deviceScaleFactor: 1, mobile: false });
for (const scheme of ["light", "dark"]) {
  await call("Emulation.setEmulatedMedia", { features: [{ name: "prefers-color-scheme", value: scheme }] });
  await call("Page.navigate", { url: `http://127.0.0.1:${port}/device/camera/Cam1` });
  await sleep(1500);
  await js("localStorage.clear(); location.reload(); 1");
  await sleep(2500);
  await js(`(async () => {
    const $ = (id) => document.getElementById(id);
    for (let i = 0; i < 50 && $("raw").disabled; i++) await new Promise((r) => setTimeout(r, 100));
    $("raw").click();
    $("scale").value = "log"; $("scale").dispatchEvent(new Event("change"));
    $("auto-live").checked = true; $("auto-live").dispatchEvent(new Event("change"));
    await new Promise((r) => setTimeout(r, 1500));
    $("auto-once").click();
    $("zoom-one").click();
    return [$("level-lo").value, $("level-hi").value, $("levels-scale").textContent, $("raw-note").textContent, $("fps").textContent, $("image-error").hidden];
  })()`).then((v) => console.log(scheme, JSON.stringify(v)));
  await sleep(600);
  // the pointer on the hot pixel: searched by the cursor readout (step px, ±span) around a guess
  const find = (gx, gy, step, span) => js(`(() => { const v = document.getElementById("viewport"), r = v.getBoundingClientRect();
    for (let dy = -${span}; dy <= ${span}; dy += ${step}) for (let dx = -${span}; dx <= ${span}; dx += ${step}) {
      const x = ${gx} + dx, y = ${gy} + dy;
      v.dispatchEvent(new PointerEvent("pointermove", { clientX: x, clientY: y, bubbles: true }));
      if (/counts 4000$/.test(document.getElementById("cursor").textContent)) return [x, y];
    }
    return null; })()`);
  const r = await js(`(() => { const r = document.getElementById("viewport").getBoundingClientRect(); return [r.left + r.width / 2 + 112.5, r.top + r.height / 2 - 112.5]; })()`);
  const at = await find(r[0], r[1], 1, 3);
  console.log(scheme, "hot pixel at", at);
  await sleep(300);
  await shot(`raw_auto_log_1to1_${scheme}`);
  // the black from the auto levels, the white at the full scale: the floor as snow, the hot pixel alone white
  await js(`(() => { const $ = (id) => document.getElementById(id);
    $("auto-live").checked = false; $("auto-live").dispatchEvent(new Event("change"));
    $("level-hi").value = 4095; $("level-hi").dispatchEvent(new Event("change")); 1 })()`);
  await find(at[0], at[1], 1, 0);
  await sleep(300);
  await shot(`raw_floor_to_full_log_1to1_${scheme}`);
  await js(`(() => { const v = document.getElementById("viewport"), [x, y] = ${JSON.stringify(at)};
    v.dispatchEvent(new WheelEvent("wheel", { clientX: x, clientY: y, deltaY: -Math.log(16) / 0.0015, bubbles: true, cancelable: true })); 1 })()`);
  await sleep(300);
  console.log(scheme, "zoomed, hot pixel at", await find(at[0], at[1], 4, 48));
  await sleep(300);
  await shot(`raw_floor_to_full_log_16x_${scheme}`);
}
ws.close();
