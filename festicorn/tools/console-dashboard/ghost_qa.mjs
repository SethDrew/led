import { chromium } from "playwright-core";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
await page.goto("http://localhost:8080/?sim=1");
await page.waitForFunction(() => window.__ledReady && window.__sim, { timeout: 20000 });

const result = await page.evaluate(() => {
  const push = () =>
    window.LED3D.applyState({ roles: { clicky: { online: true, state: window.__sim.state } } });
  let t = 0;
  const step = (dt) => { t += dt; push(); window.__setClock(t); };

  window.__sim.set(0, 0.15);
  for (let i = 0; i < 25; i++) step(0.04);

  window.__sim.btn(1);
  for (let i = 0; i < 40; i++) { window.__sim.set(0, 0.15 + 0.7 * i / 39); step(0.04); }
  window.__sim.btn(0);

  window.__sim.set(0, 0.15);
  for (let i = 0; i < 25; i++) step(0.04);

  const samples = [];
  for (let j = 0; j < 6; j++) {
    for (let i = 0; i < 10; i++) step(0.04);
    samples.push({ t, colors: window.__dumpColors().slice(0, 297 * 3) });
  }
  return samples;
});

for (const s of result) {
  const px = [];
  for (let i = 0; i < 297; i++) {
    const r = s.colors[i * 3], g = s.colors[i * 3 + 1], b = s.colors[i * 3 + 2];
    if (r > 0.02 && r > 2.5 * g && r > 2.5 * b) px.push({ i, r: +r.toFixed(3) });
  }
  const blobs = [];
  let cur = null;
  for (const p of px) {
    if (cur && p.i - cur.end <= 2) { cur.end = p.i; cur.peak = Math.max(cur.peak, p.r); }
    else { cur = { start: p.i, end: p.i, peak: p.r }; blobs.push(cur); }
  }
  console.log(`t=${s.t.toFixed(2)}s red blobs:`,
    blobs.map(b => `[${b.start}-${b.end}] peak=${b.peak.toFixed(3)}`).join("  ") || "none");
}
await browser.close();
