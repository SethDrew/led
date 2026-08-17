import { chromium } from "playwright-core";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
await page.goto("http://localhost:8080/?sim=1");
await page.waitForFunction(() => window.__ledReady && window.__sim, { timeout: 20000 });

const result = await page.evaluate(() => {
  const S = window.__sim;
  const push = () =>
    window.LED3D.applyState({ roles: { clicky: { online: true, state: S.state } } });
  let t = 0;
  const step = (dt) => { t += dt; push(); window.__setClock(t); };

  [0.2, 0.5, 0.8, 0.35, 0.6].forEach((v, k) => S.set(k, v));
  S.set(5, 1.0);
  for (let i = 0; i < 25; i++) step(0.04);

  // Phase A: pots 0 and 2 recorded SIMULTANEOUSLY, different gestures.
  S.btn(0b00101);
  for (let i = 0; i < 50; i++) {
    S.set(0, 0.2 + 0.6 * i / 49);                       // slow sweep up
    S.set(2, 0.8 + 0.15 * Math.sin(i / 49 * 6 * Math.PI)); // 3 fast wiggles
    step(0.04);
  }
  S.btn(0);

  // Phase B: pot 3 recorded while being pulled out and pushed back in.
  S.btn(0b01000);
  for (let i = 0; i < 45; i++) {
    S.set(3, 0.35 + 0.5 * i / 44);
    if (i === 12) S.pull(0b01000);
    if (i === 32) S.pull(0);
    step(0.04);
  }
  S.btn(0);

  // Phase C: pot 4 held stationary 0.6 s -> parked ghost.
  S.btn(0b10000);
  for (let i = 0; i < 15; i++) step(0.04);
  S.btn(0);

  // Park live particles away from the action.
  [0.05, 0.5, 0.05, 0.05, 0.95].forEach((v, k) => S.set(k, v));
  for (let i = 0; i < 25; i++) step(0.04);

  // Live pull on pot 1 while ghosts replay (burst + comb amid ghosts),
  // then push back in and settle so the sampled frames show ghosts only.
  S.pull(0b00010);
  for (let i = 0; i < 20; i++) step(0.04);
  S.pull(0);
  for (let i = 0; i < 30; i++) step(0.04);

  const samples = [];
  for (let j = 0; j < 6; j++) {
    for (let i = 0; i < 12; i++) step(0.04);
    samples.push({ t: +t.toFixed(2), colors: window.__dumpColors().slice(0, 297 * 3) });
  }
  return samples;
});

const hueOf = (r, g, b) => {
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  if (d < 1e-6) return -1;
  let h;
  if (mx === r) h = ((g - b) / d) % 6;
  else if (mx === g) h = (b - r) / d + 2;
  else h = (r - g) / d + 4;
  return (h * 60 + 360) % 360;
};
const family = (h) => h < 0 ? "grey"
  : h < 30 || h >= 340 ? "RED(p0)" : h < 90 ? "YEL(p1)" : h < 150 ? "GRN(p2)"
  : h < 215 ? "CYN(p3)" : h < 290 ? "PUR(p4)" : "MAG";

for (const s of result) {
  const px = [];
  for (let i = 0; i < 297; i++) {
    const r = s.colors[i * 3], g = s.colors[i * 3 + 1], b = s.colors[i * 3 + 2];
    const v = Math.max(r, g, b);
    if (v > 0.05) px.push({ i, v, h: hueOf(r, g, b) });
  }
  const blobs = [];
  let cur = null;
  for (const p of px) {
    if (cur && p.i - cur.end <= 2) {
      cur.end = p.i;
      if (p.v > cur.peak) { cur.peak = p.v; cur.h = p.h; }
    } else { cur = { start: p.i, end: p.i, peak: p.v, h: p.h }; blobs.push(cur); }
  }
  console.log(`t=${s.t}s  ` + blobs.map(b =>
    `${family(b.h)}[${b.start}-${b.end}]p${b.peak.toFixed(2)}`).join("  "));
}

await page.locator("#viewerbox").screenshot({ path: "shots/ghost_multi.png" });
await browser.close();
console.log("screenshot: shots/ghost_multi.png");
