import { chromium } from "playwright-core";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
await page.goto("http://localhost:8080/?sim=1");
await page.waitForFunction(() => window.__ledReady && window.__sim, { timeout: 20000 });

const out = await page.evaluate(() => {
  const S = window.__sim;
  let t = 0;
  const step = (dt) => {
    t += dt;
    window.LED3D.applyState({ roles: { clicky: { online: true, state: S.state } } });
    window.__setClock(t);
  };
  const purpleBlobs = () => {
    const c = window.__dumpColors();
    const px = [];
    // strap strips start at LED 700 (14 canopy strips x 50); use strip 0
    // (canopy_0, 50 LEDs) — small, but blob centroids in [0,50) still separate.
    // Better: scan the whole first canopy spoke ring group? Use LEDs 0..49.
    for (let i = 0; i < 50; i++) {
      const r = c[i * 3], g = c[i * 3 + 1], b = c[i * 3 + 2];
      if (b > 0.03 && r > 0.3 * b && g < 0.5 * b) px.push({ i, v: b });
    }
    const blobs = [];
    let cur = null;
    for (const p of px) {
      if (cur && p.i - cur.end <= 2) {
        cur.end = p.i; cur.num += p.i * p.v; cur.den += p.v;
      } else {
        cur = { start: p.i, end: p.i, num: p.i * p.v, den: p.v };
        blobs.push(cur);
      }
    }
    return blobs.map(bl => +(bl.num / bl.den).toFixed(1));
  };

  [0.5, 0.5, 0.5, 0.5, 0.2, 1.0].forEach((v, k) => S.set(k, v));
  for (let i = 0; i < 30; i++) step(0.04);

  // Gesture 1 on button E (bit 4): purple sweeps low, 0.2 -> 0.35.
  S.btn(1 << 4);
  for (let i = 0; i < 30; i++) { S.set(4, 0.2 + 0.15 * i / 29); step(0.04); }
  S.btn(0);
  // Gesture 2 on button F (bit 5): purple sweeps high, 0.7 -> 0.85.
  S.set(4, 0.7);
  for (let i = 0; i < 15; i++) step(0.04);
  S.btn(1 << 5);
  for (let i = 0; i < 30; i++) { S.set(4, 0.7 + 0.15 * i / 29); step(0.04); }
  S.btn(0);
  // Park live purple in the middle, away from both ghosts.
  S.set(4, 0.5);
  for (let i = 0; i < 30; i++) step(0.04);

  const seen = new Set();
  for (let j = 0; j < 12; j++) {
    for (let i = 0; i < 6; i++) step(0.04);
    for (const cn of purpleBlobs()) seen.add(Math.round(cn / 3) * 3);
  }
  // Tap button F: only its ghost clears.
  S.btn(1 << 5); step(0.04); step(0.04); S.btn(0);
  for (let i = 0; i < 15; i++) step(0.04);
  const afterTap = new Set();
  for (let j = 0; j < 12; j++) {
    for (let i = 0; i < 6; i++) step(0.04);
    for (const cn of purpleBlobs()) afterTap.add(Math.round(cn / 3) * 3);
  }
  return { both: [...seen].sort((a, b) => a - b),
           afterTapF: [...afterTap].sort((a, b) => a - b) };
});

console.log("purple blob centroids (bucketed, live at ~24):", out.both);
console.log("after tapping button F (high ghost should vanish):", out.afterTapF);
await browser.close();
