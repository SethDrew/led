import { chromium } from "playwright-core";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
await page.goto("http://localhost:8080/?sim=1");
await page.waitForFunction(() => window.__ledReady && window.__sim, { timeout: 20000 });

const out = await page.evaluate(() => {
  const S = window.__sim;
  const pushC = () =>
    window.LED3D.applyState({ roles: { clicky: { online: true, state: S.state } } });
  let enc = 0, kbtn = [0, 0, 0, 0, 0], up = 0;
  const pushK = () => window.LED3D.applyState(
    { roles: { kettle: { online: true, state: { c: enc, btn: kbtn, up: up += 40 } } } });
  let t = 0;
  const step = (dt) => { t += dt; pushC(); pushK(); window.__setClock(t); };

  const redPeakAndCentroid = () => {
    const c = window.__dumpColors();
    let peak = 0, num = 0, den = 0;
    for (let i = 0; i < 297; i++) {
      const r = c[i * 3], g = c[i * 3 + 1], b = c[i * 3 + 2];
      if (r > 0.03 && r > 2.5 * g && r > 2.5 * b) {
        if (r > peak) peak = r;
        num += i * r; den += r;
      }
    }
    return { peak: +peak.toFixed(3), centroid: den ? +(num / den).toFixed(1) : null };
  };

  [0.5, 0.3, 0.7, 0.4, 0.6, 1.0].forEach((v, k) => S.set(k, v));
  for (let i = 0; i < 25; i++) step(0.04);

  // Record a short red sweep, then park red far away at 0.05.
  S.btn(1);
  for (let i = 0; i < 30; i++) { S.set(0, 0.5 + 0.3 * i / 29); step(0.04); }
  S.btn(0);
  S.set(0, 0.05);
  const fade = [];
  const sampleAt = [2, 10, 20, 29, 31];
  let age = 0;
  for (const target of sampleAt) {
    while (age < target) { step(0.04); age += 0.04; }
    fade.push({ age: target, ...redPeakAndCentroid() });
  }

  // Spin gain: one full knob revolution should now shift the frame HALF a ring.
  const before = redPeakAndCentroid();
  enc += 60;                      // 2 counts/detent x 30 detents = 1 rev
  for (let i = 0; i < 50; i++) step(0.04);
  const spun = redPeakAndCentroid();

  // Knob press -> integrals reset -> frame back to unrotated.
  kbtn[0] = 1; step(0.04); step(0.04);
  kbtn[0] = 0;
  for (let i = 0; i < 30; i++) step(0.04);
  const reset = redPeakAndCentroid();

  return { fade, before, spun, reset };
});

console.log("ghost fade (red peak vs age):");
for (const f of out.fade) console.log(`  age ${f.age}s  peak=${f.peak}  centroid=${f.centroid}`);
console.log("spin gain: live red centroid before spin =", out.before.centroid,
  "| after 1 knob rev =", out.spun.centroid, "(297-LED ring, half = ~148 shift)");
console.log("knob-press reset: centroid =", out.reset.centroid,
  "(should match 'before')");
await browser.close();
