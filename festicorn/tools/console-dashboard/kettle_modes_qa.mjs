import { chromium } from "playwright-core";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
await page.goto("http://localhost:8080/?sim=1");
await page.waitForFunction(() => window.__ledReady && window.__sim, { timeout: 20000 });

const out = await page.evaluate(() => {
  const S = window.__sim;
  let enc = 0, kbtn = [0, 0, 0, 0, 0], up = 0, t = 0;
  const step = (dt) => {
    t += dt;
    window.LED3D.applyState({ roles: {
      clicky: { online: true, state: S.state },
      kettle: { online: true, state: { c: enc, btn: kbtn, up: up += 40 } },
    } });
    window.__setClock(t);
  };
  const pulse = (bit) => {
    kbtn[bit] = 1; step(0.04); step(0.04); kbtn[bit] = 0; step(0.04);
  };
  const redBlobs = () => {
    const c = window.__dumpColors();
    const px = [];
    for (let i = 0; i < 297; i++) {
      const r = c[i * 3], g = c[i * 3 + 1], b = c[i * 3 + 2];
      if (r > 0.04 && r > 2.5 * g && r > 2.5 * b) px.push({ i, r });
    }
    const blobs = [];
    let cur = null;
    for (const p of px) {
      if (cur && p.i - cur.end <= 3) {
        cur.end = p.i; cur.num += p.i * p.r; cur.den += p.r;
        if (p.r > cur.peak) cur.peak = p.r;
      } else {
        cur = { start: p.i, end: p.i, peak: p.r, num: p.i * p.r, den: p.r };
        blobs.push(cur);
      }
    }
    return {
      count: blobs.length,
      lit: px.length,
      blobs: blobs.map(b => ({
        c: +(b.num / b.den).toFixed(1), peak: +b.peak.toFixed(2),
        span: b.end - b.start + 1 })),
    };
  };

  [0.1, 0.45, 0.6, 0.75, 0.9, 1.0].forEach((v, k) => S.set(k, v));
  for (let i = 0; i < 40; i++) step(0.04);
  const R = {};

  R.rotBase = redBlobs();
  for (let i = 0; i < 30; i++) { enc += 2; step(0.04); }
  for (let i = 0; i < 10; i++) step(0.04);
  R.rotSpun = redBlobs();
  for (let i = 0; i < 300; i++) step(0.04);
  R.rotLeaked = redBlobs();

  pulse(3);
  for (let i = 0; i < 10; i++) step(0.04);
  R.dupBase = redBlobs();
  for (let i = 0; i < 25; i++) { enc += 2; step(0.04); }
  R.dupSpinning = redBlobs();
  for (let i = 0; i < 100; i++) step(0.04);
  R.dupFaded = redBlobs();

  pulse(4);
  for (let i = 0; i < 10; i++) step(0.04);
  R.combBase = redBlobs();
  for (let i = 0; i < 50; i++) { enc += 3; step(0.04); }
  R.combSpinning = redBlobs();
  for (let i = 0; i < 100; i++) step(0.04);
  R.combRelaxed = redBlobs();

  pulse(1);
  for (let i = 0; i < 10; i++) step(0.04);
  R.backToRot = redBlobs();
  return R;
});

for (const [k, v] of Object.entries(out))
  console.log(k.padEnd(13), `blobs=${v.count} lit=${v.lit}`,
    v.blobs.map(b => `[c${b.c} p${b.peak} w${b.span}]`).join(" "));
await browser.close();
