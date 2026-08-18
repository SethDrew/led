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
  const red = () => {
    const c = window.__dumpColors();
    let n = 0, num = 0, den = 0, lo = 999, hi = -1;
    for (let i = 0; i < 50; i++) {
      const r = c[i * 3], g = c[i * 3 + 1], b = c[i * 3 + 2];
      if (r > 0.04 && r > 2.5 * g && r > 2.5 * b) {
        n++; num += i * r; den += r;
        if (i < lo) lo = i; if (i > hi) hi = i;
      }
    }
    return { n, c: den ? +(num / den).toFixed(1) : null, span: n ? hi - lo : 0 };
  };

  [0.5, 0.02, 0.98, 0.02, 0.98, 1.0].forEach((v, k) => S.set(k, v));
  for (let i = 0; i < 40; i++) step(0.04);
  const R = {};
  R.base = red();

  // SPLAY held: spin one knob rev -> teeth part way; keep spinning -> further.
  kbtn[4] = 1;
  for (let i = 0; i < 30; i++) { enc += 2; step(0.04); }
  R.splay1rev = red();
  for (let i = 0; i < 30; i++) { enc += 2; step(0.04); }
  R.splay2rev = red();
  for (let i = 0; i < 30; i++) { enc += 2; step(0.04); }
  R.splay3rev = red();
  // Dial back two revs -> spacing shrinks toward original.
  for (let i = 0; i < 60; i++) { enc -= 2; step(0.04); }
  for (let i = 0; i < 10; i++) step(0.04);
  R.dialback = red();
  // Spin out again then RELEASE -> drift home.
  for (let i = 0; i < 60; i++) { enc += 2; step(0.04); }
  R.beforeRelease = red();
  kbtn[4] = 0;
  for (let i = 0; i < 100; i++) step(0.04);
  R.released = red();

  // Frame must not have rotated during all that: centroid ~= base.
  R.frameAfterSplay = red();

  // DUP held: spin half a knob rev -> second red blob; release -> fades.
  kbtn[3] = 1;
  for (let i = 0; i < 25; i++) { enc += 2; step(0.04); }
  const c2 = (() => {
    const c = window.__dumpColors();
    const px = [];
    for (let i = 0; i < 50; i++) {
      const r = c[i * 3], g = c[i * 3 + 1], b = c[i * 3 + 2];
      if (r > 0.04 && r > 2.5 * g && r > 2.5 * b) px.push(i);
    }
    const blobs = [];
    let cur = null;
    for (const i of px) {
      if (cur && i - cur.end <= 2) cur.end = i;
      else { cur = { start: i, end: i }; blobs.push(cur); }
    }
    return blobs.map(b => Math.round((b.start + b.end) / 2));
  })();
  R.dupSpinningBlobs = c2;
  kbtn[3] = 0;
  for (let i = 0; i < 100; i++) step(0.04);
  R.dupFaded = red();

  // Bare spin still rotates the frame.
  for (let i = 0; i < 30; i++) { enc += 2; step(0.04); }
  R.bareSpun = red();
  return R;
});

for (const [k, v] of Object.entries(out)) console.log(k.padEnd(15), JSON.stringify(v));
await browser.close();
