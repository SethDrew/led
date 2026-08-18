import { chromium } from "playwright-core";

const browser = await chromium.launch({ channel: "chrome", headless: true });
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
await page.goto("http://localhost:8080/?sim=1");
await page.waitForFunction(() => window.__ledReady && window.__sim, { timeout: 20000 });

const n = await page.evaluate(() => {
  const S = window.__sim;
  let t = 0;
  const step = (dt) => {
    t += dt;
    window.LED3D.applyState({ roles: { clicky: { online: true, state: S.state } } });
    window.__setClock(t);
  };
  [0.15, 0.35, 0.55, 0.75, 0.9, 1.0].forEach((v, k) => S.set(k, v));
  S.pull(0b00100);
  for (let i = 0; i < 60; i++) step(0.04);
  return window.__dumpColors().length / 3;
});
console.log("LED count in engine:", n);
await page.waitForTimeout(400);
await page.locator("#viewerbox").screenshot({ path: "shots/strap_shape.png" });
await browser.close();
console.log("screenshot: shots/strap_shape.png");
