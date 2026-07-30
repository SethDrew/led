// shoot.mjs — headless screenshots of the Axis Mundi dashboard.
//
// The AI-loop hook: drive the same page a human uses, headless, so an agent can
// capture the 3D sculpture panel and reason about frames without hardware.
// Uses the system Google Chrome via channel:'chrome' (no browser download).
//
//   node shoot.mjs [--url URL] [--t SECONDS] [--out shots]
//
// Page hooks (installed by viz/led3d.mjs):
//   window.__ledReady          true once the render loop produces frames
//   window.__setClock(t)       pause and render the effect at time t
//   window.__ledPattern = fn   overwrite each frame's colors (display QA)
//   window.__dumpColors()      current linear RGB per LED, pre-gain
//   window.__setGlow(h,c,r)    per-section glow diameter (m)

import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const arg = (k, d) => {
  const i = process.argv.indexOf(k);
  return i >= 0 ? process.argv[i + 1] : d;
};
const URL = arg("--url", "http://localhost:8080/");
const T = parseFloat(arg("--t", "0"));
const OUT = arg("--out", "shots");
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ channel: "chrome", args: ["--use-gl=angle"] });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
page.on("pageerror", (e) => console.error("PAGE ERROR:", e.message));
page.on("console", (m) => { if (m.type() === "error") console.error("CONSOLE:", m.text()); });

await page.goto(URL, { waitUntil: "domcontentloaded" });
await page.waitForFunction(() => window.__ledReady === true, { timeout: 15000 });

if (T > 0) await page.evaluate((t) => window.__setClock(t), T);
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/dashboard_t${T}.png` });

const box = await page.locator("#viewerbox").boundingBox();
if (box) await page.screenshot({ path: `${OUT}/sculpture_t${T}.png`, clip: box });

const n = await page.evaluate(() => window.LED3D?.nLeds);
console.log(`captured t=${T}s · ${n} LEDs → ${OUT}/`);
await browser.close();
