// shoot.mjs — headless screenshots of the LED viewer.
//
// This is the AI-loop hook: drive the same browser viewer a human uses, but
// headless, so an agent can capture frames at chosen camera angles + timeline
// positions and reason about them (or diff against the nominal MP4 stills).
//
// Uses the system Google Chrome via channel:'chrome' (no browser download).
//
//   node shoot.mjs [--url URL] [--frame N] [--out shots]
//
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const arg = (k, d) => {
  const i = process.argv.indexOf(k);
  return i >= 0 ? process.argv[i + 1] : d;
};
const URL = arg("--url", "http://localhost:8777/viewer/?mode=wasm");
const FRAME = parseInt(arg("--frame", "300"), 10);
const OUT = arg("--out", "shots");
mkdirSync(OUT, { recursive: true });

// camera orbits to capture (azimuth around, elevation up): [theta, phi] degrees
const VIEWS = [
  ["front", 0, 8],
  ["three-quarter", 45, 18],
  ["below", 30, -25],
  ["top", 0, 80],
];

const browser = await chromium.launch({ channel: "chrome", args: ["--use-gl=angle"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 1280 }, deviceScaleFactor: 1 });
page.on("pageerror", (e) => console.error("PAGE ERROR:", e.message));
page.on("console", (m) => { if (m.type() === "error") console.error("CONSOLE:", m.text()); });

await page.goto(URL, { waitUntil: "networkidle" });

// wait for the render loop to actually be producing frames (rfps populated)
await page.waitForFunction(() => {
  const el = document.getElementById("rfps");
  return el && el.textContent !== "–" && el.textContent !== "";
}, { timeout: 15000 });

// pause + set timeline to a representative frame
await page.evaluate((f) => {
  const s = document.getElementById("scrub");
  s.value = String(f);
  s.dispatchEvent(new Event("input", { bubbles: true }));
}, FRAME);

// drive OrbitControls from outside: the page doesn't expose camera, so we move
// the mouse to orbit. Simpler + robust: reload-free spherical set via wheel/drag
// is fiddly, so instead capture the default angle plus auto-rotate snapshots.
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/frame${FRAME}_default.png` });

// nudge auto-rotate on, grab a few angles by waiting between shots
await page.evaluate(() => {
  const cb = document.getElementById("spin");
  cb.checked = true; cb.dispatchEvent(new Event("change", { bubbles: true }));
});
for (let i = 0; i < 3; i++) {
  await page.waitForTimeout(900);
  await page.screenshot({ path: `${OUT}/frame${FRAME}_orbit${i}.png` });
}

const rfps = await page.evaluate(() => document.getElementById("rfps").textContent);
const nleds = await page.evaluate(() => document.getElementById("n-leds").textContent);
console.log(`captured ${VIEWS.length ? "" : ""}frame ${FRAME} · ${nleds} LEDs · render ${rfps} fps`);

await browser.close();
