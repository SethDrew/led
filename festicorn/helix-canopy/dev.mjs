// dev.mjs — one-command hot-reload dev server for the LED IDE.
//
//   node dev.mjs            # then open http://localhost:8777/viewer/
//
// Serves the helix-canopy tree (so /viewer/ and /wasm/ both resolve), watches
// the effect + geometry sources, rebuilds the WASM engine (and regenerates
// geometry when geometry_gen.py changes) on save, and pushes a live-reload to
// the browser over SSE. Build errors are streamed to the viewer's error overlay
// instead of silently failing. Plain Node, no dependencies.

import http from "node:http";
import fs from "node:fs";
import crypto from "node:crypto";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));   // helix-canopy/
const PORT = 8777;

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json",
  ".css": "text/css", ".bin": "application/octet-stream", ".png": "image/png",
};

// ── SSE live-reload channel ────────────────────────────────────────────────
const clients = new Set();
function broadcast(event, data) {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const res of clients) res.write(payload);
}

// ── static file server + SSE endpoint ──────────────────────────────────────
const server = http.createServer((req, res) => {
  const url = decodeURIComponent(req.url.split("?")[0]);

  if (url === "/__livereload") {
    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    });
    res.write("retry: 1000\n\n");
    clients.add(res);
    req.on("close", () => clients.delete(res));
    return;
  }

  let rel = url === "/" ? "/index.html" : url;
  let file = path.join(ROOT, rel);
  if (!file.startsWith(ROOT)) { res.writeHead(403).end(); return; }    // no escape
  fs.stat(file, (err, st) => {
    if (err) { res.writeHead(404).end("not found"); return; }
    if (st.isDirectory()) file = path.join(file, "index.html");
    fs.readFile(file, (e, buf) => {
      if (e) { res.writeHead(404).end("not found"); return; }
      res.writeHead(200, {
        "Content-Type": MIME[path.extname(file)] || "application/octet-stream",
        "Cache-Control": "no-store",                                   // always fresh in dev
      });
      res.end(buf);
    });
  });
});

// ── build pipeline ─────────────────────────────────────────────────────────
function run(cmd, args, cwd) {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { cwd });
    let out = "";
    p.stdout.on("data", (d) => (out += d));
    p.stderr.on("data", (d) => (out += d));
    p.on("close", (code) => resolve({ code, out }));
  });
}

const WATCH = [
  "src/effects.h",
  "wasm/engine.cpp",
  "wasm/build.sh",
  "tools/geometry_gen.py",
];

// Content hash of all watched files — the gate that ignores no-op touches (an
// editor autosave/format watcher re-stamps mtimes without changing bytes, which
// would otherwise rebuild forever).
function hashWatched() {
  const h = crypto.createHash("sha1");
  for (const rel of WATCH) {
    try { h.update(rel); h.update(fs.readFileSync(path.join(ROOT, rel))); } catch {}
  }
  return h.digest("hex");
}

let building = false;
let lastHash = hashWatched();
async function maybeBuild(reason) {
  const h = hashWatched();
  if (h === lastHash) return;                  // mtime touched but bytes identical — skip
  if (building) { setTimeout(() => maybeBuild(reason), 300); return; }  // serialize
  lastHash = h;
  building = true;
  const t0 = Date.now();
  process.stdout.write(`\n⟳ ${reason} — rebuilding… `);
  try {
    if (reason.includes("geometry_gen")) {
      const g = await run("python3", ["geometry_gen.py"], path.join(ROOT, "tools"));
      if (g.code !== 0) throw new Error("geometry_gen.py failed:\n" + g.out);
    }
    const b = await run("bash", ["build.sh"], path.join(ROOT, "wasm"));
    if (b.code !== 0) throw new Error(b.out);
    console.log(`ok (${Date.now() - t0}ms) → reload`);
    broadcast("reload", "1");
  } catch (err) {
    console.log("FAILED");
    console.error(err.message);
    broadcast("builderror", err.message);
  } finally {
    building = false;
  }
}

// ── watch sources (debounced + hash-gated) ─────────────────────────────────
let timer = null, pending = "";
for (const rel of WATCH) {
  const abs = path.join(ROOT, rel);
  try {
    fs.watch(abs, () => {
      pending = rel;
      clearTimeout(timer);
      timer = setTimeout(() => maybeBuild(pending), 250);
    });
  } catch { console.warn("watch skip (missing):", rel); }
}

server.listen(PORT, () => {
  console.log(`LED IDE dev server  →  http://localhost:${PORT}/viewer/`);
  console.log(`watching: ${WATCH.join(", ")}`);
  console.log("edit src/effects.h (or geometry_gen.py) and save — the browser reloads.\n");
});
