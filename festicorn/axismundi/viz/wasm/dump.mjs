// dump.mjs — headless frame dump from the WASM engine (Node, no GPU).
//
// The same engine.mjs the browser runs, driven under Node to emit the exact CSV
// the host C++ sim used to produce (frame,t_s,pixel,x,y,z,r,g,b). This retires
// effect_sim for the data path: analyze.py and the viewer's build_frames.py keep
// working unchanged, but the colors now come from the one WASM artifact.
//
//   node dump.mjs <effect> <fps> <seconds>  > frames.csv
//   node dump.mjs nebula 30 20 > nebula_frames.csv
//
import initEngine from "./engine.mjs";

const name    = process.argv[2] ?? "nebula";
const fps     = parseFloat(process.argv[3] ?? "30");
const seconds = parseFloat(process.argv[4] ?? "20");

const M = await initEngine();
const render      = M.cwrap("render", "number", ["number"]);
const positions   = M.cwrap("positions", "number", []);
const ledCount    = M.cwrap("led_count", "number", []);
const effectCount = M.cwrap("effect_count", "number", []);
const effectName  = M.cwrap("effect_name", "string", ["number"]);
const setEffect   = M.cwrap("set_effect", null, ["number"]);

// resolve effect name -> index
let idx = -1;
for (let e = 0; e < effectCount(); e++) if (effectName(e) === name) idx = e;
if (idx < 0) {
  const all = Array.from({ length: effectCount() }, (_, e) => effectName(e)).join(" ");
  process.stderr.write(`unknown effect '${name}'. available: ${all}\n`);
  process.exit(1);
}
setEffect(idx);

const N = ledCount();
// positions are static for the run; read once. (ptr/4 indexes HEAPF32.)
const pPtr = positions() >> 2;
const pos = M.HEAPF32.subarray(pPtr, pPtr + N * 3);
const px = new Float32Array(pos);   // copy out before render() reuses the heap region? (separate buffers, but be safe)

// gamma 2.2 display encode -> 0..255, mirroring effect_sim.cpp's enc().
const enc = (lin) => {
  if (lin < 0) lin = 0; else if (lin > 1) lin = 1;
  return Math.round(Math.pow(lin, 1 / 2.2) * 255);
};

const nFrames = Math.floor(fps * seconds);
const out = [];
out.push("frame,t_s,pixel,x,y,z,r,g,b");
for (let f = 0; f < nFrames; f++) {
  const t = f / fps;
  const cPtr = render(t) >> 2;
  const c = M.HEAPF32.subarray(cPtr, cPtr + N * 3);   // re-acquired each frame
  for (let i = 0; i < N; i++) {
    out.push(`${f},${t.toFixed(4)},${i},` +
      `${px[i*3].toFixed(4)},${px[i*3+1].toFixed(4)},${px[i*3+2].toFixed(4)},` +
      `${enc(c[i*3])},${enc(c[i*3+1])},${enc(c[i*3+2])}`);
  }
}
process.stdout.write(out.join("\n") + "\n");
process.stderr.write(`rendered ${nFrames} frames x ${N} LEDs (effect=${name}, ${fps}fps)\n`);
