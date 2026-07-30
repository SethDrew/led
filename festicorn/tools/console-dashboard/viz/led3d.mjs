// led3d.mjs — the Axis Mundi 3D sculpture panel, mounted directly in the
// dashboard page (no iframe, no postMessage handshake, no second server).
//
// Render core ported from the retired helix-canopy viewer. The WASM engine runs
// the real effect math (axismundi_fx.h + kettle_energy.h — the same headers the
// installation firmware includes); this module owns only the display transform
// and the three.js scene. Console state arrives via applyState(st) from the
// dashboard's own /api/state poll — the engine mirrors ClickyPacketV1 /
// KettlePacketV1 field for field.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

// Colors reach the vertex-color attribute LINEAR, exactly as the effect emitted
// them: WS2812 photon output is linear in the value the firmware writes, and
// three.js's output pass is already an sRGB encode — encoding here as well
// would put gamma on twice. A panel-sized view reads genuinely dark at gain
// 1.0, so the linear signal gets a labelled soft-knee boost: the knee holds the
// ×3 slope where the effects live and compresses smoothly onto 1.0, so no two
// LED values land on one screen value (a plain min(1, v*3) would clip the whole
// top of the range onto 1.0 and silently reorder the picture).
const GAIN = 3.0;
const softGain = (v) => v <= 0 ? 0 : (GAIN * v) / (1 + (GAIN - 1) * v);

const bitsOf = (a) => {
  let bits = 0;
  if (Array.isArray(a)) for (let i = 0; i < a.length && i < 8; i++) if (a[i]) bits |= 1 << i;
  return bits;
};

export default async function mountLed3d(container, { fx = "axismundi", gainLabel = null } = {}) {
  const initEngine = (await import("./engine/engine.mjs")).default;
  const M = await initEngine();
  const render    = M.cwrap("render", "number", ["number"]);
  const N         = M.cwrap("led_count", "number", [])();
  const fxCount   = M.cwrap("effect_count", "number", [])();
  const fxName    = M.cwrap("effect_name", "string", ["number"]);
  const setFx     = M.cwrap("set_effect", null, ["number"]);
  const N8 = ["number","number","number","number","number","number","number","number"];
  const setClicky = M.cwrap("set_clicky", null, N8);
  const setKettle = M.cwrap("set_kettle", null, ["number", "number", "number"]);
  const setDuck   = M.cwrap("set_duck", null, N8);

  const pPtr = M.cwrap("positions", "number", [])() >> 2;
  const pos = new Float32Array(M.HEAPF32.subarray(pPtr, pPtr + N * 3));
  const kPtr = M.cwrap("kinds", "number", [])() >> 2;
  const kinds = new Int32Array(M.HEAP32.subarray(kPtr, kPtr + N));

  let found = -1;
  for (let i = 0; i < fxCount; i++) if (fxName(i) === fx) found = i;
  if (found < 0) throw new Error(`effect "${fx}" not in engine`);
  setFx(found);

  if (gainLabel) gainLabel.textContent = `×${GAIN} soft gain`;

  // ---- scene ----------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  // No tone mapping: this view is a predictor of what the strips will do, and a
  // filmic curve is a lie about that — ACES measured as a 1.31-gamma expansion
  // end to end, darkening dim output up to 3x exactly where the effects live.
  renderer.toneMapping = THREE.NoToneMapping;
  container.appendChild(renderer.domElement);
  renderer.domElement.style.display = "block";

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0d0d);

  const bmin = [Infinity, Infinity, Infinity], bmax = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
    bmin[k] = Math.min(bmin[k], pos[i*3+k]); bmax[k] = Math.max(bmax[k], pos[i*3+k]);
  }
  const center = new THREE.Vector3((bmin[0]+bmax[0])/2, (bmin[1]+bmax[1])/2, (bmin[2]+bmax[2])/2);
  const span = Math.hypot(bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]);

  const boxW = () => Math.max(1, container.clientWidth);
  const boxH = () => Math.max(1, container.clientHeight);
  renderer.setSize(boxW(), boxH());

  const camera = new THREE.PerspectiveCamera(50, boxW() / boxH(), 0.01, 1000);
  let fitRadius = 0;
  for (let i = 0; i < N; i++) fitRadius = Math.max(fitRadius, Math.hypot(
    pos[i*3] - center.x, pos[i*3+1] - center.y, pos[i*3+2] - center.z));

  const EYE = new THREE.Vector3(0.55, -0.9, 0.25).normalize();
  const FIT_MARGIN = 1.12;   // glow sprites have width the LED positions don't
  const aim = center.clone();
  camera.up.set(0, 0, 1);                                    // sculpture is Z-up

  // Framing solves for the LEDs' PROJECTED extent, converged over a few passes,
  // rather than the bounding sphere: the panel is whatever shape the browser
  // window makes it, and a sphere fit leaves a band of dead space above.
  function placeCamera() {
    const fovY = camera.fov * Math.PI / 180;
    const fovX = 2 * Math.atan(Math.tan(fovY / 2) * camera.aspect);
    let d = fitRadius / Math.sin(Math.min(fovY, fovX) / 2);
    aim.copy(center);
    const p = new THREE.Vector3(), right = new THREE.Vector3(), up = new THREE.Vector3();
    for (let it = 0; it < 8; it++) {
      camera.position.copy(aim).addScaledVector(EYE, d);
      camera.lookAt(aim);
      camera.updateMatrixWorld(true);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < N; i++) {
        p.set(pos[i*3], pos[i*3+1], pos[i*3+2]).project(camera);
        if (p.x < x0) x0 = p.x;  if (p.x > x1) x1 = p.x;
        if (p.y < y0) y0 = p.y;  if (p.y > y1) y1 = p.y;
      }
      right.setFromMatrixColumn(camera.matrixWorld, 0);
      up.setFromMatrixColumn(camera.matrixWorld, 1);
      const hH = d * Math.tan(fovY / 2), hW = hH * camera.aspect;
      aim.addScaledVector(right, (x0 + x1) / 2 * hW)
         .addScaledVector(up,    (y0 + y1) / 2 * hH);
      d *= FIT_MARGIN * Math.max((x1 - x0) / 2, (y1 - y0) / 2);
    }
    camera.position.copy(aim).addScaledVector(EYE, d);
  }
  placeCamera();

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(aim);
  controls.enableDamping = true;
  controls.update();
  // once someone has orbited, a resize must not yank the camera back
  let camMoved = false;
  controls.addEventListener("start", () => { camMoved = true; });

  const grid = new THREE.GridHelper(span * 0.8, 24, 0x0d1420, 0x080b11);
  grid.rotation.x = Math.PI / 2;
  grid.position.set(center.x, center.y, 0);
  scene.add(grid);

  // LED point cloud. Each LED carries its own world-space glow DIAMETER, so a
  // dense section can be drawn small enough to read as individual pixels while
  // a sparse one glows large. Raw ShaderMaterial does perspective-correct world
  // sizing (PointsMaterial can't do per-vertex size).
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const col = new Float32Array(N * 3);
  const colAttr = new THREE.BufferAttribute(col, 3);
  geo.setAttribute("color", colAttr);

  const sizeArr = new Float32Array(N);
  const sizeAttr = new THREE.BufferAttribute(sizeArr, 1);
  geo.setAttribute("aSize", sizeAttr);

  // per-section glow diameter (m), indexed by kind 0=helix 1=canopy 2=root
  const sectionSize = [0.028, 0.06, 0.06];
  function applySectionSizes() {
    for (let i = 0; i < N; i++) sizeArr[i] = sectionSize[kinds[i]] ?? 0.05;
    sizeAttr.needsUpdate = true;
  }
  applySectionSizes();

  const ptsMat = new THREE.ShaderMaterial({
    uniforms: { uScale: { value: 1.0 } },
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    vertexShader: `
      attribute float aSize;
      attribute vec3 color;
      varying vec3 vColor;
      uniform float uScale;
      void main() {
        vColor = color;
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = aSize * uScale / max(0.001, -mv.z);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      varying vec3 vColor;
      void main() {
        float r = length(gl_PointCoord - vec2(0.5)) * 2.0;
        if (r > 1.0) discard;
        float a = pow(1.0 - r, 1.6);
        gl_FragColor = vec4(vColor * a, a);
      }`,
  });
  scene.add(new THREE.Points(geo, ptsMat));

  function updatePointScale() {
    const h = renderer.domElement.height;
    ptsMat.uniforms.uScale.value = h / (2 * Math.tan(camera.fov * Math.PI / 360));
  }
  updatePointScale();

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(boxW(), boxH()), 0.9, 0.6, 0.0));

  new ResizeObserver(() => {
    camera.aspect = boxW() / boxH();
    camera.updateProjectionMatrix();
    if (!camMoved) { placeCamera(); controls.target.copy(aim); controls.update(); }
    renderer.setSize(boxW(), boxH());
    composer.setSize(boxW(), boxH());
    updatePointScale();
  }).observe(container);

  // ---- frame loop -----------------------------------------------------------
  let clock = 0, playing = true, last = performance.now();

  // headless QA hooks (same contract the retired viewer exposed)
  window.__ledReady = false;
  window.__ledPattern = null;
  window.__dumpColors = () => Array.from(col);
  window.__setClock = (t) => { playing = false; clock = t; apply(t); };
  window.__setGlow = (h, c, r) => {
    if (h != null) sectionSize[0] = h; if (c != null) sectionSize[1] = c; if (r != null) sectionSize[2] = r;
    applySectionSizes();
  };

  function apply(t) {
    const ptr = render(t) >> 2;                        // re-acquire view each
    const c = M.HEAPF32.subarray(ptr, ptr + N * 3);    // frame (detach-safe)
    for (let i = 0; i < N * 3; i++) {
      const v = c[i];
      col[i] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
    if (window.__ledPattern) window.__ledPattern(col);
    for (let i = 0; i < N * 3; i++) col[i] = softGain(col[i]);
    colAttr.needsUpdate = true;
  }
  apply(0);

  function loop(now) {
    const dt = (now - last) / 1000; last = now;
    if (playing) { clock += dt; apply(clock); }
    controls.update();
    composer.render();
    window.__ledReady = true;
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  return {
    nLeds: N,
    // Dashboard /api/state → engine console mirrors. Push only while a console
    // is online: the engine's staleness clock (AX_KETTLE_AGE) is what stops an
    // unplugged console's last state from acting forever.
    applyState(st) {
      const ck = st.roles?.clicky, kt = st.roles?.kettle, dk = st.roles?.duck;
      if (ck?.online && ck.state?.p) {
        const p = ck.state.p;
        setClicky(bitsOf(ck.state.pull), bitsOf(ck.state.btn),
          ...Array.from({ length: 6 }, (_, i) => Math.round((p[i] ?? 0.5) * 10000)));
      }
      if (kt?.online && kt.state?.c !== undefined) {
        setKettle(bitsOf(kt.state.btn), kt.state.c, kt.state.up ?? 0);
      }
      if (dk?.online && dk.state?.rms_mean !== undefined) {
        const d = dk.state;
        setDuck(d.seq ?? 0, d.rms_mean, d.rms_max, d.ax, d.ay, d.az,
          d.flags ?? 0, d.up ?? 0);
      }
    },
  };
}
