/**
 * Headless WebGL renderer — reproduces the jam platform's tree+LED glow
 * in a Node.js process and saves a PNG screenshot.
 *
 * Usage: npx tsx tools/headless-render.ts
 */

import createGL from 'gl';
import sharp from 'sharp';
import { PNG } from 'pngjs';
import { readFileSync, writeFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const JAM_ROOT = resolve(__dirname, '..');

// --- Shader sources (exact copy from client/src/renderer.ts) ---

const VERT_SRC = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAG_SRC = `
precision highp float;

uniform sampler2D u_treeMask;
uniform sampler2D u_ledPositions;
uniform sampler2D u_ledColors;
uniform float u_numLeds;
uniform float u_sigma;
uniform float u_aspect;

varying vec2 v_uv;

void main() {
  vec2 maskUV = vec2(v_uv.x, 1.0 - v_uv.y);
  vec4 treeSample = texture2D(u_treeMask, maskUV);
  float treeAlpha = treeSample.a;

  if (treeAlpha < 0.01) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  vec3 totalColor = vec3(0.0);
  float sigma2 = 2.0 * u_sigma * u_sigma;

  for (float i = 0.0; i < 4096.0; i += 1.0) {
    if (i >= u_numLeds) break;

    float tx = (i + 0.5) / 4096.0;
    vec2 ledPos = texture2D(u_ledPositions, vec2(tx, 0.5)).rg;
    vec3 ledColor = texture2D(u_ledColors, vec2(tx, 0.5)).rgb;

    if (ledColor.r + ledColor.g + ledColor.b < 0.01) continue;

    vec2 diff = v_uv - ledPos;
    diff.x *= u_aspect;
    float dist2 = dot(diff, diff);

    float intensity = exp(-dist2 / sigma2);

    totalColor += ledColor * intensity;
  }

  vec3 treeColor = treeSample.rgb;
  vec3 masked = totalColor * treeAlpha;
  vec3 final = masked + treeColor * treeAlpha * 0.04;

  gl_FragColor = vec4(final, 1.0);
}
`;

// --- Topology loader (mirrored from client/src/topology/loader.ts) ---

interface Branch {
  id: string;
  parent: string | null;
  path: [number, number][];
  ledCount: number;
}

interface Topology {
  name: string;
  numLeds: number;
  branches: Branch[];
}

function polylineLength(path: [number, number][]): number {
  let len = 0;
  for (let i = 1; i < path.length; i++) {
    const dx = path[i][0] - path[i - 1][0];
    const dy = path[i][1] - path[i - 1][1];
    len += Math.sqrt(dx * dx + dy * dy);
  }
  return len;
}

function interpolateAlongPolyline(
  path: [number, number][],
  distance: number
): [number, number] {
  let remaining = distance;
  for (let i = 1; i < path.length; i++) {
    const dx = path[i][0] - path[i - 1][0];
    const dy = path[i][1] - path[i - 1][1];
    const segLen = Math.sqrt(dx * dx + dy * dy);
    if (remaining <= segLen || i === path.length - 1) {
      const t = segLen > 0 ? remaining / segLen : 0;
      return [
        path[i - 1][0] + dx * t,
        path[i - 1][1] + dy * t,
      ];
    }
    remaining -= segLen;
  }
  return path[path.length - 1];
}

function computeLedPositions(topology: Topology) {
  const positions: [number, number][] = [];
  const depthMap = new Map<string, number>();

  function computeDepth(branch: Branch): number {
    if (branch.parent === null) return 0;
    const parentDepth = depthMap.get(branch.parent);
    return parentDepth !== undefined ? parentDepth + 1 : 0;
  }

  for (const branch of topology.branches) {
    depthMap.set(branch.id, computeDepth(branch));
  }

  for (const branch of topology.branches) {
    const totalLen = polylineLength(branch.path);
    for (let i = 0; i < branch.ledCount; i++) {
      const t = branch.ledCount > 1 ? i / (branch.ledCount - 1) : 0.5;
      const distance = t * totalLen;
      const pos = interpolateAlongPolyline(branch.path, distance);
      positions.push(pos);
    }
  }

  return positions;
}

// --- GL helpers ---

function createShader(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type)!;
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const info = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile error: ${info}`);
  }
  return shader;
}

function createProgram(gl: WebGLRenderingContext, vertSrc: string, fragSrc: string): WebGLProgram {
  const program = gl.createProgram()!;
  gl.attachShader(program, createShader(gl, gl.VERTEX_SHADER, vertSrc));
  gl.attachShader(program, createShader(gl, gl.FRAGMENT_SHADER, fragSrc));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const info = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link error: ${info}`);
  }
  return program;
}

// --- HSV to RGB ---

function hsvToRgb(h: number, s: number, v: number): [number, number, number] {
  const i = Math.floor(h * 6);
  const f = h * 6 - i;
  const p = v * (1 - s);
  const q = v * (1 - f * s);
  const t = v * (1 - (1 - f) * s);
  switch (i % 6) {
    case 0: return [v, t, p];
    case 1: return [q, v, p];
    case 2: return [p, v, t];
    case 3: return [p, q, v];
    case 4: return [t, p, v];
    case 5: return [v, p, q];
    default: return [v, t, p];
  }
}

// --- Main ---

async function main() {
  const WIDTH = 800;
  const HEIGHT = 800;

  // Allow overriding topology path and output path via CLI args
  const args = process.argv.slice(2);
  const topoArg = args.find(a => a.startsWith('--topology='));
  const outArg = args.find(a => a.startsWith('--output='));
  const sigmaArg = args.find(a => a.startsWith('--sigma='));
  const sweepArg = args.find(a => a.startsWith('--sweep'));
  const sweepFrames = sweepArg ? parseInt(sweepArg.split('=')[1] || '30') : 0;

  // 1. Load topology
  const topoPath = topoArg
    ? resolve(topoArg.split('=')[1])
    : resolve(JAM_ROOT, 'client/src/topology/yggdrasil.json');
  const topology: Topology = JSON.parse(readFileSync(topoPath, 'utf-8'));
  const positions = computeLedPositions(topology);
  const numLeds = positions.length;
  console.log(`Topology: ${topology.name}, ${numLeds} LEDs`);

  // 2. Load tree mask PNG via sharp → raw RGBA
  const maskPath = resolve(JAM_ROOT, 'client/public/yggdrasil-mask.png');
  const maskImage = sharp(maskPath);
  const maskMeta = await maskImage.metadata();
  const maskW = maskMeta.width!;
  const maskH = maskMeta.height!;
  const maskRGBA = await maskImage.ensureAlpha().raw().toBuffer();
  console.log(`Tree mask: ${maskW}x${maskH}`);

  // 3. Create headless GL context
  const gl = createGL(WIDTH, HEIGHT, { preserveDrawingBuffer: true });
  if (!gl) throw new Error('Failed to create headless GL context');

  // Check for float texture support
  const floatExt = gl.getExtension('OES_texture_float');
  const useFloatTextures = !!floatExt;
  console.log(`OES_texture_float: ${useFloatTextures ? 'available' : 'NOT available, using UNSIGNED_BYTE fallback'}`);

  gl.viewport(0, 0, WIDTH, HEIGHT);

  // 4. Compile shaders
  const program = createProgram(gl, VERT_SRC, FRAG_SRC);
  gl.useProgram(program);

  // 5. Fullscreen quad
  const posBuffer = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1, 1, -1, -1, 1,
    -1, 1, 1, -1, 1, 1,
  ]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(program, 'a_position');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  // 6. Uniform locations
  const uTreeMask = gl.getUniformLocation(program, 'u_treeMask')!;
  const uLedPositions = gl.getUniformLocation(program, 'u_ledPositions')!;
  const uLedColors = gl.getUniformLocation(program, 'u_ledColors')!;
  const uNumLeds = gl.getUniformLocation(program, 'u_numLeds')!;
  const uSigma = gl.getUniformLocation(program, 'u_sigma')!;
  const uAspect = gl.getUniformLocation(program, 'u_aspect')!;

  // 7. Tree mask texture
  const treeMaskTex = gl.createTexture()!;
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, treeMaskTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, maskW, maskH, 0, gl.RGBA, gl.UNSIGNED_BYTE,
    new Uint8Array(maskRGBA));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  // 8. LED position data texture (512x1)
  const viewBounds = { minX: -0.85, maxX: 0.85, minY: -0.55, maxY: 0.95 };

  if (useFloatTextures) {
    // Float path — exact match with browser renderer
    const TEX_WIDTH = 4096;
    const posData = new Float32Array(TEX_WIDTH * 4);
    for (let i = 0; i < numLeds; i++) {
      const [tx, ty] = positions[i];
      posData[i * 4] = (tx - viewBounds.minX) / (viewBounds.maxX - viewBounds.minX);
      posData[i * 4 + 1] = (ty - viewBounds.minY) / (viewBounds.maxY - viewBounds.minY);
    }

    const ledPosTex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, ledPosTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TEX_WIDTH, 1, 0, gl.RGBA, gl.FLOAT, posData);

    // 9. LED color data texture — rainbow gradient
    const colorData = new Float32Array(TEX_WIDTH * 4);
    for (let i = 0; i < numLeds; i++) {
      const hue = i / numLeds;
      const [r, g, b] = hsvToRgb(hue, 1.0, 1.0);
      colorData[i * 4] = r;
      colorData[i * 4 + 1] = g;
      colorData[i * 4 + 2] = b;
      colorData[i * 4 + 3] = 1.0;
    }

    const ledColorTex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, ledColorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TEX_WIDTH, 1, 0, gl.RGBA, gl.FLOAT, colorData);
  } else {
    // UNSIGNED_BYTE fallback
    const TEX_WIDTH = 4096;
    const posDataBytes = new Uint8Array(TEX_WIDTH * 4);
    for (let i = 0; i < numLeds; i++) {
      const [tx, ty] = positions[i];
      const u = (tx - viewBounds.minX) / (viewBounds.maxX - viewBounds.minX);
      const v = (ty - viewBounds.minY) / (viewBounds.maxY - viewBounds.minY);
      posDataBytes[i * 4] = Math.round(Math.max(0, Math.min(1, u)) * 255);
      posDataBytes[i * 4 + 1] = Math.round(Math.max(0, Math.min(1, v)) * 255);
    }

    const ledPosTex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, ledPosTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TEX_WIDTH, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, posDataBytes);

    const colorDataBytes = new Uint8Array(TEX_WIDTH * 4);
    for (let i = 0; i < numLeds; i++) {
      const hue = i / numLeds;
      const [r, g, b] = hsvToRgb(hue, 1.0, 1.0);
      colorDataBytes[i * 4] = Math.round(r * 255);
      colorDataBytes[i * 4 + 1] = Math.round(g * 255);
      colorDataBytes[i * 4 + 2] = Math.round(b * 255);
      colorDataBytes[i * 4 + 3] = 255;
    }

    const ledColorTex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, ledColorTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, TEX_WIDTH, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colorDataBytes);
  }

  // 10. Set uniforms
  gl.uniform1i(uTreeMask, 0);
  gl.uniform1i(uLedPositions, 1);
  gl.uniform1i(uLedColors, 2);
  gl.uniform1f(uNumLeds, numLeds);
  const sigma = sigmaArg ? parseFloat(sigmaArg.split('=')[1]) : 0.045;
  gl.uniform1f(uSigma, sigma);
  console.log(`Sigma: ${sigma}`);
  gl.uniform1f(uAspect, WIDTH / HEIGHT); // 1.0 for 800x800

  // Compute LED UV positions for sweep mode
  const ledUVs: [number, number][] = positions.map(([tx, ty]) => [
    (tx - viewBounds.minX) / (viewBounds.maxX - viewBounds.minX),
    (ty - viewBounds.minY) / (viewBounds.maxY - viewBounds.minY),
  ]);

  // Helper: render one frame and return flipped RGBA buffer
  function renderFrame(): Buffer {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    const pixels = new Uint8Array(WIDTH * HEIGHT * 4);
    gl.readPixels(0, 0, WIDTH, HEIGHT, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    const rowBytes = WIDTH * 4;
    const flipped = new Uint8Array(WIDTH * HEIGHT * 4);
    for (let y = 0; y < HEIGHT; y++) {
      const srcOffset = (HEIGHT - 1 - y) * rowBytes;
      const dstOffset = y * rowBytes;
      flipped.set(pixels.subarray(srcOffset, srcOffset + rowBytes), dstOffset);
    }
    return Buffer.from(flipped);
  }

  // Helper: update LED color texture with new color data
  function uploadColors(colorData: Float32Array): void {
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture()!);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4096, 1, 0, gl.RGBA, gl.FLOAT, colorData);
  }

  const outPath = outArg
    ? resolve(outArg.split('=')[1])
    : resolve(JAM_ROOT, 'tools/render-preview.png');

  if (sweepFrames > 0) {
    // --- Sweep mode: vertical band scans left to right ---
    const bandWidth = 0.06; // UV width of the lit band
    const frameDir = resolve(JAM_ROOT, 'tools/sweep-frames');
    const { mkdirSync } = await import('fs');
    mkdirSync(frameDir, { recursive: true });

    for (let f = 0; f < sweepFrames; f++) {
      const bandCenter = f / (sweepFrames - 1); // 0 to 1
      const colorData = new Float32Array(4096 * 4);
      for (let i = 0; i < numLeds; i++) {
        const u = ledUVs[i][0];
        if (Math.abs(u - bandCenter) < bandWidth / 2) {
          colorData[i * 4] = 1.0;
          colorData[i * 4 + 1] = 1.0;
          colorData[i * 4 + 2] = 1.0;
          colorData[i * 4 + 3] = 1.0;
        }
      }
      uploadColors(colorData);
      const frameBuf = renderFrame();
      const png = new PNG({ width: WIDTH, height: HEIGHT });
      png.data = frameBuf;
      const pngBuffer = PNG.sync.write(png);
      const framePath = resolve(frameDir, `frame-${String(f).padStart(3, '0')}.png`);
      writeFileSync(framePath, pngBuffer);
    }
    console.log(`Saved ${sweepFrames} frames to ${frameDir}`);
    console.log(`Combine with: ffmpeg -framerate 15 -i ${frameDir}/frame-%03d.png -vf palettegen=max_colors=128 /tmp/pal.png && ffmpeg -framerate 15 -i ${frameDir}/frame-%03d.png -i /tmp/pal.png -lavfi paletteuse=dither=bayer ${outPath.replace('.png', '.gif')}`);
  } else {
    // --- Single frame mode (existing behavior) ---
    // 11. Draw
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    const err = gl.getError();
    if (err !== gl.NO_ERROR) {
      console.error(`GL error after draw: ${err}`);
    }

    const frameBuf = renderFrame();
    const png = new PNG({ width: WIDTH, height: HEIGHT });
    png.data = frameBuf;
    const pngBuffer = PNG.sync.write(png);
    writeFileSync(outPath, pngBuffer);
    console.log(`Saved: ${outPath}`);
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
