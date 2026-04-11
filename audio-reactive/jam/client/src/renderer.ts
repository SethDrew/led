import type { Topology } from './topology/loader';
import { computeLedPositions } from './topology/loader';

/**
 * WebGL renderer with two modes:
 *   0 = Gaussian glow masked by tree image (GPU per-pixel)
 *   1 = Small LED dots drawn via gl.POINTS over dimmed tree background
 */

// --- Glow mode shaders (fullscreen quad, per-pixel Gaussian sum) ---

const GLOW_VERT = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const GLOW_FRAG = `
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

// --- Background shader (dim tree image) ---

const BG_VERT = `
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const BG_FRAG = `
precision highp float;
uniform sampler2D u_treeMask;
uniform float u_dimLevel;
varying vec2 v_uv;
void main() {
  vec2 maskUV = vec2(v_uv.x, 1.0 - v_uv.y);
  vec4 t = texture2D(u_treeMask, maskUV);
  gl_FragColor = vec4(t.rgb * t.a * u_dimLevel, 1.0);
}
`;

// --- Dot mode shaders (gl.POINTS, one vertex per LED) ---

const DOT_VERT = `
attribute vec2 a_ledUV;   // LED position in UV space (0-1)
attribute vec3 a_ledColor;
uniform float u_aspect;
uniform float u_pointSize;
varying vec3 v_color;
void main() {
  v_color = a_ledColor;
  // UV (0-1) → clip space (-1,+1)
  vec2 clip = a_ledUV * 2.0 - 1.0;
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = u_pointSize;
}
`;

const DOT_FRAG = `
precision highp float;
varying vec3 v_color;
void main() {
  // Circle mask from point coord
  vec2 c = gl_PointCoord - 0.5;
  if (dot(c, c) > 0.25) discard;
  gl_FragColor = vec4(v_color, 1.0);
}
`;

// --- Helper functions ---

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

function createDataTexture(gl: WebGLRenderingContext, width: number): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const data = new Float32Array(width * 4);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, 1, 0, gl.RGBA, gl.FLOAT, data);
  return tex;
}

export class Renderer {
  private canvas: HTMLCanvasElement;
  private gl: WebGLRenderingContext;
  private container: HTMLElement;

  // LED data
  private ledPositions: [number, number][] = [];
  private _numLeds = 0;

  // UV-space LED positions (0-1 range)
  private ledUVs: Float32Array;

  // WebGL textures
  private treeMaskTex: WebGLTexture;
  private ledPosTex: WebGLTexture;
  private ledColorTex: WebGLTexture;

  // Glow mode program + uniforms
  private glowProgram: WebGLProgram;
  private glowQuadBuf: WebGLBuffer;
  private glow_uTreeMask: WebGLUniformLocation;
  private glow_uLedPositions: WebGLUniformLocation;
  private glow_uLedColors: WebGLUniformLocation;
  private glow_uNumLeds: WebGLUniformLocation;
  private glow_uSigma: WebGLUniformLocation;
  private glow_uAspect: WebGLUniformLocation;

  // Background program (dim tree for dot mode)
  private bgProgram: WebGLProgram;
  private bg_uTreeMask: WebGLUniformLocation;
  private bg_uDimLevel: WebGLUniformLocation;

  // Dot mode program + buffers
  private dotProgram: WebGLProgram;
  private dotUVBuf: WebGLBuffer;
  private dotColorBuf: WebGLBuffer;
  private dot_uAspect: WebGLUniformLocation;
  private dot_uPointSize: WebGLUniformLocation;

  // Render mode: 0 = glow behind mask, 1 = dots on top
  private renderMode = 1;

  // Data buffers
  private posData: Float32Array;
  private colorData: Float32Array;
  private dotColorData: Float32Array; // interleaved RGB floats for dot attribute

  // Topology bounds
  private viewBounds = { minX: -0.85, maxX: 0.85, minY: -0.55, maxY: 0.95 };

  constructor(container: HTMLElement, topology: Topology) {
    this.container = container;

    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.canvas.style.display = 'block';
    this.canvas.style.background = '#000';
    container.appendChild(this.canvas);

    const gl = this.canvas.getContext('webgl', { alpha: false, premultipliedAlpha: false })!;
    this.gl = gl;

    gl.getExtension('OES_texture_float');

    // --- Glow program ---
    this.glowProgram = createProgram(gl, GLOW_VERT, GLOW_FRAG);
    this.glowQuadBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowQuadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1,
    ]), gl.STATIC_DRAW);

    gl.useProgram(this.glowProgram);
    this.glow_uTreeMask = gl.getUniformLocation(this.glowProgram, 'u_treeMask')!;
    this.glow_uLedPositions = gl.getUniformLocation(this.glowProgram, 'u_ledPositions')!;
    this.glow_uLedColors = gl.getUniformLocation(this.glowProgram, 'u_ledColors')!;
    this.glow_uNumLeds = gl.getUniformLocation(this.glowProgram, 'u_numLeds')!;
    this.glow_uSigma = gl.getUniformLocation(this.glowProgram, 'u_sigma')!;
    this.glow_uAspect = gl.getUniformLocation(this.glowProgram, 'u_aspect')!;

    // --- Background program ---
    this.bgProgram = createProgram(gl, BG_VERT, BG_FRAG);
    gl.useProgram(this.bgProgram);
    this.bg_uTreeMask = gl.getUniformLocation(this.bgProgram, 'u_treeMask')!;
    this.bg_uDimLevel = gl.getUniformLocation(this.bgProgram, 'u_dimLevel')!;

    // --- Dot program ---
    this.dotProgram = createProgram(gl, DOT_VERT, DOT_FRAG);
    this.dotUVBuf = gl.createBuffer()!;
    this.dotColorBuf = gl.createBuffer()!;

    gl.useProgram(this.dotProgram);
    this.dot_uAspect = gl.getUniformLocation(this.dotProgram, 'u_aspect')!;
    this.dot_uPointSize = gl.getUniformLocation(this.dotProgram, 'u_pointSize')!;

    // --- LED positions ---
    const computed = computeLedPositions(topology);
    this.ledPositions = computed.positions;
    this._numLeds = computed.positions.length;

    // Convert topology coords → UV space (0-1)
    this.ledUVs = new Float32Array(this._numLeds * 2);
    const { minX, maxX, minY, maxY } = this.viewBounds;
    for (let i = 0; i < this._numLeds; i++) {
      const [tx, ty] = this.ledPositions[i];
      this.ledUVs[i * 2] = (tx - minX) / (maxX - minX);
      this.ledUVs[i * 2 + 1] = (ty - minY) / (maxY - minY);
    }

    // Upload static LED UV positions for dot mode
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dotUVBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.ledUVs, gl.STATIC_DRAW);

    // --- Data textures for glow mode ---
    this.posData = new Float32Array(4096 * 4);
    this.colorData = new Float32Array(4096 * 4);
    this.dotColorData = new Float32Array(this._numLeds * 3);

    for (let i = 0; i < this._numLeds; i++) {
      this.posData[i * 4] = this.ledUVs[i * 2];
      this.posData[i * 4 + 1] = this.ledUVs[i * 2 + 1];
    }

    this.ledPosTex = createDataTexture(gl, 4096);
    this.ledColorTex = createDataTexture(gl, 4096);

    gl.bindTexture(gl.TEXTURE_2D, this.ledPosTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4096, 1, 0, gl.RGBA, gl.FLOAT, this.posData);

    // --- Tree mask texture ---
    this.treeMaskTex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.treeMaskTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([255, 255, 255, 255]));
    this.loadTreeMask('/yggdrasil-mask.png');

    // --- Set glow uniforms ---
    gl.useProgram(this.glowProgram);
    gl.uniform1i(this.glow_uTreeMask, 0);
    gl.uniform1i(this.glow_uLedPositions, 1);
    gl.uniform1i(this.glow_uLedColors, 2);
    gl.uniform1f(this.glow_uNumLeds, this._numLeds);
    gl.uniform1f(this.glow_uSigma, 0.035);

    // Initial sizing
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  private loadTreeMask(url: string): void {
    const img = new Image();
    img.onload = () => {
      const gl = this.gl;
      gl.bindTexture(gl.TEXTURE_2D, this.treeMaskTex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    };
    img.onerror = () => {
      console.warn('Tree mask not found — rendering without mask');
    };
    img.src = url;
  }

  get numLeds(): number {
    return this._numLeds;
  }

  get positions(): [number, number][] {
    return this.ledPositions;
  }

  setMode(mode: number): void {
    this.renderMode = mode;
  }

  updateLeds(colors: Uint8Array): void {
    const count = Math.min(this._numLeds, Math.floor(colors.length / 3));
    for (let i = 0; i < count; i++) {
      const r = colors[i * 3] / 255;
      const g = colors[i * 3 + 1] / 255;
      const b = colors[i * 3 + 2] / 255;
      // Glow mode texture data
      this.colorData[i * 4] = r;
      this.colorData[i * 4 + 1] = g;
      this.colorData[i * 4 + 2] = b;
      this.colorData[i * 4 + 3] = 1.0;
      // Dot mode vertex data
      this.dotColorData[i * 3] = r;
      this.dotColorData[i * 3 + 1] = g;
      this.dotColorData[i * 3 + 2] = b;
    }
  }

  render(): void {
    const gl = this.gl;

    if (this.renderMode === 1) {
      this.renderDots(gl);
    } else {
      this.renderGlow(gl);
    }
  }

  private renderGlow(gl: WebGLRenderingContext): void {
    // Upload color data
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.ledColorTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 4096, 1, 0, gl.RGBA, gl.FLOAT, this.colorData);

    gl.useProgram(this.glowProgram);

    // Bind textures
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.treeMaskTex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.ledPosTex);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.ledColorTex);

    // Bind quad
    const aPos = gl.getAttribLocation(this.glowProgram, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowQuadBuf);
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  private renderDots(gl: WebGLRenderingContext): void {
    // 1) Draw dim tree background
    gl.useProgram(this.bgProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.treeMaskTex);
    gl.uniform1i(this.bg_uTreeMask, 0);
    gl.uniform1f(this.bg_uDimLevel, 0.15);

    const bgPos = gl.getAttribLocation(this.bgProgram, 'a_position');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.glowQuadBuf);
    gl.enableVertexAttribArray(bgPos);
    gl.vertexAttribPointer(bgPos, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.disableVertexAttribArray(bgPos);

    // 2) Draw LED dots as gl.POINTS
    gl.useProgram(this.dotProgram);

    // Compute point size: ~4px at 1000px canvas height, scale with DPR
    const canvasH = this.canvas.height;
    const pointSize = Math.max(2, Math.round(canvasH / 250));
    gl.uniform1f(this.dot_uPointSize, pointSize);

    // Bind LED UV positions
    const aUV = gl.getAttribLocation(this.dotProgram, 'a_ledUV');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dotUVBuf);
    gl.enableVertexAttribArray(aUV);
    gl.vertexAttribPointer(aUV, 2, gl.FLOAT, false, 0, 0);

    // Upload and bind LED colors
    const aColor = gl.getAttribLocation(this.dotProgram, 'a_ledColor');
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dotColorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.dotColorData, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(aColor);
    gl.vertexAttribPointer(aColor, 3, gl.FLOAT, false, 0, 0);

    gl.drawArrays(gl.POINTS, 0, this._numLeds);

    gl.disableVertexAttribArray(aUV);
    gl.disableVertexAttribArray(aColor);
  }

  resize(): void {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    // Update aspect for glow mode
    this.gl.useProgram(this.glowProgram);
    this.gl.uniform1f(this.glow_uAspect, w / h);
  }
}
