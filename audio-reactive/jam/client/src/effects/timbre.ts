import { BaseEffect, type EffectConfig } from './base';
import { buildMelFilterbank, type MelFilter } from './mel';

/**
 * Timbre — MFCC-based visualization that shows spectral texture as color
 * saturation and per-coefficient brightness. Includes palette system that
 * selects complementary gradients based on a dominant color.
 *
 * Ported from shuffle3d/led-strip.js TimbreEffect.
 */

// 12-stop gradient palettes, each complements a dominant color family
const TIMBRE_PALETTES: Record<string, [number, number, number][]> = {
  ember: [
    [255,30,20],[255,70,10],[255,120,0],[255,170,10],
    [255,210,20],[240,230,50],[255,140,10],[255,80,30],
    [240,40,60],[200,20,100],[255,60,20],[255,100,10],
  ],
  arctic: [
    [0,240,220],[0,210,255],[20,160,255],[60,100,255],
    [100,60,255],[160,40,240],[210,30,220],[0,230,200],
    [30,180,255],[80,130,255],[140,70,250],[180,40,230],
  ],
  acid: [
    [255,20,160],[255,60,200],[240,30,255],[180,20,255],
    [130,10,255],[255,200,0],[255,160,20],[255,100,150],
    [220,0,255],[200,30,200],[255,40,140],[255,80,220],
  ],
  citrus: [
    [255,240,0],[210,255,0],[150,255,20],[80,255,60],
    [40,230,100],[255,220,0],[230,255,20],[190,255,10],
    [255,200,20],[240,240,0],[120,255,30],[60,240,80],
  ],
  neon: [
    [255,0,80],[255,60,0],[255,200,0],[0,255,80],
    [0,200,255],[80,0,255],[255,0,200],[255,100,0],
    [0,255,180],[180,0,255],[255,40,80],[0,240,120],
  ],
  rose: [
    [0,220,180],[20,240,140],[60,255,100],[120,240,60],
    [0,200,220],[40,230,200],[80,255,160],[0,180,200],
    [30,210,160],[100,240,80],[160,220,40],[0,200,180],
  ],
};

function pickPalette(r: number, g: number, b: number): string {
  const mx = Math.max(r, g, b);
  const mn = Math.min(r, g, b);
  const sat = mx > 0 ? (mx - mn) / mx : 0;
  if (sat < 0.15 || mx < 50) return 'neon';
  const hue =
    mx === mn ? 0
    : mx === r ? 60 * (((g - b) / (mx - mn)) % 6)
    : mx === g ? 60 * ((b - r) / (mx - mn) + 2)
    : 60 * ((r - g) / (mx - mn) + 4);
  const h = (hue + 360) % 360;
  if (h < 30 || h >= 330) return 'arctic';
  if (h < 60) return 'arctic';
  if (h < 90) return 'acid';
  if (h < 165) return 'acid';
  if (h < 200) return 'ember';
  if (h < 260) return 'ember';
  if (h < 290) return 'citrus';
  return 'rose';
}

export class TimbreEffect extends BaseEffect {
  private nMfcc = 13;
  private nDisplay = 12;
  private nMels = 40;

  private melFb: MelFilter[];
  private dct: Float32Array;

  // LED → MFCC interpolation
  private ledIdx: Int32Array;
  private ledWt: Float32Array;

  // Per-LED hues from palette
  private ledHues: [number, number, number][];

  // Adaptive MFCC normalization
  private mfccMin: Float32Array;
  private mfccMax: Float32Array;
  private adaptCount = 0;

  // RMS normalization
  private rmsPeak = 1e-10;

  // Smooth state
  private smoothSat = 1;
  private smoothBr: Float32Array;
  private smoothBright = 0;
  private silenceFactor = 1;

  // Cached audio data
  private lastFreqData: Float32Array | null = null;
  private lastTimeData: Float32Array | null = null;

  constructor(config: EffectConfig) {
    super(config);
    const n = this.numLeds;
    const freqBins = 1024;

    this.melFb = buildMelFilterbank(config.sampleRate, freqBins, this.nMels, 20, 8000);
    this.dct = this.buildDct(this.nMfcc, this.nMels);

    this.ledIdx = new Int32Array(n);
    this.ledWt = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const f = (i / (n - 1)) * (this.nDisplay - 1);
      this.ledIdx[i] = Math.min(f | 0, this.nDisplay - 2);
      this.ledWt[i] = f - this.ledIdx[i];
    }

    this.ledHues = new Array(n);
    this.applyPalette(pickPalette(128, 128, 128));

    this.mfccMin = new Float32Array(this.nMfcc);
    this.mfccMax = new Float32Array(this.nMfcc).fill(1);
    this.smoothBr = new Float32Array(n);
  }

  get name(): string {
    return 'Timbre';
  }

  updateColor(r: number, g: number, b: number): void {
    this.applyPalette(pickPalette(r, g, b));
  }

  processAudio(freqData: Float32Array, timeData: Float32Array): void {
    this.lastFreqData = freqData;
    this.lastTimeData = timeData;
  }

  render(dt: number): Uint8Array {
    const n = this.numLeds;
    const out = new Uint8Array(n * 3);
    const freqData = this.lastFreqData;
    const timeData = this.lastTimeData;
    if (!freqData || !timeData) return out;

    // RMS from time-domain
    let sum = 0;
    for (let i = 0; i < timeData.length; i++) sum += timeData[i] * timeData[i];
    const rms = Math.sqrt(sum / timeData.length);
    this.rmsPeak = Math.max(rms, this.rmsPeak * 0.9995);
    const rmsNorm = this.rmsPeak > 1e-10 ? rms / this.rmsPeak : 0;

    // Power spectrum from dB
    const nBins = freqData.length;
    const spec = new Float32Array(nBins);
    for (let k = 0; k < nBins; k++) spec[k] = 10 ** (freqData[k] / 10);

    // Mel energies
    const melE = new Float32Array(this.nMels);
    for (let m = 0; m < this.nMels; m++) {
      let s = 0;
      const { w, s: start, e: end } = this.melFb[m];
      for (let k = start; k < end; k++) s += w[k] * spec[k];
      melE[m] = Math.log(Math.max(s, 1e-10));
    }

    // DCT → MFCCs
    const mfcc = new Float32Array(this.nMfcc);
    for (let k = 0; k < this.nMfcc; k++) {
      let s = 0;
      const off = k * this.nMels;
      for (let m = 0; m < this.nMels; m++) s += this.dct[off + m] * melE[m];
      mfcc[k] = s;
    }

    // Adaptive min/max per coefficient
    this.adaptCount++;
    const EXPAND_RATE = 0.6;
    const CONTRACT_RATE = 0.3;
    const WARMUP_CONTRACT = 3;
    const expandAlpha = 1 - Math.exp(-EXPAND_RATE * dt);
    const contractAlpha = 1 - Math.exp(
      -(this.adaptCount < 60 ? WARMUP_CONTRACT : CONTRACT_RATE) * dt
    );
    for (let i = 0; i < this.nMfcc; i++) {
      if (mfcc[i] < this.mfccMin[i]) {
        this.mfccMin[i] += expandAlpha * (mfcc[i] - this.mfccMin[i]);
      } else {
        this.mfccMin[i] += contractAlpha * (mfcc[i] - this.mfccMin[i]);
      }
      if (mfcc[i] > this.mfccMax[i]) {
        this.mfccMax[i] += expandAlpha * (mfcc[i] - this.mfccMax[i]);
      } else {
        this.mfccMax[i] += contractAlpha * (mfcc[i] - this.mfccMax[i]);
      }
    }

    // Normalize MFCCs 1-12 (skip MFCC0)
    const mfccNorm = new Float32Array(this.nDisplay);
    for (let i = 0; i < this.nDisplay; i++) {
      const span = Math.max(this.mfccMax[i + 1] - this.mfccMin[i + 1], 0.1);
      mfccNorm[i] = Math.max(
        0,
        Math.min((mfcc[i + 1] - this.mfccMin[i + 1]) / span, 1)
      );
    }

    // Silence factor
    const db = 20 * Math.log10(Math.max(rmsNorm, 1e-10));
    const silRaw = Math.max(0, Math.min((db + 40) / 34, 1));
    const silAlpha = 1 - Math.exp(-5.0 * dt);
    this.silenceFactor += silAlpha * (silRaw ** 0.3 - this.silenceFactor);

    // Rate constants
    const ATTACK = 55;
    const BR_DECAY = 6.93;
    const aAttack = 1 - Math.exp(-ATTACK * dt);
    const brDecay = Math.exp(-BR_DECAY * dt);

    // MFCC spectral contrast → saturation
    let mfccMean = 0;
    for (let i = 0; i < this.nDisplay; i++) mfccMean += mfccNorm[i];
    mfccMean /= this.nDisplay;
    let mfccVar = 0;
    for (let i = 0; i < this.nDisplay; i++) {
      const d = mfccNorm[i] - mfccMean;
      mfccVar += d * d;
    }
    const mfccContrast = Math.sqrt(mfccVar / this.nDisplay);
    const satTarget = 1 - Math.exp(-mfccContrast * 16);

    const SAT_ATTACK = 6;
    const SAT_DECAY_RATE = 3;
    const satRate = satTarget > this.smoothSat ? SAT_ATTACK : SAT_DECAY_RATE;
    const satAlpha = 1 - Math.exp(-satRate * dt);
    this.smoothSat += satAlpha * (satTarget - this.smoothSat);

    // Per-LED brightness
    for (let i = 0; i < n; i++) {
      const lo = mfccNorm[this.ledIdx[i]];
      const hi = mfccNorm[Math.min(this.ledIdx[i] + 1, this.nDisplay - 1)];
      const target = lo * (1 - this.ledWt[i]) + hi * this.ledWt[i];
      if (target > this.smoothBr[i]) {
        this.smoothBr[i] += aAttack * (target - this.smoothBr[i]);
      } else {
        this.smoothBr[i] *= brDecay;
      }
    }

    // Overall brightness from RMS
    if (rmsNorm > this.smoothBright) {
      const brEnvAlpha = 1 - Math.exp(-21 * dt);
      this.smoothBright += brEnvAlpha * (rmsNorm - this.smoothBright);
    } else {
      this.smoothBright *= Math.exp(-1.0 * dt);
    }
    const baseBr = Math.min(this.smoothBright, 1) ** 0.7 * this.silenceFactor;

    // Color output
    const satVal = this.smoothSat;
    const GRAY = 80;
    for (let i = 0; i < n; i++) {
      const br = baseBr * this.smoothBr[i];
      if (br < 0.12) continue; // clean black
      const h = this.ledHues[i];
      out[i * 3] = Math.min(255, ((h[0] * satVal + GRAY * (1 - satVal)) * br) | 0);
      out[i * 3 + 1] = Math.min(255, ((h[1] * satVal + GRAY * (1 - satVal)) * br) | 0);
      out[i * 3 + 2] = Math.min(255, ((h[2] * satVal + GRAY * (1 - satVal)) * br) | 0);
    }
    return out;
  }

  private applyPalette(name: string): void {
    const palette = TIMBRE_PALETTES[name] || TIMBRE_PALETTES.neon;
    for (let i = 0; i < this.numLeds; i++) {
      const idx = this.ledIdx[i];
      const idx2 = Math.min(idx + 1, this.nDisplay - 1);
      const w = this.ledWt[i];
      this.ledHues[i] = [
        palette[idx][0] * (1 - w) + palette[idx2][0] * w,
        palette[idx][1] * (1 - w) + palette[idx2][1] * w,
        palette[idx][2] * (1 - w) + palette[idx2][2] * w,
      ];
    }
  }

  private buildDct(nMfcc: number, nMels: number): Float32Array {
    const dct = new Float32Array(nMfcc * nMels);
    const s0 = 1 / Math.sqrt(nMels);
    const s1 = Math.sqrt(2 / nMels);
    for (let k = 0; k < nMfcc; k++) {
      for (let n = 0; n < nMels; n++) {
        dct[k * nMels + n] =
          Math.cos((Math.PI * k * (2 * n + 1)) / (2 * nMels)) *
          (k === 0 ? s0 : s1);
      }
    }
    return dct;
  }
}
