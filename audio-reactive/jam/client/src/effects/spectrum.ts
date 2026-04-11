import { BaseEffect, type EffectConfig } from './base';
import { buildMelFilterbank, type MelFilter } from './mel';

/**
 * Spectro Chroma — mel-frequency spectrum mapped to LED strip with
 * peak-normalized bands and RMS brightness envelope.
 *
 * Ported from shuffle3d/led-strip.js SpectrumEffect.
 */
export class SpectrumEffect extends BaseEffect {
  // Theme color (max-channel normalized to 255)
  private color: [number, number, number] = [80, 180, 255];

  private nMels = 64;
  private melFb: MelFilter[];

  // LED → mel bin interpolation
  private mIdx: Int32Array;
  private mWt: Float32Array;

  // Per-mel-band peaks
  private mPeaks: Float32Array;

  // RMS peak
  private rPeak = 1e-10;

  // Smoothed state
  private sChroma: Float32Array;
  private sBright = 0;

  // Cached audio data
  private lastFreqData: Float32Array | null = null;
  private lastTimeData: Float32Array | null = null;

  constructor(config: EffectConfig) {
    super(config);
    const n = this.numLeds;
    const freqBins = 1024; // fftSize=2048 → 1024 bins

    this.melFb = buildMelFilterbank(config.sampleRate, freqBins, this.nMels, 20, 8000);

    this.mIdx = new Int32Array(n);
    this.mWt = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const f = (i / (n - 1)) * (this.nMels - 1);
      this.mIdx[i] = Math.min(f | 0, this.nMels - 2);
      this.mWt[i] = f - this.mIdx[i];
    }

    this.mPeaks = new Float32Array(this.nMels).fill(1e-10);
    this.sChroma = new Float32Array(n);
  }

  get name(): string {
    return 'Spectrum';
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

    // RMS from time domain
    let sum = 0;
    for (let i = 0; i < timeData.length; i++) sum += timeData[i] * timeData[i];
    const rms = Math.sqrt(sum / timeData.length);
    this.rPeak = Math.max(rms, this.rPeak * 0.9995);
    const rN = this.rPeak > 1e-10 ? rms / this.rPeak : 0;
    const emaAlpha = dt ? 1 - Math.exp(-18 * dt) : 0.3;

    // Mel energies
    const mel = new Float32Array(this.nMels);
    for (let m = 0; m < this.nMels; m++) {
      let s = 0;
      const { w, s: start, e: end } = this.melFb[m];
      for (let k = start; k < end; k++) s += w[k] * 10 ** (freqData[k] / 10);
      mel[m] = s;
    }

    // dB + normalize
    let mx = -Infinity;
    for (let i = 0; i < this.nMels; i++) {
      mel[i] = 10 * Math.log10(Math.max(mel[i], 1e-10));
      if (mel[i] > mx) mx = mel[i];
    }
    let mn = Infinity;
    for (let i = 0; i < this.nMels; i++) {
      mel[i] = Math.max(mel[i], mx - 80);
      if (mel[i] < mn) mn = mel[i];
    }
    for (let i = 0; i < this.nMels; i++) {
      mel[i] -= mn;
      this.mPeaks[i] = Math.max(mel[i], this.mPeaks[i] * 0.9995);
      mel[i] = this.mPeaks[i] > 1e-10 ? mel[i] / this.mPeaks[i] : 0;
    }

    // Interpolate to LEDs + smooth
    for (let i = 0; i < n; i++) {
      const lo = mel[this.mIdx[i]];
      const hi = mel[Math.min(this.mIdx[i] + 1, this.nMels - 1)];
      const c = lo * (1 - this.mWt[i]) + hi * this.mWt[i];
      this.sChroma[i] += emaAlpha * (c - this.sChroma[i]);
    }
    this.sBright += emaAlpha * (rN - this.sBright);

    // Color output
    const br = Math.max(0, Math.min(this.sBright, 1));
    for (let i = 0; i < n; i++) {
      const c = Math.max(0, Math.min(this.sChroma[i], 1));
      out[i * 3] = ((90 * (1 - c) + this.color[0] * c) * br) | 0;
      out[i * 3 + 1] = ((90 * (1 - c) + this.color[1] * c) * br) | 0;
      out[i * 3 + 2] = ((90 * (1 - c) + this.color[2] * c) * br) | 0;
    }
    return out;
  }
}
