import { BaseEffect, type EffectConfig } from './base';

/**
 * Energy Waterfall — dual-band (bass + vocal) energy scrolls inward from
 * strip edges toward a breathing center dead-band, with phosphor persistence.
 *
 * Ported from shuffle3d/led-strip.js WaterfallEffect.
 */
export class WaterfallEffect extends BaseEffect {
  // Theme color (max-channel normalized to 255)
  private color: [number, number, number] = [255, 120, 40];

  // Frequency band bin ranges
  private bassLo: number;
  private bassHi: number;
  private vocalLo: number;
  private vocalHi: number;

  // Sticky-floor normalization state
  private floorAlpha: number;
  private floorUpMult = 1.0;
  private floorDownMult = 4.0;
  private dbWindow = 15.0;
  private peakDecay = 0.9995;

  private bassFloor = 1e-10;
  private bassPeak = 1e-10;
  private vocalFloor = 1e-10;
  private vocalPeak = 1e-10;

  // Scroll buffers
  private mid: number;
  private lBuf: Float32Array;
  private rBuf: Float32Array;

  // Center source buffers
  private breatheMin = 5;
  private breatheAmp = 15;
  private ctrL: Float32Array;
  private ctrR: Float32Array;

  // Breathing dead band
  private breathePeriod = 40.0;
  private breatheHold = 8.0;
  private breatheFade = 3;
  private breatheTime = 0;

  // Background phosphor
  private bg: Float32Array;

  // Scroll timing
  private scrollAccum = 0;
  private scrollRate = 1 / 18;
  private pendingVal = 0;

  // Cached freq data for render()
  private lastFreqData: Float32Array | null = null;

  constructor(config: EffectConfig) {
    super(config);
    const n = this.numLeds;
    this.mid = n >> 1;

    const freqBins = 1024; // fftSize=2048 → 1024 bins
    const binHz = config.sampleRate / (freqBins * 2);
    this.bassLo = Math.max(0, Math.round(20 / binHz));
    this.bassHi = Math.min(freqBins, Math.round(250 / binHz));
    this.vocalLo = Math.max(0, Math.round(600 / binHz));
    this.vocalHi = Math.min(freqBins, Math.round(3000 / binHz));

    const fps = 60;
    this.floorAlpha = 2.0 / (10.0 * fps + 1);

    this.lBuf = new Float32Array(this.mid);
    this.rBuf = new Float32Array(n - this.mid);
    this.ctrL = new Float32Array(this.breatheAmp);
    this.ctrR = new Float32Array(this.breatheAmp);
    this.bg = new Float32Array(n);
  }

  get name(): string {
    return 'Waterfall';
  }

  processAudio(freqData: Float32Array, _timeData: Float32Array): void {
    this.lastFreqData = freqData;
  }

  render(dt: number): Uint8Array {
    const freqData = this.lastFreqData;
    const n = this.numLeds;
    const out = new Uint8Array(n * 3);

    if (!freqData) return out;

    // Extract bass band RMS (dB → power → mean → sqrt)
    let bassSum = 0;
    let bassCnt = 0;
    for (let k = this.bassLo; k < this.bassHi; k++) {
      bassSum += Math.pow(10, freqData[k] / 10);
      bassCnt++;
    }
    const bassRms = bassCnt > 0 ? Math.sqrt(bassSum / bassCnt) : 0;

    // Extract vocal band RMS
    let vocalSum = 0;
    let vocalCnt = 0;
    for (let k = this.vocalLo; k < this.vocalHi; k++) {
      vocalSum += Math.pow(10, freqData[k] / 10);
      vocalCnt++;
    }
    const vocalRms = vocalCnt > 0 ? Math.sqrt(vocalSum / vocalCnt) : 0;

    // Sticky floor + dB normalization per band
    let bassVal: number, vocalVal: number;
    [bassVal, this.bassFloor, this.bassPeak] = this.stickyFloorDb(
      bassRms, this.bassFloor, this.bassPeak
    );
    [vocalVal, this.vocalFloor, this.vocalPeak] = this.stickyFloorDb(
      vocalRms, this.vocalFloor, this.vocalPeak
    );

    const v = Math.max(bassVal, vocalVal);
    this.pendingVal = Math.max(this.pendingVal, v);

    // Scroll at fixed rate
    this.scrollAccum += dt;
    while (this.scrollAccum >= this.scrollRate) {
      this.scrollAccum -= this.scrollRate;
      this.lBuf.copyWithin(1, 0);
      this.lBuf[0] = this.pendingVal;
      this.rBuf.copyWithin(1, 0);
      this.rBuf[0] = this.pendingVal;
      this.ctrL.copyWithin(1, 0);
      this.ctrL[0] = this.pendingVal;
      this.ctrR.copyWithin(1, 0);
      this.ctrR[0] = this.pendingVal;
      this.pendingVal = 0;
    }

    // Three-phase breathing: expand → hold → contract
    this.breatheTime += dt;
    const cycleT = this.breatheTime % this.breathePeriod;
    const moveTime = (this.breathePeriod - this.breatheHold) / 2.0;

    let rawOffset: number;
    if (cycleT < moveTime) {
      const t = cycleT / moveTime;
      rawOffset = this.breatheAmp * 0.5 * (1 - Math.cos(Math.PI * t));
    } else if (cycleT < moveTime + this.breatheHold) {
      rawOffset = this.breatheAmp;
    } else {
      const t = (cycleT - moveTime - this.breatheHold) / moveTime;
      rawOffset = this.breatheAmp * 0.5 * (1 + Math.cos(Math.PI * t));
    }
    const offset = Math.max(
      Math.min(rawOffset | 0, this.breatheAmp),
      this.breatheMin
    );

    // Combine outer waterfall into full strip
    const mid = this.mid;
    const strip = new Float32Array(n);
    strip.set(this.lBuf);
    for (let i = 0; i < this.rBuf.length; i++) {
      strip[mid + i] = this.rBuf[this.rBuf.length - 1 - i];
    }

    // Replace dead band region with center source
    const deadStart = mid - offset;
    const deadEnd = mid + offset;
    if (offset > 0) {
      for (let i = 0; i < offset; i++) {
        strip[deadStart + i] = this.ctrL[offset - 1 - i];
        if (mid + i < n) strip[mid + i] = this.ctrR[i];
      }
    }

    // Background phosphor
    for (let i = 0; i < n; i++) {
      this.bg[i] = Math.min(this.bg[i] * 0.97 + strip[i] * 0.02, 0.5);
    }

    // Final brightness
    const brightness = new Float32Array(n);
    for (let i = 0; i < n; i++) brightness[i] = Math.max(strip[i], this.bg[i]);

    // Soft fade at boundaries
    if (offset > 0) {
      const fw = this.breatheFade;
      for (let i = 0; i < fw; i++) {
        const alpha = (i + 1) / (fw + 1);
        let idx: number;
        idx = deadStart - 1 - i;
        if (idx >= 0) brightness[idx] *= alpha;
        idx = deadStart + i;
        if (idx < mid) brightness[idx] *= alpha;
        idx = deadEnd - 1 - i;
        if (idx >= mid) brightness[idx] *= alpha;
        idx = deadEnd + i;
        if (idx < n) brightness[idx] *= alpha;
      }
    }

    // Color output
    for (let i = 0; i < n; i++) {
      const br = brightness[i] ** 0.65;
      out[i * 3] = (this.color[0] * br) | 0;
      out[i * 3 + 1] = (this.color[1] * br) | 0;
      out[i * 3 + 2] = (this.color[2] * br) | 0;
    }
    return out;
  }

  private stickyFloorDb(
    rms: number,
    floor: number,
    peak: number
  ): [number, number, number] {
    const alpha =
      rms > floor
        ? this.floorAlpha * this.floorUpMult
        : this.floorAlpha * this.floorDownMult;
    floor = floor + alpha * (rms - floor);
    floor = Math.max(floor, 1e-10);
    const ratio = Math.max(rms / floor, 1.0);
    const dbAbove = 20.0 * Math.log10(ratio);
    let val = Math.max(0, Math.min(dbAbove / this.dbWindow, 1.0));
    peak = Math.max(val, peak * this.peakDecay);
    if (peak > 1e-10) val = val / peak;
    return [val, floor, peak];
  }
}
