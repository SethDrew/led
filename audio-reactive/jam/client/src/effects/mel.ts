// Sparse mel filterbank — shared by Spectrum and Timbre effects

export interface MelFilter {
  w: Float32Array;  // weights (length = nBins, sparse: only [s..e) nonzero)
  s: number;        // start bin (inclusive)
  e: number;        // end bin (exclusive)
}

export function buildMelFilterbank(
  sampleRate: number,
  nBins: number,
  nMels: number,
  fmin: number,
  fmax: number
): MelFilter[] {
  const h2m = (f: number) => 2595 * Math.log10(1 + f / 700);
  const m2h = (m: number) => 700 * (10 ** (m / 2595) - 1);
  const fftSz = nBins * 2;
  const freqs = new Float32Array(nBins);
  for (let k = 0; k < nBins; k++) freqs[k] = (k * sampleRate) / fftSz;

  const mMin = h2m(fmin);
  const mMax = h2m(fmax);
  const pts = new Float32Array(nMels + 2);
  for (let i = 0; i < nMels + 2; i++) {
    pts[i] = m2h(mMin + (i * (mMax - mMin)) / (nMels + 1));
  }

  const fb: MelFilter[] = [];
  for (let i = 0; i < nMels; i++) {
    const lo = pts[i];
    const ctr = pts[i + 1];
    const hi = pts[i + 2];
    let start = nBins;
    let end = 0;
    const w = new Float32Array(nBins);
    for (let k = 0; k < nBins; k++) {
      const up = (freqs[k] - lo) / (ctr - lo + 1e-10);
      const dn = (hi - freqs[k]) / (hi - ctr + 1e-10);
      const v = Math.max(0, Math.min(up, dn));
      w[k] = v;
      if (v > 0) {
        start = Math.min(start, k);
        end = k + 1;
      }
    }
    fb.push({ w, s: start, e: end });
  }
  return fb;
}
