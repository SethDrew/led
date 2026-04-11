import type { Topology } from '../topology/loader';

export interface EffectConfig {
  numLeds: number;
  sampleRate: number;
  topology?: Topology;
}

export abstract class BaseEffect {
  protected numLeds: number;
  protected sampleRate: number;

  constructor(config: EffectConfig) {
    this.numLeds = config.numLeds;
    this.sampleRate = config.sampleRate;
  }

  // Process audio features (called each frame before render)
  abstract processAudio(freqData: Float32Array, timeData: Float32Array): void;

  // Produce LED colors (called at frame rate)
  // Returns [R,G,B, R,G,B, ...] length = numLeds * 3
  abstract render(dt: number): Uint8Array;

  // Display name
  abstract get name(): string;
}
