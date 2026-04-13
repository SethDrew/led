import { BaseEffect, type EffectConfig } from './base';
import { computeLedPositions } from '../topology/loader';

/**
 * Sap Flow: particles rise from roots through trunk to canopy tips.
 * Port of SapFlowForeground.h — stochastic particle spawn,
 * constant velocity, soft Gaussian falloff, additive blending.
 * Uses globalProgress (physical height) as the flow axis.
 */

interface Particle {
  height: number;    // 0 (roots) → 1 (tips)
  velocity: number;  // units per second
  brightness: number; // 0-1
  active: boolean;
}

const MAX_PARTICLES = 16;
const VELOCITY = 0.12;          // full tree height in ~8 seconds
const MIN_SPAWN_INTERVAL = 0.4; // seconds
const MAX_SPAWN_INTERVAL = 2.5; // force spawn
const SPAWN_CHANCE = 0.08;      // per frame when eligible
const SOFT_RADIUS = 0.04;       // falloff distance in height units

// Deep forest green background (~1%)
const BG_R = 1;
const BG_G = 6;
const BG_B = 1;

// Sap color: bright lime green
const SAP_R = 100;
const SAP_G = 255;
const SAP_B = 100;

export class SapFlowEffect extends BaseEffect {
  private buf: Uint8Array;
  private heights: Float32Array; // globalProgress per LED
  private particles: Particle[] = [];
  private timeSinceSpawn = 0;

  constructor(config: EffectConfig) {
    super(config);
    this.buf = new Uint8Array(this.numLeds * 3);
    this.heights = new Float32Array(this.numLeds);

    if (config.topology) {
      const computed = computeLedPositions(config.topology);
      for (let i = 0; i < this.numLeds; i++) {
        this.heights[i] = computed.globalProgress[i];
      }
    } else {
      // Fallback: linear mapping
      for (let i = 0; i < this.numLeds; i++) {
        this.heights[i] = i / (this.numLeds - 1);
      }
    }

    for (let i = 0; i < MAX_PARTICLES; i++) {
      this.particles.push({ height: 0, velocity: 0, brightness: 0, active: false });
    }
  }

  processAudio(_freqData: Float32Array, _timeData: Float32Array): void {
    // Non-interactive — ignores audio
  }

  render(dt: number): Uint8Array {
    this.timeSinceSpawn += dt;

    // Spawn logic
    let shouldSpawn = false;
    if (this.timeSinceSpawn >= MAX_SPAWN_INTERVAL) {
      shouldSpawn = true;
    } else if (this.timeSinceSpawn >= MIN_SPAWN_INTERVAL && Math.random() < SPAWN_CHANCE) {
      shouldSpawn = true;
    }

    if (shouldSpawn) {
      for (const p of this.particles) {
        if (!p.active) {
          p.active = true;
          p.height = 0;
          p.velocity = VELOCITY * (0.8 + Math.random() * 0.4); // slight variation
          p.brightness = 0.3 + Math.random() * 0.7;
          this.timeSinceSpawn = 0;
          break;
        }
      }
    }

    // Update particles
    for (const p of this.particles) {
      if (!p.active) continue;
      p.height += p.velocity * dt;
      if (p.height > 1.0 + SOFT_RADIUS * 2) {
        p.active = false;
      }
    }

    // Render: background + additive particles
    for (let i = 0; i < this.numLeds; i++) {
      const off = i * 3;
      let r = BG_R;
      let g = BG_G;
      let b = BG_B;

      const h = this.heights[i];

      for (const p of this.particles) {
        if (!p.active) continue;
        const dist = Math.abs(h - p.height);
        if (dist >= SOFT_RADIUS) continue;

        const falloff = 1.0 - dist / SOFT_RADIUS;
        const intensity = falloff * falloff * p.brightness;

        r += SAP_R * intensity;
        g += SAP_G * intensity;
        b += SAP_B * intensity;
      }

      this.buf[off] = Math.min(255, r);
      this.buf[off + 1] = Math.min(255, g);
      this.buf[off + 2] = Math.min(255, b);
    }

    return this.buf;
  }

  get name(): string {
    return 'sap flow';
  }
}
