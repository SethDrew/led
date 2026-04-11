import type { EffectConfig } from './base';
import { BaseEffect } from './base';
import { WaterfallEffect } from './waterfall';
import { SpectrumEffect } from './spectrum';
import { TimbreEffect } from './timbre';
export { BaseEffect };
export type { EffectConfig };

export type EffectName = 'waterfall' | 'spectrum' | 'timbre';

const EFFECT_CONSTRUCTORS: Record<EffectName, new (config: EffectConfig) => BaseEffect> = {
  waterfall: WaterfallEffect,
  spectrum: SpectrumEffect,
  timbre: TimbreEffect,
};

export const EFFECT_NAMES: EffectName[] = ['waterfall', 'spectrum', 'timbre'];

export function createEffect(name: EffectName, config: EffectConfig): BaseEffect {
  const Ctor = EFFECT_CONSTRUCTORS[name];
  return new Ctor(config);
}
