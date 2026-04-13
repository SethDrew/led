import type { EffectConfig } from './base';
import { BaseEffect } from './base';
import { SapFlowEffect } from './sapflow';
import { WaterfallEffect } from './waterfall';
import { SpectrumEffect } from './spectrum';
import { TimbreEffect } from './timbre';
export { BaseEffect };
export type { EffectConfig };

export type EffectName = 'sap flow' | 'waterfall' | 'spectrum' | 'timbre';

const EFFECT_CONSTRUCTORS: Record<EffectName, new (config: EffectConfig) => BaseEffect> = {
  'sap flow': SapFlowEffect,
  waterfall: WaterfallEffect,
  spectrum: SpectrumEffect,
  timbre: TimbreEffect,
};

export const EFFECT_NAMES: EffectName[] = ['sap flow', 'waterfall', 'spectrum', 'timbre'];

export function createEffect(name: EffectName, config: EffectConfig): BaseEffect {
  const Ctor = EFFECT_CONSTRUCTORS[name];
  return new Ctor(config);
}
