#!/usr/bin/env python3
"""sim_energy_stretch.py — validate the bloom audio-energy contrast-stretch offline.

Ports the tree-of-record firmware's EXACT fxEnergy derivation (adaptive floor +
log scaling, audioDeriveFeaturesRms), adds the proposed 5s leaky integrator
(energy5s), and shows how the contrast-stretch responds across a real telemetry
trace. No mic / hardware needed.

Run:  ../../../.venv/bin/python sim_energy_stretch.py audio_envelope_profile.csv
"""
import sys, math
from replay import load

# ── Firmware constants (mirrored from src/main.cpp) ──────────────────
SNS_RMS_FLOOR_MIN  = 20000.0
SNS_FLOOR_HEADROOM = 1.4
SNS_FLOOR_LEAK     = 0.005
SNS_FLOOR_SNAP_EPS = 0.05
SNS_FLOOR_SOFT_SIG = 0.6

# ── Proposed new params ──────────────────────────────────────────────
ENERGY5S_TAU = 5.0    # integrator time constant (s) — "integral over 5s"
STRETCH_GAIN = 1.6    # G: max extra contrast at energy5s = 1.0


class EnergyState:
    """Faithful port of the firmware adaptive floor/ceiling + fxEnergy."""
    def __init__(self):
        self.floor = 0.0
        self.ceiling = SNS_RMS_FLOOR_MIN
        self.below = 0
        self.energy = 0.0

    def update_floor(self, rms, dt):
        if self.floor < 1.0:
            self.floor = max(rms, SNS_RMS_FLOOR_MIN)
            return
        if rms < self.floor * (1.0 + SNS_FLOOR_SNAP_EPS):
            self.below += 1
            if self.below >= 3:
                target = max(rms, SNS_RMS_FLOOR_MIN)
                a = min(1.0, dt / 0.11)
                self.floor += a * (target - self.floor)
        else:
            self.below = 0
            ratio = rms / max(self.floor, 1.0)
            d = (ratio - 1.0) / SNS_FLOOR_SOFT_SIG
            self.floor *= (1.0 + SNS_FLOOR_LEAK * dt * math.exp(-(d * d)))
        self.floor = max(self.floor, SNS_RMS_FLOOR_MIN)

    def step(self, rms_mean, dt):
        self.ceiling = max(SNS_RMS_FLOOR_MIN, self.ceiling * math.exp(-0.0025 * dt))
        if rms_mean > self.ceiling:
            self.ceiling = rms_mean
        self.update_floor(rms_mean, dt)
        eff = self.floor * SNS_FLOOR_HEADROOM
        if rms_mean < eff:
            self.energy = 0.0
        else:
            db = 20.0 * math.log10(rms_mean / eff)
            rng = max(1.0, 20.0 * math.log10(self.ceiling / eff))
            self.energy = max(0.0, min(1.0, db / rng))
        return self.energy


# Real breath field, perceptual (breathGlow) domain, with floor=0.32:
FIELD_FLOOR = 0.32     # bloomBreathFloor (live-tuned this session)
FIELD_MEAN  = 0.57     # ~ floor + 0.5*(avgPeak-floor), avgPeak≈0.825
FIELD_PEAK  = 0.83     # brightest pixel's breath peak (breathPeak max ≈ 1.0 → glow)


def load_phases(path):
    import csv
    with open(path) as f:
        rows = [r for r in f if not r.lstrip().startswith("#")]
    rdr = csv.reader(rows)
    next(rdr)  # header
    return [row[3] if len(row) > 3 else "" for row in rdr if row]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "audio_envelope_profile.csv"
    samples = list(load(path))
    phases = load_phases(path)
    st = EnergyState()
    energy5s = 0.0
    prev_t = samples[0][0]

    print(f"{'t':>5} {'phase':<8} {'rms':>7} {'fxE':>4} {'e5s':>4} "
          f"{'×':>5}  field span [dim·····bright]")
    for i, (t, mean, mx) in enumerate(samples):
        dt = max(1e-3, t - prev_t)
        prev_t = t
        e = st.step(mean, dt)
        energy5s += min(1.0, dt / ENERGY5S_TAU) * (e - energy5s)
        mult = 1.0 + STRETCH_GAIN * energy5s

        # Stretch the real field's extremes around the mean.
        dim = max(0.0, FIELD_MEAN + (FIELD_FLOOR - FIELD_MEAN) * mult)
        brt = min(1.0, FIELD_MEAN + (FIELD_PEAK  - FIELD_MEAN) * mult)
        bar = render_bar(dim, brt)
        ph = phases[i] if i < len(phases) else ""
        print(f"{t:5.1f} {ph:<8} {mean:7.0f} {e:4.2f} {energy5s:4.2f} "
              f"{mult:5.2f}  {bar}")


def render_bar(dim, brt):
    # 30-wide bar; the lit span shows the field's dim→bright extent
    lo = int(round(dim * 29))
    hi = int(round(brt * 29))
    return "".join("█" if lo <= i <= hi else "·" for i in range(30))


if __name__ == "__main__":
    main()
