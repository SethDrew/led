"""Adaptive noise floor stress tests.

Loads real airport sensor recording, builds empirical CDF + jitter model,
synthesizes longer adversarial sequences, runs three candidate algorithms,
prints per-scenario comparison tables.
"""
from __future__ import annotations
import struct, math, os, sys
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

BIN_PATH = "/Users/sethdrew/Documents/projects/led/library/test-vectors/phone-sensor-profiles/s24_airport_ambient_30s.bin"
TIMES_PATH = BIN_PATH + ".times"

PACKET_FMT = "<hhhhhhHB"
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # 15

HEADROOM = 1.4            # trigger when rms > floor * HEADROOM
MIC_NOISE_FLOOR_MIN = 50  # hardcoded physical floor (counts)

# ---------------------------------------------------------------------------
# Data loading

def load_bin(path, fallback_dt=0.105):
    with open(path, "rb") as f: raw = f.read()
    n = len(raw) // PACKET_SIZE
    rms = np.empty(n, dtype=np.float64)
    mic_on = np.empty(n, dtype=bool)
    for i in range(n):
        u = struct.unpack_from(PACKET_FMT, raw, i * PACKET_SIZE)
        rms[i] = u[6]; mic_on[i] = bool(u[7])
    times_path = path + ".times"
    if os.path.exists(times_path):
        with open(times_path) as f:
            times = np.array([float(x) for x in f.read().split()])
        times = times - times[0]
    else:
        times = np.arange(n) * fallback_dt
    return rms[mic_on], times[mic_on]

def load_airport(): return load_bin(BIN_PATH)

# ---------------------------------------------------------------------------
# Synthetic sequence generators (rms_sample, dt) iterables

def from_distribution(rms_pool, n_samples, dt=0.1, scale=1.0, offset=0.0, rng=None):
    """Draw with replacement from empirical airport distribution, scaled/shifted."""
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(rms_pool), n_samples)
    return [(max(0.0, rms_pool[i] * scale + offset), dt) for i in idx]

def constant(level, n, dt=0.1, jitter=0.0, rng=None):
    rng = rng or np.random.default_rng(0)
    return [(max(0.0, level + rng.normal(0, jitter * level)), dt) for _ in range(n)]

def ramp(lo, hi, n, dt=0.1, jitter=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    out = []
    for i in range(n):
        base = lo + (hi - lo) * i / max(n - 1, 1)
        out.append((max(0.0, base + rng.normal(0, jitter * base)), dt))
    return out

def shout_burst(base_rms, peak_rms, n, dt=0.1, rng=None):
    """Voice-like: 4 Hz AM, bursts and pauses."""
    rng = rng or np.random.default_rng(0)
    out = []
    for i in range(n):
        t = i * dt
        # 4 Hz AM with breath pauses every ~2s
        am = 0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)
        breath = 0.0 if (math.sin(2 * math.pi * 0.5 * t) < -0.7) else 1.0
        v = base_rms + (peak_rms - base_rms) * am * breath
        v += rng.normal(0, 0.05 * v)
        out.append((max(0.0, v), dt))
    return out

def concat(*seqs):
    out = []
    for s in seqs: out.extend(s)
    return out

# ---------------------------------------------------------------------------
# Algorithms — each is a class with .update(rms, dt) -> floor

class P4LinearClamped:
    """P4 (min-snap + linear leak) + clamped ceiling: MIN_FLOOR <= f <= 5*long_min."""
    def __init__(self, snap_alpha=0.3, leak_rate=1.5, long_drift_up=0.05, ceiling_mult=5.0):
        self.snap_alpha = snap_alpha
        self.leak_rate = leak_rate
        self.long_drift_up = long_drift_up
        self.ceiling_mult = ceiling_mult
        self.floor = None
        self.long_min = None

    def update(self, rms, dt):
        if self.floor is None: self.floor = rms; self.long_min = rms
        # long_min: snap down on lows, very slow drift up
        if rms < self.long_min: self.long_min = rms
        else: self.long_min += self.long_drift_up * dt
        # floor: snap down / linear leak up
        if rms < self.floor: self.floor += self.snap_alpha * (rms - self.floor)
        else: self.floor += self.leak_rate * dt
        # clamp
        ceil = self.ceiling_mult * self.long_min
        self.floor = max(MIC_NOISE_FLOOR_MIN, min(self.floor, ceil))
        return self.floor

class P4MultSoft:
    """P4-style snap + multiplicative leak gated by soft-weight ratio."""
    def __init__(self, snap_alpha=0.3, leak_rate=0.005, sigma=0.6, snap_epsilon=0.05):
        self.snap_alpha = snap_alpha
        self.leak_rate = leak_rate
        self.sigma = sigma
        self.snap_epsilon = snap_epsilon
        self.floor = None

    def update(self, rms, dt):
        if self.floor is None: self.floor = max(rms, MIC_NOISE_FLOOR_MIN); return self.floor
        if rms < self.floor * (1 + self.snap_epsilon):
            self.floor += self.snap_alpha * (rms - self.floor)
        else:
            ratio = rms / max(self.floor, 1.0)
            weight = math.exp(-((ratio - 1.0) / self.sigma) ** 2)
            self.floor *= (1 + self.leak_rate * dt * weight)
        self.floor = max(MIC_NOISE_FLOOR_MIN, self.floor)
        return self.floor

class P4Guarded:
    """P4-mult + soft-weight + breath-pause guard. Snap-down requires N
    consecutive below-floor samples AND target floored at fraction of long
    minimum, so single dips don't collapse floor below true ambient."""
    def __init__(self, snap_alpha=0.3, leak_rate=0.005, sigma=0.6,
                 snap_epsilon=0.05, snap_consecutive=3,
                 snap_min_floor_ratio=0.4, long_drift_rate=0.001):
        self.snap_alpha = snap_alpha
        self.leak_rate = leak_rate
        self.sigma = sigma
        self.snap_epsilon = snap_epsilon
        self.snap_consecutive = snap_consecutive
        self.snap_min_floor_ratio = snap_min_floor_ratio
        self.long_drift_rate = long_drift_rate
        self.floor = None
        self.long_min = None
        self.below_count = 0

    def update(self, rms, dt):
        if self.floor is None:
            self.floor = max(rms, MIC_NOISE_FLOOR_MIN)
            self.long_min = self.floor
            return self.floor
        if rms < self.long_min: self.long_min = max(rms, MIC_NOISE_FLOOR_MIN)
        else: self.long_min *= (1 + self.long_drift_rate * dt)
        if rms < self.floor * (1 + self.snap_epsilon):
            self.below_count += 1
            if self.below_count >= self.snap_consecutive:
                target = max(rms, self.snap_min_floor_ratio * self.long_min)
                self.floor += self.snap_alpha * (target - self.floor)
        else:
            self.below_count = 0
            ratio = rms / max(self.floor, 1.0)
            weight = math.exp(-((ratio - 1.0) / self.sigma) ** 2)
            self.floor *= (1 + self.leak_rate * dt * weight)
        return max(MIC_NOISE_FLOOR_MIN, self.floor)

class P2Soft:
    """P2 lower-envelope with soft Gaussian weight replacing hard 1.5x gate."""
    def __init__(self, tau_down=8.0, tau_up=120.0, sigma=0.5):
        self.tau_down = tau_down
        self.tau_up = tau_up
        self.sigma = sigma
        self.floor = None

    def update(self, rms, dt):
        if self.floor is None: self.floor = max(rms, MIC_NOISE_FLOOR_MIN); return self.floor
        if rms < self.floor:
            self.floor += (rms - self.floor) * (1 - math.exp(-dt / self.tau_down))
        else:
            ratio = rms / max(self.floor, 1.0)
            weight = math.exp(-((ratio - 1.0) / self.sigma) ** 2)
            self.floor += weight * (rms - self.floor) * (1 - math.exp(-dt / self.tau_up))
        self.floor = max(MIC_NOISE_FLOOR_MIN, self.floor)
        return self.floor

ALGORITHMS = {
    "P4-mult+soft":    P4MultSoft,
    "P2-soft":         P2Soft,
    "P4-guarded":      P4Guarded,
}

# ---------------------------------------------------------------------------
# Scenario driver

@dataclass
class SegmentLabel:
    name: str
    start_idx: int
    end_idx: int
    intent: str  # "ambient" or "event"

def run_algo(algo, seq):
    floors = np.empty(len(seq))
    rmss = np.empty(len(seq))
    triggers = np.empty(len(seq), dtype=bool)
    for i, (rms, dt) in enumerate(seq):
        f = algo.update(rms, dt)
        floors[i] = f
        rmss[i] = rms
        triggers[i] = rms > f * HEADROOM
    return rmss, floors, triggers

def metrics(triggers, labels):
    """Return per-segment trigger rates and overall."""
    out = {}
    for lab in labels:
        seg = triggers[lab.start_idx:lab.end_idx]
        rate = float(seg.mean()) if len(seg) else 0.0
        out[lab.name] = (lab.intent, rate, len(seg))
    return out

# ---------------------------------------------------------------------------
# Scenarios

def build_scenarios(rms_pool):
    rng = np.random.default_rng(42)
    scenarios = {}

    # 1. Boot in airport: replay loop of empirical distribution for 5 min
    seq = from_distribution(rms_pool, 3000, dt=0.1, rng=rng)
    scenarios["airport_5min_loop"] = (seq, [
        SegmentLabel("warmup_first_30s", 0, 300, "ambient"),
        SegmentLabel("settled_after_60s", 600, 3000, "ambient"),
    ])

    # 2. Quiet room boot -> 5 min sustained shouting -> quiet
    quiet1 = constant(120, 300, dt=0.1, jitter=0.1, rng=rng)         # 30s quiet
    shout = shout_burst(400, 2200, 3000, dt=0.1, rng=rng)            # 5 min shouting
    quiet2 = constant(120, 300, dt=0.1, jitter=0.1, rng=rng)         # 30s quiet
    scenarios["quiet_shout5min_quiet"] = (concat(quiet1, shout, quiet2), [
        SegmentLabel("initial_quiet", 0, 300, "ambient"),
        SegmentLabel("shout_first_30s", 300, 600, "event"),
        SegmentLabel("shout_mid", 600, 3000, "event"),
        SegmentLabel("shout_last_30s", 3000, 3300, "event"),
        SegmentLabel("after_quiet", 3300, 3600, "ambient"),
    ])

    # 3. Gradual crowd ramp 400 -> 1800 over 15 min, no events
    seq = ramp(400, 1800, 9000, dt=0.1, rng=rng)
    scenarios["crowd_ramp_15min"] = (seq, [
        SegmentLabel("ramp_first_5min", 0, 3000, "ambient"),
        SegmentLabel("ramp_last_5min", 6000, 9000, "ambient"),
    ])

    # 4. Airport loop + shout train at minute 3 (10 events over 30s, then back to ambient)
    pre = from_distribution(rms_pool, 1800, dt=0.1, rng=rng)         # 3 min ambient
    train = []
    for _ in range(10):
        train.extend(shout_burst(800, 2500, 20, dt=0.1, rng=rng))    # 2s shout
        train.extend(from_distribution(rms_pool, 10, dt=0.1, rng=rng))  # 1s gap
    post = from_distribution(rms_pool, 1800, dt=0.1, rng=rng)        # 3 min ambient
    scenarios["airport_with_shout_train"] = (concat(pre, train, post), [
        SegmentLabel("ambient_pre", 600, 1800, "ambient"),
        SegmentLabel("shout_train", 1800, 1800 + len(train), "event"),
        SegmentLabel("ambient_post", 1800 + len(train) + 300, len(pre) + len(train) + len(post), "ambient"),
    ])

    # 5. Loud->quiet step at minute 5
    loud = from_distribution(rms_pool, 3000, dt=0.1, scale=1.0, rng=rng)
    quiet = constant(120, 3000, dt=0.1, jitter=0.1, rng=rng)
    scenarios["loud_to_quiet_step"] = (concat(loud, quiet), [
        SegmentLabel("loud_pre", 600, 3000, "ambient"),
        SegmentLabel("quiet_first_10s", 3000, 3100, "ambient"),
        SegmentLabel("quiet_settled", 3300, 6000, "ambient"),
    ])

    # 6. Pure HVAC: flat 600 +/- 5% for 10 min
    seq = constant(600, 6000, dt=0.1, jitter=0.05, rng=rng)
    scenarios["hvac_flat_10min"] = (seq, [
        SegmentLabel("hvac_first_min", 0, 600, "ambient"),
        SegmentLabel("hvac_last_min", 5400, 6000, "ambient"),
    ])

    # 7. Sparse intentional: quiet room with one clap every 30s for 10 min
    seq = constant(120, 6000, dt=0.1, jitter=0.1, rng=rng)
    clap_idxs = []
    for t_clap_s in range(30, 600, 30):
        i = t_clap_s * 10
        # 0.5s clap burst
        for j in range(5):
            if i + j < len(seq):
                seq[i + j] = (1500 + rng.normal(0, 100), 0.1)
                clap_idxs.append(i + j)
    scenarios["sparse_claps_10min"] = (seq, [
        SegmentLabel("between_claps", 1000, 6000, "ambient"),
        SegmentLabel("clap_samples", -1, -1, "event"),
    ])
    scenarios["sparse_claps_10min_clap_idxs"] = clap_idxs
    return scenarios

# ---------------------------------------------------------------------------
# Report

def fmt_pct(x): return f"{x*100:5.1f}%"

def print_report(scenarios):
    for name, value in scenarios.items():
        if name.endswith("_clap_idxs"): continue
        seq, labels = value
        print(f"\n=== {name}  (n={len(seq)})")
        header = f"{'segment':28s} {'intent':8s} " + " ".join(f"{a:>16s}" for a in ALGORITHMS)
        print(header)
        results_by_algo = {}
        for aname, AlgoCls in ALGORITHMS.items():
            algo = AlgoCls()
            rmss, floors, triggers = run_algo(algo, seq)
            results_by_algo[aname] = (rmss, floors, triggers)
        # Per-segment trigger rate
        for lab in labels:
            if lab.start_idx < 0:
                # special: clap samples for sparse scenario
                if name == "sparse_claps_10min":
                    clap_idxs = scenarios["sparse_claps_10min_clap_idxs"]
                    row = f"{lab.name:28s} {lab.intent:8s} "
                    for aname in ALGORITHMS:
                        _, _, triggers = results_by_algo[aname]
                        rate = float(np.mean([triggers[i] for i in clap_idxs])) if clap_idxs else 0.0
                        row += f"{fmt_pct(rate):>16s} "
                    print(row)
                continue
            row = f"{lab.name:28s} {lab.intent:8s} "
            for aname in ALGORITHMS:
                _, _, triggers = results_by_algo[aname]
                seg = triggers[lab.start_idx:lab.end_idx]
                rate = float(seg.mean()) if len(seg) else 0.0
                row += f"{fmt_pct(rate):>16s} "
            print(row)
        # Floor at end
        row = f"{'final floor':28s} {'-':8s} "
        for aname in ALGORITHMS:
            _, floors, _ = results_by_algo[aname]
            row += f"{floors[-1]:>16.0f} "
        print(row)

def replay_with_dt(rms, times, n_loops=1):
    """Convert real recording into (rms, dt) sequence, looped."""
    dts = np.diff(times, prepend=times[0]); dts[0] = dts[1] if len(dts) > 1 else 0.1
    one_loop = list(zip(rms, dts))
    return one_loop * n_loops

def build_real_scenarios():
    """Scenarios using ONLY real recordings, stitched."""
    base = "/Users/sethdrew/Documents/projects/led/library/test-vectors/phone-sensor-profiles"
    quiet, qt = load_bin(f"{base}/s24_home_quiet_10s.bin")
    close, ct = load_bin(f"{base}/s24_home_close_talk_10s.bin")
    far,  ft  = load_bin(f"{base}/s24_home_far_talk_10s.bin")
    airport, at = load_bin(f"{base}/s24_airport_ambient_30s.bin")

    scenarios = {}

    # REAL-1: quiet home + close-talk + quiet home (the breath-pause stress case)
    seq = (replay_with_dt(quiet, qt, 3) +     # 30s quiet
           replay_with_dt(close, ct, 6) +     # 60s close talking (with natural pauses)
           replay_with_dt(quiet, qt, 6))      # 60s quiet again
    q_end = len(quiet) * 3
    talk_end = q_end + len(close) * 6
    scenarios["REAL_quiet_close_talk_quiet"] = (seq, [
        SegmentLabel("quiet_pre", 0, q_end, "ambient"),
        SegmentLabel("talk_first_10s", q_end, q_end + len(close), "event"),
        SegmentLabel("talk_full", q_end, talk_end, "event"),
        SegmentLabel("quiet_post_first_10s", talk_end, talk_end + len(quiet), "ambient"),
        SegmentLabel("quiet_post_full", talk_end, talk_end + len(quiet)*6, "ambient"),
    ])

    # REAL-2: airport loop only (steady loud) — verifies airport-as-ambient
    seq = replay_with_dt(airport, at, 30)  # ~5 min of airport
    n = len(seq)
    scenarios["REAL_airport_loop"] = (seq, [
        SegmentLabel("warmup_first_30s", 0, len(airport) * 3, "ambient"),
        SegmentLabel("settled_after_60s", len(airport) * 6, n, "ambient"),
    ])

    # REAL-3: airport ambient + close talking (shouting in loud venue)
    seq = (replay_with_dt(airport, at, 6) +   # 3 min airport
           replay_with_dt(close, ct, 3) +     # 30s close shouting at mic
           replay_with_dt(airport, at, 6))    # 3 min airport
    pre_end = len(airport) * 6
    talk_end = pre_end + len(close) * 3
    scenarios["REAL_airport_close_airport"] = (seq, [
        SegmentLabel("airport_pre_settled", len(airport)*3, pre_end, "ambient"),
        SegmentLabel("close_talk", pre_end, talk_end, "event"),
        SegmentLabel("airport_post_first_30s", talk_end, talk_end + len(airport)*3, "ambient"),
        SegmentLabel("airport_post_settled", talk_end + len(airport)*3, len(seq), "ambient"),
    ])

    # REAL-4: far talking in quiet home (low SNR event detection)
    seq = (replay_with_dt(quiet, qt, 3) +
           replay_with_dt(far, ft, 6) +
           replay_with_dt(quiet, qt, 3))
    q_end = len(quiet) * 3
    talk_end = q_end + len(far) * 6
    scenarios["REAL_quiet_far_talk_quiet"] = (seq, [
        SegmentLabel("quiet_pre", 0, q_end, "ambient"),
        SegmentLabel("far_talk", q_end, talk_end, "event"),
        SegmentLabel("quiet_post", talk_end, len(seq), "ambient"),
    ])

    return scenarios

if __name__ == "__main__":
    rms_pool, times = load_airport()
    print(f"Loaded airport: n={len(rms_pool)} packets, "
          f"min={rms_pool.min():.0f} median={np.median(rms_pool):.0f} "
          f"p95={np.percentile(rms_pool,95):.0f} max={rms_pool.max():.0f}")
    print(f"Time span: {times[-1]:.1f}s, mean dt={np.mean(np.diff(times)):.3f}s")
    print("\n--- REAL RECORDING SCENARIOS ---")
    print_report(build_real_scenarios())
    print("\n--- SYNTHETIC SCENARIOS ---")
    print_report(build_scenarios(rms_pool))
