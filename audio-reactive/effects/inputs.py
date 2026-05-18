"""
First-class inputs for audio-reactive effects.

Each function converts raw bytes from a physical sensor into a normalized
representation effects can rely on. Each function has a 1:1 entry in
INPUT_CATALOG so the UI can show what an effect needs.

Effects own their own state dicts (no globals). All helpers stateless on
the module side.
"""

import math


# ── Shared constants ─────────────────────────────────────────────
POT_MAX = 1023
ACCEL_BASELINE_ALPHA = 0.008   # ~5s @ 25Hz — slow gravity tracker
GYRO_COUNTS_PER_DPS = 131.0    # MPU-6050 ±250 dps full-scale
AMAG_TAP_THRESH = 130          # v1 telemetry amag_max threshold for tap


# ── Pot ───────────────────────────────────────────────────────────
def pot_position(raw_0_1023, state, alpha=0.3, deadzone_raw=6.0):
    """Pot raw 0-1023 → smoothed value in [0..1].

    state = {'smoothed': float}. Caller seeds with starting value (e.g. 512).
    EMA with absolute deadzone to reject ADC jitter.
    """
    if 'smoothed' not in state:
        state['smoothed'] = float(raw_0_1023)
    raw = float(raw_0_1023)
    if abs(raw - state['smoothed']) > deadzone_raw:
        state['smoothed'] += (raw - state['smoothed']) * alpha
    return state['smoothed'] / POT_MAX


# ── Accel ─────────────────────────────────────────────────────────
def accel_gravity(raw_xyz_g, state, alpha=ACCEL_BASELINE_ALPHA):
    """Slow-EMA gravity vector: tracks the orientation baseline.

    Input: (x, y, z) in g units (already scaled, e.g. raw/16384.0).
    state = {'baseline': [x,y,z], 'ready': bool}.
    Returns (gx, gy, gz) — the slow EMA estimate of gravity.
    """
    x, y, z = float(raw_xyz_g[0]), float(raw_xyz_g[1]), float(raw_xyz_g[2])
    if not state.get('ready'):
        state['baseline'] = [x, y, z]
        state['ready'] = True
    b = state['baseline']
    b[0] += (x - b[0]) * alpha
    b[1] += (y - b[1]) * alpha
    b[2] += (z - b[2]) * alpha
    return (b[0], b[1], b[2])


def accel_shake(raw_xyz_g, state, alpha=ACCEL_BASELINE_ALPHA):
    """High-pass accel: raw - gravity baseline.

    Same baseline math as accel_gravity, but returns the AC component
    (motion/shake) rather than the DC gravity. state must be the SAME
    dict passed to accel_gravity if both are wanted; this function
    updates the baseline too.
    """
    x, y, z = float(raw_xyz_g[0]), float(raw_xyz_g[1]), float(raw_xyz_g[2])
    if not state.get('ready'):
        state['baseline'] = [x, y, z]
        state['ready'] = True
    b = state['baseline']
    b[0] += (x - b[0]) * alpha
    b[1] += (y - b[1]) * alpha
    b[2] += (z - b[2]) * alpha
    return (x - b[0], y - b[1], z - b[2])


# ── Gyro ──────────────────────────────────────────────────────────
def gyro_rate(raw_xyz_deg_s):
    """3-axis gyro raw counts → scalar magnitude in deg/sec.

    Input: (gx, gy, gz) in raw MPU-6050 counts at ±250 dps full-scale.
    Returns sqrt(gx²+gy²+gz²)/131.0.
    """
    gx, gy, gz = raw_xyz_deg_s
    return math.sqrt(gx * gx + gy * gy + gz * gz) / GYRO_COUNTS_PER_DPS


# ── Tap / key events ──────────────────────────────────────────────
def tap_event(magnitude, state, threshold=AMAG_TAP_THRESH, cooldown_s=0.15, now=0.0):
    """Edge-triggered tap detector with cooldown.

    state = {'last_t': float}. Returns True once when magnitude crosses
    threshold and cooldown has elapsed. Caller passes its own `now`
    (typically self.elapsed).
    """
    if 'last_t' not in state:
        state['last_t'] = -1e9
    if magnitude >= threshold and (now - state['last_t']) > cooldown_s:
        state['last_t'] = now
        return True
    return False


def key_event(key, current_keys, prev_keys):
    """True iff key just transitioned from up to down."""
    return key in current_keys and key not in prev_keys


# ── Catalog ───────────────────────────────────────────────────────
# Each entry: label (UI badge text), wiring (one-line concrete blurb).
# Single source of truth for what each canonical input is + how it gets
# to the runner.
INPUT_CATALOG = {
    'pot_position': {
        'label': 'Pot',
        'wiring': 'Noodles controller ADC (10-bit, 0-1023) over USB at ~30 Hz; CMD 0xFC.',
    },
    'accel_gravity': {
        'label': 'Gravity',
        'wiring': 'Duck sender (38:44:be:45:d9:cc) MPU-6050 accel via ESP-NOW → noodles → USB CMD 0xFB. Slow EMA gives gravity vector.',
    },
    'accel_shake': {
        'label': 'Shake',
        'wiring': 'Duck sender accel minus the slow gravity baseline — high-passed motion only.',
    },
    'gyro_rate': {
        'label': 'Gyro',
        'wiring': 'Duck sender MPU-6050 gyro ±250 deg/s; magnitude in deg/sec.',
    },
    'tap_event': {
        'label': 'Tap',
        'wiring': 'v1 telemetry sender (gyro-sense, 200 Hz inner sampling) amag_max @ 25 Hz over ESP-NOW; CMD 0xFA.',
    },
    'key_event': {
        'label': 'Key',
        'wiring': 'Viewer stdin → runner key listener. Digit1 etc.',
    },
    'audio': {
        'label': 'Audio',
        'wiring': 'sounddevice input (physical mic, BlackHole virtual device, or --wav file playback), mono 44.1 kHz chunks → process_audio.',
    },
}
