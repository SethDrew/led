#!/usr/bin/env python3
"""
Replay stored v1 wire packets through the bloom energy model.
Compares old (k=4.07) vs fixed (k=19.5, fabsf accel) parameters.
"""

import struct
import sys
import math
from pathlib import Path

MAG_FS = 57000.0
COUNTS_PER_G = 8192.0
COUNTS_PER_DPS = 32.8

SURPRISE_RATIO = 3.0
DRAIN_SCALE = 100.0
DRAIN_ENVELOPE_DECAY = 0.85
FLASH_MOTION_SCALE = 300.0
ENERGY_MULTIPLIER = 1.4
MOTION_SETTLE_MS = 300
BLOOM_RECOVERY_RAMP = 0.033
BLOOM_BREATH_FLOOR = 0.15
BLOOM_BRIGHTNESS_CAP = 0.25

PACKET_HZ = 25.0
RENDER_HZ = 150.0  # approximate bench-bulbs render rate


def mag_counts_from_byte(b):
    n = b / 255.0
    return n * n * MAG_FS

def amag_g(b):
    return mag_counts_from_byte(b) / COUNTS_PER_G

def gmag_dps(b):
    return mag_counts_from_byte(b) / COUNTS_PER_DPS


def simulate(packets, label, drain_k=19.5, use_fabsf=True, ema_tau_up=0.77, ema_tau_down=0.16,
             add_gravity=False, drain_clamp=999.0):
    """add_gravity: if True, add 1g to amag before processing (for AC-coupled sim data)."""
    motion_ema = 0.0
    drain_envelope = 0.0
    colony_energy = 1.0
    hit_intensity = 0.0
    last_motion_ms = 0

    pkt_dt = 1.0 / PACKET_HZ
    render_dt = 1.0 / RENDER_HZ
    renders_per_pkt = int(RENDER_HZ / PACKET_HZ)

    rows = []
    now_ms = 0

    for i, pkt in enumerate(packets):
        now_ms += int(1000 / PACKET_HZ)

        # ── bloomProcessMotion (runs once per packet) ──
        gyro_rate = gmag_dps(pkt['gmag_max'])
        amag_val = amag_g(pkt['amag_max'])
        if add_gravity:
            amag_val += 1.0  # sim data has gravity removed; live boards include it
        raw_accel = amag_val - 1.0
        accel_jolt = abs(raw_accel) if use_fabsf else max(0.0, raw_accel)
        motion_rate = max(gyro_rate, accel_jolt * 300.0)

        surprise = max(0.0, motion_rate - motion_ema * SURPRISE_RATIO)

        alpha = min(1.0, pkt_dt / ema_tau_up) if motion_rate > motion_ema else min(1.0, pkt_dt / ema_tau_down)
        motion_ema += alpha * (motion_rate - motion_ema)

        if surprise > 1.0:
            last_motion_ms = now_ms
            hit_raw = math.log2(1.0 + surprise) / math.log2(1.0 + FLASH_MOTION_SCALE)
            hit_raw = max(0.0, min(1.0, hit_raw))
            if hit_raw > hit_intensity:
                hit_intensity = hit_raw

            norm_motion = min(surprise / DRAIN_SCALE, drain_clamp)
            new_drain = norm_motion ** 3 * (1.0 - DRAIN_ENVELOPE_DECAY)
            if new_drain > drain_envelope:
                drain_envelope = new_drain

        # ── renderQuietBloom (runs every render frame) ──
        for _ in range(renders_per_pkt):
            dt = render_dt
            draining = drain_envelope > 0.001

            if not draining:
                hit_intensity = 0.0

            if now_ms - last_motion_ms > MOTION_SETTLE_MS:
                colony_energy = min(1.0, colony_energy + BLOOM_RECOVERY_RAMP * dt)

            if draining:
                drain = drain_envelope * dt
                drain = min(drain, colony_energy)
                colony_energy -= drain
                drain_envelope *= math.exp(-drain_k * dt)
                if drain_envelope <= 0.001:
                    drain_envelope = 0.0

        rows.append({
            'i': i,
            't_s': i / PACKET_HZ,
            'amag_g': amag_g(pkt['amag_max']),
            'gmag_dps': gmag_dps(pkt['gmag_max']),
            'motion_rate': motion_rate,
            'motion_ema': motion_ema,
            'surprise': surprise,
            'hit_intensity': hit_intensity,
            'drain_envelope': drain_envelope,
            'colony_energy': colony_energy,
        })

    return rows


def load_wire_packets(path):
    data = path.read_bytes()
    n = len(data) // 16
    packets = []
    for i in range(n):
        chunk = data[i*16:(i+1)*16]
        seq, ax_max, ay_max, az_max, ax_min, ay_min, az_min, \
            ax_mean, ay_mean, az_mean, amag_max, amag_mean, \
            gmag_max, gmag_mean, flags = struct.unpack('<HbbbbbbbbbBBBBB', chunk)
        packets.append({
            'seq': seq, 'amag_max': amag_max, 'amag_mean': amag_mean,
            'gmag_max': gmag_max, 'gmag_mean': gmag_mean, 'flags': flags,
        })
    return packets


def print_comparison(old, new):
    print(f"{'t_s':>6} | {'energy_OLD':>10} {'drain_OLD':>10} {'hit_OLD':>8} | {'energy_NEW':>10} {'drain_NEW':>10} {'hit_NEW':>8} | {'gmag':>6}")
    print('-' * 100)

    for o, n in zip(old, new):
        show = False
        if o['surprise'] > 1.0 or n['surprise'] > 1.0:
            show = True
        elif o['drain_envelope'] > 0.001 or n['drain_envelope'] > 0.001:
            show = True
        elif o['i'] % 25 == 0:
            show = True

        if show:
            print(f"{o['t_s']:6.1f} | {o['colony_energy']:10.3f} {o['drain_envelope']:10.5f} {o['hit_intensity']:8.3f} | "
                  f"{n['colony_energy']:10.3f} {n['drain_envelope']:10.5f} {n['hit_intensity']:8.3f} | {o['gmag_dps']:6.0f}")


def stats(rows, label):
    surprises = [r for r in rows if r['surprise'] > 1.0]
    min_e = min(r['colony_energy'] for r in rows)
    max_drain = max(r['drain_envelope'] for r in rows)
    # Time spent below 50% energy
    low_pct = sum(1 for r in rows if r['colony_energy'] < 0.5) / len(rows) * 100
    # Time to recover from min to 0.9
    recovery_start = None
    recovery_time = None
    for r in rows:
        if r['colony_energy'] <= min_e + 0.01 and recovery_start is None:
            recovery_start = r['t_s']
        if recovery_start and r['colony_energy'] >= 0.9:
            recovery_time = r['t_s'] - recovery_start
            break

    print(f"\n  [{label}]")
    print(f"  Surprise events: {len(surprises)}")
    print(f"  Max drain_envelope: {max_drain:.4f}")
    print(f"  Colony energy min: {min_e:.3f}")
    print(f"  Time below 50% energy: {low_pct:.1f}%")
    if recovery_time:
        print(f"  Recovery min→0.9: {recovery_time:.1f}s")


def main():
    default = Path(__file__).parent.parent.parent / 'gyro-sense/data/recordings/wire_packets/20260426T210735.bin'
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    packets = load_wire_packets(path)
    print(f"Loaded {len(packets)} packets ({len(packets)/PACKET_HZ:.1f}s) from {path.name}\n")

    old = simulate(packets, "old", drain_k=4.07, use_fabsf=False, add_gravity=True)
    new = simulate(packets, "new (k=19.5+clamp3)", drain_k=19.5, use_fabsf=True, add_gravity=True, drain_clamp=3.0)

    print_comparison(old, new)

    print("\n=== Summary ===")
    stats(old, "OLD (k=4.07, fmaxf)")
    stats(new, "NEW (k=19.5, fabsf)")


if __name__ == '__main__':
    main()
