"""
Collision sim for polycule_rainbow particles.

Algorithm:
  1. Substep when any particle would move > MAX_STEP per frame (CCD).
  2. Per substep, compute earliest pairwise collision time-of-impact (TOI)
     among approaching pairs; advance everyone to that TOI; resolve that
     one pair; repeat until no collisions remain in the substep.
  3. Wall bounces handled before pair resolution at each substep boundary.
  4. Equal-mass elastic exchange with mild randomization; clamp post-speeds
     to [SPEED_MIN, SPEED_MAX] to prevent runaway from random kicks.
  5. Post-collision, nudge particles apart by EPS along their (new) directions
     to avoid immediate re-collision (sticking).

Why earliest-TOI-first: order-independent and naturally handles 3-body
chains (A hits B, then B hits C as a separate, later TOI event). Naive
pairwise sweeps bias toward whichever pair you check first and can
inject energy when two collisions are resolved against the same particle
in one frame.
"""

import random
import math
import sys

NUM_LEDS = 120
NUM_PARTICLES = 5
FPS = 60.0
DT = 1.0 / FPS
FRAMES = 10000

SPEED_MIN = 10.0
SPEED_MAX = 80.0
RANDOM_KICK = 0.08      # +/- 8% speed randomization on collision
COLLISION_RADIUS = 0.5  # particles "touch" when |dx| <= this
EPS = 1e-3              # separation nudge after resolve
MAX_SUBSTEP_DIST = 0.5  # LEDs per substep max => guarantees no tunneling


def init_particles():
    base = (NUM_LEDS - 1) / 3.0
    ps = []
    for i in range(NUM_PARTICLES):
        pos = NUM_LEDS * (i + 0.5) / NUM_PARTICLES
        d = 1 if i % 2 == 0 else -1
        speed = base * random.uniform(0.5, 1.5)
        ps.append([pos, d, speed])
    return ps


def wall_bounce(p):
    if p[0] > NUM_LEDS - 1:
        p[0] = (NUM_LEDS - 1) - (p[0] - (NUM_LEDS - 1))
        p[1] = -1
    elif p[0] < 0:
        p[0] = -p[0]
        p[1] = 1


def earliest_toi(ps, dt_remaining):
    """Return (t, i, j) for earliest approaching pair collision, or None."""
    best = None
    n = len(ps)
    for i in range(n):
        for j in range(i + 1, n):
            xi, di, si = ps[i]
            xj, dj, sj = ps[j]
            vi = di * si
            vj = dj * sj
            dx = xj - xi
            dv = vj - vi
            # approaching if dx and dv have opposite signs
            if dx == 0:
                t = 0.0
            elif dx * dv >= 0:
                continue  # separating or parallel
            else:
                # solve |xi+vi*t - (xj+vj*t)| = COLLISION_RADIUS
                # => (xi-xj) + (vi-vj)*t = +/- R
                # closing: (xi+vi*t) approaches (xj+vj*t)
                # time to contact: (|dx| - R) / |dv|
                t = (abs(dx) - COLLISION_RADIUS) / abs(dv)
                if t < 0:
                    t = 0.0
            if t <= dt_remaining and (best is None or t < best[0]):
                best = (t, i, j)
    return best


def resolve_pair(ps, i, j, stats):
    a, b = ps[i], ps[j]
    va = a[1] * a[2]
    vb = b[1] * b[2]
    # equal mass elastic: swap velocities
    new_va = vb
    new_vb = va
    # randomize magnitudes slightly
    sa = abs(new_va) * random.uniform(1 - RANDOM_KICK, 1 + RANDOM_KICK)
    sb = abs(new_vb) * random.uniform(1 - RANDOM_KICK, 1 + RANDOM_KICK)
    sa = max(SPEED_MIN, min(SPEED_MAX, sa))
    sb = max(SPEED_MIN, min(SPEED_MAX, sb))
    a[1] = 1 if new_va > 0 else (-1 if new_va < 0 else a[1])
    b[1] = 1 if new_vb > 0 else (-1 if new_vb < 0 else b[1])
    # if either ended up zero velocity, give it a small kick
    if new_va == 0:
        a[1] = -b[1]
    if new_vb == 0:
        b[1] = -a[1]
    a[2] = sa
    b[2] = sb
    # separate
    if a[0] < b[0]:
        a[0] -= EPS
        b[0] += EPS
    else:
        a[0] += EPS
        b[0] -= EPS
    stats['collisions'] += 1


def step(ps, dt, stats):
    # substep so no particle moves more than MAX_SUBSTEP_DIST
    max_v = max(p[2] for p in ps)
    n_sub = max(1, int(math.ceil(max_v * dt / MAX_SUBSTEP_DIST)))
    sub_dt = dt / n_sub
    for _ in range(n_sub):
        remaining = sub_dt
        guard = 0
        while remaining > 1e-12:
            guard += 1
            if guard > 50:
                stats['guard_aborts'] += 1
                break
            ev = earliest_toi(ps, remaining)
            if ev is None:
                for p in ps:
                    p[0] += p[1] * p[2] * remaining
                remaining = 0.0
            else:
                t, i, j = ev
                for p in ps:
                    p[0] += p[1] * p[2] * t
                resolve_pair(ps, i, j, stats)
                remaining -= t
        for p in ps:
            wall_bounce(p)


def kinetic(ps):
    return sum(0.5 * p[2] * p[2] for p in ps)


def run(label, ps, frames=FRAMES):
    stats = {'collisions': 0, 'guard_aborts': 0, 'tunnels': 0,
             'energy_spikes': 0, 'stuck_frames': 0}
    ke0 = kinetic(ps)
    ke_max = ke0
    last_positions = [p[0] for p in ps]
    stuck_run = [0] * len(ps)
    for f in range(frames):
        prev = [(p[0], p[1]) for p in ps]
        step(ps, DT, stats)
        # tunneling check: did any pair swap order without a collision event?
        # (we count via order changes vs collision count proxy)
        for i, p in enumerate(ps):
            if abs(p[0] - last_positions[i]) < 1e-4 and p[2] > SPEED_MIN:
                stuck_run[i] += 1
                if stuck_run[i] > 10:
                    stats['stuck_frames'] += 1
            else:
                stuck_run[i] = 0
            last_positions[i] = p[0]
        ke = kinetic(ps)
        if ke > ke_max:
            ke_max = ke
        if ke > ke0 * 5 and ke0 > 0:
            stats['energy_spikes'] += 1
        # bounds check
        for p in ps:
            if p[0] < -1 or p[0] > NUM_LEDS:
                stats['tunnels'] += 1
    print(f"\n=== {label} ===")
    print(f"  frames: {frames}")
    print(f"  collisions: {stats['collisions']}")
    print(f"  KE start: {ke0:.1f}  max: {ke_max:.1f}  end: {kinetic(ps):.1f}")
    print(f"  energy_spike_frames(>5x): {stats['energy_spikes']}")
    print(f"  out_of_bounds: {stats['tunnels']}")
    print(f"  stuck_frames: {stats['stuck_frames']}")
    print(f"  guard_aborts: {stats['guard_aborts']}")
    return stats


def main():
    random.seed(0xC0FFEE)

    # 1. nominal
    run("nominal", init_particles())

    # 2. all stacked at same position
    stacked = [[60.0, 1 if i % 2 == 0 else -1, 40.0 + i] for i in range(5)]
    run("all-stacked-at-60", stacked, frames=2000)

    # 3. corner pile-up (3 against left wall)
    corner = [[0.5, -1, 70.0], [1.0, -1, 60.0], [1.5, -1, 50.0],
              [80.0, 1, 30.0], [100.0, -1, 30.0]]
    run("corner-pileup-left", corner, frames=2000)

    # 4. high-speed head-on (would tunnel without CCD)
    fast = [[10.0, 1, SPEED_MAX], [11.0, -1, SPEED_MAX],
            [50.0, 1, SPEED_MAX], [80.0, -1, SPEED_MAX],
            [115.0, -1, SPEED_MAX]]
    run("high-speed-headon", fast, frames=2000)

    # 5. tight 3-body near wall
    tight = [[0.6, -1, 50.0], [2.0, 1, 45.0], [3.5, -1, 55.0],
             [60.0, 1, 30.0], [110.0, -1, 35.0]]
    run("tight-3body-wall", tight, frames=2000)


if __name__ == '__main__':
    main()
