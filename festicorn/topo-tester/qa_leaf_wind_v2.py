#!/usr/bin/env python3
"""QA v2 of bulb-fleet Leaf Wind effect (post-fix).

Updated to match bulb_fleet.cpp:
  wind angle -45deg, wind_speed 0.35, damping 0.85, boost 0.25,
  kill bounds ±0.05, spawn cycles vertices [0, 1, 4, 5].
"""
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = '/Users/sethdrew/Documents/projects/led/festicorn/topo-tester'

VERTICES = [
    (0.00, 0.50), (0.25, 0.85), (0.50, 0.95), (0.78, 0.80),
    (0.95, 0.50), (0.55, 0.10), (0.25, 0.15),
]
STRIP_PATHS = {
    0: [(0.00,0.50), (0.25,0.15), (0.50,0.95), (0.95,0.50)],
    2: [(0.00,0.50), (0.25,0.85), (0.55,0.10), (0.72,0.70)],
    4: [(0.00,0.50), (0.78,0.80), (0.55,0.10), (0.40,0.12)],
    5: [(0.00,0.50), (0.95,0.50), (0.25,0.85)],
}
STRIP_COLORS = {0:'blue', 2:'orange', 4:'black', 5:'deeppink'}
ACTIVE_STRIPS = [0, 2, 4, 5]
LEDS_PER_STRIP = 100

def interpolate_path(waypoints, n):
    cum = [0.0]
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0]-waypoints[i-1][0]
        dy = waypoints[i][1]-waypoints[i-1][1]
        cum.append(cum[-1] + math.sqrt(dx*dx+dy*dy))
    total = cum[-1]
    pts=[]
    for i in range(n):
        t = i/(n-1); dist = t*total; seg = 0
        for w in range(1,len(waypoints)):
            if dist <= cum[w] or w == len(waypoints)-1:
                seg = w-1; break
        seg_len = cum[seg+1]-cum[seg]
        frac = (dist-cum[seg])/seg_len if seg_len>1e-6 else 0.0
        frac = min(frac,1.0)
        x = waypoints[seg][0] + frac*(waypoints[seg+1][0]-waypoints[seg][0])
        y = waypoints[seg][1] + frac*(waypoints[seg+1][1]-waypoints[seg][1])
        pts.append((x,y))
    return pts
LED_POS = {s: np.array(interpolate_path(STRIP_PATHS[s], LEDS_PER_STRIP))
           for s in ACTIVE_STRIPS}

# ── v2 constants ────────────────────────────────────────────────
LW_MAX_LEAVES     = 10
LW_GLOW_RADIUS    = 0.08
LW_GLOW_SQ2       = 2.0 * LW_GLOW_RADIUS * LW_GLOW_RADIUS
LW_WIND_SPEED     = 0.35
LW_SPAWN_INTERVAL = 0.5
LW_FADE_IN        = 0.4
LW_DAMPING        = 0.85
LW_TURBULENCE     = 0.3
LW_BOOST_SPEED    = 0.25
LW_BOOST_TC       = 1.5
LW_WIND_ANGLE     = -45.0 * math.pi / 180.0
LW_SPAWN_VERTS    = [0, 1, 4, 5]

LW_PALETTE = [(255,140,20),(240,100,10),(220,60,5),(200,40,10),
              (180,30,5),(255,180,40),(160,25,5)]
WIND_DX = math.cos(LW_WIND_ANGLE)
WIND_DY = math.sin(LW_WIND_ANGLE)

def lw_noise_1d(pos, t, seed):
    return (math.sin(pos*0.4 + t*0.3 + seed*7.3)
          * math.cos(pos*0.17 - t*0.19 + seed*3.1)
          + math.sin(pos*0.09 + t*0.13 + seed*1.7) * 0.5) / 1.5

class Leaf:
    __slots__ = ('x','y','vx','vy','bx','by','age','brightness',
                 'r','g','b','active','id','trail','spawn_frame',
                 'die_frame','spawn_vert')
    def __init__(self): self.active=False

random.seed(42)
def esp_random(): return random.randint(0, 2**32-1)

leaves = [Leaf() for _ in range(LW_MAX_LEAVES)]
next_leaf_id = [0]
next_spawn_vert = [0]
spawned = []

def spawn_leaf(frame):
    for lf in leaves:
        if lf.active: continue
        lf.active = True
        lf.id = next_leaf_id[0]; next_leaf_id[0] += 1
        vi = LW_SPAWN_VERTS[next_spawn_vert[0]]
        next_spawn_vert[0] = (next_spawn_vert[0] + 1) % len(LW_SPAWN_VERTS)
        lf.spawn_vert = vi
        lf.x = VERTICES[vi][0] + ((esp_random() % 100) - 50)/2000.0
        lf.y = VERTICES[vi][1] + ((esp_random() % 100) - 50)/2000.0
        boost_mag = LW_BOOST_SPEED * (0.5 + (esp_random() % 100)/200.0)
        lf.bx = WIND_DX * boost_mag; lf.by = WIND_DY * boost_mag
        lf.vx=0.0; lf.vy=0.0; lf.age=0.0; lf.brightness=0.0
        ci = esp_random() % len(LW_PALETTE)
        lf.r,lf.g,lf.b = LW_PALETTE[ci]
        lf.trail = [(lf.x, lf.y)]
        lf.spawn_frame = frame; lf.die_frame = None
        spawned.append({'id': lf.id, 'leaf': lf, 'vert': vi, 'spawn_frame': frame})
        return

FPS=40; DT=0.025; DURATION=10.0
N_FRAMES = int(DURATION/DT)
GLOBAL_SPEED = 1.0
lw_time=0.0; lw_spawn_timer=0.0

hist = {s: np.zeros((N_FRAMES, LEDS_PER_STRIP, 3), dtype=np.uint8) for s in ACTIVE_STRIPS}
leaf_positions_per_frame=[]
strip_touch={}

def render_frame(f):
    global lw_time, lw_spawn_timer
    dt = min(DT, 0.1); spd = GLOBAL_SPEED
    lw_time += dt*spd; lw_spawn_timer += dt*spd
    interval = LW_SPAWN_INTERVAL / max(spd, 0.1)
    while lw_spawn_timer >= interval:
        lw_spawn_timer -= interval; spawn_leaf(f)
    boost_decay = math.exp(-dt / LW_BOOST_TC)
    for i, lf in enumerate(leaves):
        if not lf.active: continue
        noise = lw_noise_1d(lf.x*5.0 + lf.y*3.0, lw_time, i)
        speed_mult = max(0.1, 1.0 + noise * LW_TURBULENCE)
        noise_perp = lw_noise_1d(lf.y*4.0 - lf.x*2.0, lw_time + 100.0, i + 37)
        fx = WIND_DX * LW_WIND_SPEED * speed_mult * spd
        fy = WIND_DY * LW_WIND_SPEED * speed_mult * spd
        fx += (-WIND_DY) * noise_perp * LW_WIND_SPEED * 0.3 * spd
        fy += ( WIND_DX) * noise_perp * LW_WIND_SPEED * 0.3 * spd
        lf.vx = lf.vx * LW_DAMPING + fx * (1.0 - LW_DAMPING)
        lf.vy = lf.vy * LW_DAMPING + fy * (1.0 - LW_DAMPING)
        lf.bx *= boost_decay; lf.by *= boost_decay
        lf.x += (lf.vx + lf.bx) * dt
        lf.y += (lf.vy + lf.by) * dt
        lf.age += dt
        lf.brightness = (lf.age/LW_FADE_IN) if lf.age < LW_FADE_IN else 1.0
        lf.trail.append((lf.x, lf.y))
        if lf.x < -0.05 or lf.x > 1.05 or lf.y < -0.05 or lf.y > 1.05:
            lf.active = False; lf.die_frame = f
    snap = [(lf.x, lf.y, lf.r, lf.g, lf.b, lf.brightness, lf.id, lf.spawn_vert)
            for lf in leaves if lf.active]
    leaf_positions_per_frame.append(snap)
    active = [lf for lf in leaves if lf.active]
    for s in ACTIVE_STRIPS:
        if not active: continue
        pos = LED_POS[s]
        total_glow = np.zeros(LEDS_PER_STRIP)
        cr = np.zeros(LEDS_PER_STRIP); cg = np.zeros(LEDS_PER_STRIP); cb = np.zeros(LEDS_PER_STRIP)
        for lf in active:
            dx = pos[:,0]-lf.x; dy = pos[:,1]-lf.y
            distSq = dx*dx + dy*dy
            intensity = np.exp(-distSq / LW_GLOW_SQ2) * lf.brightness
            mask = intensity >= 0.005
            total_glow[mask] += intensity[mask]
            cr[mask] += intensity[mask]*lf.r
            cg[mask] += intensity[mask]*lf.g
            cb[mask] += intensity[mask]*lf.b
            touched = np.where(intensity >= 0.05)[0]
            if len(touched):
                strip_touch.setdefault(lf.id, {}).setdefault(s, set()).update(touched.tolist())
        nz = total_glow > 0.01
        if not nz.any(): continue
        cr[nz] /= total_glow[nz]; cg[nz] /= total_glow[nz]; cb[nz] /= total_glow[nz]
        bright = np.minimum(total_glow, 1.0)
        hist[s][f, nz, 0] = (cr[nz]*bright[nz]).astype(np.uint8)
        hist[s][f, nz, 1] = (cg[nz]*bright[nz]).astype(np.uint8)
        hist[s][f, nz, 2] = (cb[nz]*bright[nz]).astype(np.uint8)

for f in range(N_FRAMES): render_frame(f)
for lf in leaves:
    if lf.active and lf.die_frame is None:
        lf.die_frame = N_FRAMES - 1

# Ledogram
fig, axes = plt.subplots(len(ACTIVE_STRIPS), 1, figsize=(14,9), sharex=True)
for ax, s in zip(axes, ACTIVE_STRIPS):
    img = hist[s].transpose(1,0,2)
    ax.imshow(img, aspect='auto', origin='lower',
              extent=[0, DURATION, 0, LEDS_PER_STRIP], interpolation='nearest')
    ax.set_ylabel(f'strip {s}\n({STRIP_COLORS[s]})\nLED idx')
    ax.set_facecolor('black')
axes[-1].set_xlabel('time (s)')
fig.suptitle('Leaf Wind v2 LED-o-gram — fixed params', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/qa_leaf_wind_v2_ledogram.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/qa_leaf_wind_v2_ledogram.png")

# Keyframes
N_KEY=8
key_frames = np.linspace(int(N_FRAMES*0.05), N_FRAMES-1, N_KEY, dtype=int)
fig, axes = plt.subplots(2, 4, figsize=(20,10))
vert_color = {0:'cyan', 1:'magenta', 4:'lime', 5:'yellow'}
for ax, kf in zip(axes.flat, key_frames):
    ax.set_xlim(-0.1, 1.15); ax.set_ylim(-0.1, 1.05)
    ax.set_aspect('equal'); ax.set_facecolor('#101010')
    ax.set_title(f't={kf*DT:.2f}s (frame {kf})', fontsize=10, color='white')
    for s in ACTIVE_STRIPS:
        pos = LED_POS[s]
        ax.plot(pos[:,0], pos[:,1], '-', color=STRIP_COLORS[s], alpha=0.15, lw=1)
    for s in ACTIVE_STRIPS:
        pos = LED_POS[s]
        rgb = hist[s][kf].astype(float)/255.0
        bright = rgb.sum(axis=1)
        ax.scatter(pos[:,0], pos[:,1], c=rgb, s=4 + bright*40, edgecolors='none', zorder=3)
    for (x,y,r,g,b,br,lid,vi) in leaf_positions_per_frame[kf]:
        ax.plot(x, y, 'o', color=(r/255,g/255,b/255), markersize=14,
                markeredgecolor='white', markeredgewidth=0.8, alpha=0.9, zorder=10)
        ax.annotate(f'#{lid}', (x,y), color='white', fontsize=7,
                    ha='center', va='center', zorder=11)
    for vi in LW_SPAWN_VERTS:
        ax.plot(VERTICES[vi][0], VERTICES[vi][1], 'x', color=vert_color[vi],
                markersize=10, mew=2, zorder=12)
        ax.annotate(f'v{vi}', VERTICES[vi], color=vert_color[vi],
                    fontsize=8, xytext=(4,4), textcoords='offset points', zorder=12)
    ax.tick_params(colors='gray', labelsize=7)
fig.suptitle('Leaf Wind v2 keyframes — spawn verts marked (cyan=0, magenta=1, lime=4, yellow=5)',
             fontsize=13, color='black')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/qa_leaf_wind_v2_keyframes.png', dpi=110,
            bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"Saved: {OUT_DIR}/qa_leaf_wind_v2_keyframes.png")

print("\n" + "="*70)
print(f"SIM v2: {DURATION}s @ {FPS}fps, dt={DT}, {N_FRAMES} frames")
print(f"Spawned {len(spawned)} leaves total")
print(f"Wind: -45deg → dir=({WIND_DX:+.3f},{WIND_DY:+.3f}), speed={LW_WIND_SPEED}")
print(f"Spawn cycles vertices {LW_SPAWN_VERTS}")
print("="*70)

vert_counts = {v:0 for v in LW_SPAWN_VERTS}
for rec in spawned:
    vert_counts[rec['vert']] += 1
    lf = rec['leaf']
    t = lf.trail
    dist = math.dist(t[0], t[-1])
    spawn_f = rec['spawn_frame']
    end_f = lf.die_frame if lf.die_frame is not None else N_FRAMES-1
    alive = end_f - spawn_f + 1
    touched = strip_touch.get(rec['id'], {})
    ts = []
    for s in ACTIVE_STRIPS:
        if s in touched:
            idxs = sorted(touched[s])
            ts.append(f"s{s}[{min(idxs)}-{max(idxs)},n={len(idxs)}]")
        else:
            ts.append(f"s{s}[--]")
    status = 'died' if (lf.die_frame is not None and lf.die_frame < N_FRAMES-1) else 'alive@end'
    print(f"#{rec['id']:2d} v{rec['vert']}: start=({t[0][0]:+.2f},{t[0][1]:+.2f}) "
          f"end=({t[-1][0]:+.2f},{t[-1][1]:+.2f}) travel={dist:.2f} "
          f"frames={alive:3d} {status}  {' '.join(ts)}")

print("\n" + "="*70)
print("ANALYSIS v2")
print("="*70)
print(f"Spawn-vertex distribution: {vert_counts}")

travels = [math.dist(rec['leaf'].trail[0], rec['leaf'].trail[-1]) for rec in spawned]
print(f"Movement: mean travel={np.mean(travels):.3f}, max={max(travels):.3f}, min={min(travels):.3f}")

multi = sum(1 for st in strip_touch.values() if len(st) >= 2)
all4  = sum(1 for st in strip_touch.values() if len(st) >= 4)
print(f"Multi-strip leaves: {multi}/{len(strip_touch)} touched ≥2 strips, {all4} touched all 4")

quick = sum(1 for rec in spawned
            if rec['leaf'].die_frame is not None
            and (rec['leaf'].die_frame - rec['spawn_frame']) < 5)
print(f"Leaves dying in <5 frames (slot freed fast): {quick}/{len(spawned)}")

alive_end = sum(1 for lf in leaves if lf.active)
print(f"Leaves still alive at end: {alive_end}/{LW_MAX_LEAVES} slots")

print("Per-strip coverage:")
for s in ACTIVE_STRIPS:
    af = int((hist[s].sum(axis=(1,2)) > 0).sum())
    al = int((hist[s].sum(axis=(0,2)) > 0).sum())
    print(f"  strip {s} ({STRIP_COLORS[s]:>9s}): {af}/{N_FRAMES} frames lit, "
          f"{al}/{LEDS_PER_STRIP} unique LEDs ever lit")

all_four = all((hist[s].sum() > 0) for s in ACTIVE_STRIPS)
print(f"\nALL 4 STRIPS COVERED: {'✓ YES' if all_four else '✗ NO'}")
