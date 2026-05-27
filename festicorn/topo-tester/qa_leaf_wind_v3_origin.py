#!/usr/bin/env python3
"""v3: investigate flashes near origin (vertex 0).

Reuses v2 sim, adds origin-specific diagnostics:
  - per-frame brightness at LED 0 on every strip (all strips start at v0)
  - sudden brightness spikes (jump >50% then drop back)
  - leaf count within glow radius of v0
  - boost magnitude vs steady-state velocity
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
ACTIVE_STRIPS = [0,2,4,5]
LEDS_PER_STRIP = 100

def interpolate_path(waypoints, n):
    cum=[0.0]
    for i in range(1,len(waypoints)):
        dx=waypoints[i][0]-waypoints[i-1][0]; dy=waypoints[i][1]-waypoints[i-1][1]
        cum.append(cum[-1]+math.sqrt(dx*dx+dy*dy))
    total=cum[-1]; pts=[]
    for i in range(n):
        t=i/(n-1); dist=t*total; seg=0
        for w in range(1,len(waypoints)):
            if dist<=cum[w] or w==len(waypoints)-1: seg=w-1; break
        sl=cum[seg+1]-cum[seg]
        frac=(dist-cum[seg])/sl if sl>1e-6 else 0.0
        frac=min(frac,1.0)
        x=waypoints[seg][0]+frac*(waypoints[seg+1][0]-waypoints[seg][0])
        y=waypoints[seg][1]+frac*(waypoints[seg+1][1]-waypoints[seg][1])
        pts.append((x,y))
    return pts
LED_POS = {s: np.array(interpolate_path(STRIP_PATHS[s], LEDS_PER_STRIP)) for s in ACTIVE_STRIPS}

LW_MAX_LEAVES=10; LW_GLOW_RADIUS=0.08
LW_GLOW_SQ2=2.0*LW_GLOW_RADIUS*LW_GLOW_RADIUS
LW_WIND_SPEED=0.35; LW_SPAWN_INTERVAL=0.5; LW_FADE_IN=0.4
LW_DAMPING=0.85; LW_TURBULENCE=0.3
LW_BOOST_SPEED=0.25; LW_BOOST_TC=1.5
LW_WIND_ANGLE=-45.0*math.pi/180.0
LW_SPAWN_VERTS=[0,1,4,5]
LW_PALETTE=[(255,140,20),(240,100,10),(220,60,5),(200,40,10),
            (180,30,5),(255,180,40),(160,25,5)]
WIND_DX=math.cos(LW_WIND_ANGLE); WIND_DY=math.sin(LW_WIND_ANGLE)

def lw_noise_1d(pos,t,seed):
    return (math.sin(pos*0.4+t*0.3+seed*7.3)*math.cos(pos*0.17-t*0.19+seed*3.1)
          + math.sin(pos*0.09+t*0.13+seed*1.7)*0.5)/1.5

class Leaf:
    __slots__=('x','y','vx','vy','bx','by','age','brightness',
               'r','g','b','active','id','spawn_vert','spawn_frame','die_frame',
               'spawn_speed','spawn_boost_mag')
    def __init__(self): self.active=False

random.seed(42)
def esp_random(): return random.randint(0,2**32-1)

leaves=[Leaf() for _ in range(LW_MAX_LEAVES)]
next_leaf_id=[0]; next_spawn_vert=[0]
spawned=[]
spawn_events=[]  # (frame, leaf_id, vert, boost_mag)

def spawn_leaf(frame):
    for lf in leaves:
        if lf.active: continue
        lf.active=True
        lf.id=next_leaf_id[0]; next_leaf_id[0]+=1
        vi=LW_SPAWN_VERTS[next_spawn_vert[0]]
        next_spawn_vert[0]=(next_spawn_vert[0]+1)%len(LW_SPAWN_VERTS)
        lf.spawn_vert=vi
        lf.x=VERTICES[vi][0]+((esp_random()%100)-50)/2000.0
        lf.y=VERTICES[vi][1]+((esp_random()%100)-50)/2000.0
        boost_mag=LW_BOOST_SPEED*(0.5+(esp_random()%100)/200.0)
        lf.bx=WIND_DX*boost_mag; lf.by=WIND_DY*boost_mag
        lf.vx=0.0; lf.vy=0.0; lf.age=0.0; lf.brightness=0.0
        ci=esp_random()%len(LW_PALETTE)
        lf.r,lf.g,lf.b=LW_PALETTE[ci]
        lf.spawn_frame=frame; lf.die_frame=None
        lf.spawn_boost_mag=boost_mag
        spawned.append({'id':lf.id,'vert':vi,'frame':frame,'boost_mag':boost_mag,
                        'leaf':lf, 'spawn_pos':(lf.x,lf.y)})
        spawn_events.append((frame, lf.id, vi, boost_mag))
        return

FPS=40; DT=0.025; DURATION=10.0
N_FRAMES=int(DURATION/DT); GLOBAL_SPEED=1.0
lw_time=0.0; lw_spawn_timer=0.0

# Per-frame brightness at LED 0 of each strip (the v0-anchored LED)
origin_led_brightness = {s: np.zeros(N_FRAMES) for s in ACTIVE_STRIPS}
origin_led_total_glow = {s: np.zeros(N_FRAMES) for s in ACTIVE_STRIPS}
leaves_near_v0_count = np.zeros(N_FRAMES, dtype=int)
leaf_velocities = {}  # leaf_id -> list of (frame, |v|, |v+b|)

V0 = VERTICES[0]

def render_frame(f):
    global lw_time, lw_spawn_timer
    dt=min(DT,0.1); spd=GLOBAL_SPEED
    lw_time+=dt*spd; lw_spawn_timer+=dt*spd
    interval=LW_SPAWN_INTERVAL/max(spd,0.1)
    while lw_spawn_timer>=interval:
        lw_spawn_timer-=interval; spawn_leaf(f)
    boost_decay=math.exp(-dt/LW_BOOST_TC)
    for i,lf in enumerate(leaves):
        if not lf.active: continue
        noise=lw_noise_1d(lf.x*5.0+lf.y*3.0, lw_time, i)
        speed_mult=max(0.1, 1.0+noise*LW_TURBULENCE)
        noise_perp=lw_noise_1d(lf.y*4.0-lf.x*2.0, lw_time+100.0, i+37)
        fx=WIND_DX*LW_WIND_SPEED*speed_mult*spd
        fy=WIND_DY*LW_WIND_SPEED*speed_mult*spd
        fx+=(-WIND_DY)*noise_perp*LW_WIND_SPEED*0.3*spd
        fy+=( WIND_DX)*noise_perp*LW_WIND_SPEED*0.3*spd
        lf.vx=lf.vx*LW_DAMPING + fx*(1.0-LW_DAMPING)
        lf.vy=lf.vy*LW_DAMPING + fy*(1.0-LW_DAMPING)
        lf.bx*=boost_decay; lf.by*=boost_decay
        lf.x+=(lf.vx+lf.bx)*dt; lf.y+=(lf.vy+lf.by)*dt
        lf.age+=dt
        lf.brightness=(lf.age/LW_FADE_IN) if lf.age<LW_FADE_IN else 1.0
        v_mag=math.sqrt(lf.vx*lf.vx+lf.vy*lf.vy)
        vb_mag=math.sqrt((lf.vx+lf.bx)**2+(lf.vy+lf.by)**2)
        leaf_velocities.setdefault(lf.id, []).append((f, v_mag, vb_mag, lf.brightness))
        if lf.x<-0.05 or lf.x>1.05 or lf.y<-0.05 or lf.y>1.05:
            lf.active=False; lf.die_frame=f

    # Count leaves currently within glow_radius of v0
    near = 0
    for lf in leaves:
        if not lf.active: continue
        d = math.sqrt((lf.x-V0[0])**2 + (lf.y-V0[1])**2)
        if d <= LW_GLOW_RADIUS * 1.5:  # 1.5x for "near"
            near += 1
    leaves_near_v0_count[f] = near

    # Render only LED 0 per strip (origin-anchored) and full hist briefly
    active=[lf for lf in leaves if lf.active]
    for s in ACTIVE_STRIPS:
        pos = LED_POS[s][0]  # LED 0
        total_glow=0.0; cr=cg=cb=0.0
        for lf in active:
            dx=pos[0]-lf.x; dy=pos[1]-lf.y
            distSq=dx*dx+dy*dy
            intensity=math.exp(-distSq/LW_GLOW_SQ2)*lf.brightness
            if intensity<0.005: continue
            total_glow+=intensity
            cr+=intensity*lf.r; cg+=intensity*lf.g; cb+=intensity*lf.b
        origin_led_total_glow[s][f] = total_glow
        if total_glow > 0.01:
            bright = min(total_glow, 1.0)
            cr/=total_glow; cg/=total_glow; cb/=total_glow
            # sum of channels / 765 = normalized luminosity approximation
            origin_led_brightness[s][f] = (cr+cg+cb)*bright / 765.0

for f in range(N_FRAMES): render_frame(f)
for lf in leaves:
    if lf.active and lf.die_frame is None: lf.die_frame=N_FRAMES-1

# ── Detect brightness spikes (>50% jump then drop within 3 frames) ───
spikes = {s: [] for s in ACTIVE_STRIPS}
for s in ACTIVE_STRIPS:
    b = origin_led_brightness[s]
    for f in range(2, N_FRAMES-3):
        jump = b[f] - b[f-1]
        if jump > 0.15 and b[f] > 0.3:
            # check if it drops back within 3 frames
            future_min = b[f+1:f+4].min() if f+4 <= N_FRAMES else b[f+1:].min()
            if future_min < b[f] * 0.5:
                spikes[s].append((f, b[f-1], b[f], future_min))

# Glow stacking detection: total_glow > 1.0 means saturation/clipping
saturation = {s: int((origin_led_total_glow[s] > 1.0).sum()) for s in ACTIVE_STRIPS}
peak_glow = {s: float(origin_led_total_glow[s].max()) for s in ACTIVE_STRIPS}

# ── Plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
times = np.arange(N_FRAMES) * DT

# Panel 1: brightness at origin LED per strip
ax = axes[0]
for s in ACTIVE_STRIPS:
    ax.plot(times, origin_led_brightness[s], label=f'strip {s} ({STRIP_COLORS[s]}) LED 0',
            color=STRIP_COLORS[s], lw=1.2)
ax.set_ylabel('LED 0 brightness\n(normalized 0-1)')
ax.set_title('Origin LED (LED 0) brightness over time — per strip')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)
# mark spike events
for s in ACTIVE_STRIPS:
    for (f, b_prev, b_peak, b_after) in spikes[s]:
        ax.axvline(f*DT, color=STRIP_COLORS[s], alpha=0.3, lw=0.5)

# Panel 2: total un-clipped glow at LED 0 (shows stacking)
ax = axes[1]
for s in ACTIVE_STRIPS:
    ax.plot(times, origin_led_total_glow[s], label=f'strip {s}',
            color=STRIP_COLORS[s], lw=1.0)
ax.axhline(1.0, color='red', ls='--', lw=0.8, label='clipping threshold (1.0)')
ax.set_ylabel('TOTAL glow at LED 0\n(pre-clip)')
ax.set_title('Raw glow sum at LED 0 — values >1.0 indicate multi-leaf stacking → clipped flash')
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

# Panel 3: leaves currently within 1.5×glow_radius of v0, plus spawn events at v0
ax = axes[2]
ax.fill_between(times, 0, leaves_near_v0_count, color='gray', alpha=0.4,
                step='pre', label='leaves within 1.5×radius of v0')
ax.set_ylabel('# leaves near v0')
ax.set_xlabel('time (s)')
ax.set_title('Leaves crowding the origin (v0) — and spawn events at v0')
v0_spawns = [(f, bm) for (f, lid, vi, bm) in spawn_events if vi == 0]
for (f, bm) in v0_spawns:
    ax.axvline(f*DT, color='cyan', alpha=0.8, lw=1.2)
    ax.annotate(f'spawn v0\nboost={bm:.3f}', (f*DT, leaves_near_v0_count.max()*0.85),
                fontsize=7, color='darkblue', rotation=0)
ax.set_ylim(0, max(3, leaves_near_v0_count.max() + 1))
ax.legend(loc='upper right', fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/qa_leaf_wind_v3_origin.png', dpi=120, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {OUT_DIR}/qa_leaf_wind_v3_origin.png")

# ── Diagnostics ─────────────────────────────────────────────────
print("\n" + "="*70)
print("ORIGIN (v0 = 0.00, 0.50) DIAGNOSTICS")
print("="*70)
print(f"Spawn boost_mag range: {LW_BOOST_SPEED*0.5:.3f} to {LW_BOOST_SPEED*1.0:.3f} "
      f"(LW_BOOST_SPEED={LW_BOOST_SPEED}, jitter 0.5-1.0)")
print(f"Steady-state wind velocity (terminal): {LW_WIND_SPEED * 1.0:.3f} units/s "
      f"(terminal=fx since EMA settles to fx, i.e. WIND_SPEED at speed_mult=1)")
print(f"Initial frame displacement at spawn from boost only: "
      f"{LW_BOOST_SPEED * DT:.4f} units (one dt step)")
print(f"vs steady-state per-frame: {LW_WIND_SPEED * DT:.4f} units")
print(f"→ boost is {LW_BOOST_SPEED/LW_WIND_SPEED:.2f}x the wind speed → "
      f"initial motion is dominated by boost for ~{LW_BOOST_TC:.1f}s TC")

# Spawn-velocity sanity
print("\nSpawn-frame velocity (|v+b|) per leaf:")
for rec in spawned[:8]:
    lid = rec['id']
    vels = leaf_velocities.get(lid, [])
    if not vels: continue
    f0 = vels[0]
    # mid-life velocity (frame 40 after spawn or earlier)
    mid_idx = min(40, len(vels)-1)
    fmid = vels[mid_idx]
    print(f"  #{lid:2d} v{rec['vert']}: spawn |v+b|={f0[2]:.3f} "
          f"(|v|={f0[1]:.3f}), mid |v+b|={fmid[2]:.3f}, boost_mag={rec['boost_mag']:.3f}")

print(f"\nv0 spawns total: {len(v0_spawns)} → boost magnitudes: "
      f"{[f'{bm:.3f}' for (_,bm) in v0_spawns]}")

print(f"\nMax simultaneous leaves within 1.5×glow of v0: {leaves_near_v0_count.max()}")
print(f"Frames with ≥2 leaves near v0: "
      f"{int((leaves_near_v0_count >= 2).sum())} / {N_FRAMES}")
print(f"Frames with ≥3 leaves near v0: "
      f"{int((leaves_near_v0_count >= 3).sum())} / {N_FRAMES}")

print(f"\nGlow stacking at LED 0 (origin):")
for s in ACTIVE_STRIPS:
    print(f"  strip {s}: peak total glow = {peak_glow[s]:.3f} "
          f"({'CLIPPED' if peak_glow[s] > 1.0 else 'no clip'}), "
          f"{saturation[s]}/{N_FRAMES} frames saturated")

print(f"\nDetected brightness spikes (>50% jump up, drop back within 3 frames):")
total_spikes = sum(len(s) for s in spikes.values())
print(f"  Total: {total_spikes}")
for s in ACTIVE_STRIPS:
    for (f, b_prev, b_peak, b_after) in spikes[s]:
        print(f"  strip {s} t={f*DT:.2f}s: {b_prev:.2f} → {b_peak:.2f} → {b_after:.2f} "
              f"(jump+{b_peak-b_prev:+.2f}, then drop {b_peak-b_after:+.2f})")

# Boost-flash analysis: brightness at spawn frame for v0 spawns
print(f"\nBrightness at LED 0 over first 0.5s of each v0 spawn:")
for (sf, lid, vi, bm) in spawn_events:
    if vi != 0: continue
    end_f = min(sf + 20, N_FRAMES)
    for s in ACTIVE_STRIPS:
        peak = origin_led_total_glow[s][sf:end_f].max()
        peak_f = sf + int(np.argmax(origin_led_total_glow[s][sf:end_f]))
        if peak > 0.5:
            print(f"  spawn #{lid} v0 @ t={sf*DT:.2f}s, boost={bm:.3f}: "
                  f"strip {s} LED 0 peak glow={peak:.2f} at t+{(peak_f-sf)*DT:.2f}s")
