"""Power-wiring model of the helix-canopy sculpture, for Blender (headless).

Renders the LED cloud + the power-injection scheme we worked out:
  - LEDs colored by group (helix / canopy / roots), dim, for context
  - 18 tap markers (bright): 4 helix (every 150, single-fed), 7 canopy (at the
    hub, each feeds 2x50 spokes), 7 roots (at the base, one per root)
  - a THIN 12 V trunk run up to a buck at the canopy hub (orange)
  - a short 5 V pair to the two mid-helix taps at ~1.77 m (red)
  - the base PSU with its 9-tap floor fan-out (red 5 V leads)

Geometry is read live from ../geometry/{layout.json, meta.json} so taps track
the real LED positions — no hardcoded coords. Run from this dir:

  /Applications/Blender.app/Contents/MacOS/Blender -b -P wiring_blender.py

Writes PNGs + a .blend into ../wiring/ (gitignored scratch).

Design rationale — budget, tap layout, injection scheme, AWG per run — lives in
POWER.md (helix-canopy root). This script is the visual of that doc; keep the
two in sync (currents, tap counts, 12 V-to-buck topology).
"""
import bpy, math, os, json
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
GEO  = os.path.join(HERE, "..", "geometry")
OUT  = os.path.join(HERE, "..", "wiring")
os.makedirs(OUT, exist_ok=True)

# ── load geometry ──────────────────────────────────────────────────────────
P = [p["point"] for p in json.load(open(os.path.join(GEO, "layout.json")))]
META = json.load(open(os.path.join(GEO, "meta.json")))
STRIPS = META["strips"]
HUB_Z = META["canopy_z_m"]                       # canopy plane height

# ── scene reset ──────────────────────────────────────────────────────────────
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)
COLL = bpy.context.scene.collection

# ── material / mesh helpers ──────────────────────────────────────────────────
def mat(name, rgb, emit=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.5
    if emit > 0:
        b.inputs["Emission Color"].default_value = (*rgb, 1)
        b.inputs["Emission Strength"].default_value = emit
    return m

def _mesh(name, verts, faces, material):
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    ob = bpy.data.objects.new(name, me); ob.data.materials.append(material)
    COLL.objects.link(ob); return ob

CUBE_V = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
CUBE_F = [(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]

def dotcloud(name, pts, size, material):
    """One merged mesh of tiny cubes — visible point cloud, cheap."""
    h = size / 2.0; verts = []; faces = []
    for (x, y, z) in pts:
        b = len(verts)
        verts += [(x+vx*h, y+vy*h, z+vz*h) for vx, vy, vz in CUBE_V]
        faces += [tuple(b+i for i in f) for f in CUBE_F]
    return _mesh(name, verts, faces, material)

def node(name, p, r, material):
    x, y, z = p
    v = [(x+r,y,z),(x-r,y,z),(x,y+r,z),(x,y-r,z),(x,y,z+r),(x,y,z-r)]
    f = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    return _mesh(name, v, f, material)

def cyl(name, p0, p1, radius, material, seg=14):
    p0, p1 = Vector(p0), Vector(p1); axis = p1 - p0
    if axis.length < 1e-9: return None
    z = axis.normalized()
    up = Vector((0,0,1)) if abs(z.z) < 0.9 else Vector((1,0,0))
    x = z.cross(up).normalized(); y = z.cross(x).normalized()
    ring = lambda c: [tuple(c + (x*math.cos(2*math.pi*i/seg) +
                                 y*math.sin(2*math.pi*i/seg))*radius) for i in range(seg)]
    verts = ring(p0) + ring(p1); faces = []
    for i in range(seg):
        j = (i+1) % seg; faces.append((i, j, seg+j, seg+i))
    c0 = len(verts); verts.append(tuple(p0)); c1 = len(verts); verts.append(tuple(p1))
    for i in range(seg):
        j = (i+1) % seg; faces += [(c0, j, i), (c1, seg+i, seg+j)]
    return _mesh(name, verts, faces, material)

def box(name, center, dims, material):
    cx, cy, cz = center; dx, dy, dz = (d/2 for d in dims)
    v = [(cx+vx*dx, cy+vy*dy, cz+vz*dz) for vx, vy, vz in CUBE_V]
    return _mesh(name, v, CUBE_F, material)

# ── palette ──────────────────────────────────────────────────────────────────
M_HELIX  = mat("helix",  (0.10, 0.55, 0.85), emit=0.7)   # cyan
M_CANOPY = mat("canopy", (0.75, 0.20, 0.80), emit=0.7)   # violet
M_ROOT   = mat("root",   (0.25, 0.75, 0.25), emit=0.7)   # green
M_12V    = mat("v12",    (1.00, 0.55, 0.05), emit=1.5)   # orange wire
M_5V     = mat("v5",     (0.95, 0.10, 0.10), emit=1.2)   # red wire
M_TAP    = mat("tap",    (1.00, 0.95, 0.55), emit=3.0)   # bright tap node
M_PSU    = mat("psu",    (0.30, 0.30, 0.33))             # grey box
M_BUCK   = mat("buck",   (0.55, 0.55, 0.20))             # olive box

# ── LED cloud, by group ──────────────────────────────────────────────────────
groups = {"helix": [], "canopy": [], "root": []}
for s in STRIPS:
    groups[s["kind"]] += P[s["start"]:s["start"]+s["count"]]
dotcloud("leds_helix",  groups["helix"],  0.022, M_HELIX)
dotcloud("leds_canopy", groups["canopy"], 0.022, M_CANOPY)
dotcloud("leds_root",   groups["root"],   0.022, M_ROOT)

# ── derive taps from geometry ────────────────────────────────────────────────
TAP_R = 0.07
helix_strips  = [s for s in STRIPS if s["kind"] == "helix"]
canopy_strips = [s for s in STRIPS if s["kind"] == "canopy"]
root_strips   = [s for s in STRIPS if s["kind"] == "root"]

# helix: tap at local 0 and 150 (single-fed every 150)
helix_taps = []   # (pos, is_mid)
for s in helix_strips:
    for li in (0, 150):
        if li < s["count"]:
            helix_taps.append((P[s["start"]+li], li > 0))

# canopy: pair adjacent spokes -> tap at midpoint of their inner ends, on the hub ring
canopy_taps = []
for k in range(0, len(canopy_strips), 2):
    a = Vector(P[canopy_strips[k]["start"]])
    b = Vector(P[canopy_strips[k+1]["start"]]) if k+1 < len(canopy_strips) else a
    canopy_taps.append(tuple((a + b) / 2.0))

# roots: tap at start, nudged outward along the root so the 7 don't overlap at origin
root_taps = []
for s in root_strips:
    p0 = Vector(P[s["start"]]); p1 = Vector(P[s["start"]+1])
    d = (p1 - p0); d = d.normalized() if d.length > 1e-6 else Vector((1, 0, 0))
    root_taps.append(tuple(p0 + d*0.18 + Vector((0, 0, 0.03))))

for i, (p, _) in enumerate(helix_taps):  node(f"tap_helix_{i}",  p, TAP_R, M_TAP)
for i, p in enumerate(canopy_taps):      node(f"tap_canopy_{i}", p, TAP_R, M_TAP)
for i, p in enumerate(root_taps):        node(f"tap_root_{i}",   p, TAP_R, M_TAP)

# ── power distribution ───────────────────────────────────────────────────────
PSU  = (0.0, -0.45, 0.10)          # base PSU, just off the trunk foot
FOOT = (0.0,  0.0,  0.10)          # trunk base
box("PSU", PSU, (0.42, 0.26, 0.18), M_PSU)

# 12 V thin trunk run: base -> canopy buck at the hub.
# The two trunk cables are offset off-axis (±x) so the thin 12 V and the 5 V
# read as distinct runs instead of overlapping on the spine.
HUB = (0.0, 0.0, HUB_Z)
cyl("bus_12v", (0.055, 0.0, FOOT[2]), (0.0, 0.0, HUB_Z), 0.008, M_12V)  # thin
box("buck_canopy", HUB, (0.16, 0.16, 0.07), M_BUCK)

# 5 V short run up to the mid-helix taps (the two with is_mid=True)
mid = [Vector(p) for p, is_mid in helix_taps if is_mid]
mid_z = min(m.z for m in mid) if mid else 1.77
cyl("bus_5v_helix", (-0.055, 0.0, FOOT[2]), (-0.055, 0.0, mid_z), 0.015, M_5V)  # fatter
for i, m in enumerate(mid):
    cyl(f"lead_helixmid_{i}", (-0.055, 0.0, m.z), tuple(m), 0.006, M_5V)

# leads: buck -> 7 canopy taps (5 V, post-buck)
for i, p in enumerate(canopy_taps):
    cyl(f"lead_canopy_{i}", HUB, p, 0.006, M_5V)

# leads: PSU -> 9 base taps (2 helix-base + 7 roots), 5 V
base_taps = [p for p, is_mid in helix_taps if not is_mid] + root_taps
for i, p in enumerate(base_taps):
    cyl(f"lead_base_{i}", PSU, p, 0.006, M_5V)

# ── world / light / camera (meters) ──────────────────────────────────────────
world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
bpy.context.scene.world = world; world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.02, 0.025, 0.03, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 1.0

sun_d = bpy.data.lights.new("Sun", 'SUN'); sun_d.energy = 3.0
sun = bpy.data.objects.new("Sun", sun_d)
sun.rotation_euler = (math.radians(55), math.radians(10), math.radians(35))
COLL.objects.link(sun)

cam_d = bpy.data.cameras.new("Cam"); cam = bpy.data.objects.new("Cam", cam_d)
COLL.objects.link(cam); bpy.context.scene.camera = cam
def look_at(obj, frm, to):
    obj.location = Vector(frm)
    obj.rotation_euler = (Vector(frm) - Vector(to)).to_track_quat('Z', 'Y').to_euler()

scn = bpy.context.scene
scn.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in \
    [i.identifier for i in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] \
    else 'BLENDER_EEVEE'
scn.render.resolution_x = 1400; scn.render.resolution_y = 1050

MID = Vector((0, 0, 2.3))
VIEWS = [
    ("wiring_angle", (7.0, -8.0, 4.5), MID),
    ("wiring_base",  (3.0, -3.2, 0.9), Vector((0, 0, 0.6))),   # the floor fan-out
    ("wiring_hub",   (2.2, -2.4, 4.6), Vector((0, 0, 3.05))),  # the canopy split
    ("wiring_top",   (0.01, 0, 12.0),  Vector((0, 0, 0))),
]
for name, pos, tgt in VIEWS:
    look_at(cam, pos, tgt)
    scn.render.filepath = os.path.join(OUT, name + ".png")
    bpy.ops.render.render(write_still=True)
    print("wrote", scn.render.filepath)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "wiring.blend"))
print("saved blend")
