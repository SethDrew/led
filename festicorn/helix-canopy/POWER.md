# helix-canopy — Power & Wiring

Running design doc for the power distribution of the helix-canopy install.
Numbers derive from `geometry/meta.json` (1697 LEDs: helix 2×297, canopy 14×50,
roots 7 strands ≈ 403). The Blender wiring model is `tools/wiring_blender.py`
→ renders + `wiring.blend` in `wiring/`. Last updated 2026-07-07.

## Ground rules

- **All LEDs are 5 V.** (helix WS2812, canopy 12 mm WS2811 bullets, roots — all 5 V.)
- **Brightness cap: 2.5 A per 200 LEDs → 12.5 mA/LED** at full effect brightness
  (~21% of WS2812 full-white). This cap is what makes single-end feed viable.
- **Single-end feed everywhere.** At 12.5 mA/LED a 5 m / 150-LED run drops only
  ~0.9–1.5 V at the tip (vs the "inject every 5 m" rule, which is set at full
  white / 60 mA). Confirm with a multimeter on one strand's 5 V rail if the
  strip copper is unusually thin (≥0.5 Ω/m loop → red-shift at the tip).

## Power budget (at the cap)

| Group | LEDs | Current | Power @ 5 V |
|---|---|---|---|
| Helix (2×297) | 594 | 7.43 A | 37 W |
| Canopy (14×50) | 700 | 8.75 A | 44 W |
| Roots (7 strands) | 403 | 5.04 A | 25 W |
| **Total** | **1697** | **21.2 A** | **106 W** |

Base PSU: **one 5 V / 30 A (150 W)** at the trunk foot, ~70% loaded.

## Injection scheme — 18 taps in 3 clusters

- **Base (z ≈ 0), 9 taps:** 2 helix-base (r = 0.38 m, ±x) + 7 root (all from
  trunk center, splaying out). Land straight on the PSU bus bars.
- **Mid-helix (z ≈ 1.77 m / 5.8 ft), 2 taps:** helix LED 150 on each strand.
  (Not halfway up — the wide base turns eat more LEDs per meter of arc.)
- **Canopy hub (z = 3.05 m / 10 ft), 7 taps:** on a 0.35 m ring, each feeds
  2×50 spokes single-fed outward.

Per-tap current: helix 1.88 A (150 LEDs), canopy 1.25 A (100), roots 0.7 A (~56).

## Distribution topology

- **Thin 12 V run up the trunk** to a **12 V→5 V buck at the canopy hub**.
  Carries ~4.1 A (44 W / 12 V ÷ 0.9 buck eff); the high voltage + buck input
  tolerance is why this run can be thin. Eliminates the fat 5 V trunk bus and
  the need for a second PSU up top.
- **Short 5 V run** to the two mid-helix taps at 5.8 ft.
- Everything else taps locally at the base or the hub.

## Recommended AWG per run

At these currents DC gauge is set by **voltage drop**, not heat (every run is
far under even 20 AWG ampacity). The AC cord is set by durability/code, not
current. Lengths are assumptions — **drop scales linearly with length**, so bump
one step if a run is longer (or the mains cord is >50 ft).

| Run | V | Current | ~Length (1-way) | **AWG** | Drop | Driver |
|---|---|---|---|---|---|---|
| AC mains → base PSU | 120 VAC | ~1.2 A | varies | **14–16 (outdoor SJTW)** | negligible | ruggedness/code |
| 12 V trunk → canopy buck | 12 V | 4.1 A | ~3 m | **18** | ~0.5 V | buck tolerates drop |
| 5 V → mid-helix taps | 5 V | 3.76 A | ~1.8 m | **16** | ~0.2 V | 5 V is drop-sensitive |
| 5 V helix-base leads (×2) | 5 V | 1.88 A | <0.5 m | **18** | tiny | short |
| 5 V root leads (×7) | 5 V | 0.7 A | <0.5 m | **18** (20 fine) | tiny | short + low current |
| 5 V canopy leads (×7, buck→tap) | 5 V | 1.25 A | <0.5 m | **18** (20 fine) | tiny | short |

## Open / to confirm

- Base topology: 12 V PSU + two bucks (base + canopy) vs 5 V PSU at base + a
  separate 12 V feed for the canopy run. (Doc assumes 5 V base PSU + 12 V feed.)
- Measure one helix strand's 5 V rail resistance to lock the single-feed margin.
