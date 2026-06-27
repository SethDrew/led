# Cubby Presence Sensing — Method Comparison

**Concept:** A series of *dispersed, open-front cubbies* (cloth / wood / etc.). When an item is
placed inside, the cubby lights up. The sensing must be **invisible and undetectable** — an
observer (even an engineer) should not be able to tell *how* it knows.

**Item profile:** light, soft, arbitrary — sock (~30–60g), stuffed animal (~80g–1kg), etc.
No tags/magnets allowed (must work on *any* item the user did not modify).

**Stretch goal:** trigger on the **item resting**, NOT on a **hand** reaching in / pressing.

---

## Comparison

| # | Method | $ | Invisible | Light/soft item (sock) | Hand-vs-item | Dispersed-OK | Main catch |
|---|--------|---|-----------|------------------------|--------------|--------------|------------|
| 1 | Micro-switch under floor | 0.20 | ✓ | ✗ too light to press | ✗ | ✓ | needs real weight |
| 2 | FSR pressure pad | 3 | ✓ | ⚠ borderline | ✗ | ✓ | drifts, hand fools it |
| 3 | Load cell + HX711 | 4 | ✓ | ⚠ near noise floor | ✓ *(w/ tremor)* | ✓ | big board = worse |
| **4** | **ToF (VL53L0X)** ⭐ | **3** | **✓** | **✓ beam don't care** | **✓ *(persistence)*** | **✓** | **no native hand/item** |
| 5 | Reflective IR | 1 | ✓ | ✗ dark cloth absorbs | ✗ | ✓ | item color fools it |
| 6 | Ultrasonic HC-SR04 | 1 | ✗ | ✓ | ✗ | ✓ | big, ugly — skip |
| 7 | LDR shadow | 0.20 | ✓ | ⚠ | ✗ | ✓ | lighting-flaky |
| 8 | Capacitive touch | ~0 | ✓ | ✗ wants hand | (wants hand) | ✓ | backwards from wish |
| 9 | PIR motion | 1 | ~ | ✗ cold/still = nothing | (hand only) | ✓ | no static object |
| 10 | mmWave radar | 5 | ✓✓ thru wall | ⚠ small=weak | (motion) | ✓ | fiddly, loves motion |
| 11 | COM triangulation | 16 | ✓✓ | ✗ position mush | ✗ | ✗ needs 1 board | many-items ambiguous |
| 12 | Tremor liveness | 4 | ✓ | ✗ too light | ✓✓ *exact wish* | ✓ | needs heavier items |
| 13 | Resonance detune | 5 | ✓✓ | ✗ tiny shift | ~ | ~ | hardest to tune |
| 14 | RFID/NFC | 2 | ✓ | ✓ | ✓ knows which | ✓ | item needs tag |
| 15 | Hall + magnet | 0.50 | ✓ | ✓ | ✓ | ✓ | item needs magnet |
| 16 | Inductive | 2 | ✓ | metal only | ✗ | ✓ | metal items only |
| 17 | Camera + recognition | 7 | ✗ | ✓ | ✓ knows WHAT | ~ | must see + light it |

**Legend:** ✓ good · ⚠ borderline · ✗ bad · ~ depends · ✓✓ exceptional. Prices are rough per-cubby USD.

---

## The three "confuse an engineer" tricks (detail)

### 11. Center-of-mass (COM) triangulation
3–4 load cells under one **shared rigid board**; corners share an item's weight differently, so
moment math gives the item's X,Y → map to a cubby. No sensor visible in/near any cubby.
- **Magic:** "nothing's in there — how does it know *which* cubby?" Proven smart-shelf tech.
- **Kills our concept:** needs ONE shared board (not dispersed). Light item → position smears to
  mush. Two items placed at once → centroid lands between them = ambiguous (only works tracking
  one weight-change event at a time, with memory). No hand-vs-item discrimination.

### 12. Tremor liveness
A living hand always shakes — neuromuscular tremor ~7–10 Hz. An inanimate object is dead silent.
A sensitive load cell watches the *wiggle*, not just the weight: 8 Hz buzz → "hand, ignore";
stable + silent → "object, light up."
- **Magic:** "I pressed hard with my hand and nothing; I set down a sock and it lit — it can't be
  weight." It's weight + liveness detection. Real (hospital beds use it).
- **Catch:** fights light items — a 40g sock's weight *and* its (zero) tremor are both whisper-level.
  Best on heavier items.

### 13. Resonance detune
The cubby floor / frame is a tuning fork. A tiny buzzer hums it at its natural pitch; adding mass
lowers the resonant frequency (more inertia). Detect the pitch shift.
- **Magic:** the item touches nothing wired — "it's just a shelf, how?" A scale made of pitch.
- **Catch:** hardest to tune; light item = tiny shift; temperature drift.

---

## Why ToF (#4) fits this concept best
Dispersed + light/soft + invisible together kill the weight-based tricks (light item = whisper
weight; COM = position mush). An optical time-of-flight sensor doesn't care about weight at all —
a sock blocks the beam exactly like a brick.

- **VL53L0X (~$3)**, one per cubby, hidden behind dark **IR-pass acrylic** at the back wall.
- Empty distance is the baseline; an item shrinks the distance → ON. Works on feather-light/soft.
- Truly invisible (acrylic reads as a solid panel), self-contained per cubby (dispersed ✓), cheap.
- **Hand-vs-item via persistence-time:** a hand dips the distance briefly then it recovers; an item
  drops the distance and *stays*. Light only after it's stayed ~2s. The hand placing the item is
  seen and ignored; the item that remains triggers.

### Recommendation summary
- **#4 ToF + "did it stay" logic** — most robust for dispersed light/soft items.
- **#12 tremor** — coolest "hand no, item yes," *if* items get heavier.
- **#11 COM** — most magical to an observer, *but* incompatible with dispersed cubbies.

---

## Sources
- COM / smart-shelf weight localization: Loadstar Center-of-Gravity Scale; 4-load-cell 2D position
  (ResearchGate fig 220714711); Amazon smart-shelf patents (US 11017350, 10121121, 10614415).
- Tremor liveness (7–10 Hz human vs inanimate): Stryker person-support load-cell patent
  (US 10898400); "hand tremor spectrum modified by inertial sensor mass" (Nature Sci. Reports
  s41598-022-21310-4).
- Mass lowers resonant frequency: PMC7917976 (coupled resonant mass sensors).
- Capacitive grounded-hand vs dielectric ambiguity: discriminative touch-panel patent (US 9105255).
