# Visual QA Preview System

LED effect iteration without physical hardware: a composite PNG preview that lets Claude visually judge effects and give actionable semantic feedback.

## Tools

- **`tools/preview_effect.py`** — CLI wrapper
  ```
  python tools/preview_effect.py <effect_name> \
      --wav <wav_path> \
      --out <output.png> \
      [--zoom]           # 2.5s zoomed crop at peak energy
      [--start N]        # start time in seconds
      [--leds N]         # LED count (default 150)
      [--duration N]     # window duration (default 15s)
  ```
  Outputs JSON to stdout with metrics + PNG path. With `--zoom`, also saves `<output>_zoom.png`.

- **`tools/preview_lib.py`** — Rendering library (called by the CLI)

## Composite PNG Layout (900px wide)

```
Band energy (4-band mel, 80px)     ← time-aligned with LED-ogram
Waveform (amplitude envelope, 60px) ← time-aligned
LED-ogram (3px/LED, 2px/frame, 450px) ← main visualization
Key frames (8 strip snapshots, 280px) ← evenly spaced
Metrics bar (40px)                  ← PASS/WARN/FAIL
```

### Reading the LED-ogram
- Diagonal lines = motion along strip (slope = speed, direction = travel direction)
- Horizontal bands = region stays lit over time
- Vertical lines = all LEDs change at once (flash/pulse)
- Scattered dots = sparkle/noise
- Compare with band energy to judge audio-reactivity (or its absence)

### Rendering
- Gamma 0.6 applied (LEDs are emissive in dark rooms; low RGB values need lifting for screen)
- Low brightness cutoff: max(R,G,B) < 3 snaps to black
- Zoom view: 2.5s window centered on peak energy, ~5-10px per frame

## QA Team Pattern

Three roles for effect creation/iteration:

### Implementer
Writes effect code, runs `preview_effect.py`, checks metrics for FAIL/BROKEN. If metrics pass, hands preview PNG to QA.

### QA Agent
Receives ONLY the composite PNG + user's intent description. Never sees code. Outputs structured verdict:

```
VERDICT: MATCH | CLOSE | OFF | BROKEN
WHAT I SEE: [plain language description of the visual output]
WHAT NEEDS TO CHANGE: [specific, actionable changes]
PRIORITY: [single most important change]
```

The "WHAT I SEE" section forces description before judgment, preventing pattern-matching on the intent.

### Architect (optional)
Designs preview format changes, tests whether the visual approach captures what's needed. Use when building new capabilities, not for routine effect work.

## Iteration Protocol

1. User describes intent ("make shooting stars")
2. Implementer writes effect, runs preview, checks metrics
3. QA evaluates PNG vs intent (never sees code)
4. If MATCH → present to user
5. If CLOSE/OFF → feed QA feedback to implementer, iterate
6. Max 4 iterations before escalating to user

## QA Agent System Prompt (core)

```
You are a QA engineer evaluating LED strip effects. You will see a composite
preview image alongside the user's description of what they wanted.

THE IMAGE CONTAINS (top to bottom):
1. AUDIO BAND ENERGY — 4-band frequency plot, time-aligned. Use to judge
   audio-reactivity.
2. WAVEFORM — Audio amplitude envelope, time-aligned.
3. LED-OGRAM — Time on X, LED position on Y, color = RGB.
   Diagonal lines = motion (steep = fast). Horizontal = static region.
4. KEY FRAMES — 8 strip snapshots at evenly-spaced moments.
5. METRICS — Basic health numbers.

Does the visual output match the user's intent?

Focus on: motion pattern, color, density/sparsity, audio-reactivity,
overall aesthetic.

Output: VERDICT, WHAT I SEE, WHAT NEEDS TO CHANGE, PRIORITY
```

## Validated Test Results (2026-03-20)

Tested on 4 effects with deliberate failure-mode probes:

| Effect | Probes | Iterations to MATCH |
|--------|--------|-------------------|
| Meteor Shower | Motion direction/speed from diagonals | 1 |
| Campfire Embers | Warm color discrimination, spatial zones | 2 |
| Digital Rain | Density/coherence vs noise | 3 |
| Ocean Breath | Absence of audio-reactivity | 0 |

### What works well
- LED-ogram is the primary workhorse — carries most information
- Band energy alignment essential for audio-reactivity judgment
- QA feedback is specific and actionable ("heads are blue not white", "columns too fat")
- Semantic judgment works ("shooting stars vs falling leaves")

### What's harder
- Subtle motion direction (e.g., spark drift vs stationary flicker) needs zoom view
- Fine brightness variation between similar elements needs zoom view
- Very similar warm hues (red vs orange vs amber) at LED-ogram resolution

## Metrics (degenerate-case floor only)

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| avg_brightness < 2 | FAIL | Strip is blank |
| max_brightness = 0 | FAIL | Nothing ever lit |
| pct_black_frames > 0.95 | WARN | Almost always dark |
| motion_score < 0.5 | WARN | Frozen/static |
| unique_hues < 2 (with brightness > 10) | WARN | Single color flood |
