# Install-Day Checklist

Everything below changed in code since the tree was last watched running
(commits `5065ff7` → `da2b6f8`) and has been build-verified but **never
perceptually QA'd on hardware**. Budget ~20 minutes before final placement.

## 0. Setup

- Identify board by MAC, not port: v2 = `F4:2D:C9:6D:B2:58` (boards
  re-enumerate; scan with esptool `chip_id`, `--after hard_reset`).
- **Reflash required**: three effects (heat diffusion, worley, polycule)
  and a dial reorder landed after the board was last flashed.

## 1. Wipe / re-record the 5 slots

Old recordings are invalid twice over: the effect enum was reordered
(recorded decouter indices now select different effects) and the
saturation knob was remapped (recorded decinner indices replay through the
new curve). Record fresh material once everything below passes.

## 2. Knob sweeps (per the new mappings)

- **decinner (saturation)**: 5 = original. Sweep 5→9: 6–7 should stay
  colorful, only 8–9 read washed→white. Sweep 5→0: vivids slam to full,
  mid-tones (nebula, fire) saturate hard, whites/grays stay neutral —
  watch for any gray pixel turning red (would mean the neutral guard
  failed). Judge whether knob-0 "push harder" feels right.
- **ones (speed)**: at 0.2× and 8×, talk at the tree on each audio effect.
  Reaction latency must NOT change with speed — sparkle lands on the
  syllable, gravity springs on the word, at any speed. Only motion/fade
  rates should change.
- **decouter (effect select)**: full 12-position dial, grouped —
  0–3 audio field effects (bloom, fire, heat diffusion, worley),
  4–6 audio particle effects (gravity, sparkle, creatures),
  7–8 ambient particles (polycule, leaf-wind),
  9–11 passive (light-through, nebula, rainbow).
  Confirm the grouping reads intuitively on the physical knob.

## 3. New audio behaviors (first hardware exposure)

- **Fire**: now audio-wired — energy = flame height + color temp, onsets =
  flicker. Was a stub before; tune by ear.
- **Creatures**: excitement now follows speech energy (accel-shake removed).
- **Bloom**: contrast stretch — loud speech should open the field
  (some pixels brighter, some dimmer, net steady), brightness cap now 1.0
  (was 0.15 — much brighter; confirm acceptable at venue).
- **Onset detector**: rebuilt (energy-flux). If sparkle over- or
  under-fires in the venue's noise floor, tune live via serial:
  `ONSET_RISEFRAC` (sensitivity) and `ONSET_REFRAC` (max fire rate).
- **Heat diffusion** (decouter 2, never seen on hardware): sustained loud
  speech should build a white-hot spot that drifts along each strip and
  cools to red when you stop. Tune `HEAT_INJ_RATE` via serial if it
  saturates too easily or never gets past red. Passive 1–3 px warm-white
  sparkles should fade in/out over a few seconds regardless of speech.
- **Worley** (decouter 3, never seen on hardware): dim rainbow boundary
  lines at rest; each syllable kicks them outward and flares bright.
  Tune `WORLEY_KICK` via serial for how hard syllables shove the cells.
- **Polycule** (decouter 7, never seen on hardware): five rainbow-tailed
  particles per strip, bouncing and shoving on collision. Ambient only —
  confirm it does NOT react to speech (by design).

## 4. Record/playback end-to-end

Record a slot with speech, play it back: LED frames must stay locked to
the audio out the jack (new audio-clock playback). Let it loop once —
no drift, clean realign.

## 5. Low-brightness pass (after dark)

tens at 1–2: check fire and nebula for stepping/flicker in dim regions —
they spend the most time in low 8-bit codes and there's no dithering in
this firmware. Known limitation; note severity, don't fix on site.
