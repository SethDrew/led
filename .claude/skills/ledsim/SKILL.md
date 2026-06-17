---
name: ledsim
description: >
  Run a firmware (C++) LED effect off-hardware, dump its pixel data over time to CSV,
  and answer questions about that data with a data-analysis framework. Ephemeral
  self-check for an agent verifying its own work in the moment — NOT a regression suite
  and NOT a blind trust of any committed sim. The render math under test must match the
  current firmware AT RUN TIME. Use when asked to measure, compare, QA, or tune an effect,
  or via /ledsim.
argument-hint: "what to measure/compare, e.g. 'avg brightness per pixel of bloom over 10s' or 'match brightness of fire and bloom by tuning'"
allowed-tools: Read, Edit, Write, Bash, Grep, Glob, Task
---

# /ledsim — run a C++ effect, dump pixels-over-time to CSV, analyze

Help an agent **check its own work against a goal, right now**, using the firmware's real
effect math run on the desktop. Two load-bearing consequences:

1. **No regression machinery.** No golden images, no CI gates. Produce CSV, analyze it,
   answer the question, move on.
2. **A check is only worth running if it's sound.** The sim's render math must equal the
   firmware math *at run time*. A stale copy doesn't rot quietly — it manufactures false
   confidence, worse than no check. **Never trust a committed sim blind.**

## Entry points (phases are composable)

The user usually supplies the goal (Phase 0) and handles cleanup/commit (Phase 5) themselves.
Detect what they're asking for:

- **Measure-and-answer (Phases 1–4):** "figure out the average brightness per pixel of this
  effect over 10 seconds." Run, dump CSV, analyze, report the number/finding.
- **Full tuning loop (0–5):** "make the brightness of these two effects similar by tuning the
  relevant knobs and hyperparameters without disturbing the effect's intent — relative timing,
  color, relative brightness." Measure both → tune → re-measure → converge → report.

Run only the phases the request needs.

## Phase 1 — Locate the sim and pass the freshness gate

Always the C++ host-compiled sim in the install's `tools/sim/` (e.g.
`festicorn/tree-of-record/tools/sim/`, `festicorn/road-bulbs/tools/sim/`).

**Before trusting any output, prove the sim math equals the current firmware math:**
- Diff the render fn(s) under test against the current firmware source
  (`src/main.cpp` / `src/receiver.cpp`). road-bulbs `effects.cpp` is a verbatim copy —
  drift-prone. tree `bloom_dither_sim.cpp` includes the real shared libs but copies the
  render fn — diff it too.
- If drifted: re-sync — copy the current render fn into the stub, rebuild. Treat the sim as
  **scratch**.
- The only legitimate difference between a render fn and its stub is the pixel sink
  (`strip.setPixelColor` → framebuffer write) plus removed hardware glue (WiFi/UDP/freertos/
  `millis`/`esp_random`). Float math, LUTs, gamma, delta-sigma must be byte-for-byte the logic.
- If found drift in an existing committed sim, surface it — prior checks against it were unsound.
- **No sim for the target install yet?** Scaffold an ephemeral stub from the current render
  fn (same rules). Don't commit it as authoritative.

## Phase 2 — Emit CSV (pixels over time)

The sim writes one CSV per run. **Duration is the agent's call** — pick enough time to
exercise the property under test (idle behavior wants a long quiet window; transient response
wants the loud section). Tidy long format reads cleanly into a dataframe:

```
frame,t_s,pixel,r,g,b        # add ,w for RGBW; add target16/out columns if studying dither
```

If the install's sim already emits per-pixel CSV (tree-of-record does), reuse it. If it only
streams binary frames (road-bulbs), add a CSV dump to the scratch stub or pipe its frames
through a small converter. The CSV file is a **point-in-time artifact — scratch, not committed.**

## Phase 3 — Analyze with the data framework

Read the CSV into a dataframe (`.venv` python + pandas) and compute what the question needs:
per-pixel mean/max brightness, brightness envelope over time, frame-to-frame motion, hue
distribution, idle floor, two-effect comparison (align on `t_s`, compare envelopes), etc.

**The analysis framework may be developed and committed.** Unlike the sim (which mirrors
drifting firmware) and the CSV (point-in-time), analysis functions operate on data and don't
drift — so if a function would be valuable to future agents (e.g. `per_pixel_brightness(df)`,
`compare_envelopes(df_a, df_b)`, `motion_score(df)`), factor it into a reusable module and
commit it. Per-install analyzers already exist (`tools/sim/analyze.py`,
`tools/sim/qa_analyze.py`) — extend those, and lift a function to a shared analysis module
only when it's genuinely install-independent. Keep it light; grow it as real questions arrive.

## Phase 4 — Answer / judge / tune

Describe before concluding (per `library/visual-qa-preview-system.md` discipline):

- **WHAT THE DATA SHOWS** — the measured facts (the number, the envelope shape, the comparison).
- **ANSWER / VERDICT** — answer the question, or pass/fail against the Phase-0 goal. Not vibes.
- **If tuning:** adjust the relevant knobs/hyperparameters, re-run Phases 2–3, converge.
  Hold the effect's intent fixed — relative timing, color, relative brightness — change only
  what the goal targets.
- **Check in with the user** when a real question arises: an ambiguous goal, a tradeoff, a
  surprising result, or "is this close enough?" Don't guess past a genuine fork.

## Phase 5 — Leave no lie (user often handles this)

- The CSV and any one-off plots are scratch — don't commit them.
- Don't commit a re-synced or scaffolded **sim** as current truth; the worst-of-both state is a
  committed copy that looks authoritative but reflects old math. Leave scratch, or mark loud:
  "point-in-time, re-sync before trust."
- The **analysis framework** is the exception — commit reusable functions that future agents
  will want.
