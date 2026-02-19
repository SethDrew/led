# Research Ledger Guide

The ledger (`audio-reactive/research/ledger.yaml`) is the permanent record of all research
findings, experiments, intuitions, and dead ends for this project.

This is a creative research project where ideas aren't binary. Things exist on a spectrum
from "spark" to "integrated", with many in "hasn't found its place yet."

## What Belongs in the Ledger

The ledger tracks **how we think about the problem** — not what we built.

**Log these:**
- Findings with numbers: "HPSS and flux disagree 30% — they're different detectors"
- Dead ends with reasoning: "Centroid clustering doesn't use the full LED range because..."
- Hardware constraints that affect design: "WS2812B shifts hue at low brightness"
- Artistic observations: "The tree feels underutilized when sparkles cluster mid-zone"
- Hypotheses worth revisiting: "Rolling integral might distinguish drops from builds"
- Connections between ideas: "WLED proves visual design matters more than analysis quality"

**Don't log these:**
- Bug fixes: "Fixed race condition in async handler"
- UI changes: "Moved annotations to a separate tab"
- Layout tweaks: "Adjusted matplotlib subplot margins"
- Feature additions: "Added notes field to effect cards"
- Implementation details: "New endpoint at /api/render-band-analysis/"

**Rule of thumb:** If it changes how we *think* about audio, LEDs, or feelings — log it.
If it's something we *did* to the code — that's what git is for.

## Entry Format

```yaml
- id: kebab-case-unique-id
  date: 2026-02-18           # when first explored
  touched: 2026-02-18        # when last revisited
  title: Short descriptive title
  summary: >
    1-3 sentences. The core finding or idea.
  status: validated           # see vocabulary below
  warmth: high                # artistic/intuitive pull: high | medium | low
  confidence: high            # technical certainty: high | medium | low
  source: path/to/evidence    # file path(s), or [] if none
  tags: [relevant, tags]
  relates_to: [other-entry-ids]
  notes: >
    Narrative context, caveats, feelings about this finding.
    Why it matters. What it connects to. What's unresolved.
```

## Status Vocabulary

| Status | Meaning |
|--------|---------|
| `spark` | Just an idea, untested |
| `exploring` | Actively investigating |
| `validated` | Technically confirmed to work |
| `resonates` | Feels right artistically, not fully proven |
| `integrated` | Built into the codebase/workflow |
| `dormant` | Hasn't found its place yet (not dismissed) |
| `superseded` | Better approach exists (but might return) |

## Warmth vs Confidence

Independent axes. Something can be:
- **High confidence, low warmth**: Technically true but not exciting
- **Low confidence, high warmth**: Unproven but pulls you toward it
- **High both**: Core insights driving the project
- **Low both**: Noted for completeness, not currently active

## Adding Entries (Concurrent-Safe)

When multiple agents may be active, write session files instead of editing the ledger directly.

1. Create: `audio-reactive/research/sessions/<date>-<brief-topic>.yaml`
2. Use the same format with an `entries:` key
3. Session files get merged into the main ledger later

Example session file:
```yaml
# Session: 2026-02-18 — Topology adjustments
entries:
  - id: gamma-height-curve
    date: 2026-02-18
    touched: 2026-02-18
    title: Gamma curve for non-linear branch height mapping
    summary: >
      Added per-branch gamma to sculpture topology...
    status: integrated
    warmth: high
    confidence: high
    source: audio-reactive/hardware/sculptures.json
    tags: [hardware, topology]
    relates_to: [diamond-topology]
    notes: >
      Applied in runner.py apply_topology().
```

To UPDATE an existing entry's status/warmth/connections, add an update entry:
```yaml
  - id: update-some-entry-id
    updates: some-entry-id
    touched: 2026-02-18
    status: validated        # only fields that changed
    warmth: high
    notes: >
      Why the update — what happened this session.
```

## Common Tags

Add new ones freely. Currently in use:
- **Research areas**: `feature-extraction`, `beat-detection`, `source-separation`, `tactus`
- **Creative**: `feelings`, `taste`, `art`, `perception`
- **Technical**: `effects`, `hardware`, `topology`, `architecture`, `tools`
- **Organization**: `pillar-1` (internal), `pillar-2` (external/WLED)
- **Data**: `user-data`, `data-quality`, `annotations`
- **Status**: `failure-mode`, `dead-end`, `future`

## Searching

```
Grep pattern="airiness" path="audio-reactive/research/ledger.yaml"
Grep pattern="dormant" path="audio-reactive/research/ledger.yaml"
Grep pattern="tags:.*feelings" path="audio-reactive/research/ledger.yaml"
```

Or search session files too:
```
Grep pattern="beat-detection" path="audio-reactive/research/" glob="*.yaml"
```
