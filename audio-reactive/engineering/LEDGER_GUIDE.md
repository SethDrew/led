# Engineering Ledger Guide

The engineering ledger (`audio-reactive/engineering/ledger.yaml`) records hardware decisions,
protocol designs, firmware fixes, and performance findings.

This is the counterpart to the **research ledger** (`audio-reactive/research/ledger.yaml`),
which tracks how we *think* about audio, LEDs, and feelings.

## What Belongs Here vs Research

| Engineering Ledger | Research Ledger |
|---|---|
| Protocol design decisions (sync bytes, checksums) | How we think about audio decomposition |
| Firmware architecture (state machines, ISR timing) | Perceptual insights about music/light |
| Hardware constraints and workarounds | Artistic observations about LED behavior |
| Performance measurements (latency, FPS, RAM) | Hypotheses about feeling/mood mapping |
| Serial/wireless reliability fixes | Connections between musical and visual ideas |
| Deployment procedures and gotchas | Dead ends in signal processing approaches |

**Rule of thumb:** If it changes how we *build or deploy* the system — engineering ledger.
If it changes how we *think* about audio, LEDs, or feelings — research ledger.

## Entry Format

```yaml
- id: kebab-case-unique-id
  date: 2026-03-03           # when first encountered
  touched: 2026-03-03        # when last revisited
  title: Short descriptive title
  summary: >
    1-3 sentences. The core problem, decision, or finding.
  status: resolved            # see vocabulary below
  severity: high              # impact if ignored: high | medium | low
  scope: firmware             # what area: firmware | protocol | hardware | deployment | performance
  source: path/to/evidence    # file path(s), or [] if none
  tags: [relevant, tags]
  relates_to: [other-entry-ids]
  notes: >
    Root cause analysis, design rationale, alternatives considered.
    What would break if reverted. What to watch for at scale.
```

## Status Vocabulary

| Status | Meaning |
|--------|---------|
| `identified` | Problem known, no fix yet |
| `investigating` | Actively debugging |
| `resolved` | Fix implemented and working |
| `mitigated` | Workaround in place, not fully solved |
| `wont-fix` | Accepted limitation (with reasoning) |
| `monitoring` | Fixed but watching for recurrence |

## Common Tags

- **Area**: `firmware`, `serial`, `protocol`, `neopixel`, `esp32`, `nano`, `deployment`
- **Type**: `root-cause`, `design-decision`, `performance`, `reliability`, `workaround`
- **Hardware**: `ws2812b`, `ws2811`, `ch340`, `uart`, `timing`
