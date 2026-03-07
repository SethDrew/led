# Engineering Ledger Guide

The engineering ledger (`engineering/ledger.yaml`) records **physical facts** about
the hardware we work with — LEDs, controllers, chips, electricity, protocols on the wire.

This is the counterpart to the **research ledger** (`audio-reactive/research/ledger.yaml`),
which tracks how we *think* about audio, LEDs, and feelings.

## What Belongs Here

The ledger tracks **properties and constraints of physical entities** — things that are
true regardless of what software we write.

**Log these:**
- Hardware constraints: "ATmega328 UART buffer is 64 bytes"
- Electrical requirements: "Nano needs dedicated 5V power for stable flashing"
- LED physics: "WS2812B uses linear PWM with no hardware gamma — 8-bit quantization
  crushes low-brightness resolution"
- Physical properties: "COB strips diffuse light between pixels, changing how gradients appear"
- Protocol characteristics: "WS2812B requires 800kHz bit-level timing, 30µs per LED"
- Controller capabilities: "ESP32 RMT peripheral drives NeoPixels without disabling interrupts"
- Hardware selection rationale: "ESP32 chosen over STM32 because WiFi built-in"
- Deployment specs: "Diamond sculpture: 73 LEDs, 3 branches, Nano CH340"

**Don't log these:**
- Software bug fixes: "Fixed brightness dead zone by flooring gamma output to 2"
- Algorithm design: "Autocorrelation needs full buffer before first estimate"
- Software protocol implementations: "Added state machine with WAIT_SYNC1 → WAIT_SYNC2"
- Code changes: "Refactored palette rendering to support custom stops"

**Rule of thumb:** If it's a fact about a physical thing — the strip, the chip, the wire,
the power supply — it belongs here. If it's about what our *software* does — that's what
git is for.

**Grey area:** When a hardware limitation forces a specific software architecture (e.g.
WS2812B timing forces callback/render split), log the *hardware constraint*. The software
response to it is implementation detail.

## What Belongs Here vs Research

| Engineering Ledger | Research Ledger |
|---|---|
| Physical properties of LEDs, chips, strips | How we think about audio decomposition |
| Electrical constraints and power requirements | Perceptual insights about music/light |
| Controller capabilities and limits | Artistic observations about LED behavior |
| Wire protocol characteristics | Hypotheses about feeling/mood mapping |
| Hardware selection decisions | Connections between musical and visual ideas |
| Deployment specs (physical wiring, pin assignments) | Dead ends in signal processing approaches |

## Entry Format

```yaml
- id: kebab-case-unique-id
  date: 2026-03-03           # when first encountered
  touched: 2026-03-03        # when last revisited
  title: Short descriptive title
  summary: >
    1-3 sentences. The physical constraint, capability, or hardware fact.
  status: resolved            # see vocabulary below
  severity: high              # impact if ignored: high | medium | low
  scope: hardware             # what area: hardware | electrical | protocol | deployment
  source: path/to/evidence    # file path(s), or [] if none
  tags: [relevant, tags]
  relates_to: [other-entry-ids]
  notes: >
    Physical explanation. Why this constraint exists.
    What it means for our hardware choices. Measurements if available.
```

## Status Vocabulary

| Status | Meaning |
|--------|---------|
| `identified` | Constraint known, no workaround yet |
| `investigating` | Actively measuring/testing |
| `resolved` | Understood and accounted for |
| `mitigated` | Workaround in place, not fully solved |
| `wont-fix` | Accepted limitation (with reasoning) |
| `monitoring` | Accounted for but watching for recurrence |

## Common Tags

- **Hardware**: `ws2812b`, `ws2811`, `cob`, `esp32`, `nano`, `ch340`, `stm32`
- **Physical**: `timing`, `uart`, `power`, `electrical`, `gamma`, `quantization`
- **Properties**: `constraint`, `capability`, `measurement`, `design-decision`
- **Deployment**: `sculpture`, `topology`, `wiring`
