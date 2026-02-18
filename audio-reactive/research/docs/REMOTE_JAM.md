# Remote LED Jam — Design Concept

**Problem**: LED jams are fun in person but we want to do them with remote friends.
Streaming LED frames from one computer to a remote controller has too much latency
(human A/V desync threshold ~50ms, internet RTT 20-100ms+).

**Key insight**: Don't stream LED data — stream the recipe, cook locally.

## Architecture: Shared Music, Local Processing, Synced Parameters

```
  Person A (you)                               Person B (friend)
 ┌──────────────┐                             ┌──────────────┐
 │ Same audio   │◄── shared ahead of time ──► │ Same audio   │
 │ file (local) │                             │ file (local) │
 ├──────────────┤                             ├──────────────┤
 │ Audio        │                             │ Audio        │
 │ analysis     │    ~0ms local latency       │ analysis     │
 ├──────────────┤                             ├──────────────┤
 │ Effect       │◄── parameter tweaks ──────► │ Effect       │
 │ engine       │    (WebSocket, ~50-100ms)   │ engine       │
 ├──────────────┤                             ├──────────────┤
 │ Serial out   │                             │ Serial out   │
 └──────┬───────┘                             └──────┬───────┘
        │                                            │
    Tree (197)                                  Their sculpture
```

### Why This Works

- Audio-to-LED latency is **local** on both sides (~18-58ms, existing pipeline)
- Music sync only needs ~10ms — NTP achieves this trivially
- Jam data (parameter changes, effect switches) is tiny and **latency-tolerant** —
  turning a knob 80ms late is imperceptible
- Both sides run the same Python effect code, just with different LED topology

## Three Sync Layers

### 1. Music Sync (tight, ~10ms)

Both sides have the same audio file. A coordinator sends an NTP-referenced
"start at T" message. Both `sounddevice` streams begin at the same wall-clock
moment. Within 1-10ms. Done.

### 2. Effect Code Sync (pre-session)

Both pull the same commit from the repo. Git solves this. Hetzner server
can serve as the hub.

### 3. Parameter Sync (loose, ~100ms is fine)

This is the jam surface. WebSocket messages like:
```json
{"effect": "three_voices", "params": {"bass_brightness": 0.8, "palette": "lava"}}
```
Hetzner box already runs nginx — it can relay these.

## Jam Modes

- **Lanes** — Each person controls different aspects (you: rhythm, friend: color).
  Like two musicians playing different instruments.
- **Turns** — Pass "the wheel" back and forth. One drives, one watches. Trading solos.
- **Layers** — Each person owns a compositing layer (bass response + harmonic sparkles).
  Output is the blend. Different hardware makes it MORE interesting.
- **Call-and-response** — Asymmetric: your knobs control THEIR LEDs and vice versa.
  Network latency becomes part of the creative conversation.

## Live/Improvised Music (harder)

If you can't pre-share the audio file:

- **WebRTC audio** — ~100-300ms latency. Each side runs effects on what they hear.
  Sculptures are offset but each is locally synced. Video call delay partially compensates.
- **MIDI clock sync** — Agree on BPM, share a network tempo clock. Effects lock to
  beats rather than audio samples. Good for rhythmic effects, less so for ambient.

## What Needs to Be Built

1. **WebSocket parameter relay** (~100 lines on Hetzner) — relay parameter messages
   between connected clients
2. **NTP-synced playback start** (~50 lines in Python runner) — "start track X at
   epoch T" handshake
3. **Topology abstraction** — effects render to any LED count/layout, not just tree
4. **Jam session UI** — who controls what, ready/start flow, lane assignment

## Precedent

- **ArtNet/sACN** — professional lighting protocols for network distribution
- **Ableton Link** — tempo sync across devices (LAN, WAN hacks exist)
- **JackTrip/Jamulus** — networked music performance, "same input, local processing"
- **Resolume Arena** — networked VJ sync for multi-venue shows

## Open Questions

- How to handle different sample rates / audio device clocks drifting over time?
  (Probably resync every N minutes, or use audio fingerprint alignment)
- Should the parameter protocol be OSC (standard) or custom WebSocket JSON (simpler)?
- How to handle the friend's onboarding? (install Python, clone repo, configure audio...)
  Could a Docker container or standalone binary simplify this?
- Video feed of each other's sculptures — built into the jam UI or separate call?
