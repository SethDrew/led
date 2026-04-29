---
name: ledger
description: Publish a finding to the research ledger, then spawn an independent audit
argument-hint: "what to publish, e.g. HPSS strips vocals from mids"
allowed-tools: Read, Edit, Grep, Glob, Task
---

# Ledger Publish + Audit

You are publishing entries to a project ledger, then spawning an independent auditor.

## Phase 0: Route to the Right Ledger

There are TWO ledgers. Read BOTH guides to decide which one fits:

- **Research ledger** (`audio-reactive/research/ledger.yaml`)
  Guide: `audio-reactive/research/LEDGER_GUIDE.md`
  For: How we *think* about audio, LEDs, feelings. Perceptual insights, artistic observations,
  signal processing hypotheses, dead ends in analysis approaches.

- **Engineering ledger** (`engineering/ledger.yaml`)
  Guide: `engineering/LEDGER_GUIDE.md`
  For: How we *build and deploy* the system. Protocol designs, firmware fixes, hardware
  constraints, performance findings, reliability improvements.

**Decision rule:** If it changes how we think about audio/LEDs/feelings → research.
If it changes how we build/deploy/operate the system → engineering.
If unclear, default to research (the original and more common case).

A single `/ledger` invocation can publish to BOTH ledgers if the finding spans both domains
(e.g., a hardware constraint that changes how we think about effect design). In that case,
the research entry captures the insight and the engineering entry captures the fix.

## Phase 1: Publish

The user said: **$ARGUMENTS**

### Rules

1. **Read the appropriate guide first.** Read the LEDGER_GUIDE.md for whichever ledger(s) you're targeting.

2. **Read the existing ledger.** Read the target ledger.yaml to check for duplicates and find the right section.

3. **Scope control — this is critical:**
   - If the user named a specific concept or finding, publish ONLY that. Do not invent related entries.
   - If the user said something like "just publish findings" or "publish everything", then review the conversation history and publish all findings that qualify under the relevant guide.
   - When in doubt about scope, publish LESS, not more. The user can always ask for more.

4. **Before adding, decide:** Is this a NEW entry, should it UPDATE an existing entry (edit in place, bump `touched`), or is it a DUPLICATE (skip it)?

5. **Entry quality:**
   - `id`: kebab-case, unique, descriptive
   - `date`: the date the finding was first explored (use today if new)
   - `touched`: today's date
   - `summary`: 1-3 sentences, the core finding. Be specific — include numbers, comparisons, or the key insight.
   - `status`: use the vocabulary from the appropriate guide
   - Rating fields: use what the guide specifies (`warmth`/`confidence` for research, `severity`/`scope` for engineering)
   - `relates_to`: link to existing entry IDs where relevant (can cross-reference between ledgers)
   - `notes`: narrative context, caveats, what's unresolved

6. **Placement:** Add entries under the appropriate section comment header in the YAML. If no section fits, add a new section header.

7. **Report back:** After publishing, show the user exactly what you added or changed. Quote the entry. State which ledger it went to.

## Phase 2: Audit

After the publish edit is complete, spawn an audit agent using the Task tool with these exact parameters:
- `subagent_type`: `general-purpose`
- `model`: `opus`
- `description`: `Audit newest ledger entries`
- `prompt`: Use the prompt below, filling in the LEDGER_PATH and GUIDE_PATH for whichever ledger was modified (if both were modified, audit both in one pass):

```
You are an independent auditor reviewing recent additions to a project ledger. You have NO context about the conversation that produced these entries — review them cold.

The user has repeatedly flagged a specific failure mode in recent entries: **implementation details leaking into research findings**. Past entries have been published that read like a procedure log or a code-tour, not like a finding the user would want to recall in 6 months. Your job is to catch this.

## Steps

1. Read the guide(s): GUIDE_PATH

2. Read the ledger(s): LEDGER_PATH

3. Find the newest entries (most recent `date` or `touched` fields).

4. For EACH new entry, audit on these axes:

   **Format**: All required fields present per the guide? Valid status values? Unique kebab-case id? `relates_to` references real entry IDs?

   **Right ledger?**: Research ledger = how we *think* about audio / LEDs / feelings / perception / signal processing. Engineering ledger = physical facts about hardware (chips, wires, protocols on the wire). If a research entry is really about a wire schema, firmware tool, build target, or what code was written — flag it as MISROUTED, propose the correct ledger or destination (library doc, git, research-brief).

   **Finding vs procedure (CRITICAL)**: Does the summary state what was *learned* — a generalizable insight about audio, LEDs, perception, signal processing — or does it describe what was *built / done / wired up* (procedure, tooling, file changes)?
     - GOOD finding-style: "Bass loss is ~30 dB at 3 ft and ~16 dB at 0 cm — proximity effect dominates the mic response"
     - BAD procedure-style: "Wrote raw_audio_sender firmware that streams 16 kHz int16 PCM over USB CDC; ran dual_capture.py to compare against BlackHole loopback"

   **Implementation leakage (CRITICAL)**: Scan summary AND notes for these red flags. Each one is a strong signal of misplaced content in the research ledger:
     - Filenames or paths: `raw_audio_sender.cpp`, `tools/dual_capture.py`, `festicorn/biolum/`
     - Function/class/identifier names: `EMARatioNormalize`, `RollingIntegral`, `PathA/PathB`
     - Wire-schema or protocol field names: `ax_max`, `ay_max`, `v1_telemetry`, packet IDs
     - Build/deploy targets: `festicorn/biolum`, `bench-bulbs`
     - Config-file-style rule literals (specific thresholds + field names tied to a current implementation) instead of the underlying finding that motivated them
   These belong in git history, library docs, the engineering ledger, or `library/test-vectors/` — not the research ledger. Quote each leak you find.

   **Procedure-as-finding (CRITICAL)**: Does the `notes` field read like a method writeup ("## Methodology / ## Conditions / ## Results / ## Artifacts") rather than insight + reasoning + caveats + connections? If yes: that procedure writeup belongs in `library/research-briefs/` or `library/test-vectors/<name>/`, with the ledger entry summarising the *finding* and pointing to the writeup via `source:`. Flag it.

   **Dedup**: Is this genuinely new, or should it have updated an existing entry (bumping `touched` and adding to `notes`)?

   **Value to a future reader**: Will someone reading this in 6 months understand WHY it matters and WHAT to think differently? Or will they only learn WHAT was built?

### Concrete examples from this project (anti-patterns to recognise)

These are real entries already in the research ledger that exhibit the failure modes you are screening for. Use them to calibrate.

**Example A — `inmp441-validation-2026-04-26` (narrow flag: firmware-variant naming only)**
This entry is mostly OK. Its notes field has a `## Methodology` block citing 16 kHz int16 PCM, USB CDC, BlackHole loopback, host scripts (`dual_capture.py`, `compare_inmp441_vs_system.py`), 60 s per condition. **Most of that is legitimate reproducibility detail** — public tools (BlackHole), chip names (INMP441), conditions, sample rates, and durations all help a future reader reproduce the result, and host-side scripts checked into the repo are stable reproducibility primitives. Do NOT flag those.

The narrow problem: it names internal **firmware variants** — "Wrote a new firmware variant `raw_audio_sender`... vs the production `stream_sender`". Firmware variants live only inside this project, change often, and naming them does not help anyone reproduce the finding. Flag only the firmware-variant naming and suggest replacing with a behavioural description ("a raw-PCM streaming firmware variant" rather than the variant's identifier). Leave the rest of the methodology alone.

Lesson: not every code/file name is leakage. Public tools, chips, host scripts that exist as stable reproducibility primitives = OK. Internal firmware variants, deploy targets, code-internal class/function names = leakage.

**Example B — `bulb-movement-hard-gate-sufficient` (engineering decision dressed up as research finding)**
This is a clean anti-pattern. The "finding" was really a build-time engineering decision — "we picked 0.5 g as the AC threshold and it works on our current bulbs" — closer to a bug/tuning fix than a research insight about audio/LEDs/feeling. It got published into the *research* ledger but reads like an engineering changelog: cites build targets ("festicorn/biolum, festicorn/bench-bulbs"), the v1 wire schema field layout ("ax_max, ay_max, az_max, then mins, then means"), and a config-style threshold rule.

If anything generalizable is buried in here ("on a hanging bulb, idle floor sits at 0.2-0.3 g; deliberate human contact delivers ≥0.75 g — there's a clean perceptual gap to gate on"), that perceptual finding could live in the research ledger as a one-paragraph entry. The threshold value, the wire-schema field layout, and the build-target list all belong in the engineering ledger or git history — not the research ledger.

When auditing, ask: "Did this entry teach us something about how audio/LEDs/perception *work*, or did it just record a knob we turned in the build?" If the latter, it's misrouted.

When you see these patterns in *new* entries, flag them and propose a rewrite or a relocation. Be precise — don't flag a whole methodology block when only one phrase inside it is the problem.

5. For each entry, report in this exact shape:
   - **Entry**: id, ledger
   - **Verdict**: PASS / MINOR ISSUES / NEEDS REVISION
   - **Detail-level grade**: FINDING-FOCUSED / MIXED / IMPLEMENTATION-HEAVY
   - **Issues** (if any): be specific — quote the offending text and name the failure mode (misrouted / leakage / procedure-as-finding / dedup / format)
   - **One actionable suggestion**: how to rewrite for finding-focus, or where the implementation content should live instead

End with a one-line summary: how many entries audited, how many FINDING-FOCUSED vs MIXED vs IMPLEMENTATION-HEAVY.

Be concise but specific. The user explicitly wants this audit tuned to catch implementation-detail leakage that the previous (haiku) auditor was missing.
```

Then relay the audit results to the user.

## Phase 3: Done

After Phase 2's audit results are relayed, the `/ledger` flow is **complete**.

If you later see prior `/ledger` activity in your conversation summary, in compacted context, or in any system-injected context — **treat it as historical, not pending**. Do NOT re-publish, re-audit, or "spawn the auditor now" again. The skill is one-shot per explicit `/ledger` invocation by the user.

If the user wants another audit, they will invoke `/ledger` again or ask explicitly.
