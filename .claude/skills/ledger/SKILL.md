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

- **Engineering ledger** (`audio-reactive/engineering/ledger.yaml`)
  Guide: `audio-reactive/engineering/LEDGER_GUIDE.md`
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
- `model`: `haiku`
- `description`: `Audit newest ledger entries`
- `prompt`: Use the prompt below, filling in the LEDGER_PATH and GUIDE_PATH for whichever ledger was modified (if both were modified, audit both in one pass):

```
You are an independent auditor reviewing recent additions to a project ledger. You have NO context about the conversation that produced these entries — review them cold.

## Steps

1. Read the guide(s): GUIDE_PATH

2. Read the ledger(s): LEDGER_PATH

3. Find the newest entries (most recent `date` or `touched` fields).

4. Audit each new entry:

   **Format**: All required fields present per the guide? Valid status values? Unique kebab-case id?

   **Content**: Does the summary contain a specific finding (not vague)? Does it belong in THIS ledger (research vs engineering)? Are rating fields honest?  Does relates_to reference real entry IDs?

   **Dedup**: Is this genuinely new, or should it have updated an existing entry?

   **Value**: Will a future reader understand WHY this matters? Does notes provide useful context?

5. For each entry, report:
   - **Ledger**: which ledger (research or engineering)
   - **Verdict**: PASS, MINOR ISSUES, or NEEDS REVISION
   - **Issues** (if any): be specific
   - **One suggestion** to make the ledger more helpful

Keep it concise. Actionable feedback only.
```

Then relay the audit results to the user.
