---
name: ledger
description: Publish a finding to the research ledger, then spawn an independent audit
argument-hint: "what to publish, e.g. HPSS strips vocals from mids"
allowed-tools: Read, Edit, Grep, Glob, Task
---

# Ledger Publish + Audit

You are publishing entries to the research ledger at `audio-reactive/research/ledger.yaml`, then spawning an independent auditor.

## Phase 1: Publish

The user said: **$ARGUMENTS**

### Rules

1. **Read the guide first.** Read `audio-reactive/research/LEDGER_GUIDE.md` for the entry format, status vocabulary, and what belongs in the ledger.

2. **Read the existing ledger.** Read `audio-reactive/research/ledger.yaml` to check for duplicates and find the right section to place new entries.

3. **Scope control — this is critical:**
   - If the user named a specific concept or finding, publish ONLY that. Do not invent related entries.
   - If the user said something like "just publish findings" or "publish everything", then review the conversation history and publish all research findings that qualify under the ledger guide (changes how we THINK about audio/LEDs/feelings — not implementation details).
   - When in doubt about scope, publish LESS, not more. The user can always ask for more.

4. **Before adding, decide:** Is this a NEW entry, should it UPDATE an existing entry (edit in place, bump `touched`), or is it a DUPLICATE (skip it)?

5. **Entry quality:**
   - `id`: kebab-case, unique, descriptive
   - `date`: the date the finding was first explored (use today if new)
   - `touched`: today's date
   - `summary`: 1-3 sentences, the core finding. Be specific — include numbers, comparisons, or the key insight.
   - `status`: use the vocabulary from the guide (spark, exploring, validated, resonates, integrated, dormant, superseded)
   - `warmth` and `confidence`: rate honestly on independent axes
   - `relates_to`: link to existing entry IDs where relevant
   - `notes`: narrative context, caveats, what's unresolved

6. **Placement:** Add entries under the appropriate section comment header in the YAML. If no section fits, add a new section header.

7. **Report back:** After publishing, show the user exactly what you added or changed. Quote the entry.

## Phase 2: Audit

After the publish edit is complete, spawn an audit agent using the Task tool with these exact parameters:
- `subagent_type`: `general-purpose`
- `model`: `haiku`
- `description`: `Audit newest ledger entries`
- `prompt`: Use the prompt below, copying it verbatim:

```
You are an independent auditor reviewing recent additions to a research ledger. You have NO context about the conversation that produced these entries — review them cold.

## Steps

1. Read `audio-reactive/research/LEDGER_GUIDE.md` to understand the format and what belongs.

2. Read `audio-reactive/research/ledger.yaml`.

3. Find the newest entries (most recent `date` or `touched` fields).

4. Audit each new entry:

   **Format**: All required fields present? (id, date, touched, title, summary, status, warmth, confidence, source, tags, relates_to, notes) Valid status/warmth/confidence values? Unique kebab-case id?

   **Content**: Does the summary contain a specific finding (not vague)? Does it change how we THINK about audio/LEDs/feelings (not a bug fix or implementation detail)? Are warmth/confidence rated honestly? Does relates_to reference real entry IDs?

   **Dedup**: Is this genuinely new, or should it have updated an existing entry?

   **Value**: Will a future reader understand WHY this matters? Does notes provide useful context?

5. For each entry, report:
   - **Verdict**: PASS, MINOR ISSUES, or NEEDS REVISION
   - **Issues** (if any): be specific
   - **One suggestion** to make the ledger more helpful

Keep it concise. Actionable feedback only.
```

Then relay the audit results to the user.
