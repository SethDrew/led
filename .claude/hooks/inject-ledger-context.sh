#!/usr/bin/env bash
# Hook: UserPromptSubmit — inject relevant research ledger entries as context.
# Reads {"prompt": "..."} from stdin, searches the ledger, prints results to stdout.

set -euo pipefail

VENV_PYTHON="/Users/sethdrew/Documents/projects/led/venv/bin/python"
SEARCH_SCRIPT="/Users/sethdrew/Documents/projects/led/tools/search_ledger.py"

# Read stdin (hook provides JSON with prompt)
INPUT=$(cat)

# Extract prompt text — try jq first, fall back to python
if command -v jq &>/dev/null; then
    PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty' 2>/dev/null || true)
else
    PROMPT=""
fi

# Fall back to python json parsing if jq failed or returned empty
if [ -z "$PROMPT" ]; then
    PROMPT=$("$VENV_PYTHON" -c "
import sys, json
try:
    data = json.loads(sys.stdin.read())
    print(data.get('prompt', ''))
except Exception:
    pass
" <<< "$INPUT" 2>/dev/null || true)
fi

# If we still have no prompt, exit silently
if [ -z "$PROMPT" ]; then
    exit 0
fi

# Fast gate: skip search for short follow-ups and conversational messages.
# Only run the expensive TF-IDF search when the prompt looks like a new topic
# or implementation request (>12 words, or contains implementation signals).
WORD_COUNT=$(echo "$PROMPT" | wc -w | tr -d ' ')
if [ "$WORD_COUNT" -lt 25 ]; then
    # Short message — check for implementation keywords before skipping
    if ! echo "$PROMPT" | grep -qiE '(build|implement|add|create|effect|normali|fix|design|write|pulse|color|rainbow|animation|topology|brightness|gamma|blend|composit|beat|tempo|audio|led|firmware|esp32)'; then
        exit 0
    fi
fi

# Run search — suppress stderr, only emit stdout results
"$VENV_PYTHON" "$SEARCH_SCRIPT" "$PROMPT" 2>/dev/null || true

exit 0
