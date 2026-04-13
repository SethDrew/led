#!/usr/bin/env bash
# Hook: UserPromptSubmit — inject relevant research ledger entries as context.
# Reads {"prompt": "..."} from stdin, searches the ledger, prints results to stdout.

set -euo pipefail

VENV_PYTHON="/Users/sethdrew/Documents/projects/led/venv/bin/python"
SEARCH_SCRIPT="/Users/sethdrew/Documents/projects/led/tools/search_ledger_combined.py"

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
# Only run the embedding search when the prompt looks like a new topic
# or implementation request (>25 words, or contains domain keywords).
WORD_COUNT=$(echo "$PROMPT" | wc -w | tr -d ' ')
if [ "$WORD_COUNT" -lt 25 ]; then
    # Short message — check for domain keywords before skipping
    if ! echo "$PROMPT" | grep -qiE '(build|implement|add|create|effect|normali|fix|design|write|pulse|color|rainbow|animation|topology|brightness|gamma|blend|composit|beat|tempo|audio|led|firmware|esp32|uart|serial|buffer|hardware|sensor|power|chip|ws2812|sk6812|rgbw|dither|flicker|latency|pipeline|sculpture|neopixel|osc|web.?server|ota|deploy)'; then
        exit 0
    fi
fi

# Run search — suppress stderr, only emit stdout results
"$VENV_PYTHON" "$SEARCH_SCRIPT" "$PROMPT" 2>/dev/null || true

exit 0
