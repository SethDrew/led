#!/usr/bin/env bash
# Hook: UserPromptSubmit — inject relevant research ledger entries as context.
# Reads {"prompt": "..."} from stdin, searches the ledger, prints results to stdout.

set -euo pipefail

VENV_PYTHON="/Users/KO16K39/Documents/led/venv/bin/python"
SEARCH_SCRIPT="/Users/KO16K39/Documents/led/tools/search_ledger.py"

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

# Run search — suppress stderr, only emit stdout results
"$VENV_PYTHON" "$SEARCH_SCRIPT" "$PROMPT" 2>/dev/null || true

exit 0
