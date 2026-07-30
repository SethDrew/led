#!/usr/bin/env bash
# One command for the Axis Mundi UI: the console dashboard (serial + state +
# 3D sculpture panel) on :8080.
#
#   festicorn/tools/start-ui.sh              # then open http://127.0.0.1:8080
#
# Extra arguments pass through to the dashboard (e.g. --port kettle=/dev/cu.x,
# --no-discover).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LED_ROOT="$(cd "$HERE/../.." && pwd)"
DASH_PORT="${DASH_PORT:-8080}"
PY="$LED_ROOT/.venv/bin/python3"
[ -x "$PY" ] || PY=python3

if lsof -nP -iTCP:"$DASH_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $DASH_PORT is already in use — stop that process first, or set DASH_PORT" >&2
  exit 1
fi

echo "dashboard → http://127.0.0.1:$DASH_PORT"
exec "$PY" "$HERE/console-dashboard/dashboard.py" --http "$DASH_PORT" "$@"
