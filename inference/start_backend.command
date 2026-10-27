#!/bin/zsh
set -eu
RUN_DIR="${0:A:h}"
PROJECT_ROOT="${RUN_DIR:h}"
export BIOVERIFY_PROJECT_ROOT="$PROJECT_ROOT"
export BIOVERIFY_RESCUE_LIVE=1
PYTHON_BIN="${BIOVERIFY_PYTHON:-python3}"
cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" -m uvicorn dashboard_backend:create_app --factory --app-dir "$RUN_DIR" --host 127.0.0.1 --port 8000
