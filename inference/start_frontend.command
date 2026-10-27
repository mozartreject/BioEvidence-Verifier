#!/bin/zsh
set -eu
RUN_DIR="${0:A:h}"
PYTHON_BIN="${BIOVERIFY_PYTHON:-python3}"
exec "$PYTHON_BIN" "$RUN_DIR/../dashboard/serve.py"
