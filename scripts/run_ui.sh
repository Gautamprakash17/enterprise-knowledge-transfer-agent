#!/usr/bin/env bash
# Start API + open web UI (latest code — required for projects / upload).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8000}"
export PYTHONPATH="${PYTHONPATH:-}:src"

if ! .venv/bin/python -c "import multipart" 2>/dev/null; then
  .venv/bin/pip install -q python-multipart
fi

echo "Starting API on http://127.0.0.1:${PORT} (UI: http://127.0.0.1:${PORT}/app/)"
exec .venv/bin/python -m uvicorn knowledge_transfer_agent.api.main:app \
  --host 127.0.0.1 --port "$PORT" --reload
