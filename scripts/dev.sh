#!/usr/bin/env bash
# Start the iris recognition backend (FastAPI on :8000) and the dashboard
# (TanStack Start / Vite on :5173) in parallel. Stop both with Ctrl-C.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

cleanup() { trap - EXIT; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "→ starting FastAPI backend on http://127.0.0.1:8000"
uvicorn backend.main:app --reload --port 8000 --host 127.0.0.1 &

echo "→ starting dashboard on http://localhost:5173"
(
  cd dashboard-reimagined-main
  if [ ! -d node_modules ]; then
    echo "  (installing npm deps — first run only)"
    npm install --silent
  fi
  npm run dev
) &

wait
