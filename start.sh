#!/bin/bash
# ─────────────────────────────────────────────────────────────
# start.sh  — Start the XAI Admin backend + frontend
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "ERROR: .venv not found at $ROOT/.venv"
  echo "Run: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

source "$VENV/bin/activate"

echo "▶ Starting backend on http://localhost:8000 ..."
# PYTHONPATH=. is required so `import backend` works
PYTHONPATH="$ROOT" uvicorn backend.app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000 \
  --reload-dir "$ROOT/backend" &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

echo "▶ Starting frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "✓ Backend: http://localhost:8000"
echo "✓ Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID
