#!/bin/bash
# ─────────────────────────────────────────────────────────────
# start.sh  — Kill previous instances, then restart backend + frontend
# Usage: bash start.sh
# ─────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

# ── Kill any process currently holding port 8000 or 5173 ────
kill_port() {
  local port=$1
  local pids
  pids=$(lsof -ti tcp:"$port" 2>/dev/null)
  if [ -n "$pids" ]; then
    echo "  Killing process(es) on port $port: $pids"
    kill -9 $pids 2>/dev/null || true
  fi
}

echo "▶ Stopping any previous backend/frontend instances..."
kill_port 8000
kill_port 5173

# Belt-and-suspenders: also kill lingering uvicorn / vite workers
pkill -f "uvicorn backend.app.main" 2>/dev/null || true
pkill -f "vite"                     2>/dev/null || true

sleep 1   # give the OS a moment to release the ports

# ── Venv check ───────────────────────────────────────────────
if [ ! -f "$VENV/bin/activate" ]; then
  echo "ERROR: .venv not found at $ROOT/.venv"
  echo "Run: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

source "$VENV/bin/activate"

# ── Backend ──────────────────────────────────────────────────
echo "▶ Starting backend on http://localhost:8000 ..."
PYTHONPATH="$ROOT" uvicorn backend.app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000 \
  --reload-dir "$ROOT/backend" &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# ── Frontend ─────────────────────────────────────────────────
echo "▶ Starting frontend on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "✓ Backend:  http://localhost:8000"
echo "✓ Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

# Graceful shutdown on Ctrl+C
trap 'echo ""; echo "Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT TERM

wait $BACKEND_PID $FRONTEND_PID
