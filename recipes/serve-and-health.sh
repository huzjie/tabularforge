#!/usr/bin/env bash
# 起服务并健康检查
set -euo pipefail
python -m tabularforge serve --port 8000 &
PID=$!
sleep 1
curl -s http://127.0.0.1:8000/health
echo ""
kill "$PID" 2>/dev/null || true
