#!/usr/bin/env bash
# 用 Docker 起服务
set -euo pipefail
docker compose up --build -d
echo "health check:"
curl -s http://127.0.0.1:8000/health
