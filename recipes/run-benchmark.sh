#!/usr/bin/env bash
# 跑完整 TabArena 基准 + 训练演示
set -euo pipefail
python -m tabularforge doctor
python -m tabularforge bench
python -m tabularforge train --epochs 10
