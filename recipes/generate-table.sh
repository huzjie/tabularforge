#!/usr/bin/env bash
# 按 schema 生成一张合成表并保存为 CSV
set -euo pipefail
python -m tabularforge.examples.generate_table
echo "done: generated_table.csv"
