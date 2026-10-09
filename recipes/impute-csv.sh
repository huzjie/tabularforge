#!/usr/bin/env bash
# 一键补全 CSV 缺失值
set -euo pipefail
INPUT="${1:?usage: ./impute-csv.sh input.csv [output.csv]}"
OUTPUT="${2:-${INPUT%.csv}-imputed.csv}"
python -m tabularforge.examples.impute_csv "$INPUT" "$OUTPUT"
echo "done: $OUTPUT"
