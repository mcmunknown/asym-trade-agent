#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CPP_DIR="${PROJECT_ROOT}/cpp"
OUTPUT_LIB="${CPP_DIR}/libmathcore.dylib"

mkdir -p "${CPP_DIR}"

clang -std=c11 -O3 -fPIC \
  -I"${CPP_DIR}" \
  "${CPP_DIR}/math_core.c" \
  -shared -o "${OUTPUT_LIB}"

echo "Built ${OUTPUT_LIB}"
