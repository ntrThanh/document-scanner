#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-prod}"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate deeplearning || true
fi

export SCAN_MAX_CONCURRENT_JOBS="${SCAN_MAX_CONCURRENT_JOBS:-2}"
export SCAN_MAX_CONCURRENT_OCR="${SCAN_MAX_CONCURRENT_OCR:-1}"

if [ "$MODE" = "dev" ]; then
    exec uvicorn main:app --reload --host 0.0.0.0 --port "${PORT:-8888}"
fi

exec gunicorn main:app -c gunicorn_conf.py
