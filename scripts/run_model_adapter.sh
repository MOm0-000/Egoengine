#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <conda-env> <adapter-module> [args...]" >&2
  exit 2
fi

adapter_env="$1"
adapter_module="$2"
shift 2

exec conda run --no-capture-output -n "$adapter_env" python -m "$adapter_module" "$@"

