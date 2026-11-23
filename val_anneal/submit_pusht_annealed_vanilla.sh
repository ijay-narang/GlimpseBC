#!/usr/bin/env bash
# PushT annealed-vanilla grid with held-out offline validation only.
set -euo pipefail
[[ $# -le 1 ]] || { echo "Usage: bash $0 [--dry-run|--submit|--help]" >&2; exit 2; }
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
exec "${PLANNER_PYTHON:-${repo_root}/.venv/bin/python}" "${script_dir}/submit_grid.py" \
  "${repo_root}" pusht vanilla "${1:---dry-run}"
