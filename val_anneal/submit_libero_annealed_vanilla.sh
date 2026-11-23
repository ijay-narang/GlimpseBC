#!/usr/bin/env bash
# LIBERO Spatial tasks 1-3: annealed vanilla with offline validation only.
set -euo pipefail
[[ $# -le 1 ]] || { echo "Usage: bash $0 [--dry-run|--submit|--help]" >&2; exit 2; }
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
exec "${PLANNER_PYTHON:-${repo_root}/.venv/bin/python}" "${script_dir}/submit_grid.py" \
  "${repo_root}" libero vanilla "${1:---dry-run}"
