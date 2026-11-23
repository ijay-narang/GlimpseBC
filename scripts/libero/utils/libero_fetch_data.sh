#!/bin/bash
#SBATCH --job-name=libero-fetch
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=run_logs/slurm/slurm_%x_%j.out
#SBATCH --error=run_logs/slurm/slurm_%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] &&
  [[ -f "${SLURM_SUBMIT_DIR}/scripts/libero/utils/libero_fetch_data.sh" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "${REPO_ROOT}"

: "${PYTHON_BIN:=${REPO_ROOT}/.venv-libero/bin/python}"
: "${LIBERO_DATA_ROOT:=${REPO_ROOT}/libero_datasets}"
: "${LIBERO_SUITE:=libero_spatial}"
: "${LIBERO_CONFIG_PATH:=${REPO_ROOT}/.cache/libero_config}"
: "${HF_HOME:=${REPO_ROOT}/.cache/hf_cache}"
: "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}/hub}"

export HF_HOME HUGGINGFACE_HUB_CACHE LIBERO_CONFIG_PATH LIBERO_DATA_ROOT
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${LIBERO_DATA_ROOT}"

"${PYTHON_BIN}" examples/process_dataset/download_libero_data.py \
  --suite "${LIBERO_SUITE}" \
  --output-dir "${LIBERO_DATA_ROOT}"
