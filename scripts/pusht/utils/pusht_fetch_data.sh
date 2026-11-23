#!/bin/bash
#SBATCH --job-name=pusht-fetch
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --output=run_logs/slurm/slurm_%x_%j.out
#SBATCH --error=run_logs/slurm/slurm_%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
cd "${REPO_ROOT}"

IS_LOCAL_TRAIN=0
[ -z "${SLURM_JOB_ID:-}" ] && IS_LOCAL_TRAIN=1

if [ "${IS_LOCAL_TRAIN}" -eq 1 ]; then
  : "${HF_HOME:=${REPO_ROOT}/.cache/hf_cache}"
  : "${TMPDIR:=${REPO_ROOT}/.cache/tmp}"
  : "${SLURM_JOB_NAME:=localpushtfetch}"
  : "${SLURM_JOB_ID:=$(date -u +%Y%m%d%H%M)}"
  : "${SLURM_NODELIST:=$(hostname)}"
else
  : "${HF_HOME:=${REPO_ROOT}/.cache/hf_cache}"
  : "${TMPDIR:=${SLURM_TMPDIR:-/tmp/${USER:-pvl}/pvl-sacc-bc/${SLURM_JOB_ID:-manual}}}"
  : "${SLURM_NODELIST:=$(hostname)}"
fi

: "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}/hub}"
: "${HF_XET_CACHE:=${HF_HOME}/xet}"
: "${UV_CACHE_DIR:=${REPO_ROOT}/.cache/uv_cache}"
: "${XDG_CACHE_HOME:=${REPO_ROOT}/.cache}"
: "${PYTHON_BIN:=${REPO_ROOT}/.venv/bin/python}"
: "${PUSHT_DATASET_REPO:=ChaoyiPan/mip-dataset}"
: "${PUSHT_DATASET_FILENAME:=pusht/pusht_cchi_v7_replay.zarr.zip}"
: "${LOG_BASE:=${REPO_ROOT}/run_logs/pusht_fetch}"

case "${UV_CACHE_DIR}" in
  /var/spool*|"") UV_CACHE_DIR="${REPO_ROOT}/.cache/uv_cache" ;;
esac
case "${XDG_CACHE_HOME}" in
  /var/spool*|"") XDG_CACHE_HOME="${REPO_ROOT}/.cache" ;;
esac

export HF_HOME HUGGINGFACE_HUB_CACHE HF_XET_CACHE UV_CACHE_DIR XDG_CACHE_HOME TMPDIR PYTHON_BIN LOG_BASE
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${HF_XET_CACHE}" "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${LOG_BASE}"

log_file="${LOG_BASE}/fetch_pusht_${SLURM_JOB_ID}.log"
cmd=(
  "${PYTHON_BIN}" -c
  'import sys; from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], repo_type="dataset"))'
  "${PUSHT_DATASET_REPO}"
  "${PUSHT_DATASET_FILENAME}"
)

echo "===== PushT fetch job=${SLURM_JOB_ID} node=${SLURM_NODELIST} cache=${HUGGINGFACE_HUB_CACHE} repo=${PUSHT_DATASET_REPO}/${PUSHT_DATASET_FILENAME} ====="
printf "Command:"
printf " %q" "${cmd[@]}"
printf "\n"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}" 2>&1 | tee "${log_file}"
