#!/usr/bin/env bash
#SBATCH --job-name=robomimic-fetch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=run_logs/slurm/slurm_%x_%j.out
#SBATCH --error=run_logs/slurm/slurm_%x_%j.err

# Cache the native 84x84 absolute-action datasets used by the paper.
# Training uses the same Hugging Face cache automatically.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"
: "${PYTHON_BIN:=${PROJECT_ROOT}/.venv/bin/python}"
: "${HF_HOME:=${PROJECT_ROOT}/.cache/hf_cache}"
: "${HUGGINGFACE_HUB_CACHE:=${HF_HOME}/hub}"
export HF_HOME HUGGINGFACE_HUB_CACHE

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Download ChaoyiPan/mip-dataset native 84x84 image_abs.hdf5 for can, lift, square, tool_hang (ph) into ${HUGGINGFACE_HUB_CACHE}"
  exit 0
fi

"${PYTHON_BIN}" - <<'PY'
from huggingface_hub import hf_hub_download

for task in ("can", "lift", "square", "tool_hang"):
    path = hf_hub_download(
        repo_id="ChaoyiPan/mip-dataset",
        repo_type="dataset",
        filename=f"robomimic/{task}/ph/image_abs.hdf5",
    )
    print(f"{task}: {path}")
PY
