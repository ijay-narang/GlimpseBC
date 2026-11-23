#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

checkpoint_name="ibot_vits_16_checkpoint_teacher.pth"
default_pretrained_dir="${REPO_ROOT}/glitr/pretrained"
pretrained_dir="${GLITR_PRETRAINED_DIR:-${default_pretrained_dir}}"
checkpoint_url="${GLITR_IBOT_CHECKPOINT_URL:-https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vits_16/checkpoint_teacher.pth}"
checkpoint_path="${pretrained_dir}/${checkpoint_name}"
tmp_path="${checkpoint_path}.tmp.$$"

mkdir -p "${pretrained_dir}"

if [[ -f "${checkpoint_path}" ]]; then
  echo "iBOT checkpoint already exists: ${checkpoint_path}"
  exit 0
fi

cleanup() {
  rm -f "${tmp_path}"
}
trap cleanup EXIT

echo "Downloading iBOT ViT-S/16 teacher backbone:"
echo "  ${checkpoint_url}"
echo "Saving as:"
echo "  ${checkpoint_path}"

if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 --retry-delay 5 "${checkpoint_url}" -o "${tmp_path}"
elif command -v wget >/dev/null 2>&1; then
  wget --tries=3 --waitretry=5 -O "${tmp_path}" "${checkpoint_url}"
else
  "${PYTHON_BIN:-python3}" -c \
    "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
    "${checkpoint_url}" "${tmp_path}"
fi

mv "${tmp_path}" "${checkpoint_path}"
trap - EXIT

echo "Downloaded ${checkpoint_path}"
echo "Use: export GLITR_PRETRAINED_DIR=${pretrained_dir}"
