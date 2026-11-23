#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VENV_DIR="${LIBERO_VENV:-${REPO_ROOT}/.venv-libero}"
PYTHON_VERSION="${LIBERO_PYTHON_VERSION:-3.10}"
SETUP_CACHE_DIR="${LIBERO_SETUP_CACHE_DIR:-${REPO_ROOT}/.cache/libero_setup}"
LIBERO_SOURCE_DIR="${LIBERO_SOURCE_DIR:-${SETUP_CACHE_DIR}/LIBERO}"
LIBERO_GIT_URL="${LIBERO_GIT_URL:-https://github.com/Lifelong-Robot-Learning/LIBERO.git}"
LIBERO_GIT_COMMIT="${LIBERO_GIT_COMMIT:-8f1084e3132a39270c3a13ebe37270a43ece2a01}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/.cache/libero_config}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SETUP_CACHE_DIR}/uv}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SETUP_CACHE_DIR}/xdg}"
export TMPDIR="${TMPDIR:-${SETUP_CACHE_DIR}/tmp}"

cd "${REPO_ROOT}"
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TMPDIR}"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
fi
uv pip install \
  --python "${VENV_DIR}/bin/python" \
  -r "${SCRIPT_DIR}/../requirements.txt"
"${VENV_DIR}/bin/python" -c \
  "import mujoco; assert mujoco.__version__ == '2.3.7', mujoco.__version__"

if [[ ! -d "${LIBERO_SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${LIBERO_SOURCE_DIR}")"
  git clone --filter=blob:none --no-checkout \
    "${LIBERO_GIT_URL}" "${LIBERO_SOURCE_DIR}"
fi
git -C "${LIBERO_SOURCE_DIR}" fetch --depth=1 origin "${LIBERO_GIT_COMMIT}"
git -C "${LIBERO_SOURCE_DIR}" checkout --detach FETCH_HEAD

# LIBERO's outer `libero/` directory is a namespace package without an
# __init__.py. Modern wheel and PEP 660 editable builds omit that namespace,
# so register the pinned source root directly with this isolated environment.
SITE_PACKAGES="$("${VENV_DIR}/bin/python" -c \
  "import site; print(site.getsitepackages()[0])")"
printf "%s\n" "${LIBERO_SOURCE_DIR}" > "${SITE_PACKAGES}/libero_source.pth"

mkdir -p "${LIBERO_CONFIG_PATH}"
LIBERO_CONFIG_FILE="${LIBERO_CONFIG_PATH}/config.yaml"
if [[ ! -f "${LIBERO_CONFIG_FILE}" ]]; then
  LIBERO_PACKAGE_ROOT="${LIBERO_SOURCE_DIR}/libero/libero"
  {
    printf "benchmark_root: %s\n" "${LIBERO_PACKAGE_ROOT}"
    printf "bddl_files: %s\n" "${LIBERO_PACKAGE_ROOT}/bddl_files"
    printf "init_states: %s\n" "${LIBERO_PACKAGE_ROOT}/init_files"
    printf "datasets: %s\n" "${REPO_ROOT}/libero_datasets"
    printf "assets: %s\n" "${LIBERO_PACKAGE_ROOT}/assets"
  } > "${LIBERO_CONFIG_FILE}"
fi
export LIBERO_CONFIG_PATH

"${VENV_DIR}/bin/python" -c \
  "from libero.libero import benchmark; assert 'libero_object' in benchmark.get_benchmark_dict()"

echo "LIBERO environment ready: ${VENV_DIR}"
echo "Use PYTHON_BIN=${VENV_DIR}/bin/python with scripts/libero launchers."
