#!/usr/bin/env bash
# Install Ubuntu system libraries for headless MuJoCo rendering.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/install_rendering_deps.sh"
  echo "Installs OSMesa and OpenGL/EGL system libraries on Ubuntu."
  echo "CUDA rendering also requires an NVIDIA driver installed separately."
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "Unexpected arguments. Use --help for usage." >&2
  exit 2
fi

if [[ ! -f /etc/os-release ]]; then
  echo "This installer supports Ubuntu. Install OSMesa/OpenGL libraries through your system package manager." >&2
  exit 1
fi
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This installer supports Ubuntu; detected ${ID:-unknown}." >&2
  exit 1
fi

privilege=()
if [[ ${EUID} -ne 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Run as root or install sudo to install system packages." >&2
    exit 1
  fi
  privilege=(sudo)
fi

"${privilege[@]}" apt-get update
"${privilege[@]}" apt-get install -y --no-install-recommends \
  libosmesa6 \
  libosmesa6-dev \
  libgl1-mesa-dri \
  libegl1

echo "Rendering libraries installed. Continue with the benchmark's Python environment setup."
