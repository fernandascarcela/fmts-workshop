#!/usr/bin/env bash
# Install the tested environment. See SETUP.md for the reasons behind each step.
set -euo pipefail

# PyTorch wheel index. Override for CPU-only machines or another CUDA version:
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash install.sh
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

echo "=== Checking virtual environment ==="

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: No virtual environment is active."
  echo "Create and activate one first:"
  echo "  python3.12 -m venv .venv"
  echo "  source .venv/bin/activate"
  exit 1
fi

echo "Python: $(which python)"
python --version

echo
echo "=== Updating pip tooling ==="

# Toto and Lightning still import pkg_resources, so setuptools is pinned.
python -m pip install --upgrade pip wheel "setuptools==80.10.2"

echo
echo "=== Installing PyTorch 2.8.0 from ${TORCH_INDEX_URL} ==="

python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url "${TORCH_INDEX_URL}"

echo
echo "=== Installing project dependencies ==="

python -m pip install -r requirements.txt

echo
echo "=== Installing Toto and Uni2TS without dependency resolution ==="

# Their declared pins would downgrade NumPy, SciPy, GluonTS and PyTorch.
python -m pip install --no-deps toto-ts==0.2.0 uni2ts==2.0.0

echo
echo "=== Restoring the project einops version ==="

python -m pip install "einops>=0.8.1,<0.9"

echo
echo "=== Installation complete ==="

python - <<'PY'
import sys
import numpy
import torch
import einops

print("python:", sys.version.split()[0])
print("numpy:", numpy.__version__)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("einops:", einops.__version__)
PY

echo
echo "Notes:"
echo "- Toto and Uni2TS are installed with --no-deps on purpose."
echo "- 'pip check' will report metadata conflicts for them; this is expected."
echo "- Do not downgrade NumPy or PyTorch to silence those warnings."
