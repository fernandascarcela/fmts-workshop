#!/usr/bin/env bash
set -euo pipefail

echo "=== Checking virtual environment ==="

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: No virtual environment is active."
  echo "Create and activate one first:"
  echo "  python3 -m venv .venv"
  echo "  source .venv/bin/activate"
  exit 1
fi

echo "Python: $(which python)"
python --version

echo
echo "=== Updating pip tooling ==="

python -m pip install --upgrade \
  pip \
  wheel \
  "setuptools<81"

echo
echo "=== Installing PyTorch 2.8.0 + CUDA 12.6 ==="

python -m pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126

echo
echo "=== Installing main project dependencies ==="

python -m pip install -r requirements.txt

echo
echo "=== Installing Toto without dependency resolution ==="

python -m pip install \
  --no-deps \
  toto-ts==0.2.0

echo
echo "=== Installing Uni2TS without dependency resolution ==="

python -m pip install \
  --no-deps \
  uni2ts==2.0.0

echo
echo "=== Restoring project-selected einops version ==="

python -m pip install \
  "einops>=0.8.1,<0.9"

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
echo "Note:"
echo "- Toto and Uni2TS are intentionally installed with --no-deps."
echo "- 'pip check' will therefore report metadata conflicts for those packages."
echo "- Do not downgrade NumPy or PyTorch to satisfy those warnings."