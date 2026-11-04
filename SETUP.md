# Environment setup

This file explains how to recreate the Python environment used by the probe
and why a few packages are installed in a non-standard way.

The tested environment contains:

| Component | Tested version |
|---|---:|
| Python | 3.12.3 |
| DoTime | 0.1.3 |
| NumPy | 2.1.3 |
| SciPy | 1.15.3 |
| PyTorch | 2.8.0+cu126 |
| JAX | 0.5.3 (CPU) |
| GluonTS | 0.16.2 |
| Lightning | 2.3.3 |
| setuptools | 80.10.2 |
| Chronos Forecasting | 2.3.1 |
| TabPFN-TS | 1.2.0 |
| TabPFN (Static TabPFN reference) | 8.3.0 |
| TimesFM package | 2.0.2 |
| TiRex-2 | 0.2.1 |
| Toto-TS | 0.2.0 |
| Uni2TS | 2.0.0 |

> The TimesFM Python package version is `2.0.2`; the adapter uses the
> TimesFM 2.5 checkpoint API exposed by that package.

## 1. Create and activate the virtual environment

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Confirm that the environment is active:

```bash
which python
python --version
```

## 2. Install the tested environment

From the repository root:

```bash
bash install.sh
```

This installs PyTorch 2.8.0 with CUDA 12.6, the shared dependencies from
`requirements.txt`, and Toto and Uni2TS with the `--no-deps` handling
described below.

For a CPU-only machine or a different CUDA version, point the script to the
matching PyTorch wheel index:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash install.sh
```

Do **not** run a plain `pip install uni2ts` in this environment.

## 3. Uni2TS / Moirai

`install.sh` installs only the Uni2TS package itself and keeps the shared
dependencies:

```bash
python -m pip install --no-deps "uni2ts==2.0.0"
```

A normal Uni2TS 2.0.0 installation requests older ranges such as NumPy 1.26,
SciPy 1.11 and GluonTS 0.14, which conflict with the stack used by the other
models. The project keeps the modern stack

```text
NumPy   2.1.3
SciPy   1.15.3
GluonTS 0.16.2
PyTorch 2.8.0+cu126
```

and installs Uni2TS without its dependency pins. The runtime packages Moirai
still needs (`hydra-core==1.3.0`, `orjson`) are listed in `requirements.txt`.

Test the Moirai import:

```bash
python -c "from uni2ts.model.moirai import MoiraiForecast, MoiraiModule; print('Moirai imports OK')"
```

The checkpoints `Salesforce/moirai-1.1-R-base` and
`Salesforce/moirai-2.0-R-small` are downloaded automatically on first use.

## 4. TimesFM XReg and JAX

TimesFM's XReg path uses JAX, while every other model uses PyTorch.
`requirements.txt` installs the CPU build of JAX, and the adapter runs XReg on
the CPU (`force_xreg_cpu=True`), so JAX never competes with PyTorch for GPU
memory.

If you install a CUDA build of JAX instead, set these variables **before
importing JAX or TimesFM** to stop JAX from pre-allocating the GPU:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

## 5. Toto and `pkg_resources`

Toto and Lightning import `pkg_resources`, which newer setuptools releases no
longer provide. `install.sh` therefore pins `setuptools==80.10.2`. If you see

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

restore it with:

```bash
python -m pip install "setuptools==80.10.2"
```

A deprecation warning about `pkg_resources` may still be printed; it does not
prevent Toto from running.

## 6. TabPFN-TS dependency pins

The project pins `tabpfn-time-series==1.2.0` in `requirements.txt`. Its
compatible transitive dependencies in the tested environment include
`datasets==2.17.1` and `fev==0.9.0`; they are recorded here for provenance.

## 7. Gated checkpoints

Some checkpoints, for example `NX-AI/TiRex-2`, require accepting the model's
terms on Hugging Face. Accept them on the model page, then authenticate once:

```bash
huggingface-cli login
```

or keep `HF_TOKEN=...` in a local `.env` file (git-ignored) and load it before
running the experiments:

```bash
set -a; source .env; set +a
```

## 8. Verify the installed packages

```bash
python -m pip show \
    dotime \
    chronos-forecasting \
    tabpfn \
    tabpfn-time-series \
    timesfm \
    tirex-2 \
    toto-ts \
    uni2ts
```

Expected versions:

```text
dotime               0.1.3
chronos-forecasting  2.3.1
tabpfn               8.3.0
tabpfn-time-series   1.2.0
timesfm              2.0.2
tirex-2              0.2.1
toto-ts              0.2.0
uni2ts               2.0.0
```

And the core numerical stack:

```bash
python -c "
import sys, numpy, scipy, torch, gluonts, lightning
print('Python    :', sys.version.split()[0])
print('NumPy     :', numpy.__version__)
print('SciPy     :', scipy.__version__)
print('PyTorch   :', torch.__version__)
print('GluonTS   :', gluonts.__version__)
print('Lightning :', lightning.__version__)
"
```

## 9. Expected `pip check` warnings

Because Toto and Uni2TS are installed with `--no-deps`, `pip check` reports
that they declare older versions of NumPy, SciPy, GluonTS, PyTorch or
python-dotenv than the environment provides. This is intentional.

What matters is that the imports and the runs succeed. If a real runtime error
appears, fix that specific package rather than running a plain
`pip install uni2ts`, which would change the dependencies of the other models.

## 10. Installing from a notebook

In an existing environment, the installation corresponds roughly to:

```python
%pip install -r requirements.txt
%pip install --no-deps "toto-ts==0.2.0"
%pip install --no-deps "uni2ts==2.0.0"
%pip install "einops>=0.8.1,<0.9" "setuptools==80.10.2"
```

Restart the kernel before importing the model libraries. The notebooks in
`notebooks/` additionally need `notebook` and `matplotlib`.

## 11. GPU memory

Do not keep every model loaded on the GPU at once. The runners load one model,
run all of its protocols and release it before loading the next. When
experimenting by hand, do the same:

```python
import gc
import torch

model = make_model()
# run experiment...
del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

Otherwise the last models in a sequence can fail simply because earlier ones
still occupy GPU memory.
