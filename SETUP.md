# Environment Setup

This file explains how to recreate the Python environment used by the
DoTime TSFM probing code.

The tested environment currently contains:

| Component | Tested version |
|---|---:|
| Python | 3.12.3 |
| DoTime | 0.1.3 |
| NumPy | 2.1.3 |
| SciPy | 1.15.3 |
| PyTorch | 2.8.0+cu126 |
| GluonTS | 0.16.2 |
| Lightning | 2.3.3 |
| Chronos Forecasting | 2.3.1 |
| TabPFN-TS | 1.2.0 |
| TimesFM package | 2.0.2 |
| Toto-TS | 0.2.0 |
| Uni2TS | 2.0.0 |

> The TimesFM Python package version is `2.0.2`; the adapter uses the
> TimesFM 2.5 model/checkpoint API exposed by that package.

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

The recommended installation from the repository root is:

```bash
bash install.sh
```

This installs PyTorch 2.8.0 with CUDA 12.6, the shared dependencies from
`requirements.txt`, and Toto/Uni2TS with the required `--no-deps` handling.

For a CPU-only environment or a different CUDA version, install the matching
PyTorch build first and then follow the manual steps below.

Do **not** run a normal `pip install uni2ts` in this shared environment.

## 3. Install Uni2TS / Moirai separately

Install only the Uni2TS package itself and keep the already-tested shared
dependencies:

```bash
python -m pip install --no-deps "uni2ts==2.0.0"
```

Why this is necessary:

A normal Uni2TS 2.0.0 installation requests older dependency ranges such as
NumPy 1.26, SciPy 1.11, and GluonTS 0.14. These conflict with the shared,
tested environment used by the other models.

The project therefore keeps the tested modern stack:

```text
NumPy   2.1.3
SciPy   1.15.3
GluonTS 0.16.2
PyTorch 2.8.0+cu126
```

and installs Uni2TS with `--no-deps`.

The additional runtime packages currently needed by Moirai are already listed
in `requirements.txt`, including:

```text
hydra-core==1.3.0
orjson
```

Test the Moirai import:

```bash
python -c "from uni2ts.model.moirai import MoiraiForecast, MoiraiModule; print('Moirai imports OK')"
```

The project uses the checkpoint:

```text
Salesforce/moirai-1.1-R-base
```

The checkpoint is downloaded automatically by `from_pretrained(...)`.

## 4. TimesFM / JAX GPU configuration

TimesFM XReg uses JAX. The other foundation models use PyTorch.

To avoid JAX reserving GPU memory needed by the PyTorch models, set these
variables **before importing JAX or TimesFM**:

```python
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

PyTorch may still use CUDA normally.

## 5. Toto and `pkg_resources`

Toto/Lightning code currently imports `pkg_resources`.

The tested environment therefore pins:

```text
setuptools==80.10.2
```

If you see:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

restore the tested setuptools version:

```bash
python -m pip install "setuptools==80.10.2"
```

A deprecation warning about `pkg_resources` may still be printed. It did not
prevent Toto from running.

## 6. TabPFN-TS dependency pins

The working environment uses `tabpfn-time-series==1.2.0`. Its compatible
transitive dependencies in the tested environment include `datasets==2.17.1`
and `fev==0.9.0`. The project pins TabPFN-TS directly in `requirements.txt`;
the transitive versions are recorded here for provenance.

## 7. Verify the installed model packages

Run:

```bash
python -m pip show \
    chronos-forecasting \
    tabpfn-time-series \
    timesfm \
    toto-ts \
    uni2ts
```

Expected model-package versions:

```text
chronos-forecasting  2.3.1
tabpfn-time-series   1.2.0
timesfm              2.0.2
toto-ts              0.2.0
uni2ts               2.0.0
```

You can also inspect the core numerical environment:

```bash
python -c "
import sys
import numpy
import scipy
import torch
import gluonts
import lightning

print('Python    :', sys.version)
print('NumPy     :', numpy.__version__)
print('SciPy     :', scipy.__version__)
print('PyTorch   :', torch.__version__)
print('GluonTS   :', gluonts.__version__)
print('Lightning :', lightning.__version__)
"
```

## 8. Known intentional Uni2TS metadata conflicts

Because Uni2TS is installed with `--no-deps`, `pip check` may report that
Uni2TS 2.0.0 declares older versions than the shared environment, for example
NumPy, SciPy, GluonTS, PyTorch, or python-dotenv.

This is intentional in the current shared research environment.

The important test is that the actual Moirai imports and smoke tests pass.
If a real runtime error appears, fix that specific missing/incompatible package
rather than running a normal `pip install uni2ts`, which would modify the
working dependencies of the other models.

## 9. Notebook installation equivalent

For an existing environment, the installation cells correspond roughly to:

```python
%pip install -r requirements.txt
%pip install --no-deps "toto-ts==0.2.0"
%pip install --no-deps "uni2ts==2.0.0"
%pip install "einops>=0.8.1,<0.9"
```

Then restart the notebook kernel before importing the model libraries.

The JAX environment variables from Section 4 must be set before the first
TimesFM/JAX import.

## 10. GPU memory during smoke tests

Do not keep every foundation model loaded on the GPU simultaneously.

Prefer creating one adapter/model, running it, and releasing it before loading
the next:

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

This prevents the final models in the sequence from failing simply because
previous models are still occupying GPU memory.
