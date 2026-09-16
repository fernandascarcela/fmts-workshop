# Counterfactual probing of time-series foundation models

This repository evaluates whether time-series foundation models can receive an
intervention and predict its effect. Experiments use the DoTime
Identifiability and Continuous benchmark suites.

## 1. Environment

The tested environment uses Python 3.12, PyTorch 2.8.0 with CUDA 12.6, and the
package versions listed in `requirements.txt` and `SETUP.md`.

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the tested environment:

```bash
bash install.sh
```

Some models require authenticated access. Store credentials only in a local
`.env` file and never commit that file. If needed, load it into the shell
before running the experiments:

```bash
set -a
source .env
set +a
```

See `SETUP.md` for model-specific installation details and known dependency
constraints.

## 2. Run the experiments

Run all commands from the repository root. The benchmark datasets and model
checkpoints are downloaded automatically on first use.

Run the TSFMs, deterministic anchors, and oracle checks:

```bash
python run_probe.py
```

Run the public DoT-PFN causal reference on all three protocols:

```bash
python run_dotpfn.py
```

Run Static TabPFN on the complete Identifiability suite:

```bash
python run_tabpfn_adjustment.py --full
```

Completed runs are validated and skipped automatically. Use `--force` only
with the reference runners when an existing result must intentionally be
replaced.

Raw predictions are written to:

```text
results/runs/identifiability/
results/runs/continuous_in_window/
results/runs/continuous_post_window/
```

The frozen Continuous episode selections are stored in `results/selections/`.
Identifiability uses all 10,800 episodes. Each Continuous regime uses 1,000
episodes from each of its three causal structures, for 3,000 episodes total.

## 3. Aggregate the metrics

After all runs finish, reproduce the reported metrics and 95% bootstrap
confidence intervals with:

```bash
python aggregate_metrics.py \
  --population intersection \
  --n-boot 10000 \
  --seed 42
```

The generated files are written to `results/metrics/`:

```text
per_episode.csv
summary_pooled.csv
summary_by_structure.csv
summary_by_horizon.csv
duration_accrual.csv
```

## 4. Reproduce the paper tables and figure

Install the notebook interface if it is not already available:

```bash
python -m pip install "notebook==7.6.2" "matplotlib==3.11.1"
```

Open and run:

```text
notebooks/fmts_workshop_tables_figures.ipynb
```

The notebook reads the aggregated CSV files; it does not rerun the models. It
generates Tables 1–10 and Figure 1 from the workshop paper.

Exported tables are saved in:

```text
results/paper_artifacts/
```

The main figure is saved as:

```text
results/figures/figure1_effect_delta_mae.pdf
results/figures/figure1_effect_delta_mae.png
```

## Reproducibility settings

- Identifiability: `dot-Identifiability-v1` v1.1.0, revision `03e33050`.
- Continuous: `dot-Continuous-v1` v1.0.0.
- Continuous selection seed: 42.
- Inference seeds for Moirai-1.1: 42, 43, and 44.
- Bootstrap seed: 42.
- Bootstrap repetitions: 10,000.
- Point prediction: predictive median (`q50`) when available.

Every raw result records its benchmark version, episode IDs, inference seed,
installed package versions, model configuration, and repository state.
