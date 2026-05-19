# Spaceship Titanic ML Workshop Project

This repository contains the runnable source code for our AI3023 Machine Learning Workshop course project on the Kaggle Spaceship Titanic task.

The demo uses the original dataset provided by the instructor/Kaggle. The code expects the standard `train.csv` and `test.csv` schema, where `train.csv` contains the target column `Transported` and `test.csv` contains passengers whose `Transported` labels must be predicted.

## Method Overview

Our final modeling route is a group-aware tabular learning pipeline. Instead of placing all engineered variables into one large feature pool, we separated the features by role and used multiple tree-based models to capture complementary signals.

Main ideas:

1. Fold-safe preprocessing and feature construction.
2. Passenger-group features from `PassengerId`.
3. Cabin structure features from deck, side and cabin number.
4. Family/name features from surname patterns.
5. Behavioral consistency features from `CryoSleep`, spending columns, age and route.
6. A small ensemble of XGBoost and LightGBM models on controlled feature families.
7. A second-stage boundary calibration step based on model disagreement and interpretable passenger-structure rules.

The key entrypoint is:

```bash
python src/spaceship_raw_trio_groupaware_retrain.py --threads 8 --xgb-device cpu --apply-residual-stage --strict-candidates
```

Use `--xgb-device cuda` only if your local XGBoost installation supports GPU training.

## Repository Structure

```text
.
|-- src/
|   |-- spaceship_raw_trio_groupaware_retrain.py     # main training and inference entrypoint
|   |-- spaceship_residual_pipeline.py               # second-stage calibration helpers
|   |-- spaceship_foldsafe_v2_local_probe.py         # fold-safe feature builder
|   |-- spaceship_part1_train_groupaware_baseline.py # baseline ensemble helper
|   |-- spaceship_part2_residual_second_stage.py     # residual calibration helper
|   `-- additional feature-rule modules
|-- data/
|   `-- README.md                                    # place train/test files here
|-- artifacts/
|   |-- probability_tables/                          # compact model-output artifacts from completed runs
|   |-- submissions/                                 # generated submission examples
|   `-- summaries/                                   # audit tables and model metrics
|-- scripts/
|   |-- run_cpu.cmd
|   `-- run_gpu.cmd
|-- docs/
|   `-- GITHUB_UPLOAD_CHECKLIST.md
|-- requirements.txt
`-- README.md
```

## Data Setup

Place the original Spaceship Titanic data files in `data/`:

```text
data/train.csv
data/test.csv
data/sample_submission.csv   # optional, used only as a format reference
```

The standard task is:

- Train on `train.csv`, where `Transported` is known.
- Predict one `Transported` label for each `PassengerId` in `test.csv`.
- Save a two-column submission file with `PassengerId` and `Transported`.

Environment variables can be used if the data is stored elsewhere:

```bash
set SPACESHIP_DATA_DIR=C:\path\to\data
set SPACESHIP_OUT_DIR=C:\path\to\outputs
```

## Running

CPU:

```bash
scripts\run_cpu.cmd
```

GPU:

```bash
scripts\run_gpu.cmd
```

Manual command:

```bash
python src/spaceship_raw_trio_groupaware_retrain.py --threads 8 --xgb-device cpu --apply-residual-stage --strict-candidates
```

The generated CSV is written to:

```text
outputs/raw_trio_groupaware_retrain/submissions/groupaware_raw_retrain_plus_residual_second_stage.csv
```

## Artifacts

The `artifacts/` folder contains compact outputs from completed local runs:

- model probability tables used by the second-stage calibration code;
- example generated submission files;
- summary and audit CSV files for checking the training process.

These files are included only to make the project easier to inspect and reproduce. The core project deliverable is the source code plus the instructions above. Large exploratory folders, local notebook caches and raw downloaded data are intentionally excluded.

## Reproducibility Notes

- The main script reads only the provided train/test files and the project artifacts documented above.
- Cross-validation uses group-aware splitting through the passenger group extracted from `PassengerId`.
- The second-stage calibration is expressed as feature/model-disagreement rules rather than manual editing of individual output rows.
- For a clean demo, copy the provided original `train.csv` and `test.csv` into `data/`, install the dependencies, and run the CPU or GPU command.

