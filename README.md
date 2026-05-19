# Spaceship Titanic ML Workshop Project

This repository contains the runnable source code for our AI3023 Machine Learning Workshop course project on the Kaggle Spaceship Titanic competition.

## Method Overview

The final pipeline was built around a group-aware ensemble rather than a broad "feature stew" approach. Early experiments with large mixed feature sets and paper-inspired model stacks produced unstable validation-public behavior, so we moved to a more controlled route:

1. Fold-safe feature construction from the original train/test schema.
2. Group-aware feature families using `PassengerId`, cabin deck/side/number, surname, route, CryoSleep, and spend consistency.
3. A raw anchor ensemble:
   - `xgb@F1+F2`
   - `lgb@F1+F2`
   - `xgb@F1+F2+F3`
4. Stability-gated residual correction using heterogeneous model probability tables.
5. OOF-supported residual rescue rules based on interpretable feature/model-disagreement regions.

The key entrypoint is:

```bash
python src/spaceship_raw_trio_groupaware_retrain.py --threads 8 --xgb-device cpu --apply-residual-stage --strict-candidates
```

Use `--xgb-device cuda` if your local XGBoost installation supports GPU training.

## Repository Structure

```text
.
├── src/
│   ├── spaceship_raw_trio_groupaware_retrain.py     # main reproducible entrypoint
│   ├── spaceship_zero_to_82698_pipeline.py          # residual-stage helper functions
│   ├── spaceship_foldsafe_v2_local_probe.py         # fold-safe feature builder
│   └── spaceship_81201_* / spaceship_residual_*     # second-stage rule modules
├── data/
│   └── README.md                                    # put train/test files here
├── artifacts/
│   ├── probability_tables/                          # cached probability tables for exact reproduction
│   ├── submissions/                                 # reference submissions produced by the pipeline
│   └── summaries/                                   # run summaries, audit tables, and model metrics
├── scripts/
│   ├── run_cpu.cmd
│   └── run_gpu.cmd
├── docs/
│   └── GITHUB_UPLOAD_CHECKLIST.md
├── requirements.txt
└── README.md
```

## Data Setup

Download the Kaggle Spaceship Titanic data and place the files here:

```text
data/train.csv
data/test.csv
data/sample_submission.csv   # optional
```

The code also supports environment variables:

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

The final CSV is written to:

```text
outputs/raw_trio_groupaware_retrain/submissions/groupaware_raw_retrain_plus_residual_second_stage.csv
```

## Notes on Cached Artifacts

The course PDF requires readable runnable source code and a README. It does not require every exploratory CSV or every failed candidate submission.

The `artifacts/probability_tables/` files are included because the residual stage uses heterogeneous model signals. They are treated as reproducibility artifacts, not as additional submitted solutions. The main training script can regenerate the group-aware candidate table; the cached base probability table is kept to reproduce the submitted residual layer on the original Kaggle split.

For a new teacher-provided dataset, remove or replace cached probability tables and retrain the probability layer from the new train/test split.

## Main Output References

Reference final submission:

```text
artifacts/submissions/groupaware_raw_retrain_plus_residual_second_stage.csv
```

Main run summary:

```text
artifacts/summaries/raw_retrain_plus_residual_summary.csv
```

Leakage/CV audit:

```text
artifacts/summaries/leakage_and_cv_audit.csv
```

