from __future__ import annotations

"""
Part 1: reproduce the original 0.81201 group-aware baseline stage.

This is the baseline code path from the original local-reuse pipeline:

    candidate_oof_predictions.csv
    candidate_test_predictions.csv
        -> 0.50 * xgb@F1+F2
         + 0.30 * lgb@F1+F2
         + 0.20 * xgb@F1+F2+F3
        -> PassengerId group smoothing with alpha=0.15
        -> threshold 0.470

Important:
    The three candidate probability columns are the output of the upstream
    F1/F2/F3 model-training stage. This file reproduces the original baseline
    blend exactly; it does not invent a new feature-family approximation.

Outputs are written into zero_to_82698_pipeline_outputs so Part 2 can consume
the same baseline and probability tables.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from spaceship_residual_ml_explainable_stage2 import (
    BASELINE_GROUP_ALPHA,
    BASELINE_THRESHOLD,
    BASELINE_TRIO,
    BASELINE_WEIGHTS,
    ID_COL,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
)
from spaceship_zero_to_82698_pipeline import OUT_DIR, PROB_DIR, ensure_dirs


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_OOF = ROOT / "artifacts" / "probability_tables" / "candidate_oof_predictions.csv"
CANDIDATE_TEST = ROOT / "artifacts" / "probability_tables" / "candidate_test_predictions.csv"
BASE_TEST = ROOT / "artifacts" / "probability_tables" / "base_test_predictions.csv"


def parse_groups(ids: pd.Series) -> pd.Series:
    return ids.astype("string").str.split("_", expand=True)[0].astype(str).reset_index(drop=True)


def group_smooth(values: np.ndarray, groups: pd.Series, alpha: float) -> np.ndarray:
    tmp = pd.DataFrame({"group": groups.astype(str).to_numpy(), "p": values})
    group_mean = tmp.groupby("group")["p"].transform("mean").to_numpy()
    group_size = groups.astype(str).map(groups.astype(str).value_counts()).to_numpy(dtype=float)
    local_alpha = np.where(
        group_size >= 4,
        alpha * 1.15,
        np.where(group_size >= 2, alpha, alpha * 0.5),
    )
    return (1.0 - local_alpha) * values + local_alpha * group_mean


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def main() -> None:
    ensure_dirs()
    for path in [TRAIN_CSV, TEST_CSV, CANDIDATE_OOF, CANDIDATE_TEST, BASE_TEST]:
        if not Path(path).exists():
            raise FileNotFoundError(path)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    oof = pd.read_csv(CANDIDATE_OOF)
    candidate = pd.read_csv(CANDIDATE_TEST)
    base = pd.read_csv(BASE_TEST)

    require_columns(oof, [ID_COL, TARGET, *BASELINE_TRIO], "candidate_oof_predictions.csv")
    require_columns(candidate, [ID_COL, *BASELINE_TRIO], "candidate_test_predictions.csv")
    require_columns(base, [ID_COL, "cat_pred", "xgb_pred", "lgb_pred", "hgb_pred", "et_pred"], "base_test_predictions.csv")

    if not oof[ID_COL].equals(train[ID_COL]):
        raise ValueError("candidate_oof_predictions.csv PassengerId order does not match train.csv")
    if not candidate[ID_COL].equals(test[ID_COL]):
        raise ValueError("candidate_test_predictions.csv PassengerId order does not match test.csv")
    if not base[ID_COL].equals(test[ID_COL]):
        raise ValueError("base_test_predictions.csv PassengerId order does not match test.csv")

    y = train[TARGET].astype(int).to_numpy()
    train_groups = parse_groups(train[ID_COL])
    test_groups = parse_groups(test[ID_COL])

    w1, w2, w3 = BASELINE_WEIGHTS
    c1, c2, c3 = BASELINE_TRIO
    p_oof = w1 * oof[c1].to_numpy(float) + w2 * oof[c2].to_numpy(float) + w3 * oof[c3].to_numpy(float)
    p_test = w1 * candidate[c1].to_numpy(float) + w2 * candidate[c2].to_numpy(float) + w3 * candidate[c3].to_numpy(float)

    p_oof = group_smooth(p_oof, train_groups, BASELINE_GROUP_ALPHA)
    p_test = group_smooth(p_test, test_groups, BASELINE_GROUP_ALPHA)

    anchor = pd.DataFrame({ID_COL: test[ID_COL], TARGET: p_test >= BASELINE_THRESHOLD})
    summary = pd.DataFrame(
        [
            {
                "method": "original_local_reuse_groupaware_trio_blend",
                "probability_columns": "|".join(BASELINE_TRIO),
                "weights": "|".join(f"{w:.2f}" for w in BASELINE_WEIGHTS),
                "group_alpha": BASELINE_GROUP_ALPHA,
                "threshold": BASELINE_THRESHOLD,
                "oof_accuracy": float(accuracy_score(y, p_oof >= BASELINE_THRESHOLD)),
                "test_true_rate": float(anchor[TARGET].mean()),
                "source_oof": str(CANDIDATE_OOF),
                "source_test": str(CANDIDATE_TEST),
            }
        ]
    )

    oof.to_csv(PROB_DIR / "candidate_oof_predictions_from_train.csv", index=False)
    candidate.to_csv(PROB_DIR / "candidate_test_predictions_from_train.csv", index=False)
    base.to_csv(PROB_DIR / "base_test_predictions_from_train.csv", index=False)
    anchor.to_csv(OUT_DIR / "baseline_groupaware_anchor_from_train.csv", index=False)
    summary.to_csv(OUT_DIR / "baseline_groupaware_summary_from_train.csv", index=False)
    (OUT_DIR / "part1_source_notes.json").write_text(
        json.dumps(
            {
                "baseline_script": "spaceship_local_reuse_highscore_v_1.py",
                "baseline_trio": BASELINE_TRIO,
                "weights": BASELINE_WEIGHTS,
                "group_alpha": BASELINE_GROUP_ALPHA,
                "threshold": BASELINE_THRESHOLD,
                "note": "This stage reproduces the original local-reuse baseline from upstream F1/F2/F3 probability tables.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[part 1] original group-aware 0.81201 baseline")
    print(summary.to_string(index=False))
    print(f"Baseline submission: {OUT_DIR / 'baseline_groupaware_anchor_from_train.csv'}")
    print(f"Part 2 inputs: {PROB_DIR}")


if __name__ == "__main__":
    main()
