from __future__ import annotations

"""
Part 2: apply the interpretable residual correction layer to the Part 1 baseline.

Input produced by Part 1:
    - outputs/zero_to_submission_pipeline/baseline_groupaware_anchor_from_train.csv
    - outputs/zero_to_submission_pipeline/trained_probabilities/candidate_test_predictions_from_train.csv
    - outputs/zero_to_submission_pipeline/trained_probabilities/base_test_predictions_from_train.csv

Output:
    - outputs/zero_to_submission_pipeline/submissions/part2_residual_corrected_submission.csv
    - outputs/zero_to_submission_pipeline/part2_residual_audit.csv
    - outputs/zero_to_submission_pipeline/part2_residual_module_summary.csv

This stage reads only the baseline and model probability tables from Part 1,
plus the raw test features. It does not read external labels or
PassengerId-specific answer lists.
"""

from pathlib import Path

import pandas as pd

from spaceship_residual_pipeline import (
    ID_COL,
    OUT_DIR,
    PROB_DIR,
    SUB_DIR,
    TARGET,
    TEST_CSV,
    ensure_dirs,
    run_residual_stage,
)


BASELINE_PATH = OUT_DIR / "baseline_groupaware_anchor_from_train.csv"
CANDIDATE_PATH = PROB_DIR / "candidate_test_predictions_from_train.csv"
BASE_MODEL_PATH = PROB_DIR / "base_test_predictions_from_train.csv"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run spaceship_part1_train_groupaware_baseline.py first."
        )


def main() -> None:
    ensure_dirs()
    for path in [BASELINE_PATH, CANDIDATE_PATH, BASE_MODEL_PATH]:
        require_file(path)

    test = pd.read_csv(TEST_CSV)
    anchor = pd.read_csv(BASELINE_PATH)
    candidate = pd.read_csv(CANDIDATE_PATH)
    base = pd.read_csv(BASE_MODEL_PATH)

    missing_anchor_cols = {ID_COL, TARGET} - set(anchor.columns)
    if missing_anchor_cols:
        raise ValueError(f"Baseline is missing columns: {sorted(missing_anchor_cols)}")

    final, audit, module_summary = run_residual_stage(anchor, test, candidate, base)

    final_path = SUB_DIR / "part2_residual_corrected_submission.csv"
    audit_path = OUT_DIR / "part2_residual_audit.csv"
    module_path = OUT_DIR / "part2_residual_module_summary.csv"

    final.to_csv(final_path, index=False)
    audit.to_csv(audit_path, index=False)
    module_summary.to_csv(module_path, index=False)

    print("[part 2] residual correction")
    print(f"Input baseline: {BASELINE_PATH}")
    print(f"Final submission: {final_path}")
    print(f"Audit: {audit_path}")
    print(f"Module summary: {module_path}")


if __name__ == "__main__":
    main()
