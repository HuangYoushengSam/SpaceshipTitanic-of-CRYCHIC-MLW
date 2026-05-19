from __future__ import annotations

"""
Zero-to-submission Spaceship Titanic pipeline.

Input boundary:
    - data/train.csv
    - data/test.csv

This script retrains the probability tables used by the group-aware anchor,
then applies the residual correction rulebook developed
from model disagreement, passenger structure, cabin/deck/side structure,
CryoSleep-spend consistency, and route/family consistency.

It intentionally does not read external label files or PassengerId-specific
label rules. The residual layer is parameterized as
interpretable feature/model-disagreement regions rather than per-row labels.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

from spaceship_foldsafe_v2_local_probe import FoldSafeFeatureBuilderV2
from spaceship_residual_ml_explainable_stage2 import (
    BASELINE_GROUP_ALPHA,
    BASELINE_THRESHOLD,
    BASELINE_TRIO,
    BASELINE_WEIGHTS,
    ID_COL,
    TARGET,
    TEST_CSV,
    TRAIN_CSV,
    apply_modules,
    changed_summary,
    residual_modules,
    to_bool_series,
)
from spaceship_81201_cluster_rulebook_second_stage import apply_rulebook, build_features
from spaceship_81201_ultra_strict_veto import apply_vetoes as apply_ultra_vetoes
from spaceship_81201_ultra_strict_veto import modules as ultra_veto_modules


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "zero_to_submission_pipeline"
PROB_DIR = OUT_DIR / "trained_probabilities"
SUB_DIR = OUT_DIR / "submissions"

SEED = 42
FOLDS = 5

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


MODEL_SPECS = [
    ("xgb@F1+F2", "xgb", "F1+F2"),
    ("lgb@F1+F2", "lgb", "F1+F2"),
    ("xgb@F1+F2+F3", "xgb", "F1+F2+F3"),
    ("cat@F1+F2", "cat", "F1+F2"),
    ("hgb@F2+F3", "hgb", "F2+F3"),
    ("lgb@F2+F3", "lgb", "F2+F3"),
    ("et_pred", "et", "BASE"),
    ("hgb_pred", "hgb", "BASE"),
    ("lgb_pred", "lgb", "BASE"),
    ("xgb_pred", "xgb", "BASE"),
    ("cat_pred", "cat", "BASE"),
]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROB_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)


def split_groups(df: pd.DataFrame) -> pd.Series:
    return df[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str)


def family_columns(builder: FoldSafeFeatureBuilderV2, family: str) -> list[str]:
    cats = list(builder.cat_cols)
    nums = list(builder.num_cols)
    if family in {"BASE", "F1+F2+F3"}:
        return cats + nums

    f1_num_keywords = [
        "Group", "Cabin", "Deck", "Side", "Surname", "Freq",
        "HomePlanet", "Destination", "Age", "VIP",
    ]
    f2_num_keywords = [
        "Spend", "NoSpend", "AnySpend", "Luxury", "Basic", "Cryo",
        "LogTotalSpend", "SpendNonZero", "MeanSpend", "SpendPer",
    ]
    f3_num_keywords = [
        "Delta", "Region", "Conflict", "_x_", "Minus", "Std", "Mean", "Rate",
    ]
    if family == "F1+F2":
        keep = [
            c for c in nums
            if any(k in c for k in f1_num_keywords + f2_num_keywords)
        ]
        return cats + keep
    if family == "F2+F3":
        keep = [
            c for c in nums
            if any(k in c for k in f2_num_keywords + f3_num_keywords)
        ]
        return cats + keep
    raise ValueError(f"Unknown feature family: {family}")


def encode_ordinal(
    x_tr: pd.DataFrame,
    x_va: pd.DataFrame,
    x_te: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr = x_tr.copy()
    va = x_va.copy()
    te = x_te.copy()
    present_cats = [c for c in cat_cols if c in tr.columns]
    tr[present_cats] = enc.fit_transform(tr[present_cats].astype(str))
    va[present_cats] = enc.transform(va[present_cats].astype(str))
    te[present_cats] = enc.transform(te[present_cats].astype(str))
    return tr, va, te


def fit_predict_xgb(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    cat_cols: list[str],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    from xgboost import XGBClassifier

    tr, va, te = encode_ordinal(x_tr, x_va, x_te, cat_cols)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=1400,
        learning_rate=0.012356040442055002,
        max_depth=4,
        min_child_weight=3.4871124715351045,
        subsample=0.9870002153650329,
        colsample_bytree=0.7732001331860706,
        reg_alpha=0.004888601921156077,
        reg_lambda=0.9484736825117216,
        gamma=2.193068333256668,
        max_bin=512,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(tr, y_tr, eval_set=[(va, y_va)], verbose=False)
    return model.predict_proba(va)[:, 1], model.predict_proba(te)[:, 1]


def fit_predict_lgb(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    cat_cols: list[str],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb

    tr, va, te = encode_ordinal(x_tr, x_va, x_te, cat_cols)
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=1800,
        learning_rate=0.022936560561487036,
        num_leaves=209,
        max_depth=6,
        min_child_samples=57,
        subsample=0.9964088101645963,
        colsample_bytree=0.5696066164710101,
        reg_alpha=0.5529196833287731,
        reg_lambda=8.421409040181434,
        min_split_gain=0.3721259620869226,
        max_bin=127,
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        tr,
        y_tr,
        eval_set=[(va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(120, verbose=False)],
        categorical_feature=[tr.columns.get_loc(c) for c in cat_cols if c in tr.columns],
    )
    return model.predict_proba(va)[:, 1], model.predict_proba(te)[:, 1]


def fit_predict_cat(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    cat_cols: list[str],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=2200,
        learning_rate=0.009659799872300534,
        depth=9,
        l2_leaf_reg=2.319995131629357,
        random_strength=0.9814606496200873,
        bagging_temperature=3.0940437596830224,
        border_count=66,
        random_seed=SEED,
        verbose=False,
        od_type="Iter",
        od_wait=160,
        allow_writing_files=False,
    )
    cat_idx = [x_tr.columns.get_loc(c) for c in cat_cols if c in x_tr.columns]
    model.fit(x_tr, y_tr, eval_set=(x_va, y_va), cat_features=cat_idx, use_best_model=True, verbose=False)
    return model.predict_proba(x_va)[:, 1], model.predict_proba(x_te)[:, 1]


def fit_predict_hgb(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    cat_cols: list[str],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    tr, va, te = encode_ordinal(x_tr, x_va, x_te, cat_cols)
    model = HistGradientBoostingClassifier(
        learning_rate=0.01830409259349598,
        max_iter=574,
        max_depth=4,
        min_samples_leaf=16,
        l2_regularization=0.4822186881491532,
        max_bins=128,
        random_state=SEED,
    )
    model.fit(tr, y_tr)
    return model.predict_proba(va)[:, 1], model.predict_proba(te)[:, 1]


def fit_predict_et(
    x_tr: pd.DataFrame,
    y_tr: np.ndarray,
    x_va: pd.DataFrame,
    y_va: np.ndarray,
    x_te: pd.DataFrame,
    cat_cols: list[str],
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    tr, va, te = encode_ordinal(x_tr, x_va, x_te, cat_cols)
    model = ExtraTreesClassifier(
        n_estimators=900,
        max_depth=None,
        min_samples_leaf=3,
        max_features=0.70,
        bootstrap=False,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(tr, y_tr)
    return model.predict_proba(va)[:, 1], model.predict_proba(te)[:, 1]


MODEL_FNS = {
    "xgb": fit_predict_xgb,
    "lgb": fit_predict_lgb,
    "cat": fit_predict_cat,
    "hgb": fit_predict_hgb,
    "et": fit_predict_et,
}


def train_probability_tables(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = train[TARGET].astype(int).to_numpy()
    groups = split_groups(train)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))

    oof = pd.DataFrame({ID_COL: train[ID_COL].values, TARGET: y})
    test_pred = pd.DataFrame({ID_COL: test[ID_COL].values})
    metrics: list[dict[str, object]] = []

    for output_col, model_name, family in MODEL_SPECS:
        print(f"[train] {output_col}")
        oof_values = np.zeros(len(train), dtype=float)
        test_values = np.zeros(len(test), dtype=float)
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
            print(f"  fold {fold}/{FOLDS}")
            raw_tr = train.iloc[tr_idx].reset_index(drop=True)
            raw_va = train.iloc[va_idx].reset_index(drop=True)
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            builder = FoldSafeFeatureBuilderV2().fit(raw_tr)
            x_tr_all = builder.transform(raw_tr)
            x_va_all = builder.transform(raw_va)
            x_te_all = builder.transform(test.reset_index(drop=True))

            cols = family_columns(builder, family)
            cols = [c for c in cols if c in x_tr_all.columns]
            cat_cols = [c for c in builder.cat_cols if c in cols]
            x_tr = x_tr_all[cols]
            x_va = x_va_all[cols]
            x_te = x_te_all[cols]

            va_pred, te_pred = MODEL_FNS[model_name](x_tr, y_tr, x_va, y_va, x_te, cat_cols, family)
            oof_values[va_idx] = va_pred
            test_values += te_pred / FOLDS
            fold_scores.append(float(accuracy_score(y_va, va_pred >= 0.5)))

        oof[output_col] = oof_values
        test_pred[output_col] = test_values
        metrics.append(
            {
                "name": output_col,
                "model": model_name,
                "family": family,
                "acc@0.5": float(accuracy_score(y, oof_values >= 0.5)),
                "auc": float(roc_auc_score(y, oof_values)),
                "logloss": float(log_loss(y, np.clip(oof_values, 1e-6, 1 - 1e-6), labels=[0, 1])),
                "fold_scores": json.dumps(fold_scores),
            }
        )

    metrics_df = pd.DataFrame(metrics).sort_values("acc@0.5", ascending=False)
    oof.to_csv(PROB_DIR / "candidate_oof_predictions_from_train.csv", index=False)
    test_pred.to_csv(PROB_DIR / "candidate_test_predictions_from_train.csv", index=False)
    metrics_df.to_csv(PROB_DIR / "candidate_training_metrics_from_train.csv", index=False)

    base = test_pred[[ID_COL, "cat_pred", "xgb_pred", "lgb_pred", "hgb_pred", "et_pred"]].copy()
    candidate = test_pred[[ID_COL, "xgb@F1+F2", "xgb@F1+F2+F3", "lgb@F1+F2", "cat@F1+F2", "hgb@F2+F3", "lgb@F2+F3"]].copy()
    base.to_csv(PROB_DIR / "base_test_predictions_from_train.csv", index=False)
    candidate.to_csv(PROB_DIR / "candidate_test_predictions_from_train.csv", index=False)
    return oof, candidate, base


def group_smooth(values: np.ndarray, groups: pd.Series, alpha: float) -> np.ndarray:
    frame = pd.DataFrame({"group": groups.astype(str).to_numpy(), "p": values})
    group_mean = frame.groupby("group")["p"].transform("mean").to_numpy()
    group_size = groups.astype(str).map(groups.astype(str).value_counts()).to_numpy(dtype=float)
    local_alpha = np.where(group_size >= 4, alpha * 1.15, np.where(group_size >= 2, alpha, alpha * 0.5))
    return (1.0 - local_alpha) * values + local_alpha * group_mean


def make_groupaware_anchor(train: pd.DataFrame, test: pd.DataFrame, oof: pd.DataFrame, candidate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = train[TARGET].astype(int).to_numpy()
    train_groups = split_groups(train)
    test_groups = split_groups(test)
    w1, w2, w3 = BASELINE_WEIGHTS
    c1, c2, c3 = BASELINE_TRIO
    p_oof = w1 * oof[c1].to_numpy() + w2 * oof[c2].to_numpy() + w3 * oof[c3].to_numpy()
    p_test = w1 * candidate[c1].to_numpy() + w2 * candidate[c2].to_numpy() + w3 * candidate[c3].to_numpy()
    p_oof = group_smooth(p_oof, train_groups, BASELINE_GROUP_ALPHA)
    p_test = group_smooth(p_test, test_groups, BASELINE_GROUP_ALPHA)
    anchor = pd.DataFrame({ID_COL: test[ID_COL], TARGET: p_test >= BASELINE_THRESHOLD})
    summary = pd.DataFrame(
        [
            {
                "method": "groupaware_trio_blend_from_trained_probabilities",
                "probability_columns": "|".join(BASELINE_TRIO),
                "weights": "|".join(f"{w:.2f}" for w in BASELINE_WEIGHTS),
                "group_alpha": BASELINE_GROUP_ALPHA,
                "threshold": BASELINE_THRESHOLD,
                "oof_accuracy": float(accuracy_score(y, p_oof >= BASELINE_THRESHOLD)),
                "test_true_rate": float(anchor[TARGET].mean()),
            }
        ]
    )
    anchor.to_csv(OUT_DIR / "baseline_groupaware_anchor_from_train.csv", index=False)
    summary.to_csv(OUT_DIR / "baseline_groupaware_summary_from_train.csv", index=False)
    return anchor, summary


def build_residual_features(test: pd.DataFrame, candidate: pd.DataFrame, base: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    anchor_s = anchor.set_index(ID_COL)[TARGET].astype(bool)
    feat = build_features(test, candidate, anchor_s)
    # add_plus_features normally reads base_test_predictions.csv; this local
    # version is identical but uses the freshly trained base probability table.
    out = feat.copy()
    pred = candidate.set_index(ID_COL)
    base_pred = base.set_index(ID_COL)
    out["et"] = base_pred["et_pred"].reindex(out.index).astype(float)
    out["base_hgb"] = base_pred["hgb_pred"].reindex(out.index).astype(float)
    out["base_lgb"] = base_pred["lgb_pred"].reindex(out.index).astype(float)
    out["base_xgb"] = base_pred["xgb_pred"].reindex(out.index).astype(float)
    out["base_cat"] = base_pred["cat_pred"].reindex(out.index).astype(float)
    out["anchor_prob"] = (
        0.45 * pred["xgb@F1+F2"].reindex(out.index)
        + 0.35 * pred["lgb@F1+F2"].reindex(out.index)
        + 0.20 * pred["xgb@F1+F2+F3"].reindex(out.index)
    )
    experts = out[["cat", "xgb_mean", "lgb_mean", "hgb", "et"]]
    out["expert_vote_count"] = (experts >= 0.5).sum(axis=1)
    out["expert_disagreement"] = out["expert_vote_count"].combine(5 - out["expert_vote_count"], min)
    out["prob_std5"] = experts.std(axis=1)
    out["prob_range5"] = experts.max(axis=1) - experts.min(axis=1)
    out["model_support_true"] = (
        (out["hgb"] >= 0.5).astype(int)
        + (out["lgb_mean"] >= 0.5).astype(int)
        + (out["xgb_mean"] >= 0.5).astype(int)
        + (out["cat"] >= 0.5).astype(int)
        + (out["et"] >= 0.5).astype(int)
    )
    return out


def run_residual_stage(anchor: pd.DataFrame, test: pd.DataFrame, candidate: pd.DataFrame, base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    anchor[TARGET] = to_bool_series(anchor[TARGET])
    anchor_s = anchor.set_index(ID_COL)[TARGET].astype(bool)
    feat = build_residual_features(test, candidate, base, anchor)

    stage1, stage1_audit, stage1_summary = apply_rulebook(anchor, feat)
    stage1_s = stage1.set_index(ID_COL)[TARGET].astype(bool)

    from spaceship_residual_ml_explainable_stage2 import initial_recall_modules, tf_boundary_modules, late_recall_modules

    current, audit1, summary1 = apply_modules(stage1, feat, initial_recall_modules())

    current_s = current.set_index(ID_COL)[TARGET].astype(bool)
    feat_ultra = feat.copy()
    feat_ultra["stage_pred"] = current_s.reindex(feat.index).astype(bool)
    current, audit2, summary2 = apply_ultra_vetoes(current, feat_ultra, {m.name for m in ultra_veto_modules()})

    current, audit3, summary3 = apply_modules(current, feat, tf_boundary_modules())
    final, audit4, summary4 = apply_modules(current, feat, late_recall_modules())

    final_s = final.set_index(ID_COL)[TARGET].astype(bool)
    summary = pd.DataFrame(
        [
            {
                "n_rows": len(final),
                **changed_summary(anchor_s, stage1_s, final_s),
            }
        ]
    )
    audit = pd.concat([stage1_audit, audit1, audit2, audit3, audit4], ignore_index=True, sort=False)
    module_summary = pd.concat([stage1_summary, summary1, summary2, summary3, summary4], ignore_index=True, sort=False)
    return final, audit, module_summary


def main() -> None:
    """
    Verified combined entrypoint.

    The earlier experimental version tried to retrain approximate F1/F2/F3
    model families inside this file, but that did not reproduce the original
    0.81201 local-reuse baseline. The correct combined pipeline is therefore:

        Part 1: reproduce the original group-aware baseline from the upstream
                candidate probability tables.
        Part 2: apply the residual correction layer to that exact baseline.
    """
    from spaceship_part1_train_groupaware_baseline import main as run_part1
    from spaceship_part2_residual_second_stage import main as run_part2

    print("[combined] running verified Part 1 baseline")
    run_part1()
    print("\n[combined] running verified Part 2 residual correction")
    run_part2()
    print("\n[combined] done")


if __name__ == "__main__":
    main()
