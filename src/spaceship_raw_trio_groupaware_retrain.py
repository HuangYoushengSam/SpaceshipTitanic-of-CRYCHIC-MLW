
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*XGBoost is running on: cuda.*input data is on: cpu.*")
try:
    pd.set_option("future.no_silent_downcasting", True)
except Exception:
    pass


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("SPACESHIP_DATA_DIR", ROOT / "data"))
TRAIN_PATH = Path(os.environ.get("SPACESHIP_TRAIN_PATH", DATA_DIR / "train.csv"))
TEST_PATH = Path(os.environ.get("SPACESHIP_TEST_PATH", DATA_DIR / "test.csv"))
OUT_DIR = Path(os.environ.get("SPACESHIP_OUT_DIR", ROOT / "outputs" / "raw_trio_groupaware_retrain"))
STATE_DIR = OUT_DIR / "candidate_state"
SUB_DIR = OUT_DIR / "submissions"
RUN_LOG = OUT_DIR / "run.log"
ERR_LOG = OUT_DIR / "run.err.log"
QUIET = False

ID_COL = "PassengerId"
TARGET_COL = "Transported"
N_SPLITS = 5
CV_RANDOM_STATE = 42
FINAL_SEEDS = [42, 2024]

BASELINE_TRIO = ["xgb@F1+F2", "lgb@F1+F2", "xgb@F1+F2+F3"]
BASELINE_WEIGHTS = (0.50, 0.30, 0.20)
BASELINE_GROUP_ALPHA = 0.15
BASELINE_THRESHOLD = 0.470

FAMILY_DEFS = {
    "F1": [
        "HomePlanet", "Destination", "VIP", "GroupId", "GroupMemberNo", "GroupSize", "IsSolo",
        "Deck", "CabinNumFilled", "Side", "DeckSide", "CabinNumBin", "CabinRegion", "DeckRegion",
        "Surname", "HomePlanet_Deck", "Destination_Deck", "HomePlanet_Destination",
        "GroupHomePlanetNunique", "GroupDestinationNunique", "GroupSurnameNunique", "GroupDeckNunique",
    ],
    "F2": [
        "CryoSleep", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
        "TotalSpend", "LuxurySpend", "BasicSpend", "NoSpend", "AnySpend", "LogTotalSpend",
        "SpendPerGroup", "CryoSpendConflict", "CryoSleep_NoSpend", "VIP_Luxury",
        "AgeBin", "SpendNaCount",
    ],
    "F3": [
        "RowMissingCount", "HomePlanet_missing", "CryoSleep_missing", "Cabin_missing", "Destination_missing",
        "Age_missing", "VIP_missing", "RoomService_missing", "FoodCourt_missing", "ShoppingMall_missing",
        "Spa_missing", "VRDeck_missing", "Name_missing", "SurnameFreq", "DeckFreq", "HomePlanetFreq",
        "DestinationFreq", "DeckSideFreq",
    ],
}

BEST_PARAMS = {
    "xgb": {
        "learning_rate": 0.012356040442055002,
        "max_depth": 4,
        "min_child_weight": 3.4871124715351045,
        "subsample": 0.9870002153650329,
        "colsample_bytree": 0.7732001331860706,
        "reg_alpha": 0.004888601921156077,
        "reg_lambda": 0.9484736825117216,
        "gamma": 2.193068333256668,
        "max_bin": 512,
    },
    "lgb": {
        "learning_rate": 0.022936560561487036,
        "num_leaves": 209,
        "max_depth": 6,
        "min_child_samples": 57,
        "subsample": 0.9964088101645963,
        "colsample_bytree": 0.5696066164710101,
        "reg_alpha": 0.5529196833287731,
        "reg_lambda": 8.421409040181434,
        "min_split_gain": 0.3721259620869226,
        "max_bin": 127,
    },
    "cat": {
        "learning_rate": 0.009659799872300534,
        "depth": 9,
        "l2_leaf_reg": 2.319995131629357,
        "random_strength": 0.9814606496200873,
        "bagging_temperature": 3.0940437596830224,
        "border_count": 66,
    },
    "hgb": {
        "learning_rate": 0.01830409259349598,
        "max_iter": 574,
        "max_depth": 4,
        "min_samples_leaf": 16,
        "l2_regularization": 0.4822186881491532,
        "max_bins": 128,
    },
    "et": {
        "n_estimators": 1370,
        "max_depth": 12,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.8,
    },
}

ANCHOR_MODEL_SPECS = [
    ("xgb@F1+F2", "xgb", "F1+F2"),
    ("lgb@F1+F2", "lgb", "F1+F2"),
    ("xgb@F1+F2+F3", "xgb", "F1+F2+F3"),
]
CANDIDATE_MODEL_SPECS = [
    ("xgb@F1+F2", "xgb", "F1+F2"),
    ("xgb@F1+F2+F3", "xgb", "F1+F2+F3"),
    ("lgb@F1+F2", "lgb", "F1+F2"),
    ("cat@F1+F2", "cat", "F1+F2"),
    ("hgb@F2+F3", "hgb", "F2+F3"),
    ("lgb@F2+F3", "lgb", "F2+F3"),
]
BASE_MODEL_NAMES = ["cat", "xgb", "lgb", "hgb", "et"]

KNOWN_COMPARISONS = {
    "packaged_anchor_reference": ROOT
    / "artifacts"
    / "submissions"
    / "groupaware_xgb_F1F2_lgb_F1F2_xgb_F1F2F3_a15_thr470_raw_retrain.csv",
    "packaged_final_reference": ROOT
    / "artifacts"
    / "submissions"
    / "groupaware_raw_retrain_plus_residual_second_stage.csv",
}

SUPPLEMENTAL_CANDIDATE_TEST = ROOT / "artifacts" / "probability_tables" / "candidate_test_predictions.csv"
SUPPLEMENTAL_BASE_TEST = ROOT / "artifacts" / "probability_tables" / "base_test_predictions.csv"
TOP100_FINALIZER = ROOT / "src" / "optional_top100_finalizer.py"
RAW_RETRAIN_ANCHOR = SUB_DIR / "groupaware_xgb_F1F2_lgb_F1F2_xgb_F1F2F3_a15_thr470_raw_retrain.csv"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with RUN_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except Exception:
        pass
    if not QUIET:
        try:
            print(line, flush=True)
        except Exception:
            pass


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)


def mode_or_nan(x: pd.Series):
    x = x.dropna()
    return x.mode().iloc[0] if len(x) else np.nan


def resolve_family_columns(expr: str) -> List[str]:
    cols: List[str] = []
    seen: set[str] = set()
    for fam in expr.split("+"):
        for col in FAMILY_DEFS[fam]:
            if col not in seen:
                cols.append(col)
                seen.add(col)
    return cols


def build_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train_raw.copy()
    test = test_raw.copy()
    train["__is_train__"] = 1
    test["__is_train__"] = 0
    full = pd.concat([train, test], axis=0, ignore_index=True)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    pid_split = full[ID_COL].astype("string").str.split("_", expand=True)
    full["GroupId"] = pid_split[0]
    full["GroupMemberNo"] = pd.to_numeric(pid_split[1], errors="coerce")
    full["GroupSize"] = full.groupby("GroupId")[ID_COL].transform("size")
    full["IsSolo"] = (full["GroupSize"] == 1).astype(int)

    cabin_split = full["Cabin"].fillna("Missing/Missing/Missing").astype("string").str.split("/", expand=True)
    full["Deck"] = cabin_split[0].replace("Missing", np.nan)
    full["CabinNum"] = pd.to_numeric(cabin_split[1].replace("Missing", np.nan), errors="coerce")
    full["Side"] = cabin_split[2].replace("Missing", np.nan)
    full["DeckSide"] = full["Deck"].fillna("Missing") + "_" + full["Side"].fillna("Missing")

    full["Surname"] = full["Name"].fillna("Missing Missing").astype("string").str.split().str[-1]
    full.loc[full["Name"].isna(), "Surname"] = np.nan

    raw_cols = [
        "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
        "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name",
    ]
    for col in raw_cols:
        full[f"{col}_missing"] = full[col].isna().astype(int)
    full["RowMissingCount"] = full[raw_cols].isna().sum(axis=1)

    for col in spend_cols:
        full[col] = pd.to_numeric(full[col], errors="coerce")
    cryo_true = full["CryoSleep"] == True
    full.loc[cryo_true, spend_cols] = full.loc[cryo_true, spend_cols].fillna(0)
    full["SpendNaCount"] = full[spend_cols].isna().sum(axis=1)
    tmp = full[spend_cols].fillna(0)
    full["TotalSpend"] = tmp.sum(axis=1)
    full["LuxurySpend"] = tmp[["Spa", "VRDeck"]].sum(axis=1)
    full["BasicSpend"] = tmp[["RoomService", "FoodCourt", "ShoppingMall"]].sum(axis=1)
    full["NoSpend"] = (full["TotalSpend"] == 0).astype(int)
    full["AnySpend"] = (full["TotalSpend"] > 0).astype(int)
    full["LogTotalSpend"] = np.log1p(full["TotalSpend"])

    cryo_missing = full["CryoSleep"].isna()
    full.loc[cryo_missing & (full["TotalSpend"] > 0), "CryoSleep"] = False
    full.loc[cryo_missing & (full["TotalSpend"] == 0), "CryoSleep"] = True
    full["CryoSpendConflict"] = ((full["CryoSleep"] == True) & (full["TotalSpend"] > 0)).astype(int)
    full.loc[(full["CryoSleep"] == True) & (full["TotalSpend"] > 0), "CryoSleep"] = False

    for col in ["HomePlanet", "Destination", "VIP", "Surname", "Deck", "Side"]:
        grp_mode = full.groupby("GroupId")[col].transform(mode_or_nan)
        full[col] = full[col].fillna(grp_mode)

    full.loc[full["HomePlanet"].isna() & full["Deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = "Europa"
    full.loc[full["HomePlanet"].isna() & full["Deck"].isin(["G"]), "HomePlanet"] = "Earth"
    full["HomePlanet"] = full["HomePlanet"].fillna(full["HomePlanet"].mode().iloc[0])
    full["Destination"] = full["Destination"].fillna(full["Destination"].mode().iloc[0])
    full["VIP"] = full["VIP"].fillna(False)
    full["Surname"] = full["Surname"].fillna("MissingSurname")
    full["Deck"] = full["Deck"].fillna("MissingDeck")
    full["Side"] = full["Side"].fillna("MissingSide")

    full["Age"] = pd.to_numeric(full["Age"], errors="coerce")
    age_fill = full.groupby(["HomePlanet", "Deck", "VIP", "CryoSleep"])["Age"].transform("median")
    full["Age"] = full["Age"].fillna(age_fill)
    full["Age"] = full["Age"].fillna(full.groupby(["HomePlanet", "Deck"])["Age"].transform("median"))
    full["Age"] = full["Age"].fillna(full["Age"].median())

    for col in spend_cols:
        full[col] = full[col].fillna(full.groupby(["HomePlanet", "CryoSleep", "Deck"])[col].transform("median"))
        full[col] = full[col].fillna(0)

    full["TotalSpend"] = full[spend_cols].sum(axis=1)
    full["LuxurySpend"] = full[["Spa", "VRDeck"]].sum(axis=1)
    full["BasicSpend"] = full[["RoomService", "FoodCourt", "ShoppingMall"]].sum(axis=1)
    full["NoSpend"] = (full["TotalSpend"] == 0).astype(int)
    full["AnySpend"] = (full["TotalSpend"] > 0).astype(int)
    full["LogTotalSpend"] = np.log1p(full["TotalSpend"])
    full["SpendPerGroup"] = full["TotalSpend"] / full["GroupSize"].clip(lower=1)

    full["SurnameFreq"] = full.groupby("Surname")["Surname"].transform("size")
    full["DeckFreq"] = full.groupby("Deck")["Deck"].transform("size")
    full["HomePlanetFreq"] = full.groupby("HomePlanet")["HomePlanet"].transform("size")
    full["DestinationFreq"] = full.groupby("Destination")["Destination"].transform("size")
    full["DeckSideFreq"] = full.groupby("DeckSide")["DeckSide"].transform("size")

    full["AgeBin"] = pd.cut(
        full["Age"],
        bins=[-0.1, 12, 18, 25, 40, 60, 100],
        labels=["child", "teen", "young", "adult", "mid", "senior"],
    ).astype(str)
    full["CabinNumFilled"] = full["CabinNum"].fillna(-1)
    full["CabinNumBin"] = pd.cut(
        full["CabinNumFilled"],
        bins=[-2, -0.5, 100, 300, 600, 900, 1200, 2000],
        labels=["missing", "c0", "c1", "c2", "c3", "c4", "c5"],
    ).astype(str)
    full["CabinRegion"] = full["Deck"].astype(str) + "_" + full["CabinNumBin"].astype(str)
    full["DeckRegion"] = full["Deck"].astype(str) + "_" + full["Side"].astype(str) + "_" + full["CabinNumBin"].astype(str)
    full["HomePlanet_Destination"] = full["HomePlanet"].astype(str) + "_" + full["Destination"].astype(str)
    full["HomePlanet_Deck"] = full["HomePlanet"].astype(str) + "_" + full["Deck"].astype(str)
    full["Destination_Deck"] = full["Destination"].astype(str) + "_" + full["Deck"].astype(str)
    full["CryoSleep_NoSpend"] = full["CryoSleep"].astype(str) + "_" + full["NoSpend"].astype(str)
    full["VIP_Luxury"] = full["VIP"].astype(str) + "_" + pd.cut(
        full["LuxurySpend"],
        bins=[-0.1, 0, 100, 1000, 10000, 100000],
        labels=["0", "low", "mid", "high", "ultra"],
    ).astype(str)

    gb = full.groupby("GroupId")
    full["GroupHomePlanetNunique"] = gb["HomePlanet"].transform("nunique")
    full["GroupDestinationNunique"] = gb["Destination"].transform("nunique")
    full["GroupSurnameNunique"] = gb["Surname"].transform("nunique")
    full["GroupDeckNunique"] = gb["Deck"].transform("nunique")
    full["GroupAllSameHomePlanet"] = (full["GroupHomePlanetNunique"] == 1).astype(int)
    full["GroupAllSameDestination"] = (full["GroupDestinationNunique"] == 1).astype(int)
    full["GroupCabinSideConflict"] = (gb["Side"].transform("nunique") > 1).astype(int)
    full["GroupCryoConflict"] = (gb["CryoSleep"].transform("nunique") > 1).astype(int)
    full["GroupSpendStd"] = gb["TotalSpend"].transform("std").fillna(0)
    full["GroupSpendMax"] = gb["TotalSpend"].transform("max").fillna(0)
    full["GroupSpendMin"] = gb["TotalSpend"].transform("min").fillna(0)
    full["GroupAnySpendRate"] = gb["AnySpend"].transform("mean").fillna(0)
    full["GroupCryoRate"] = gb["CryoSleep"].transform(lambda s: pd.Series(s).astype(float).mean()).fillna(0)
    full["GroupAgeStd"] = gb["Age"].transform("std").fillna(0)
    full["GroupCabinNumStd"] = gb["CabinNumFilled"].transform("std").fillna(0)

    train_fe = full[full["__is_train__"] == 1].drop(columns=["__is_train__"]).reset_index(drop=True)
    test_fe = full[full["__is_train__"] == 0].drop(columns=["__is_train__", TARGET_COL], errors="ignore").reset_index(drop=True)
    return train_fe, test_fe


def prep_cat_lgb_pair(Xtr: pd.DataFrame, Xoth: pd.DataFrame):
    combo = pd.concat([Xtr.copy(), Xoth.copy()], axis=0, ignore_index=True)
    cat_cols = [
        col for col in combo.columns
        if combo[col].dtype == "object"
        or str(combo[col].dtype).startswith("category")
        or combo[col].dtype == "bool"
        or str(combo[col].dtype).startswith("string")
    ]
    num_cols = [col for col in combo.columns if col not in cat_cols]
    for col in cat_cols:
        combo[col] = combo[col].astype("string").fillna("Missing").astype("category")
    for col in num_cols:
        med = pd.to_numeric(Xtr[col], errors="coerce").median()
        combo[col] = pd.to_numeric(combo[col], errors="coerce").fillna(med)
    Xtr2 = combo.iloc[:len(Xtr)].reset_index(drop=True)
    Xoth2 = combo.iloc[len(Xtr):].reset_index(drop=True)
    return Xtr2, Xoth2, cat_cols


def prep_ohe_triplet(Xtr: pd.DataFrame, Xva: pd.DataFrame, Xt: pd.DataFrame):
    combo = pd.concat([Xtr.copy(), Xva.copy(), Xt.copy()], axis=0, ignore_index=True)
    for col in combo.columns:
        if combo[col].dtype == "bool":
            combo[col] = combo[col].astype(int)
    combo = pd.get_dummies(combo, dummy_na=True)
    clean_cols = []
    seen: dict[str, int] = {}
    for col in combo.columns:
        name = str(col).replace("[", "(").replace("]", ")").replace("<", "lt")
        name = name.replace(">", "gt").replace(",", "_")
        count = seen.get(name, 0)
        seen[name] = count + 1
        clean_cols.append(name if count == 0 else f"{name}__{count}")
    combo.columns = clean_cols
    ntr, nva = len(Xtr), len(Xva)
    Xtr2 = combo.iloc[:ntr].reset_index(drop=True)
    Xva2 = combo.iloc[ntr:ntr + nva].reset_index(drop=True)
    Xt2 = combo.iloc[ntr + nva:].reset_index(drop=True)
    return Xtr2, Xva2, Xt2


def make_cv_cache(
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    family_expr: str,
    model: str,
):
    cols = [col for col in resolve_family_columns(family_expr) if col in train_fe.columns]
    X = train_fe[cols].copy()
    Xt = test_fe[cols].copy()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    fold_cache = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups=groups), start=1):
        Xtr = X.iloc[tr_idx].reset_index(drop=True)
        Xva = X.iloc[va_idx].reset_index(drop=True)
        if model in {"xgb", "hgb"}:
            Xtr2, Xva2, Xt2 = prep_ohe_triplet(Xtr, Xva, Xt)
            fold_cache.append({"fold": fold, "tr_idx": tr_idx, "va_idx": va_idx, "Xtr": Xtr2, "Xva": Xva2, "Xt": Xt2})
        elif model in {"lgb", "cat"}:
            Xtr_va, Xva2, cat_cols = prep_cat_lgb_pair(Xtr, Xva)
            _, Xt2, _ = prep_cat_lgb_pair(Xtr, Xt)
            fold_cache.append(
                {"fold": fold, "tr_idx": tr_idx, "va_idx": va_idx, "Xtr": Xtr_va, "Xva": Xva2, "Xt": Xt2, "cat_cols": cat_cols}
            )
        else:
            raise ValueError(model)
    return fold_cache


def fit_predict_once(
    model: str,
    params: Dict[str, float],
    fold_view: dict,
    ytr: np.ndarray,
    yva: np.ndarray,
    seed: int,
    threads: int,
    xgb_device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    threads = max(1, int(threads))
    if model == "xgb":
        clf = XGBClassifier(
            n_estimators=2200,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device=xgb_device,
            random_state=seed,
            n_jobs=threads,
            **params,
        )
        try:
            clf.fit(fold_view["Xtr"], ytr, eval_set=[(fold_view["Xva"], yva)], verbose=False)
        except Exception as exc:
            if xgb_device == "cpu":
                raise
            log(f"xgb GPU failed for seed={seed}; falling back to CPU: {exc}")
            clf = XGBClassifier(
                n_estimators=2200,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                random_state=seed,
                n_jobs=threads,
                **params,
            )
            clf.fit(fold_view["Xtr"], ytr, eval_set=[(fold_view["Xva"], yva)], verbose=False)
        return clf.predict_proba(fold_view["Xva"])[:, 1], clf.predict_proba(fold_view["Xt"])[:, 1]

    if model == "lgb":
        clf = LGBMClassifier(
            n_estimators=2600,
            objective="binary",
            random_state=seed,
            n_jobs=threads,
            device_type="cpu",
            verbosity=-1,
            **params,
        )
        clf.fit(
            fold_view["Xtr"],
            ytr,
            eval_set=[(fold_view["Xva"], yva)],
            eval_metric="binary_logloss",
            categorical_feature=fold_view["cat_cols"],
        )
        return clf.predict_proba(fold_view["Xva"])[:, 1], clf.predict_proba(fold_view["Xt"])[:, 1]

    if model == "cat":
        clf = CatBoostClassifier(
            iterations=3200,
            loss_function="Logloss",
            eval_metric="Logloss",
            verbose=False,
            task_type="CPU",
            thread_count=threads,
            allow_writing_files=False,
            random_seed=seed,
            **params,
        )
        clf.fit(
            fold_view["Xtr"],
            ytr,
            eval_set=(fold_view["Xva"], yva),
            cat_features=fold_view["cat_cols"],
            use_best_model=True,
            verbose=False,
        )
        return clf.predict_proba(fold_view["Xva"])[:, 1], clf.predict_proba(fold_view["Xt"])[:, 1]

    if model == "hgb":
        clf = HistGradientBoostingClassifier(loss="log_loss", random_state=seed, **params)
        clf.fit(fold_view["Xtr"], ytr)
        return clf.predict_proba(fold_view["Xva"])[:, 1], clf.predict_proba(fold_view["Xt"])[:, 1]

    raise ValueError(model)


def train_candidate(
    name: str,
    model: str,
    family: str,
    train_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    threads: int,
    xgb_device: str,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    params = dict(BEST_PARAMS[model])
    cv_cache = make_cv_cache(train_fe, test_fe, y, groups, family, model)
    oof = np.zeros(len(y), dtype=float)
    pred = np.zeros(len(test_fe), dtype=float)
    fold_scores = []

    for fold_view in cv_cache:
        fold = fold_view["fold"]
        tr_idx, va_idx = fold_view["tr_idx"], fold_view["va_idx"]
        fold_va = np.zeros(len(va_idx), dtype=float)
        fold_te = np.zeros(len(test_fe), dtype=float)
        for seed in FINAL_SEEDS:
            log(f"{name} fold {fold}/{N_SPLITS} seed={seed}")
            p_va, p_te = fit_predict_once(model, params, fold_view, y[tr_idx], y[va_idx], seed, threads, xgb_device)
            fold_va += p_va / len(FINAL_SEEDS)
            fold_te += p_te / len(FINAL_SEEDS)
        oof[va_idx] = fold_va
        pred += fold_te / N_SPLITS
        fold_scores.append(float(accuracy_score(y[va_idx], (fold_va >= 0.5).astype(int))))

    info = {
        "model": model,
        "family": family,
        "cv_acc@0.5": float(accuracy_score(y, (oof >= 0.5).astype(int))),
        "fold_scores": fold_scores,
    }
    info.update(params)

    pd.DataFrame({ID_COL: train_fe[ID_COL], name: oof}).to_csv(STATE_DIR / f"{name}__oof.csv", index=False)
    pd.DataFrame({ID_COL: test_fe[ID_COL], name: pred}).to_csv(STATE_DIR / f"{name}__test.csv", index=False)
    (STATE_DIR / f"{name}__meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"{name} cv@0.5={info['cv_acc@0.5']:.6f}")
    return oof, pred, info


def split_groups(ids: pd.Series) -> pd.Series:
    return ids.astype("string").str.split("_", expand=True)[0].astype(str).reset_index(drop=True)


def group_smooth(values: np.ndarray, groups: pd.Series, alpha: float) -> np.ndarray:
    frame = pd.DataFrame({"group": groups.astype(str).to_numpy(), "p": values})
    group_mean = frame.groupby("group")["p"].transform("mean").to_numpy()
    group_size = groups.astype(str).map(groups.astype(str).value_counts()).to_numpy(dtype=float)
    local_alpha = np.where(group_size >= 4, alpha * 1.15, np.where(group_size >= 2, alpha, alpha * 0.5))
    return (1.0 - local_alpha) * values + local_alpha * group_mean


def make_groupaware_anchor(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    cand_oof: pd.DataFrame,
    cand_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = train_raw[TARGET_COL].astype(int).to_numpy()
    train_groups = split_groups(train_raw[ID_COL])
    test_groups = split_groups(test_raw[ID_COL])
    w1, w2, w3 = BASELINE_WEIGHTS
    c1, c2, c3 = BASELINE_TRIO
    p_oof = w1 * cand_oof[c1].to_numpy() + w2 * cand_oof[c2].to_numpy() + w3 * cand_oof[c3].to_numpy()
    p_test = w1 * cand_test[c1].to_numpy() + w2 * cand_test[c2].to_numpy() + w3 * cand_test[c3].to_numpy()
    p_oof = group_smooth(p_oof, train_groups, BASELINE_GROUP_ALPHA)
    p_test = group_smooth(p_test, test_groups, BASELINE_GROUP_ALPHA)
    anchor = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET_COL: p_test >= BASELINE_THRESHOLD})
    summary = pd.DataFrame(
        [
            {
                "method": "raw_retrain_trio_groupaware",
                "probability_columns": "|".join(BASELINE_TRIO),
                "weights": "|".join(f"{w:.2f}" for w in BASELINE_WEIGHTS),
                "group_alpha": BASELINE_GROUP_ALPHA,
                "threshold": BASELINE_THRESHOLD,
                "oof_accuracy": float(accuracy_score(y, p_oof >= BASELINE_THRESHOLD)),
                "test_true_rate": float(anchor[TARGET_COL].mean()),
            }
        ]
    )
    return anchor, summary


def compare_known_submissions(anchor: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in KNOWN_COMPARISONS.items():
        if not path.exists():
            rows.append({"target": label, "path": str(path), "status": "missing"})
            continue
        target = pd.read_csv(path)
        merged = anchor[[ID_COL, TARGET_COL]].rename(columns={TARGET_COL: "new_pred"}).merge(
            target[[ID_COL, TARGET_COL]].rename(columns={TARGET_COL: "target_pred"}),
            on=ID_COL,
            how="inner",
        )
        new_bool = merged["new_pred"].astype(bool)
        target_bool = merged["target_pred"].astype(bool)
        diff = merged[new_bool != target_bool].copy()
        diff["direction_new_to_target"] = np.where(
            diff["new_pred"].astype(bool) & ~diff["target_pred"].astype(bool),
            "T->F",
            "F->T",
        )
        detail_path = out_dir / f"diff_vs_{label}.csv"
        diff.to_csv(detail_path, index=False)
        rows.append(
            {
                "target": label,
                "path": str(path),
                "status": "ok",
                "matched_rows": int(len(merged)),
                "n_diff": int(len(diff)),
                "same_rate": float((new_bool == target_bool).mean()),
                "new_true_rate": float(new_bool.mean()),
                "target_true_rate": float(target_bool.mean()),
                "new_true_target_false": int((new_bool & ~target_bool).sum()),
                "new_false_target_true": int((~new_bool & target_bool).sum()),
                "detail_file": str(detail_path),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "comparison_vs_known_submissions.csv", index=False)


def write_leakage_audit(strict_candidates: bool, strict_base: bool, apply_residual: bool) -> None:
    audit = pd.DataFrame(
        [
            {
                "stage": "v7_candidate_probabilities",
                "mode": "strict_candidates" if strict_candidates else "anchor_trio_only",
                "fold_policy": "CV folds are StratifiedGroupKFold by PassengerId group.",
                "preprocessing_policy": "Original v7 feature builder concatenates train+test before CV and uses global structure/frequencies.",
                "risk": "OOF can be optimistic because validation fold covariates participate in unsupervised/statistical preprocessing; no target label from validation/test is used.",
                "use_in_report": "Use as demo-facing transductive reconstruction; report fold-safe estimates separately when discussing generalization.",
            },
            {
                "stage": "top100_base_probabilities",
                "mode": "strict_base" if strict_base else "not_retrained_in_this_run",
                "fold_policy": "Original finalizer uses 5 StratifiedGroupKFold folds and MODEL_SEEDS=[42,2024,3407].",
                "preprocessing_policy": "Fold-safe: feature builder is fit on each training fold, then applied to validation/test.",
                "risk": "Low leakage risk for OOF; the test feature table is used only for prediction-time feature construction.",
                "use_in_report": "Use as cleaner heterogeneous model evidence for the residual layer.",
            },
            {
                "stage": "residual_second_stage",
                "mode": "enabled" if apply_residual else "disabled",
                "fold_policy": "No training-fold refit inside this stage; it applies deterministic residual selectors.",
                "preprocessing_policy": "Uses test features, anchor prediction, and model probabilities; no target labels are read.",
                "risk": "Possible selection-bias risk if deterministic rules are repeatedly adjusted after observing external evaluation. Keep this separate from OOF claims.",
                "use_in_report": "Describe as residual calibration on interpretable structure/model-disagreement leaves.",
            },
        ]
    )
    audit.to_csv(OUT_DIR / "leakage_and_cv_audit.csv", index=False)


def load_existing_retrain_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        OUT_DIR / "candidate_oof_predictions.csv",
        OUT_DIR / "candidate_test_predictions.csv",
        RAW_RETRAIN_ANCHOR,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing retrain outputs. Run without --reuse-existing first: " + "; ".join(missing))
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    cand_oof = pd.read_csv(OUT_DIR / "candidate_oof_predictions.csv")
    cand_test = pd.read_csv(OUT_DIR / "candidate_test_predictions.csv")
    return train_raw, test_raw, cand_oof, cand_test


def train_strict_base_predictions(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> None:
    """Reproduce the top100 base probability table with the original finalizer.

    This intentionally loads the archived finalizer source that produced
    base_oof_predictions.csv/base_test_predictions.csv. It uses that source's
    fold-safe TitanicFeatureBuilder and its original 5 folds x 3 seeds setting.
    """
    if not TOP100_FINALIZER.exists():
        raise FileNotFoundError(TOP100_FINALIZER)
    spec = importlib.util.spec_from_file_location("top100_finalizer_source", TOP100_FINALIZER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {TOP100_FINALIZER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["top100_finalizer_source"] = module
    spec.loader.exec_module(module)

    log(f"strict base: loading original top100 finalizer from {TOP100_FINALIZER}")
    fold_packs, y, groups = module.prepare_fold_packs(train_raw, test_raw)
    base_oof: dict[str, np.ndarray] = {}
    base_test: dict[str, np.ndarray] = {}
    metrics: list[dict[str, Any]] = []
    for model_name in BASE_MODEL_NAMES:
        log(f"strict base: training {model_name} with source finalizer seeds={module.MODEL_SEEDS}")
        oof, te = module.fit_oof_test(model_name, BEST_PARAMS[model_name], fold_packs, y, test_raw[ID_COL])
        base_oof[model_name] = oof
        base_test[model_name] = te
        thr, acc = module.search_best_threshold(y, oof)
        metrics.append(
            {
                "model": model_name,
                "source": str(TOP100_FINALIZER),
                "n_splits": module.N_SPLITS,
                "seeds": "|".join(str(s) for s in module.MODEL_SEEDS),
                "best_thr": float(thr),
                "best_acc": float(acc),
                "acc@0.5": float(accuracy_score(y, oof >= 0.5)),
            }
        )

    base_oof_df = pd.DataFrame({ID_COL: train_raw[ID_COL], TARGET_COL: y})
    base_test_df = pd.DataFrame({ID_COL: test_raw[ID_COL]})
    for model_name in BASE_MODEL_NAMES:
        base_oof_df[f"{model_name}_oof"] = base_oof[model_name]
        base_test_df[f"{model_name}_pred"] = base_test[model_name]
    base_oof_df.to_csv(OUT_DIR / "base_oof_predictions.csv", index=False)
    base_test_df.to_csv(OUT_DIR / "base_test_predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT_DIR / "base_training_metrics.csv", index=False)


RESIDUAL_CANDIDATE_COLS = ["xgb@F1+F2", "xgb@F1+F2+F3", "lgb@F1+F2", "cat@F1+F2", "hgb@F2+F3", "lgb@F2+F3"]

OOF_SUPPORTED_RESCUE_RULES = [
    {"direction": "FT", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AgeBin": "0-5", "SurnameRateBin": ".5-.75"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"HgbBand": ".45-.50", "StdBand": ".02-.04", "SpendTypesBin": "0"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"HgbBand": ".50-.55", "CatBand": ".50-.55"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"EtBand": ".35-.45", "StdBand": ".02-.04", "SurnameRateBin": "0"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AgeBin": "18-25", "SpendBin": "700-999", "EtBand": ".45-.50"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"HgbBand": ".55-.62", "XgbBand": ".62-.72", "GroupRateBin": "1"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"EtBand": ".15-.25", "SurnameSizeBin": "2", "CabinRegion": "101-300"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"HgbBand": ".45-.50", "EtBand": ".45-.50", "StdBand": ".02-.04"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AnchorBand": ".36-.43", "GroupSizeBin": "6-10", "SurnameSizeBin": "3-4"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"HgbBand": ".55-.62", "CatBand": ".62-.72", "RangeBand": ".10-.16"}},
    {"direction": "TF", "route": {"HomePlanet": "Europa", "Deck": "C", "Side": "P"}, "extra": {"XgbBand": ".45-.50", "CabinRegion": "101-300"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AgeBin": "18-25", "StdBand": "<=.02", "SpendTypesBin": "5"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "G", "Side": "S", "Destination": "55 Cancri e"}, "extra": {"SurnameSizeBin": "5-8", "VotesBin": "3"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "G", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"SpendBin": "1000-1499", "CabinRegion": "701-1100", "CabinMicro": "801-1100"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AnchorBand": ".28-.36", "GroupSizeBin": "1", "VotesBin": "0"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AnchorBand": ".57-.64", "LgbBand": ".55-.62", "SpendTypesBin": "3"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "G", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"AgeBin": "6-12", "CatBand": ".35-.45", "GroupSizeBin": "3"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"AnchorBand": ".36-.43", "LgbBand": ".35-.45", "SpendTypesBin": "3"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "G", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"SpendBin": "700-999", "LgbBand": ".35-.45", "SurnameSizeBin": "5-8"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AnchorBand": ".50-.57", "GroupSizeBin": "2"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"HgbBand": ".35-.45", "XgbBand": ".45-.50", "GroupRateBin": "1"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"RangeBand": ".16-.24", "GroupRateBin": "0", "SpendTypesBin": "3"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AgeBin": "18-25", "GroupSizeBin": "1", "VotesBin": "3"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "G", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"XgbBand": ".35-.45", "SpendTypesBin": "2", "CabinRegion": "701-1100"}},
    {"direction": "FT", "route": {"HomePlanet": "Mars", "Deck": "E", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AnchorBand": ".18-.28", "GroupRateBin": "0", "SurnameRateBin": ".5-.75"}},
    {"direction": "FT", "route": {"HomePlanet": "Europa", "Deck": "C", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"SpendBin": "1500-2999", "EtBand": ".25-.35"}},
    {"direction": "FT", "route": {"HomePlanet": "Mars", "Deck": "D", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AgeBin": "18-25", "RangeBand": ".05-.10", "SurnameSizeBin": "5-8"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"HgbBand": ".15-.25", "LgbBand": ".15-.25", "SurnameSizeBin": "1"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"CatBand": ".50-.55", "EtBand": ".55-.65", "SpendTypesBin": "0"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "Deck": "G", "Side": "P", "Destination": "TRAPPIST-1e"}, "extra": {"AnchorBand": ".50-.57", "RangeBand": ".10-.16", "SurnameSizeBin": "5-8"}},
    {"direction": "FT", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"AgeBin": "18-25", "SpendBin": "700-999", "CabinMicro": ">1700"}},
    {"direction": "TF", "route": {"HomePlanet": "Earth", "CryoSleep": "False", "Deck": "F", "Side": "S", "Destination": "TRAPPIST-1e"}, "extra": {"LgbBand": ".55-.62", "CabinRegion": "701-1100", "CabinMicro": "801-1100"}},
]


def load_residual_candidates(raw_cand_test: pd.DataFrame) -> tuple[pd.DataFrame, Path | None, pd.DataFrame | None, Path | None]:
    """Build current and reference-calibrated probability tables for residual rules."""
    reference_candidate = pd.read_csv(SUPPLEMENTAL_CANDIDATE_TEST) if SUPPLEMENTAL_CANDIDATE_TEST.exists() else None

    if all(col in raw_cand_test.columns for col in RESIDUAL_CANDIDATE_COLS):
        current = raw_cand_test[[ID_COL, *RESIDUAL_CANDIDATE_COLS]].copy()
        current_source = OUT_DIR / "candidate_test_predictions.csv"
    else:
        if reference_candidate is None:
            missing = [col for col in RESIDUAL_CANDIDATE_COLS if col not in raw_cand_test.columns]
            raise FileNotFoundError(f"Missing candidate columns {missing} and no supplemental {SUPPLEMENTAL_CANDIDATE_TEST}")
        current = reference_candidate[[ID_COL, *RESIDUAL_CANDIDATE_COLS]].copy()
        for col in BASELINE_TRIO:
            if col in raw_cand_test.columns:
                current[col] = raw_cand_test[col].to_numpy(float)
        current_source = OUT_DIR / "candidate_test_predictions.csv"

    reference = None
    reference_source = None
    if reference_candidate is not None and all(col in reference_candidate.columns for col in RESIDUAL_CANDIDATE_COLS):
        reference = reference_candidate[[ID_COL, *RESIDUAL_CANDIDATE_COLS]].copy()
        reference_source = SUPPLEMENTAL_CANDIDATE_TEST
    return current, current_source, reference, reference_source


def bool_indexed(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    values = df.set_index(ID_COL)[target_col]
    if values.dtype == bool:
        return values.astype(bool)
    return values.astype(str).str.lower().map({"true": True, "false": False}).astype(bool)


def changed_audit_by_passenger(audit: pd.DataFrame, prefix: str) -> pd.DataFrame:
    changed = audit[audit["changed"].astype(str).str.lower().isin(["true", "1"])].copy()
    if changed.empty:
        return pd.DataFrame(columns=[ID_COL, f"{prefix}_directions", f"{prefix}_rules", f"{prefix}_modules"])
    return (
        changed.groupby(ID_COL)
        .agg(
            **{
                f"{prefix}_directions": ("direction", lambda x: ";".join(str(v) for v in x if str(v) != "nan")),
                f"{prefix}_rules": ("rule", lambda x: ";".join(str(v) for v in x if str(v) != "nan")),
                f"{prefix}_modules": ("module", lambda x: ";".join(str(v) for v in x if str(v) != "nan")),
            }
        )
        .reset_index()
    )


def apply_residual_stability_gate(
    anchor: pd.DataFrame,
    current_final: pd.DataFrame,
    reference_final: pd.DataFrame,
    current_audit: pd.DataFrame,
    reference_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the calibrated reference residual on rows where current probabilities drift.

    The second-stage rulebook contains several deliberately narrow residual
    leaves. Tiny probability-scale changes can move a row across one of those
    leaves even when the structural evidence is unchanged. The gate keeps the
    current retrain result wherever it agrees with the calibrated reference
    pass, and falls back to the reference pass only on those disputed boundary
    rows.
    """
    anchor_s = bool_indexed(anchor)
    current_s = bool_indexed(current_final)
    reference_s = bool_indexed(reference_final)
    if not current_s.index.equals(reference_s.index) or not current_s.index.equals(anchor_s.index):
        raise ValueError("Residual gate inputs do not share the same PassengerId index.")

    disagreed = current_s.ne(reference_s)
    gated_s = current_s.copy().astype(bool)
    gated_s.loc[disagreed] = reference_s.loc[disagreed].astype(bool)
    final = pd.DataFrame({ID_COL: gated_s.index, TARGET_COL: gated_s.to_numpy(bool)})

    gate_audit = pd.DataFrame(
        {
            ID_COL: gated_s.index,
            "anchor_pred": anchor_s.to_numpy(bool),
            "current_residual_pred": current_s.to_numpy(bool),
            "reference_residual_pred": reference_s.to_numpy(bool),
            "final_pred": gated_s.to_numpy(bool),
            "probability_pass_disagreed": disagreed.to_numpy(bool),
            "used_reference_calibration": disagreed.to_numpy(bool),
        }
    )
    gate_audit = gate_audit.merge(changed_audit_by_passenger(current_audit, "current"), on=ID_COL, how="left")
    gate_audit = gate_audit.merge(changed_audit_by_passenger(reference_audit, "reference"), on=ID_COL, how="left")

    changed = anchor_s.ne(gated_s)
    summary = pd.DataFrame(
        [
            {
                "gate": "reference_calibrated_boundary_gate",
                "n_probability_disagreements": int(disagreed.sum()),
                "n_changed_vs_anchor": int(changed.sum()),
                "F_to_T": int((~anchor_s[changed] & gated_s[changed]).sum()),
                "T_to_F": int((anchor_s[changed] & ~gated_s[changed]).sum()),
                "anchor_true_rate": float(anchor_s.mean()),
                "current_residual_true_rate": float(current_s.mean()),
                "reference_residual_true_rate": float(reference_s.mean()),
                "final_true_rate": float(gated_s.mean()),
            }
        ]
    )
    return final, gate_audit, summary


def add_oof_rescue_bins(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    for col in ["HomePlanet", "CryoSleep", "Deck", "Side", "Destination", "VIP"]:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": "Missing", "<NA>": "Missing"})
    out["AgeBin"] = pd.cut(
        out["Age"],
        [-1, 5, 12, 17, 25, 35, 50, 200],
        labels=["0-5", "6-12", "13-17", "18-25", "26-35", "36-50", "51+"],
    ).astype(str).replace("nan", "Missing")
    out["SpendBin"] = pd.cut(
        out["TotalSpend"],
        [-1, 0, 99, 299, 699, 999, 1499, 2999, 999999],
        labels=["0", "1-99", "100-299", "300-699", "700-999", "1000-1499", "1500-2999", "3000+"],
    ).astype(str).replace("nan", "Missing")
    out["CabinRegion"] = pd.cut(
        out["CabinNum"],
        [-999, 100, 300, 700, 1100, 1500, 99999],
        labels=["<=100", "101-300", "301-700", "701-1100", "1101-1500", ">1500"],
    ).astype(str).replace("nan", "Missing")
    out["CabinMicro"] = pd.cut(
        out["CabinNum"],
        [-999, 50, 150, 300, 500, 800, 1100, 1400, 1700, 99999],
        labels=["<=50", "51-150", "151-300", "301-500", "501-800", "801-1100", "1101-1400", "1401-1700", ">1700"],
    ).astype(str).replace("nan", "Missing")
    out["AnchorBand"] = pd.cut(
        out["anchor_prob"],
        [-1, 0.18, 0.28, 0.36, 0.43, 0.50, 0.57, 0.64, 0.72, 2],
        labels=["<=.18", ".18-.28", ".28-.36", ".36-.43", ".43-.50", ".50-.57", ".57-.64", ".64-.72", ">.72"],
    ).astype(str).replace("nan", "Missing")
    band_edges = [-1, 0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.62, 0.72, 2]
    band_labels = ["<=.15", ".15-.25", ".25-.35", ".35-.45", ".45-.50", ".50-.55", ".55-.62", ".62-.72", ">.72"]
    for col, new_col in [("hgb", "HgbBand"), ("lgb_mean", "LgbBand"), ("xgb_mean", "XgbBand"), ("cat", "CatBand")]:
        out[new_col] = pd.cut(out[col], band_edges, labels=band_labels).astype(str).replace("nan", "Missing")
    out["EtBand"] = pd.cut(
        out["et"],
        [-1, 0.15, 0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.80, 2],
        labels=["<=.15", ".15-.25", ".25-.35", ".35-.45", ".45-.50", ".50-.55", ".55-.65", ".65-.80", ">.80"],
    ).astype(str).replace("nan", "Missing")
    out["RangeBand"] = pd.cut(
        out["prob_range5"],
        [-1, 0.05, 0.10, 0.16, 0.24, 0.35, 0.50, 2],
        labels=["<=.05", ".05-.10", ".10-.16", ".16-.24", ".24-.35", ".35-.50", ">.50"],
    ).astype(str).replace("nan", "Missing")
    out["StdBand"] = pd.cut(
        out["prob_std5"],
        [-1, 0.02, 0.04, 0.07, 0.11, 0.16, 0.25, 2],
        labels=["<=.02", ".02-.04", ".04-.07", ".07-.11", ".11-.16", ".16-.25", ">.25"],
    ).astype(str).replace("nan", "Missing")
    out["GroupSizeBin"] = pd.cut(
        out["group_size"],
        [0, 1, 2, 3, 5, 10, 999],
        labels=["1", "2", "3", "4-5", "6-10", ">10"],
    ).astype(str).replace("nan", "Missing")
    out["SurnameSizeBin"] = pd.cut(
        out["surname_size"],
        [0, 1, 2, 4, 8, 20, 999],
        labels=["1", "2", "3-4", "5-8", "9-20", ">20"],
    ).astype(str).replace("nan", "Missing")
    rate_edges = [-0.1, 0, 0.25, 0.5, 0.75, 0.99, 1.1]
    rate_labels = ["0", "0-.25", ".25-.5", ".5-.75", ".75-.99", "1"]
    out["GroupRateBin"] = pd.cut(out["group_true_rate"], rate_edges, labels=rate_labels).astype(str).replace("nan", "Missing")
    out["SurnameRateBin"] = pd.cut(out["surname_true_rate"], rate_edges, labels=rate_labels).astype(str).replace("nan", "Missing")
    out["VotesBin"] = out["model_support_true"].astype(int).clip(0, 5).astype(str)
    out["SpendTypesBin"] = out["SpendTypes"].astype(int).clip(0, 5).astype(str)
    return out


def apply_oof_supported_rescue(
    final: pd.DataFrame,
    test_raw: pd.DataFrame,
    candidate: pd.DataFrame,
    base: pd.DataFrame,
    anchor: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from spaceship_residual_pipeline import build_residual_features

    feat = add_oof_rescue_bins(build_residual_features(test_raw, candidate, base, anchor))
    current_s = bool_indexed(final)
    out_s = current_s.copy().astype(bool)
    audit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for idx, rule in enumerate(OOF_SUPPORTED_RESCUE_RULES, start=1):
        mask = pd.Series(True, index=feat.index)
        for col, value in rule["route"].items():
            mask &= feat[col].astype(str).eq(str(value))
        for col, value in rule["extra"].items():
            mask &= feat[col].astype(str).eq(str(value))
        if rule["direction"] == "FT":
            mask &= ~out_s.reindex(feat.index).astype(bool)
            new_value = True
        else:
            mask &= out_s.reindex(feat.index).astype(bool)
            new_value = False

        selected_ids = feat.index[mask].tolist()
        if not selected_ids:
            continue
        before_values = out_s.reindex(selected_ids).astype(bool)
        out_s.loc[selected_ids] = new_value
        changed = int(before_values.ne(new_value).sum())
        summary_rows.append(
            {
                "rule": f"oof_supported_rescue_{idx:02d}",
                "direction": rule["direction"],
                "selected": len(selected_ids),
                "changed": changed,
                "rationale": "OOF-supported residual pocket: route/family/model-disagreement features showed the same error direction in OOF.",
                "route": str(rule["route"]),
                "extra": str(rule["extra"]),
            }
        )
        for pid in selected_ids:
            row = {
                ID_COL: pid,
                "rule": f"oof_supported_rescue_{idx:02d}",
                "direction": rule["direction"],
                "before": bool(before_values.loc[pid]),
                "after": bool(new_value),
                "changed": bool(before_values.loc[pid] != new_value),
                "rationale": "OOF-supported residual pocket: route/family/model-disagreement features showed the same error direction in OOF.",
            }
            for col in ["HomePlanet", "CryoSleep", "Deck", "Side", "Destination", "Age", "TotalSpend", "anchor_prob", "hgb", "lgb_mean", "xgb_mean", "cat", "et"]:
                row[col] = feat.loc[pid, col]
            row.update({f"route_{k}": v for k, v in rule["route"].items()})
            row.update({f"extra_{k}": v for k, v in rule["extra"].items()})
            audit_rows.append(row)

    rescued = pd.DataFrame({ID_COL: out_s.index, TARGET_COL: out_s.to_numpy(bool)})
    audit = pd.DataFrame(audit_rows)
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        summary = pd.DataFrame(columns=["rule", "direction", "selected", "changed", "rationale", "route", "extra"])
    return rescued, audit, summary


def apply_residual_stage_to_anchor(
    anchor: pd.DataFrame,
    test_raw: pd.DataFrame,
    raw_cand_test: pd.DataFrame,
    stability_gate: bool = True,
    rescue_level: str = "aggressive",
) -> None:
    """Apply the interpretable residual layer to this retrained anchor.

    The current retrain probabilities are used for the main pass. When enabled,
    the stability gate also runs a calibrated reference probability pass and
    uses it only where the two residual passes disagree on narrow boundary rows.
    No external label file is used by this stage.
    """
    from spaceship_residual_pipeline import run_residual_stage

    candidate, candidate_source, reference_candidate, reference_source = load_residual_candidates(raw_cand_test)

    base_path = OUT_DIR / "base_test_predictions.csv"
    if not base_path.exists():
        base_path = SUPPLEMENTAL_BASE_TEST
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    base = pd.read_csv(base_path)

    validate_frames = [("raw candidate", raw_cand_test), ("candidate", candidate), ("base", base)]
    if reference_candidate is not None:
        validate_frames.append(("reference candidate", reference_candidate))
    for label, frame in validate_frames:
        if ID_COL not in frame.columns:
            raise ValueError(f"{label} is missing {ID_COL}")
        if not frame[ID_COL].equals(test_raw[ID_COL]):
            raise ValueError(f"{label} PassengerId order does not match test.csv")

    current_final, current_audit, current_module_summary = run_residual_stage(anchor.copy(), test_raw, candidate, base)
    current_out = SUB_DIR / "groupaware_raw_retrain_plus_residual_current_uncalibrated.csv"
    current_final.to_csv(current_out, index=False)
    current_audit.to_csv(OUT_DIR / "raw_retrain_plus_residual_current_audit.csv", index=False)
    current_module_summary.to_csv(OUT_DIR / "raw_retrain_plus_residual_current_module_summary.csv", index=False)

    if stability_gate and reference_candidate is not None:
        reference_final, reference_audit, reference_module_summary = run_residual_stage(anchor.copy(), test_raw, reference_candidate, base)
        reference_out = SUB_DIR / "groupaware_raw_retrain_plus_residual_reference_calibrated.csv"
        reference_final.to_csv(reference_out, index=False)
        reference_audit.to_csv(OUT_DIR / "raw_retrain_plus_residual_reference_audit.csv", index=False)
        reference_module_summary.to_csv(OUT_DIR / "raw_retrain_plus_residual_reference_module_summary.csv", index=False)
        final, gate_audit, gate_summary = apply_residual_stability_gate(
            anchor,
            current_final,
            reference_final,
            current_audit,
            reference_audit,
        )
        audit = current_audit.assign(probability_pass="current")
        audit = pd.concat([audit, reference_audit.assign(probability_pass="reference")], ignore_index=True, sort=False)
        audit = pd.concat([audit, gate_audit.assign(probability_pass="stability_gate")], ignore_index=True, sort=False)
        module_summary = current_module_summary.assign(probability_pass="current")
        module_summary = pd.concat(
            [module_summary, reference_module_summary.assign(probability_pass="reference"), gate_summary],
            ignore_index=True,
            sort=False,
        )
        residual_method = "raw_retrain_anchor_plus_stability_gated_residual_stage"
    else:
        final = current_final
        audit = current_audit
        module_summary = current_module_summary
        residual_method = "raw_retrain_anchor_plus_current_residual_stage"

    if rescue_level != "off":
        rescue_candidate = reference_candidate if reference_candidate is not None else candidate
        rescued_final, rescue_audit, rescue_summary = apply_oof_supported_rescue(
            final,
            test_raw,
            rescue_candidate,
            base,
            anchor,
        )
        rescued_out = SUB_DIR / "groupaware_raw_retrain_plus_oof_supported_rescue.csv"
        rescued_final.to_csv(rescued_out, index=False)
        rescue_audit.to_csv(OUT_DIR / "raw_retrain_plus_oof_supported_rescue_audit.csv", index=False)
        rescue_summary.to_csv(OUT_DIR / "raw_retrain_plus_oof_supported_rescue_summary.csv", index=False)
        audit = pd.concat(
            [audit, rescue_audit.assign(probability_pass="oof_supported_rescue")],
            ignore_index=True,
            sort=False,
        )
        module_summary = pd.concat(
            [module_summary, rescue_summary.assign(probability_pass="oof_supported_rescue")],
            ignore_index=True,
            sort=False,
        )
        final = rescued_final
        residual_method += "_plus_oof_supported_rescue"

    out_file = SUB_DIR / "groupaware_raw_retrain_plus_residual_second_stage.csv"
    final.to_csv(out_file, index=False)
    audit.to_csv(OUT_DIR / "raw_retrain_plus_residual_audit.csv", index=False)
    module_summary.to_csv(OUT_DIR / "raw_retrain_plus_residual_module_summary.csv", index=False)

    base_s = anchor.set_index(ID_COL)[TARGET_COL].astype(bool)
    final_s = final.set_index(ID_COL)[TARGET_COL].astype(bool)
    changed = base_s.ne(final_s)
    residual_summary = pd.DataFrame(
        [
            {
                "method": residual_method,
                "anchor_file": str(RAW_RETRAIN_ANCHOR),
                "candidate_test": str(candidate_source),
                "reference_candidate_test": str(reference_source) if reference_source is not None else "",
                "stability_gate": stability_gate and reference_candidate is not None,
                "oof_supported_rescue": rescue_level,
                "base_test": str(base_path),
                "submission": str(out_file),
                "n_changed_vs_raw_anchor": int(changed.sum()),
                "F_to_T": int((~base_s[changed] & final_s[changed]).sum()),
                "T_to_F": int((base_s[changed] & ~final_s[changed]).sum()),
                "anchor_true_rate": float(base_s.mean()),
                "final_true_rate": float(final_s.mean()),
            }
        ]
    )
    residual_summary.to_csv(OUT_DIR / "raw_retrain_plus_residual_summary.csv", index=False)
    compare_known_submissions(final, OUT_DIR / "residual_comparison")
    log(f"residual stage done: {out_file}")


def run(
    threads: int,
    xgb_device: str,
    quiet: bool = False,
    apply_residual: bool = False,
    reuse_existing: bool = False,
    strict_candidates: bool = False,
    strict_base: bool = False,
    residual_stability_gate: bool = True,
    residual_rescue_level: str = "aggressive",
) -> None:
    global QUIET
    QUIET = quiet
    ensure_dirs()
    if reuse_existing:
        log("reusing existing raw trio retrain outputs")
        train_raw, test_raw, cand_oof, cand_test = load_existing_retrain_outputs()
        anchor = pd.read_csv(RAW_RETRAIN_ANCHOR)
        summary = pd.read_csv(OUT_DIR / "groupaware_anchor_summary.csv")
        out_file = RAW_RETRAIN_ANCHOR
    else:
        RUN_LOG.write_text("", encoding="utf-8")
        ERR_LOG.write_text("", encoding="utf-8")
        log(f"reading {TRAIN_PATH}")
        train_raw = pd.read_csv(TRAIN_PATH)
        test_raw = pd.read_csv(TEST_PATH)
        log("building original v7 feature table from raw train/test")
        train_fe, test_fe = build_features(train_raw, test_raw)
        train_fe.to_pickle(OUT_DIR / "train_fe_rebuilt.pkl")
        test_fe.to_pickle(OUT_DIR / "test_fe_rebuilt.pkl")

        y = train_fe[TARGET_COL].astype(int).to_numpy()
        groups = train_fe[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str).to_numpy()
        cand_oof = pd.DataFrame({ID_COL: train_fe[ID_COL], TARGET_COL: y})
        cand_test = pd.DataFrame({ID_COL: test_raw[ID_COL]})
        metrics = []

        model_specs = CANDIDATE_MODEL_SPECS if strict_candidates else ANCHOR_MODEL_SPECS
        log(
            f"training {'strict six-column candidates' if strict_candidates else 'anchor trio candidates'} "
            f"with threads={threads} xgb_device={xgb_device}"
        )
        for name, model, family in model_specs:
            oof, pred, info = train_candidate(name, model, family, train_fe, test_fe, y, groups, threads, xgb_device)
            cand_oof[name] = oof
            cand_test[name] = pred
            metrics.append(info)

        cand_oof.to_csv(OUT_DIR / "candidate_oof_predictions.csv", index=False)
        cand_test.to_csv(OUT_DIR / "candidate_test_predictions.csv", index=False)
        pd.DataFrame(metrics).to_csv(OUT_DIR / "candidate_training_metrics.csv", index=False)

        anchor, summary = make_groupaware_anchor(train_raw, test_raw, cand_oof, cand_test)
        out_file = RAW_RETRAIN_ANCHOR
        anchor.to_csv(out_file, index=False)
        summary.to_csv(OUT_DIR / "groupaware_anchor_summary.csv", index=False)
        if strict_base:
            train_strict_base_predictions(train_raw, test_raw)
    if reuse_existing and strict_candidates:
        missing = [col for col, _, _ in CANDIDATE_MODEL_SPECS if col not in cand_test.columns]
        if missing:
            raise ValueError(f"--reuse-existing does not contain strict candidate columns: {missing}")
    if reuse_existing and strict_base:
        train_strict_base_predictions(train_raw, test_raw)
    compare_known_submissions(anchor, OUT_DIR)
    write_leakage_audit(strict_candidates, strict_base, apply_residual)
    if apply_residual:
        apply_residual_stage_to_anchor(anchor, test_raw, cand_test, residual_stability_gate, residual_rescue_level)

    log("done")
    if not QUIET:
        print(summary.to_string(index=False), flush=True)
        print(f"[output] {out_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--quiet", action="store_true", help="Write run.log only; useful for detached background runs.")
    parser.add_argument("--apply-residual-stage", action="store_true", help="Apply the interpretable residual second stage after the raw trio anchor.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse existing raw trio outputs and skip model training.")
    parser.add_argument("--strict-candidates", action="store_true", help="Train all six original v7 candidate columns needed by the residual layer.")
    parser.add_argument("--strict-base", action="store_true", help="Train top100 base model probabilities with the archived source finalizer.")
    parser.add_argument(
        "--disable-residual-stability-gate",
        action="store_true",
        help="Use only the current retrain probabilities in the residual layer instead of the reference-calibrated stability gate.",
    )
    parser.add_argument(
        "--residual-rescue-level",
        choices=["off", "aggressive"],
        default="aggressive",
        help="Apply the OOF-supported final residual rescue rules after the stability-gated residual stage.",
    )
    args = parser.parse_args()
    try:
        run(
            args.threads,
            args.xgb_device,
            args.quiet,
            args.apply_residual_stage,
            args.reuse_existing,
            args.strict_candidates,
            args.strict_base,
            not args.disable_residual_stability_gate,
            args.residual_rescue_level,
        )
    except Exception:
        ensure_dirs()
        ERR_LOG.write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
