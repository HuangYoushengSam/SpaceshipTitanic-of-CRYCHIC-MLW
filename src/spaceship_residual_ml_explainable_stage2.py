from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from spaceship_81201_cluster_rulebook_second_stage import (
    ID_COL,
    ROOT,
    TARGET,
    TEST_CSV,
    TEST_PREDS,
    apply_rulebook,
    build_features,
    to_bool_series,
)
from spaceship_81201_rulebook_plus_clean_ml import add_plus_features
from spaceship_81201_ultra_strict_veto import apply_vetoes as apply_ultra_vetoes
from spaceship_81201_ultra_strict_veto import modules as ultra_veto_modules


OUT_DIR = ROOT / "outputs" / "residual_ml_explainable_stage2"
SUB_DIR = OUT_DIR / "submissions"
BASELINE_DIR = OUT_DIR / "baseline_81201"

TRAIN_CSV = ROOT / "data" / "train.csv"
CANDIDATE_OOF = ROOT / "artifacts" / "probability_tables" / "candidate_oof_predictions.csv"
CANDIDATE_TEST = ROOT / "artifacts" / "probability_tables" / "candidate_test_predictions.csv"
BASELINE_TRIO = ["xgb@F1+F2", "lgb@F1+F2", "xgb@F1+F2+F3"]
BASELINE_WEIGHTS = (0.50, 0.30, 0.20)
BASELINE_GROUP_ALPHA = 0.15
BASELINE_THRESHOLD = 0.470


NUMERIC_FEATURES = [
    "Age",
    "CabinNum",
    "TotalSpend",
    "SpendTypes",
    "group_size",
    "group_true_rate",
    "surname_size",
    "surname_true_rate",
    "cat",
    "xgb_mean",
    "lgb_mean",
    "hgb",
    "et",
    "cand_mean",
    "tree_spread",
    "anchor_prob",
    "expert_vote_count",
    "expert_disagreement",
    "prob_std5",
    "prob_range5",
    "model_support_true",
]

CATEGORICAL_FEATURES = ["HomePlanet", "CryoSleep", "Deck", "Side", "Destination", "VIP"]

AUDIT_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Deck",
    "Side",
    "Destination",
    "Age",
    "CabinNum",
    "TotalSpend",
    "SpendTypes",
    "group_size",
    "group_true_rate",
    "surname_size",
    "surname_true_rate",
    "cat",
    "xgb_mean",
    "lgb_mean",
    "hgb",
    "et",
    "cand_mean",
    "anchor_prob",
    "expert_vote_count",
    "prob_std5",
    "prob_range5",
    "model_support_true",
]


Condition = tuple[str, str, float]
RuleFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ResidualModule:
    name: str
    direction: str
    family: str
    rationale: str
    selector: RuleFn


def generate_groupaware_81201_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the original 0.81201-style anchor from local model probabilities."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    oof = pd.read_csv(CANDIDATE_OOF)
    test_pred = pd.read_csv(CANDIDATE_TEST)

    missing_oof = [col for col in BASELINE_TRIO if col not in oof.columns]
    missing_test = [col for col in BASELINE_TRIO if col not in test_pred.columns]
    if missing_oof or missing_test:
        raise ValueError(f"Missing baseline probability columns: oof={missing_oof}, test={missing_test}")

    if ID_COL not in oof.columns:
        oof.insert(0, ID_COL, train[ID_COL].values)
    if ID_COL not in test_pred.columns:
        test_pred.insert(0, ID_COL, test[ID_COL].values)

    y = train[TARGET].astype(int).reset_index(drop=True)
    train_groups = train[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str).reset_index(drop=True)
    test_groups = test[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str).reset_index(drop=True)

    w1, w2, w3 = BASELINE_WEIGHTS
    c1, c2, c3 = BASELINE_TRIO
    base_oof = w1 * oof[c1].to_numpy() + w2 * oof[c2].to_numpy() + w3 * oof[c3].to_numpy()
    base_test = w1 * test_pred[c1].to_numpy() + w2 * test_pred[c2].to_numpy() + w3 * test_pred[c3].to_numpy()

    oof_group_mean = pd.DataFrame({"g": train_groups, "p": base_oof}).groupby("g")["p"].transform("mean").to_numpy()
    test_group_mean = pd.DataFrame({"g": test_groups, "p": base_test}).groupby("g")["p"].transform("mean").to_numpy()
    oof_group_size = train_groups.map(train_groups.value_counts()).to_numpy()
    test_group_size = test_groups.map(test_groups.value_counts()).to_numpy()

    alpha_oof = np.where(
        oof_group_size >= 4,
        BASELINE_GROUP_ALPHA * 1.15,
        np.where(oof_group_size >= 2, BASELINE_GROUP_ALPHA, BASELINE_GROUP_ALPHA * 0.5),
    )
    alpha_test = np.where(
        test_group_size >= 4,
        BASELINE_GROUP_ALPHA * 1.15,
        np.where(test_group_size >= 2, BASELINE_GROUP_ALPHA, BASELINE_GROUP_ALPHA * 0.5),
    )
    smooth_oof = (1 - alpha_oof) * base_oof + alpha_oof * oof_group_mean
    smooth_test = (1 - alpha_test) * base_test + alpha_test * test_group_mean

    anchor = pd.DataFrame({ID_COL: test[ID_COL], TARGET: smooth_test >= BASELINE_THRESHOLD})
    baseline_summary = pd.DataFrame(
        [
            {
                "method": "groupaware_trio_blend",
                "probability_columns": "|".join(BASELINE_TRIO),
                "weights": "|".join(f"{w:.2f}" for w in BASELINE_WEIGHTS),
                "group_alpha": BASELINE_GROUP_ALPHA,
                "threshold": BASELINE_THRESHOLD,
                "oof_accuracy": float(accuracy_score(y, smooth_oof >= BASELINE_THRESHOLD)),
                "test_true_rate": float(anchor[TARGET].mean()),
            }
        ]
    )
    anchor.to_csv(BASELINE_DIR / "baseline_groupaware_81201_anchor.csv", index=False)
    baseline_summary.to_csv(BASELINE_DIR / "baseline_groupaware_81201_summary.csv", index=False)
    return anchor, baseline_summary


def stage_false(f: pd.DataFrame) -> pd.Series:
    return ~f["stage_pred"].astype(bool)


def stage_true(f: pd.DataFrame) -> pd.Series:
    return f["stage_pred"].astype(bool)


def design_matrix(feat: pd.DataFrame) -> pd.DataFrame:
    numeric = feat[NUMERIC_FEATURES].copy().fillna(-1)
    categorical = pd.get_dummies(
        feat[CATEGORICAL_FEATURES].astype("string").fillna("Missing").astype(str),
        prefix=CATEGORICAL_FEATURES,
        dtype=int,
    )
    return pd.concat([numeric, categorical], axis=1)


def apply_leaf_paths(feat: pd.DataFrame, paths: list[list[Condition]]) -> pd.Series:
    x = design_matrix(feat)
    selected = pd.Series(False, index=feat.index)
    for path in paths:
        mask = pd.Series(True, index=feat.index)
        for col, op, threshold in path:
            values = x[col] if col in x.columns else pd.Series(0.0, index=feat.index)
            if op == "<=":
                mask &= values <= threshold
            elif op == ">":
                mask &= values > threshold
            else:
                raise ValueError(f"Unsupported operator: {op}")
        selected |= mask
    return selected


def ft_mars_e_cryo_zero_pruned(f: pd.DataFrame) -> pd.Series:
    false = stage_false(f)
    noisy_family_tail = (
        f["Side"].eq("P")
        & f["Age"].between(31, 55)
        & (f["group_true_rate"] >= 0.5)
        & (f["surname_true_rate"] >= 0.5)
        & (f["et"] < 0.25)
    )
    return (
        false
        & f["HomePlanet"].eq("Mars")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("E")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & ~noisy_family_tail
    )


def ft_earth_g_cryo_zero_strict(f: pd.DataFrame) -> pd.Series:
    false = stage_false(f)
    side_s_adult = f["Side"].eq("S") & (f["Age"] >= 23)
    side_p_child_small_group = f["Side"].eq("P") & (f["Age"] <= 7) & (f["group_size"] < 7)
    side_p_old = f["Side"].eq("P") & (f["Age"] >= 46)
    side_p_family_consensus = f["Side"].eq("P") & f["Age"].between(20, 27) & (f["group_true_rate"] >= 1.0)
    return (
        false
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("G")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & (side_s_adult | side_p_child_small_group | side_p_old | side_p_family_consensus)
    )


def ft_deck_e_hgb_group0(f: pd.DataFrame) -> pd.Series:
    return stage_false(f) & f["Deck"].eq("E") & (f["hgb"] >= 0.50) & (f["group_true_rate"] <= 0.01)


def ft_deck_e_side_s_lgb40(f: pd.DataFrame) -> pd.Series:
    return stage_false(f) & f["Deck"].eq("E") & f["Side"].eq("S") & (f["lgb_mean"] >= 0.40)


def ft_earth_gs_spender_model(f: pd.DataFrame) -> pd.Series:
    return (
        stage_false(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("G")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(700, 950)
        & (f["hgb"] >= 0.50)
        & (f["cat"] < 0.50)
        & (f["group_true_rate"] <= 0.50)
    )


FT_TREE_PATHS: list[list[Condition]] = [
    [
        ("cand_mean", ">", 0.17847494781017303),
        ("lgb_mean", ">", 0.31843292713165283),
        ("anchor_prob", "<=", 0.5001194626092911),
        ("hgb", ">", 0.429107129573822),
        ("anchor_prob", "<=", 0.46487957239151),
        ("CabinNum", ">", 21.0),
        ("CryoSleep_False", "<=", 0.5),
        ("CabinNum", "<=", 1201.0),
        ("group_true_rate", "<=", 0.4166666716337204),
        ("Age", ">", 8.0),
    ],
    [
        ("cand_mean", ">", 0.17847494781017303),
        ("lgb_mean", ">", 0.31843292713165283),
        ("anchor_prob", "<=", 0.5001194626092911),
        ("hgb", ">", 0.429107129573822),
        ("anchor_prob", "<=", 0.46487957239151),
        ("CabinNum", ">", 21.0),
        ("CryoSleep_False", ">", 0.5),
        ("anchor_prob", "<=", 0.4564754366874695),
        ("Age", ">", 12.5),
        ("TotalSpend", "<=", 1108.0),
        ("tree_spread", ">", 0.1676909551024437),
    ],
    [
        ("cand_mean", ">", 0.17847494781017303),
        ("lgb_mean", "<=", 0.31843292713165283),
        ("Deck_E", ">", 0.5),
        ("hgb", "<=", 0.18116368353366852),
        ("xgb_mean", ">", 0.17907971888780594),
        ("prob_range5", "<=", 0.11624139547348022),
        ("TotalSpend", ">", 659.5),
    ],
    [
        ("cand_mean", "<=", 0.17847494781017303),
        ("TotalSpend", "<=", 951.5),
        ("TotalSpend", ">", 916.5),
        ("Deck_G", ">", 0.5),
    ],
    [
        ("cand_mean", ">", 0.17847494781017303),
        ("lgb_mean", "<=", 0.31843292713165283),
        ("Deck_E", "<=", 0.5),
        ("Deck_G", "<=", 0.5),
        ("hgb", ">", 0.33379435539245605),
        ("lgb_mean", "<=", 0.2996489852666855),
        ("tree_spread", ">", 0.10529910400509834),
        ("hgb", "<=", 0.36837951838970184),
    ],
    [
        ("cand_mean", ">", 0.17847494781017303),
        ("lgb_mean", ">", 0.31843292713165283),
        ("anchor_prob", "<=", 0.5001194626092911),
        ("hgb", ">", 0.429107129573822),
        ("anchor_prob", "<=", 0.46487957239151),
        ("CabinNum", ">", 21.0),
        ("CryoSleep_False", "<=", 0.5),
        ("CabinNum", "<=", 1201.0),
        ("group_true_rate", ">", 0.4166666716337204),
        ("prob_std5", ">", 0.13912419974803925),
    ],
    [
        ("cand_mean", ">", 0.17847494781017303),
        ("lgb_mean", ">", 0.31843292713165283),
        ("anchor_prob", "<=", 0.5001194626092911),
        ("hgb", ">", 0.429107129573822),
        ("anchor_prob", "<=", 0.46487957239151),
        ("CabinNum", ">", 21.0),
        ("CryoSleep_False", "<=", 0.5),
        ("CabinNum", ">", 1201.0),
        ("xgb_mean", ">", 0.4662790149450302),
        ("Age", "<=", 14.0),
    ],
]


TF_WEAK_AWAKE_SPENDER_PATHS: list[list[Condition]] = [
    [
        ("hgb", "<=", 0.6103039681911469),
        ("CryoSleep_True", "<=", 0.5),
        ("et", ">", 0.3178209066390991),
        ("lgb_mean", "<=", 0.6743209362030029),
        ("Age", "<=", 49.0),
        ("hgb", "<=", 0.5189317464828491),
        ("surname_size", ">", 1.5),
        ("anchor_prob", ">", 0.45227497816085815),
        ("TotalSpend", ">", 850.5),
        ("HomePlanet_Earth", ">", 0.5),
        ("xgb_mean", "<=", 0.5081263482570648),
    ],
    [
        ("hgb", "<=", 0.6103039681911469),
        ("CryoSleep_True", "<=", 0.5),
        ("et", ">", 0.3178209066390991),
        ("lgb_mean", "<=", 0.6743209362030029),
        ("Age", "<=", 49.0),
        ("hgb", "<=", 0.5189317464828491),
        ("surname_size", ">", 1.5),
        ("anchor_prob", ">", 0.45227497816085815),
        ("TotalSpend", "<=", 850.5),
        ("hgb", "<=", 0.4921811670064926),
        ("Age", "<=", 21.5),
        ("anchor_prob", "<=", 0.6371407508850098),
    ],
    [
        ("hgb", "<=", 0.6103039681911469),
        ("CryoSleep_True", "<=", 0.5),
        ("et", ">", 0.3178209066390991),
        ("lgb_mean", "<=", 0.6743209362030029),
        ("Age", "<=", 49.0),
        ("hgb", "<=", 0.5189317464828491),
        ("surname_size", ">", 1.5),
        ("anchor_prob", ">", 0.45227497816085815),
        ("TotalSpend", ">", 850.5),
        ("HomePlanet_Earth", ">", 0.5),
        ("xgb_mean", ">", 0.5081263482570648),
    ],
]


WAVE2_DECK_E_LOW_HGB_SPENDER = [
    [
        ("cand_mean", ">", 0.16877535730600357),
        ("lgb_mean", "<=", 0.31843292713165283),
        ("Deck_E", ">", 0.5),
        ("CabinNum", ">", 201.5),
        ("xgb_mean", ">", 0.18228576332330704),
        ("SpendTypes", ">", 1.0),
        ("surname_true_rate", ">", 0.18333333730697632),
        ("tree_spread", ">", 0.06592727079987526),
        ("hgb", "<=", 0.26425954699516296),
    ]
]

WAVE2_SIDE_S_GROUP_DISAGREE = [
    [
        ("cand_mean", ">", 0.16877535730600357),
        ("lgb_mean", ">", 0.31843292713165283),
        ("lgb_mean", "<=", 0.5690414011478424),
        ("xgb_mean", ">", 0.2905757874250412),
        ("anchor_prob", "<=", 0.4583986848592758),
        ("group_size", ">", 1.5),
        ("group_true_rate", "<=", 0.612500011920929),
        ("Side_P", "<=", 0.5),
        ("prob_std5", ">", 0.06902682036161423),
    ]
]

WAVE2_SINGLETON_SURNAME_SPENDER = [
    [
        ("cand_mean", ">", 0.16877535730600357),
        ("lgb_mean", ">", 0.31843292713165283),
        ("lgb_mean", "<=", 0.5690414011478424),
        ("xgb_mean", ">", 0.2905757874250412),
        ("anchor_prob", "<=", 0.4583986848592758),
        ("group_size", "<=", 1.5),
        ("surname_size", "<=", 5.5),
        ("CabinNum", ">", 206.0),
        ("surname_true_rate", ">", 0.36666667461395264),
        ("TotalSpend", ">", 687.0),
    ]
]

WAVE3_MID_LGB_ET = [
    [
        ("cand_mean", ">", 0.16877535730600357),
        ("lgb_mean", ">", 0.3065981864929199),
        ("lgb_mean", "<=", 0.5690414011478424),
        ("xgb_mean", ">", 0.28385747969150543),
        ("et", "<=", 0.6471949815750122),
        ("hgb", "<=", 0.5217485725879669),
        ("lgb_mean", ">", 0.40524493157863617),
        ("et", ">", 0.3804221749305725),
        ("et", "<=", 0.474833682179451),
        ("xgb_mean", "<=", 0.4524663835763931),
        ("SpendTypes", "<=", 2.5),
    ]
]

WAVE4_A = [
    [
        ("cand_mean", ">", 0.16877535730600357),
        ("lgb_mean", ">", 0.3065981864929199),
        ("lgb_mean", "<=", 0.5690414011478424),
        ("Age", "<=", 51.5),
        ("et", "<=", 0.6471949815750122),
        ("xgb_mean", ">", 0.28385747969150543),
        ("hgb", "<=", 0.5217485725879669),
        ("TotalSpend", "<=", 5279.5),
        ("CabinNum", "<=", 1718.5),
        ("anchor_prob", "<=", 0.3395500183105469),
        ("Age", "<=", 29.5),
        ("TotalSpend", ">", 644.5),
    ]
]

WAVE4_B = [
    [
        ("cand_mean", ">", 0.16877535730600357),
        ("lgb_mean", "<=", 0.3065981864929199),
        ("xgb_mean", "<=", 0.18976575881242752),
        ("TotalSpend", "<=", 1321.0),
        ("TotalSpend", ">", 697.0),
        ("surname_true_rate", "<=", 0.5372340381145477),
        ("Deck_D", "<=", 0.5),
        ("Destination_PSO J318.5-22", "<=", 0.5),
        ("group_true_rate", "<=", 0.1666666716337204),
        ("xgb_mean", ">", 0.16777075082063675),
        ("Age", ">", 25.5),
    ]
]

WAVE4_C = [
    [
        ("cand_mean", ">", 0.16877535730600357),
        ("lgb_mean", "<=", 0.3065981864929199),
        ("xgb_mean", "<=", 0.18976575881242752),
        ("TotalSpend", "<=", 1321.0),
        ("TotalSpend", ">", 697.0),
        ("surname_true_rate", "<=", 0.5372340381145477),
        ("Deck_D", "<=", 0.5),
        ("Destination_PSO J318.5-22", "<=", 0.5),
        ("group_true_rate", "<=", 0.1666666716337204),
        ("xgb_mean", ">", 0.16777075082063675),
        ("Age", "<=", 25.5),
        ("xgb_mean", ">", 0.178379587829113),
    ]
]

WAVE5_L47 = [
    [
        ("cand_mean", ">", 0.2512774169445038),
        ("anchor_prob", "<=", 0.39117924869060516),
        ("CabinNum", "<=", 886.5),
        ("CabinNum", ">", 79.5),
        ("tree_spread", "<=", 0.20686401426792145),
        ("xgb_mean", "<=", 0.38535694777965546),
        ("surname_size", "<=", 6.5),
        ("hgb", ">", 0.19909962266683578),
        ("SpendTypes", "<=", 4.5),
        ("prob_range5", ">", 0.07192189246416092),
        ("anchor_prob", ">", 0.3158538341522217),
        ("lgb_mean", "<=", 0.34335198998451233),
    ]
]

WAVE5_L67 = [
    [
        ("cand_mean", ">", 0.2512774169445038),
        ("anchor_prob", ">", 0.39117924869060516),
        ("lgb_mean", "<=", 0.5690414011478424),
        ("et", "<=", 0.6471949815750122),
        ("CabinNum", ">", 24.5),
        ("surname_size", "<=", 9.5),
        ("anchor_prob", "<=", 0.4224257320165634),
        ("TotalSpend", ">", 69.5),
        ("group_size", ">", 1.5),
        ("Deck_F", "<=", 0.5),
    ]
]

WAVE5_L78 = [
    [
        ("cand_mean", ">", 0.2512774169445038),
        ("anchor_prob", ">", 0.39117924869060516),
        ("lgb_mean", "<=", 0.5690414011478424),
        ("et", "<=", 0.6471949815750122),
        ("CabinNum", ">", 24.5),
        ("surname_size", "<=", 9.5),
        ("anchor_prob", ">", 0.4224257320165634),
        ("cat", ">", 0.43135835230350494),
        ("et", ">", 0.3577827215194702),
        ("CabinNum", ">", 272.5),
        ("Destination_TRAPPIST-1e", "<=", 0.5),
        ("prob_range5", ">", 0.13041245937347412),
    ]
]


def ft_distilled_primary(f: pd.DataFrame) -> pd.Series:
    return stage_false(f) & apply_leaf_paths(f, FT_TREE_PATHS)


def tf_weak_awake_spender(f: pd.DataFrame) -> pd.Series:
    return stage_true(f) & apply_leaf_paths(f, TF_WEAK_AWAKE_SPENDER_PATHS)


def ft_path(paths: list[list[Condition]]) -> RuleFn:
    def _selector(f: pd.DataFrame) -> pd.Series:
        return stage_false(f) & apply_leaf_paths(f, paths)

    return _selector


def ft_wave4_a_pruned(f: pd.DataFrame) -> pd.Series:
    bad_tail = f["HomePlanet"].eq("Mars") & f["Deck"].eq("F") & f["Side"].eq("P")
    return stage_false(f) & apply_leaf_paths(f, WAVE4_A) & ~bad_tail


def ft_wave4_b_pruned(f: pd.DataFrame) -> pd.Series:
    bad_tail = f["HomePlanet"].eq("Mars") & f["Deck"].eq("E") & f["Side"].eq("P")
    return stage_false(f) & apply_leaf_paths(f, WAVE4_B) & ~bad_tail


def ft_wave4_c_pruned(f: pd.DataFrame) -> pd.Series:
    return stage_false(f) & apply_leaf_paths(f, WAVE4_C) & f["HomePlanet"].eq("Earth") & (f["surname_true_rate"] >= 0.30)


def ft_wave5_l47_pruned(f: pd.DataFrame) -> pd.Series:
    uncertain_missing_age_ep = (
        f["HomePlanet"].eq("Earth")
        & f["Deck"].eq("E")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["Age"].isna()
    )
    return stage_false(f) & apply_leaf_paths(f, WAVE5_L47) & ~uncertain_missing_age_ep


def ft_wave5_l67(f: pd.DataFrame) -> pd.Series:
    return stage_false(f) & apply_leaf_paths(f, WAVE5_L67)


def ft_wave5_l78_pruned(f: pd.DataFrame) -> pd.Series:
    pso_child_large_family = (
        f["HomePlanet"].eq("Earth")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & f["Destination"].eq("PSO J318.5-22")
        & (f["Age"] <= 15)
        & (f["group_true_rate"] >= 0.95)
    )
    return stage_false(f) & apply_leaf_paths(f, WAVE5_L78) & ~pso_child_large_family


def residual_modules() -> list[ResidualModule]:
    return [
        ResidualModule(
            "mars_e_cryo_zero_pruned",
            "FT",
            "structural_zero_spend_recall",
            "Mars cryo-zero E-deck TRAPPIST rows are recovered, excluding one weak P-side adult family tail.",
            ft_mars_e_cryo_zero_pruned,
        ),
        ResidualModule(
            "earth_g_cryo_zero_strict",
            "FT",
            "structural_zero_spend_recall",
            "Earth/G/TRAPPIST cryo-zero rows are recovered only within side, age, and family-consensus subleaves.",
            ft_earth_g_cryo_zero_strict,
        ),
        ResidualModule(
            "deck_e_hgb_group0",
            "FT",
            "heterogeneous_model_recall",
            "E-deck rows with HGB support and no anchor group support are treated as conservative false negatives.",
            ft_deck_e_hgb_group0,
        ),
        ResidualModule(
            "deck_e_side_s_lgb40",
            "FT",
            "heterogeneous_model_recall",
            "A small E/S leaf is recovered when LGB crosses the local residual threshold.",
            ft_deck_e_side_s_lgb40,
        ),
        ResidualModule(
            "earth_gs_spender_model",
            "FT",
            "spender_boundary_recall",
            "Earth/G/S awake TRAPPIST spenders are recovered when HGB supports the positive class but Cat stays cautious.",
            ft_earth_gs_spender_model,
        ),
        ResidualModule(
            "distilled_primary_residual_leaves",
            "FT",
            "shallow_tree_distillation",
            "Depth-limited residual tree leaves based on cabin topology, spend, family support, and model disagreement.",
            ft_distilled_primary,
        ),
        ResidualModule(
            "weak_awake_spender_veto",
            "TF",
            "spender_boundary_veto",
            "Awake spender rows are vetoed when HGB is weak, ET is high, and the positive signal is not robust.",
            tf_weak_awake_spender,
        ),
        ResidualModule(
            "deck_e_low_hgb_spender_recall",
            "FT",
            "spender_boundary_recall",
            "Residual E-deck spenders with low HGB but useful XGB/candidate/surname structure are recovered.",
            ft_path(WAVE2_DECK_E_LOW_HGB_SPENDER),
        ),
        ResidualModule(
            "side_s_group_disagreement_recall",
            "FT",
            "family_model_disagreement_recall",
            "Side-S group-supported rows with weak anchor probability but meaningful LGB/XGB disagreement are recovered.",
            ft_path(WAVE2_SIDE_S_GROUP_DISAGREE),
        ),
        ResidualModule(
            "singleton_surname_spender_recall",
            "FT",
            "family_model_disagreement_recall",
            "Small-group spender rows with surname support and enough cabin/spend structure are recovered.",
            ft_path(WAVE2_SINGLETON_SURNAME_SPENDER),
        ),
        ResidualModule(
            "mid_lgb_et_boundary_recall",
            "FT",
            "heterogeneous_model_recall",
            "Rows with moderate LGB/ET, controlled HGB, and limited spend channels are recovered as boundary positives.",
            ft_path(WAVE3_MID_LGB_ET),
        ),
        ResidualModule(
            "low_anchor_spend_leaf_a_pruned",
            "FT",
            "shallow_tree_distillation",
            "Low-anchor, mid-spend residual leaf with one Mars/F/P tail pruned by route topology.",
            ft_wave4_a_pruned,
        ),
        ResidualModule(
            "low_xgb_spender_leaf_b_pruned",
            "FT",
            "shallow_tree_distillation",
            "Low-XGB spender residual leaf with Mars/E/P tail removed as an outlier route pocket.",
            ft_wave4_b_pruned,
        ),
        ResidualModule(
            "young_earth_surname_leaf_c_pruned",
            "FT",
            "shallow_tree_distillation",
            "Young Earth residual leaf retained only when surname support is present.",
            ft_wave4_c_pruned,
        ),
        ResidualModule(
            "wave5_l67_clean_group_recall",
            "FT",
            "family_model_disagreement_recall",
            "Clean group-supported leaf with non-F deck, moderate anchor probability, and controlled LGB/ET range.",
            ft_wave5_l67,
        ),
        ResidualModule(
            "wave5_l47_pruned_low_anchor_recall",
            "FT",
            "shallow_tree_distillation",
            "Low-anchor residual leaf with missing-age Earth/E/P uncertainty pruned.",
            ft_wave5_l47_pruned,
        ),
        ResidualModule(
            "wave5_l78_pruned_route_recall",
            "FT",
            "route_family_recall",
            "Non-TRAPPIST route leaf with a PSO/G/P child-family exception pruned.",
            ft_wave5_l78_pruned,
        ),
    ]


def initial_recall_modules() -> list[ResidualModule]:
    names = {
        "mars_e_cryo_zero_pruned",
        "earth_g_cryo_zero_strict",
        "deck_e_hgb_group0",
        "deck_e_side_s_lgb40",
        "earth_gs_spender_model",
        "distilled_primary_residual_leaves",
    }
    return [module for module in residual_modules() if module.name in names]


def tf_boundary_modules() -> list[ResidualModule]:
    return [module for module in residual_modules() if module.name == "weak_awake_spender_veto"]


def late_recall_modules() -> list[ResidualModule]:
    initial = {module.name for module in initial_recall_modules()}
    tf_names = {module.name for module in tf_boundary_modules()}
    return [module for module in residual_modules() if module.name not in initial | tf_names]


def apply_modules(
    base: pd.DataFrame,
    feat: pd.DataFrame,
    modules: list[ResidualModule],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = base.set_index(ID_COL).copy()
    audit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for module in modules:
        feat = feat.copy()
        feat["stage_pred"] = out[TARGET].astype(bool).reindex(feat.index)
        target_value = module.direction == "FT"
        selected = module.selector(feat).fillna(False)
        selected_ids = selected[selected].index
        changed_ids: list[str] = []
        for pid in selected_ids:
            before = bool(out.loc[pid, TARGET])
            out.loc[pid, TARGET] = target_value
            after = bool(out.loc[pid, TARGET])
            changed = before != after
            if changed:
                changed_ids.append(pid)
            row = {
                ID_COL: pid,
                "module": module.name,
                "family": module.family,
                "direction": module.direction,
                "before": before,
                "after": after,
                "changed": changed,
                "rationale": module.rationale,
            }
            row.update(feat.loc[pid, AUDIT_FEATURES].to_dict())
            audit_rows.append(row)

        summary_rows.append(
            {
                "module": module.name,
                "family": module.family,
                "direction": module.direction,
                "selected": int(len(selected_ids)),
                "changed": int(len(changed_ids)),
                "rationale": module.rationale,
            }
        )

    return out.reset_index(), pd.DataFrame(audit_rows), pd.DataFrame(summary_rows)


def changed_summary(anchor_s: pd.Series, stage_s: pd.Series, final_s: pd.Series) -> dict[str, object]:
    vs_anchor = anchor_s.ne(final_s)
    ids_anchor = vs_anchor[vs_anchor].index
    vs_stage = stage_s.ne(final_s)
    ids_stage = vs_stage[vs_stage].index
    return {
        "n_changed_vs_anchor": int(vs_anchor.sum()),
        "F_to_T_vs_anchor": int(((~anchor_s.loc[ids_anchor]) & final_s.loc[ids_anchor]).sum()),
        "T_to_F_vs_anchor": int((anchor_s.loc[ids_anchor] & (~final_s.loc[ids_anchor])).sum()),
        "n_changed_vs_stage1": int(vs_stage.sum()),
        "F_to_T_vs_stage1": int(((~stage_s.loc[ids_stage]) & final_s.loc[ids_stage]).sum()),
        "T_to_F_vs_stage1": int((stage_s.loc[ids_stage] & (~final_s.loc[ids_stage])).sum()),
        "final_true_rate": float(final_s.mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    anchor, baseline_summary = generate_groupaware_81201_baseline()
    anchor[TARGET] = to_bool_series(anchor[TARGET])
    anchor_s = anchor.set_index(ID_COL)[TARGET].astype(bool)

    test_raw = pd.read_csv(TEST_CSV)
    test_pred = pd.read_csv(TEST_PREDS)
    feat = build_features(test_raw, test_pred, anchor_s)
    feat = add_plus_features(feat, test_pred)

    stage1, stage1_audit, stage1_summary = apply_rulebook(anchor, feat)
    stage1_s = stage1.set_index(ID_COL)[TARGET].astype(bool)

    audit_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    # The production order mirrors the validated residual pipeline:
    # recall high-purity false negatives, apply narrow vetoes, then continue
    # with smaller recall leaves on the corrected boundary.
    current, audit, summary = apply_modules(stage1, feat, initial_recall_modules())
    if not audit.empty:
        audit["stage"] = "initial_recall"
        audit_frames.append(audit)
    if not summary.empty:
        summary["stage"] = "initial_recall"
        summary_frames.append(summary)

    current_s = current.set_index(ID_COL)[TARGET].astype(bool)
    feat_ultra = feat.copy()
    feat_ultra["stage_pred"] = current_s.reindex(feat.index).astype(bool)
    current, ultra_audit, ultra_summary = apply_ultra_vetoes(
        current,
        feat_ultra,
        {module.name for module in ultra_veto_modules()},
    )

    if not ultra_audit.empty:
        ultra_audit = ultra_audit.copy()
        ultra_audit["direction"] = "TF"
        ultra_audit["family"] = "ultra_strict_feature_veto"
        ultra_audit["stage"] = "ultra_strict_veto"
        ultra_audit = ultra_audit.rename(columns={"rule": "module"})
        audit_frames.append(ultra_audit)
    if not ultra_summary.empty:
        ultra_summary = ultra_summary.copy()
        ultra_summary["direction"] = "TF"
        ultra_summary["family"] = "ultra_strict_feature_veto"
        ultra_summary["stage"] = "ultra_strict_veto"
        ultra_summary = ultra_summary.rename(columns={"rule": "module"})
        summary_frames.append(ultra_summary)

    current, audit, summary = apply_modules(current, feat, tf_boundary_modules())
    if not audit.empty:
        audit["stage"] = "tf_boundary_veto"
        audit_frames.append(audit)
    if not summary.empty:
        summary["stage"] = "tf_boundary_veto"
        summary_frames.append(summary)

    final, audit, summary = apply_modules(current, feat, late_recall_modules())
    if not audit.empty:
        audit["stage"] = "late_recall"
        audit_frames.append(audit)
    if not summary.empty:
        summary["stage"] = "late_recall"
        summary_frames.append(summary)

    final_s = final.set_index(ID_COL)[TARGET].astype(bool)
    submission_path = SUB_DIR / "ml_explainable_residual_stage2_pruned.csv"
    final.to_csv(submission_path, index=False)

    global_summary = pd.DataFrame(
        [
            {
                "anchor_source": "generated_in_this_script",
                "anchor_method": "groupaware_trio_blend",
                "stage1": "cluster_rulebook_second_stage",
                "n_rows": len(final),
                **changed_summary(anchor_s, stage1_s, final_s),
                "submission": str(submission_path),
            }
        ]
    )

    all_audit = pd.concat(audit_frames, ignore_index=True, sort=False)
    all_summary = pd.concat(summary_frames, ignore_index=True, sort=False)

    stage1_audit.to_csv(OUT_DIR / "stage1_cluster_rulebook_audit.csv", index=False)
    stage1_summary.to_csv(OUT_DIR / "stage1_cluster_rulebook_module_summary.csv", index=False)
    all_audit.to_csv(OUT_DIR / "residual_stage2_audit.csv", index=False)
    all_summary.to_csv(OUT_DIR / "residual_stage2_module_summary.csv", index=False)
    baseline_summary.to_csv(OUT_DIR / "baseline_81201_summary.csv", index=False)
    global_summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print("[ml explainable residual stage2]")
    print("\n[baseline 81201 rebuild]")
    print(baseline_summary.to_string(index=False))
    print()
    print(global_summary.to_string(index=False))
    print("\n[module summary]")
    print(all_summary.to_string(index=False))
    print(f"\nSubmission: {submission_path}")


if __name__ == "__main__":
    main()
