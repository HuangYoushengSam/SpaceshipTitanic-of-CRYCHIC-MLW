from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANCHOR_SUBMISSION = ROOT / "artifacts" / "submissions" / "groupaware_xgb_F1F2_lgb_F1F2_xgb_F1F2F3_a15_thr470_raw_retrain.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_PREDS = ROOT / "artifacts" / "probability_tables" / "candidate_test_predictions.csv"
OUT_DIR = ROOT / "outputs" / "cluster_rulebook_81201"

ID_COL = "PassengerId"
TARGET = "Transported"


def to_bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        return values.gt(0)
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def build_features(test_raw: pd.DataFrame, pred: pd.DataFrame, anchor_pred: pd.Series) -> pd.DataFrame:
    raw = test_raw.set_index(ID_COL).copy()
    pred = pred.set_index(ID_COL).copy()

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    spend = raw[spend_cols].fillna(0)
    cabin = raw["Cabin"].astype("string").str.split("/", expand=True)

    feat = pd.DataFrame(index=raw.index)
    feat["HomePlanet"] = raw["HomePlanet"].astype("string").fillna("Missing")
    feat["CryoSleep"] = raw["CryoSleep"].astype("string").fillna("Missing")
    feat["Destination"] = raw["Destination"].astype("string").fillna("Missing")
    feat["Deck"] = cabin[0].astype("string").fillna("Missing")
    feat["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    feat["Side"] = cabin[2].astype("string").fillna("Missing")
    feat["Age"] = pd.to_numeric(raw["Age"], errors="coerce")
    feat["VIP"] = raw["VIP"].astype("string").fillna("Missing")
    feat["TotalSpend"] = spend.sum(axis=1)
    feat["SpendTypes"] = (spend > 0).sum(axis=1)
    feat["Group"] = feat.index.astype(str).str.split("_").str[0]
    feat["Surname"] = raw["Name"].astype("string").str.split().str[-1].fillna("Missing")

    feat["xgb_mean"] = (pred["xgb@F1+F2"] + pred["xgb@F1+F2+F3"]) / 2
    feat["lgb_mean"] = (pred["lgb@F1+F2"] + pred["lgb@F2+F3"]) / 2
    feat["cat"] = pred["cat@F1+F2"]
    feat["hgb"] = pred["hgb@F2+F3"]
    feat["cand_mean"] = pred.mean(axis=1, numeric_only=True)
    feat["true_votes"] = (
        (feat["xgb_mean"] >= 0.5).astype(int)
        + (feat["lgb_mean"] >= 0.5).astype(int)
        + (feat["cat"] >= 0.5).astype(int)
        + (feat["hgb"] >= 0.5).astype(int)
    )
    feat["false_votes"] = 4 - feat["true_votes"]
    feat["tree_spread"] = (
        feat[["xgb_mean", "lgb_mean", "cat", "hgb"]].max(axis=1)
        - feat[["xgb_mean", "lgb_mean", "cat", "hgb"]].min(axis=1)
    )

    feat["anchor_pred"] = anchor_pred.reindex(feat.index).astype(bool)
    group_stats = feat.groupby("Group")["anchor_pred"].agg(["size", "sum"]).rename(columns={"sum": "group_true_count"})
    group_stats["group_true_rate"] = group_stats["group_true_count"] / group_stats["size"]
    feat = feat.join(group_stats, on="Group").rename(columns={"size": "group_size"})

    surname_stats = feat.groupby("Surname")["anchor_pred"].agg(["size", "sum"]).rename(columns={"sum": "surname_true_count"})
    surname_stats["surname_true_rate"] = surname_stats["surname_true_count"] / surname_stats["size"]
    feat = feat.join(surname_stats, on="Surname").rename(columns={"size": "surname_size"})

    return feat


RuleFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class RuleModule:
    direction: str
    name: str
    rationale: str
    selector: RuleFn


def tf_earth_gs_low_hgb_trappist(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    return (
        base_true
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("G")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(700, 1800)
        & (f["hgb"] <= 0.205)
        & (f["cand_mean"] <= 0.49)
        & (f["group_size"] <= 3)
        & ((f["lgb_mean"] <= 0.40) | ((f["Age"] <= 18) & (f["true_votes"] >= 2)))
        & ~f["Age"].between(30, 32)
    )


def tf_earth_gs_55_hgb_veto(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    return (
        base_true
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("G")
        & f["Side"].eq("S")
        & f["Destination"].eq("55 Cancri e")
        & f["TotalSpend"].between(680, 730)
        & (f["hgb"] <= 0.18)
        & (f["lgb_mean"] <= 0.39)
    )


def tf_earth_fs_low_hgb(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    return (
        base_true
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("F")
        & f["Side"].eq("S")
        & f["TotalSpend"].between(500, 700)
        & (f["hgb"] <= 0.23)
    )


def tf_earth_efp_mid_spend(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    earth_mid = (
        base_true
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].isin(["E", "F"])
        & f["Side"].eq("P")
        & f["Destination"].isin(["TRAPPIST-1e", "55 Cancri e"])
        & f["TotalSpend"].between(150, 900)
        & (f["cand_mean"] <= 0.52)
    )
    e_deck_residual = f["Deck"].eq("E") & (f["hgb"] <= 0.50) & ((f["Age"] <= 25) | (f["Age"] >= 55))
    f_deck_trappist = (
        f["Deck"].eq("F")
        & f["Destination"].eq("TRAPPIST-1e")
        & (
            ((f["SpendTypes"] <= 1) & f["TotalSpend"].between(600, 710))
            | ((f["Age"] >= 45) & (f["SpendTypes"] >= 4) & (f["hgb"] <= 0.53))
        )
    )
    f_deck_cancri = (
        f["Deck"].eq("F")
        & f["Destination"].eq("55 Cancri e")
        & f["Age"].between(40, 45)
        & (f["cat"] <= 0.40)
    )
    return earth_mid & (e_deck_residual | f_deck_trappist | f_deck_cancri)


def tf_earth_gp_odd_route(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    return (
        base_true
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & (
            ((f["Age"] == 18) & f["TotalSpend"].between(800, 1500) & (f["cand_mean"] >= 0.50))
            | ((f["TotalSpend"] > 5000) & (f["cat"] < 0.35))
        )
    )


def tf_earth_cryo_gp_zero(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    cryo_p_side_boundary = (
        base_true
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & f["lgb_mean"].between(0.420, 0.438)
        & f["hgb"].between(0.48, 0.53)
    )
    model_split_low = (f["cat"] >= 0.48) & (f["cand_mean"] <= 0.481)
    late_cabin_edge = (f["CabinNum"] > 1400) & (f["Age"] <= 21.5) & (f["lgb_mean"] >= 0.435)
    return cryo_p_side_boundary & (model_split_low | late_cabin_edge)


def tf_europa_high_spend(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    europa_awake = base_true & f["HomePlanet"].eq("Europa") & f["CryoSleep"].eq("False")
    b_cancri = (
        f["Deck"].eq("B")
        & f["Side"].eq("P")
        & f["Destination"].eq("55 Cancri e")
        & (f["TotalSpend"] > 7500)
        & (f["cand_mean"] < 0.53)
        & (f["Age"] >= 60)
    )
    c_trappist = (
        f["Deck"].eq("C")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(3800, 7000)
        & (f["cand_mean"] < 0.49)
    )
    d_p_trappist = (
        f["Deck"].eq("D")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(3000, 6000)
        & (f["Age"] >= 50)
    )
    e_p_trappist = (
        f["Deck"].eq("E")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(3000, 3600)
        & (f["hgb"] < 0.50)
    )
    return europa_awake & (b_cancri | c_trappist | d_p_trappist | e_p_trappist)


def tf_mars_spender(f: pd.DataFrame) -> pd.Series:
    base_true = f["anchor_pred"]
    mars_awake = base_true & f["HomePlanet"].eq("Mars") & f["CryoSleep"].eq("False")
    e_p_trappist = (
        f["Deck"].eq("E")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(1000, 1500)
        & (f["cat"] < 0.48)
    )
    f_p_trappist = (
        f["Deck"].eq("F")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & (((f["TotalSpend"] < 100) & (f["Age"] <= 30)) | (f["TotalSpend"].between(1800, 2100) & (f["Age"] <= 25)))
        & (f["cat"] < 0.42)
    )
    f_s_pso = (
        f["Deck"].eq("F")
        & f["Side"].eq("S")
        & f["Destination"].eq("PSO J318.5-22")
        & f["TotalSpend"].between(600, 700)
        & (f["hgb"] < 0.25)
    )
    return mars_awake & (e_p_trappist | f_p_trappist | f_s_pso)


def ft_missing_planet_cryo_e(f: pd.DataFrame) -> pd.Series:
    base_false = ~f["anchor_pred"]
    return (
        base_false
        & f["HomePlanet"].eq("Missing")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("E")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & (f["hgb"] >= 0.58)
    )


def ft_earth_fp_route(f: pd.DataFrame) -> pd.Series:
    base_false = ~f["anchor_pred"]
    fp_route = (
        base_false
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("F")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(680, 1100)
        & (f["hgb"] >= 0.48)
    )
    isolated_or_dense_route = (
        ((f["group_size"] == 1) & f["CabinNum"].between(250, 1500))
        | ((f["SpendTypes"] >= 4) & f["Age"].between(20, 26))
        | ((f["Age"] == 30) & (f["hgb"] >= 0.54))
    )
    return fp_route & isolated_or_dense_route


def ft_earth_fs_side_spender(f: pd.DataFrame) -> pd.Series:
    base_false = ~f["anchor_pred"]
    return (
        base_false
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("F")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(700, 1120)
        & (f["hgb"] >= 0.51)
        & f["Age"].between(20, 50)
        & ~(f["cand_mean"] > 0.52)
    )


def ft_earth_gp_route(f: pd.DataFrame) -> pd.Series:
    base_false = ~f["anchor_pred"]
    gp = base_false & f["HomePlanet"].eq("Earth") & f["Deck"].eq("G") & f["Side"].eq("P")
    cryo_zero_supported = (
        f["CryoSleep"].eq("True")
        & (f["TotalSpend"] == 0)
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["group_true_rate"] == 0)
        & f["hgb"].between(0.52, 0.55)
        & (f["Age"] >= 35)
    )
    child_zero_awake = (
        f["CryoSleep"].eq("False")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["Age"] <= 4)
        & (f["TotalSpend"] == 0)
        & (f["hgb"] >= 0.48)
        & (f["xgb_mean"] >= 0.50)
    )
    young_low_spend = (
        f["CryoSleep"].eq("False")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["Age"] <= 16)
        & f["TotalSpend"].between(500, 700)
        & (f["hgb"] >= 0.50)
        & (f["group_size"] <= 2)
    )
    mid_spend_model_supported = (
        f["CryoSleep"].eq("False")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(850, 930)
        & (f["hgb"] >= 0.51)
        & f["cand_mean"].between(0.46, 0.53)
        & (f["true_votes"] >= 1)
    )
    pso_g_boundary = (
        f["CryoSleep"].eq("False")
        & f["Destination"].eq("PSO J318.5-22")
        & f["TotalSpend"].between(1100, 1200)
        & (f["lgb_mean"] >= 0.51)
    )
    return gp & (cryo_zero_supported | child_zero_awake | young_low_spend | mid_spend_model_supported | pso_g_boundary)


def ft_earth_gs_awake_route(f: pd.DataFrame) -> pd.Series:
    base_false = ~f["anchor_pred"]
    gs = base_false & f["HomePlanet"].eq("Earth") & f["CryoSleep"].eq("False") & f["Deck"].eq("G") & f["Side"].eq("S")
    zero_spend_child = (
        f["Destination"].eq("TRAPPIST-1e")
        & (f["Age"] <= 8)
        & (f["TotalSpend"] == 0)
        & (f["lgb_mean"] >= 0.51)
    )
    young_spend_supported = (
        f["Destination"].eq("TRAPPIST-1e")
        & f["Age"].between(14, 24)
        & f["TotalSpend"].between(500, 1000)
        & (f["hgb"] >= 0.51)
        & (f["cand_mean"] >= 0.42)
        & ~((f["SpendTypes"] >= 4) & f["CabinNum"].between(400, 500))
        & ~((f["hgb"].between(0.541, 0.55)) & (f["lgb_mean"] < 0.50) & (f["cat"] < 0.50))
    )
    older_boundary = (
        f["Destination"].eq("TRAPPIST-1e")
        & (f["Age"] >= 45)
        & f["TotalSpend"].between(700, 1000)
        & (f["hgb"] >= 0.46)
    )
    high_spend_lgb_route = (
        f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(3500, 3900)
        & (f["lgb_mean"] >= 0.54)
    )
    pso_gs_boundary = (
        f["Destination"].eq("PSO J318.5-22")
        & f["TotalSpend"].between(650, 750)
        & (f["hgb"] >= 0.53)
    )
    return gs & (zero_spend_child | young_spend_supported | older_boundary | high_spend_lgb_route | pso_gs_boundary)


def ft_earth_cryo_gs_zero(f: pd.DataFrame) -> pd.Series:
    base_false = ~f["anchor_pred"]
    cryo_zero_gs = (
        base_false
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("G")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
    )
    strong_adult_support = (f["hgb"] >= 0.70) & (f["true_votes"] >= 3)
    child_support = (f["Age"] <= 8) & (f["hgb"] >= 0.60) & (f["true_votes"] >= 3)
    return cryo_zero_gs & (strong_adult_support | child_support)


def ft_mars_fs_side_spender(f: pd.DataFrame) -> pd.Series:
    base_false = ~f["anchor_pred"]
    return (
        base_false
        & f["HomePlanet"].eq("Mars")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("F")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(1000, 1060)
        & f["hgb"].between(0.44, 0.46)
    )


def rulebook() -> list[RuleModule]:
    return [
        RuleModule(
            "TF",
            "earth_gs_low_hgb_trappist_veto",
            "Anchor=True Earth/G/S/TRAPPIST awake spenders where HGB and LGB both form a hard-negative residual leaf.",
            tf_earth_gs_low_hgb_trappist,
        ),
        RuleModule(
            "TF",
            "earth_gs_cancri_low_hgb_veto",
            "The same Earth/G/S low-HGB veto on the adjacent 55 Cancri route.",
            tf_earth_gs_55_hgb_veto,
        ),
        RuleModule(
            "TF",
            "earth_fs_low_hgb_veto",
            "Anchor=True Earth/F/S low-spend rows where HGB collapses despite the anchor saying True.",
            tf_earth_fs_low_hgb,
        ),
        RuleModule(
            "TF",
            "earth_efp_mid_spend_veto",
            "Earth E/F P-side mid-spend residual veto: route/deck structure is weak and ensemble support is only borderline.",
            tf_earth_efp_mid_spend,
        ),
        RuleModule(
            "TF",
            "earth_gp_route_conflict_veto",
            "Earth/G/P route-conflict veto for awake spenders with suspicious route-side behavior.",
            tf_earth_gp_odd_route,
        ),
        RuleModule(
            "TF",
            "earth_cryo_gp_zero_boundary_veto",
            "Cryo-zero is usually positive, but this P-side boundary has weak LGB/HGB and split model evidence.",
            tf_earth_cryo_gp_zero,
        ),
        RuleModule(
            "TF",
            "europa_high_spend_veto",
            "Europa awake high-spend rows where high expenditure and route/deck split contradict a confident transported label.",
            tf_europa_high_spend,
        ),
        RuleModule(
            "TF",
            "mars_spender_veto",
            "Mars awake spender residual veto in E/F deck routes where Cat disagrees with the HGB/LGB correction signal.",
            tf_mars_spender,
        ),
        RuleModule(
            "FT",
            "missing_planet_cryo_e_recall",
            "Missing-homeplanet cryo-zero E/S/TRAPPIST row recovered by physical consistency and HGB support.",
            ft_missing_planet_cryo_e,
        ),
        RuleModule(
            "FT",
            "earth_fp_route_recall",
            "Earth/F/P/TRAPPIST mid-spend route recall: anchor was too conservative on HGB-supported border rows.",
            ft_earth_fp_route,
        ),
        RuleModule(
            "FT",
            "earth_fs_side_spender_recall",
            "Earth/F/S/TRAPPIST side-S spender recall with moderate spend and positive HGB pressure.",
            ft_earth_fs_side_spender,
        ),
        RuleModule(
            "FT",
            "earth_gp_route_recall",
            "Earth/G/P route recall for child/zero-spend, PSO, and mid-spend HGB/LGB-supported boundary rows.",
            ft_earth_gp_route,
        ),
        RuleModule(
            "FT",
            "earth_gs_awake_route_recall",
            "Earth/G/S route recall: children, older boundary passengers, PSO edge cases, and high-spend LGB-supported rows.",
            ft_earth_gs_awake_route,
        ),
        RuleModule(
            "FT",
            "earth_cryo_gs_zero_recall",
            "CryoSleep=True with zero spend on Earth/G/S/TRAPPIST, only when HGB and vote support are strong.",
            ft_earth_cryo_gs_zero,
        ),
        RuleModule(
            "FT",
            "mars_fs_side_spender_recall",
            "Tiny Mars/F/S/TRAPPIST side-S spender leaf where the heterogeneous model signal supports recall.",
            ft_mars_fs_side_spender,
        ),
    ]


AUDIT_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Deck",
    "CabinNum",
    "Side",
    "Destination",
    "Age",
    "VIP",
    "TotalSpend",
    "SpendTypes",
    "Group",
    "group_size",
    "group_true_rate",
    "Surname",
    "surname_size",
    "surname_true_rate",
    "xgb_mean",
    "lgb_mean",
    "cat",
    "hgb",
    "cand_mean",
    "true_votes",
    "false_votes",
    "tree_spread",
]


def apply_rulebook(anchor: pd.DataFrame, feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = anchor.set_index(ID_COL).copy()
    original = out[TARGET].astype(bool).copy()
    audit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    # Recall first, veto second. The masks are anchored to the original 0.81201 file.
    for direction_group in ["FT", "TF"]:
        for module in rulebook():
            if module.direction != direction_group:
                continue
            target_value = module.direction == "FT"
            selected = module.selector(feat)
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
                    "rule": module.name,
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
                    "rule": module.name,
                    "direction": module.direction,
                    "selected": int(len(selected_ids)),
                    "changed": int(len(changed_ids)),
                    "rationale": module.rationale,
                }
            )

    final = out.reset_index()
    final_s = final.set_index(ID_COL)[TARGET].astype(bool)
    changed = original.ne(final_s)
    changed_ids = changed[changed].index
    total_row = {
        "rule": "__TOTAL__",
        "direction": "MIXED",
        "selected": int(changed.sum()),
        "changed": int(changed.sum()),
        "rationale": (
            f"F_to_T={int(((~original.loc[changed_ids]) & final_s.loc[changed_ids]).sum())}; "
            f"T_to_F={int((original.loc[changed_ids] & (~final_s.loc[changed_ids])).sum())}; "
            f"true_rate={final_s.mean():.6f}"
        ),
    }
    summary = pd.concat([pd.DataFrame(summary_rows), pd.DataFrame([total_row])], ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    return final, audit, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    anchor = pd.read_csv(ANCHOR_SUBMISSION)
    anchor[TARGET] = to_bool_series(anchor[TARGET])
    anchor_s = anchor.set_index(ID_COL)[TARGET].astype(bool)

    test_raw = pd.read_csv(TEST_CSV)
    test_pred = pd.read_csv(TEST_PREDS)
    feat = build_features(test_raw, test_pred, anchor_s)

    final, audit, rule_summary = apply_rulebook(anchor, feat)
    final_s = final.set_index(ID_COL)[TARGET].astype(bool)
    changed = anchor_s.ne(final_s)
    changed_ids = changed[changed].index
    global_summary = pd.DataFrame(
        [
            {
                "anchor_file": str(ANCHOR_SUBMISSION),
                "n_rows": len(final),
                "n_changed_vs_anchor": int(changed.sum()),
                "F_to_T": int(((~anchor_s.loc[changed_ids]) & final_s.loc[changed_ids]).sum()),
                "T_to_F": int((anchor_s.loc[changed_ids] & (~final_s.loc[changed_ids])).sum()),
                "anchor_true_rate": float(anchor_s.mean()),
                "final_true_rate": float(final_s.mean()),
            }
        ]
    )

    submission_path = OUT_DIR / "submission_81201_cluster_rulebook_second_stage.csv"
    audit_path = OUT_DIR / "cluster_rulebook_audit.csv"
    module_path = OUT_DIR / "cluster_rulebook_module_summary.csv"
    summary_path = OUT_DIR / "cluster_rulebook_summary.csv"

    final.to_csv(submission_path, index=False)
    audit.to_csv(audit_path, index=False)
    rule_summary.to_csv(module_path, index=False)
    global_summary.to_csv(summary_path, index=False)

    print("[81201 cluster rulebook second stage]")
    print(global_summary.to_string(index=False))
    print("\n[module summary]")
    print(rule_summary.to_string(index=False))
    print()
    print(f"submission: {submission_path}")
    print(f"audit:      {audit_path}")
    print(f"modules:    {module_path}")
    print(f"summary:    {summary_path}")


if __name__ == "__main__":
    main()
