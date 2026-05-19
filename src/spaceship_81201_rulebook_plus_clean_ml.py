from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from spaceship_81201_cluster_rulebook_second_stage import (
    ANCHOR_SUBMISSION,
    ID_COL,
    ROOT,
    TARGET,
    TEST_CSV,
    TEST_PREDS,
    apply_rulebook,
    build_features,
    to_bool_series,
)


OUT_DIR = ROOT / "outputs" / "rulebook_plus_clean_ml"
SUB_DIR = OUT_DIR / "submissions"
BASE_PREDS = ROOT / "artifacts" / "probability_tables" / "base_test_predictions.csv"


RuleFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class PlusModule:
    direction: str
    name: str
    rationale: str
    selector: RuleFn
    presets: tuple[str, ...]


AUDIT_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Deck",
    "CabinNum",
    "Side",
    "Destination",
    "Age",
    "TotalSpend",
    "SpendTypes",
    "group_size",
    "group_true_rate",
    "surname_size",
    "surname_true_rate",
    "xgb_mean",
    "lgb_mean",
    "cat",
    "hgb",
    "et",
    "cand_mean",
    "anchor_prob",
    "expert_vote_count",
    "expert_disagreement",
    "prob_std5",
    "prob_range5",
]


def add_plus_features(feat: pd.DataFrame, test_pred: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    pred = test_pred.set_index(ID_COL)
    base_pred = pd.read_csv(BASE_PREDS).set_index(ID_COL)

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


def stage_false(f: pd.DataFrame) -> pd.Series:
    return ~f["stage_pred"].astype(bool)


def stage_true(f: pd.DataFrame) -> pd.Series:
    return f["stage_pred"].astype(bool)


def ft_cryo_zero_conservative_tree_recall(f: pd.DataFrame) -> pd.Series:
    return (
        stage_false(f)
        & f["CryoSleep"].eq("True")
        & (f["TotalSpend"] == 0)
        & f["Destination"].eq("TRAPPIST-1e")
        & (
            ((f["cat"] < 0.5) & (f["xgb_mean"] < 0.5))
            | ((f["cat"] < 0.5) & (f["et"] < 0.5))
            | ((f["xgb_mean"] < 0.5) & (f["et"] < 0.5))
        )
    )


def ft_mars_cryo_zero_recall(f: pd.DataFrame) -> pd.Series:
    return (
        stage_false(f)
        & f["HomePlanet"].eq("Mars")
        & f["CryoSleep"].eq("True")
        & (f["TotalSpend"] == 0)
        & f["Destination"].eq("TRAPPIST-1e")
        & f["Deck"].isin(["D", "E", "F"])
        & ((f["hgb"] >= 0.12) | (f["lgb_mean"] >= 0.12) | (f["cat"] >= 0.12))
    )


def ft_cryo_zero_family_supported_recall(f: pd.DataFrame) -> pd.Series:
    family_support = (
        (f["group_size"] >= 2)
        | (f["surname_size"] >= 8)
        | (f["group_true_rate"] >= 0.50)
        | (f["surname_true_rate"] >= 0.50)
    )
    return (
        stage_false(f)
        & f["CryoSleep"].eq("True")
        & (f["TotalSpend"] == 0)
        & f["Destination"].eq("TRAPPIST-1e")
        & family_support
        & ((f["hgb"] >= 0.35) | (f["lgb_mean"] >= 0.35) | (f["expert_disagreement"] >= 1))
        & ~(
            f["HomePlanet"].eq("Europa")
            & f["Deck"].isin(["B", "C"])
            & (f["prob_range5"] <= 0.08)
        )
    )


def ft_hgb_disagreement_trappist_recall(f: pd.DataFrame) -> pd.Series:
    return (
        stage_false(f)
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["hgb"] >= 0.50)
        & (f["prob_std5"] >= 0.07)
        & (f["TotalSpend"] <= 1800)
        & ((f["cat"] < 0.5) | (f["xgb_mean"] < 0.5) | (f["et"] < 0.5))
    )


def ft_earth_efg_awake_route_recall(f: pd.DataFrame) -> pd.Series:
    efg_route = (
        stage_false(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["Deck"].isin(["E", "F", "G"])
        & f["TotalSpend"].between(450, 1200)
        & (f["hgb"] >= 0.45)
        & (f["lgb_mean"] >= 0.34)
        & (f["prob_std5"] >= 0.035)
    )
    veto_noise = (
        (f["SpendTypes"] >= 4)
        & (f["lgb_mean"] < 0.40)
        & (f["cat"] < 0.40)
    )
    return efg_route & ~veto_noise


def ft_earth_pso_side_route_recall(f: pd.DataFrame) -> pd.Series:
    return (
        stage_false(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Destination"].eq("PSO J318.5-22")
        & f["Deck"].isin(["F", "G"])
        & f["TotalSpend"].between(450, 1450)
        & ((f["hgb"] >= 0.38) | (f["lgb_mean"] >= 0.45))
    )


def ft_cryo_zero_pruned_recall(f: pd.DataFrame) -> pd.Series:
    raw_recall = (
        ft_cryo_zero_conservative_tree_recall(f)
        | ft_mars_cryo_zero_recall(f)
        | ft_cryo_zero_family_supported_recall(f)
    )
    large_group_mid_age_noise = (
        f["HomePlanet"].eq("Earth")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & (f["group_size"] >= 7)
        & f["Age"].between(3, 32)
    )
    weak_family_young_adult_noise = (
        f["HomePlanet"].eq("Earth")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & f["Age"].between(8, 32)
        & (f["group_true_rate"] < 0.34)
        & (f["surname_true_rate"] < 0.34)
    )
    return raw_recall & ~(large_group_mid_age_noise | weak_family_young_adult_noise)


def ft_missing_destination_disagreement_recall(f: pd.DataFrame) -> pd.Series:
    return (
        stage_false(f)
        & f["Destination"].eq("Missing")
        & (f["prob_std5"] >= 0.08)
        & ((f["group_size"] >= 2) | (f["expert_disagreement"] >= 1))
        & ((f["hgb"] >= 0.45) | (f["lgb_mean"] >= 0.45) | (f["cat"] < 0.50))
    )


def tf_gdeck_lgb_et_veto(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["Deck"].eq("G")
        & f["CryoSleep"].eq("False")
        & (f["TotalSpend"] > 0)
        & (f["lgb_mean"] <= 0.40)
        & (f["hgb"] <= 0.45)
        & (f["et"] >= 0.50)
        & (f["expert_disagreement"] >= 1)
    )


def tf_earth_f_trappist_low_model_veto(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["Deck"].isin(["E", "F", "G"])
        & (f["TotalSpend"] > 250)
        & (f["lgb_mean"] <= 0.42)
        & (f["hgb"] <= 0.43)
        & (f["cat"] <= 0.62)
        & (f["prob_std5"] >= 0.04)
    )


def plus_modules() -> list[PlusModule]:
    return [
        PlusModule(
            "FT",
            "cryo_zero_pruned_recall",
            "High-recall cryo-zero correction with an Earth/G/P weak-family tail pruned away by group and surname consistency.",
            ft_cryo_zero_pruned_recall,
            ("pruned", "pruned_plus"),
        ),
        PlusModule(
            "FT",
            "missing_destination_disagreement_recall",
            "Rows with missing destination and high expert disagreement are recalled when route uncertainty makes the anchor overly conservative.",
            ft_missing_destination_disagreement_recall,
            ("pruned_plus",),
        ),
        PlusModule(
            "FT",
            "cryo_zero_conservative_tree_recall",
            "CryoSleep=True and zero total spend is a physically coherent positive pattern; this recalls rows where conservative tree experts under-call the class.",
            ft_cryo_zero_conservative_tree_recall,
            ("strict", "balanced", "wide", "max"),
        ),
        PlusModule(
            "FT",
            "mars_cryo_zero_recall",
            "Mars cryo-zero passengers form a separate high-purity residual pocket, especially on E/F/D decks to TRAPPIST.",
            ft_mars_cryo_zero_recall,
            ("strict", "balanced", "wide", "max"),
        ),
        PlusModule(
            "TF",
            "gdeck_lgb_et_veto",
            "Earth/G awake spenders with LGB and HGB both low are a hard-negative pocket even when one high-variance expert remains positive.",
            tf_gdeck_lgb_et_veto,
            ("pruned", "pruned_plus", "strict", "balanced", "wide", "max"),
        ),
        PlusModule(
            "FT",
            "cryo_zero_family_supported_recall",
            "Cryo-zero rows are expanded when group or surname structure supplies consistency support.",
            ft_cryo_zero_family_supported_recall,
            ("balanced", "wide", "max"),
        ),
        PlusModule(
            "FT",
            "hgb_disagreement_trappist_recall",
            "HGB is used as a heterogeneous boundary expert: positive HGB plus large expert disagreement recovers false-negative TRAPPIST boundary rows.",
            ft_hgb_disagreement_trappist_recall,
            ("wide", "max"),
        ),
        PlusModule(
            "FT",
            "earth_efg_awake_route_recall",
            "Earth E/F/G awake TRAPPIST mid-spend rows are recalled only when HGB/LGB support and disagreement both exist.",
            ft_earth_efg_awake_route_recall,
            ("wide", "max"),
        ),
        PlusModule(
            "FT",
            "earth_pso_side_route_recall",
            "Earth F/G PSO side-route rows are recalled with moderate spend and HGB/LGB boundary support.",
            ft_earth_pso_side_route_recall,
            ("max",),
        ),
        PlusModule(
            "TF",
            "earth_efg_trappist_low_model_veto",
            "A final low-model veto prevents the expanded recall stage from keeping weak Earth E/F/G awake spenders as True.",
            tf_earth_f_trappist_low_model_veto,
            ("max",),
        ),
    ]


def apply_plus_modules(
    base_submission: pd.DataFrame,
    feat: pd.DataFrame,
    preset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = base_submission.set_index(ID_COL).copy()
    before_all = out[TARGET].astype(bool).copy()
    audit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for direction_group in ["FT", "TF"]:
        for module in plus_modules():
            if module.direction != direction_group or preset not in module.presets:
                continue
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
                    "preset": preset,
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
                    "preset": preset,
                    "rule": module.name,
                    "direction": module.direction,
                    "selected": int(len(selected_ids)),
                    "changed": int(len(changed_ids)),
                    "rationale": module.rationale,
                }
            )

    final = out.reset_index()
    final_s = final.set_index(ID_COL)[TARGET].astype(bool)
    changed = before_all.ne(final_s)
    changed_ids = changed[changed].index
    summary_rows.append(
        {
            "preset": preset,
            "rule": "__PLUS_TOTAL__",
            "direction": "MIXED",
            "selected": int(changed.sum()),
            "changed": int(changed.sum()),
            "rationale": (
                f"plus_F_to_T={int(((~before_all.loc[changed_ids]) & final_s.loc[changed_ids]).sum())}; "
                f"plus_T_to_F={int((before_all.loc[changed_ids] & (~final_s.loc[changed_ids])).sum())}; "
                f"true_rate={final_s.mean():.6f}"
            ),
        }
    )
    return final, pd.DataFrame(audit_rows), pd.DataFrame(summary_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    anchor = pd.read_csv(ANCHOR_SUBMISSION)
    anchor[TARGET] = to_bool_series(anchor[TARGET])
    anchor_s = anchor.set_index(ID_COL)[TARGET].astype(bool)

    test_raw = pd.read_csv(TEST_CSV)
    test_pred = pd.read_csv(TEST_PREDS)
    feat = build_features(test_raw, test_pred, anchor_s)
    feat = add_plus_features(feat, test_pred)

    stage1, stage1_audit, stage1_summary = apply_rulebook(anchor, feat)
    stage1_s = stage1.set_index(ID_COL)[TARGET].astype(bool)

    manifest_rows: list[dict[str, object]] = []
    all_summaries: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []

    for order, preset in enumerate(["strict", "balanced", "wide", "max", "pruned", "pruned_plus"], start=1):
        final, audit, summary = apply_plus_modules(stage1, feat.copy(), preset)
        final_s = final.set_index(ID_COL)[TARGET].astype(bool)
        changed_vs_anchor = anchor_s.ne(final_s)
        changed_vs_stage1 = stage1_s.ne(final_s)
        ids_anchor = changed_vs_anchor[changed_vs_anchor].index
        ids_stage1 = changed_vs_stage1[changed_vs_stage1].index

        name = f"{order:02d}_81201_rulebook_plus_{preset}.csv"
        path = SUB_DIR / name
        final.to_csv(path, index=False)

        manifest_rows.append(
            {
                "file": name,
                "preset": preset,
                "path": str(path),
                "n_changed_vs_81201": int(changed_vs_anchor.sum()),
                "F_to_T_vs_81201": int(((~anchor_s.loc[ids_anchor]) & final_s.loc[ids_anchor]).sum()),
                "T_to_F_vs_81201": int((anchor_s.loc[ids_anchor] & (~final_s.loc[ids_anchor])).sum()),
                "plus_changed_vs_82043": int(changed_vs_stage1.sum()),
                "plus_F_to_T_vs_82043": int(((~stage1_s.loc[ids_stage1]) & final_s.loc[ids_stage1]).sum()),
                "plus_T_to_F_vs_82043": int((stage1_s.loc[ids_stage1] & (~final_s.loc[ids_stage1])).sum()),
                "true_rate": float(final_s.mean()),
            }
        )
        all_summaries.append(summary)
        all_audits.append(audit)

    stage1_audit.to_csv(OUT_DIR / "stage1_82043_audit.csv", index=False)
    stage1_summary.to_csv(OUT_DIR / "stage1_82043_module_summary.csv", index=False)
    pd.concat(all_summaries, ignore_index=True).to_csv(OUT_DIR / "plus_module_summary.csv", index=False)
    pd.concat(all_audits, ignore_index=True).to_csv(OUT_DIR / "plus_audit.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "submission_manifest.csv", index=False)

    print("[81201 rulebook plus clean ML]")
    print(manifest.to_string(index=False))
    print()
    print(f"manifest: {OUT_DIR / 'submission_manifest.csv'}")
    print(f"submissions: {SUB_DIR}")


if __name__ == "__main__":
    main()
