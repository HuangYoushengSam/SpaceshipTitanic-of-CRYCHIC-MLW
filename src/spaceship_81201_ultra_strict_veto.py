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
from spaceship_81201_rulebook_plus_clean_ml import add_plus_features


OUT_DIR = ROOT / "ultra_strict_veto_81201_outputs"
SUB_DIR = OUT_DIR / "submissions"

RuleFn = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class VetoModule:
    name: str
    rationale: str
    selector: RuleFn


AUDIT_FEATURES = [
    "HomePlanet",
    "CryoSleep",
    "Deck",
    "Side",
    "Destination",
    "Age",
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
    "prob_std5",
]


def stage_true(f: pd.DataFrame) -> pd.Series:
    return f["stage_pred"].astype(bool)


def veto_cryo_gs_young_weak_family(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("G")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & f["Age"].between(5, 22)
        & (f["group_true_rate"] <= 0.01)
        & (f["surname_true_rate"] <= 0.34)
        & (f["hgb"] >= 0.60)
        & (f["cat"] >= 0.50)
        & (f["lgb_mean"] >= 0.54)
    )


def veto_awake_gs_child_family_boundary(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("G")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & f["Age"].between(6, 8)
        & (f["group_size"] >= 6)
        & (f["surname_size"] >= 8)
        & f["hgb"].between(0.52, 0.56)
        & f["lgb_mean"].between(0.50, 0.54)
        & (f["cat"] < 0.45)
    )


def veto_cryo_gs_adult_hgb_override(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("G")
        & f["Side"].eq("S")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & f["Age"].between(35, 45)
        & (f["hgb"] >= 0.70)
        & f["lgb_mean"].between(0.52, 0.55)
        & (f["cat"] < 0.50)
    )


def veto_gp_midspend_weak_family(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & f["TotalSpend"].between(820, 920)
        & f["Age"].between(25, 31)
        & (f["group_true_rate"] <= 0.01)
        & (f["surname_true_rate"] <= 0.01)
        & (f["hgb"] >= 0.58)
        & (f["cat"] >= 0.49)
    )


def veto_cryo_gp_large_surname_child(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("True")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & f["Destination"].eq("TRAPPIST-1e")
        & (f["TotalSpend"] == 0)
        & (f["Age"] <= 5)
        & (f["group_size"] >= 5)
        & (f["group_true_rate"] >= 0.75)
        & (f["surname_size"] >= 50)
        & (f["hgb"].between(0.50, 0.55))
    )


def veto_awake_gp_pso_large_group_zero(f: pd.DataFrame) -> pd.Series:
    return (
        stage_true(f)
        & f["HomePlanet"].eq("Earth")
        & f["CryoSleep"].eq("False")
        & f["Deck"].eq("G")
        & f["Side"].eq("P")
        & f["Destination"].eq("PSO J318.5-22")
        & (f["TotalSpend"] == 0)
        & f["Age"].between(10, 13)
        & (f["group_size"] >= 5)
        & (f["group_true_rate"] >= 0.75)
        & (f["surname_true_rate"] >= 0.50)
        & (f["cand_mean"] >= 0.55)
    )


def modules() -> list[VetoModule]:
    return [
        VetoModule(
            "cryo_gs_young_weak_family",
            "Earth/G/S cryo-zero boundary with young age, no group support, weak surname support, and split model evidence.",
            veto_cryo_gs_young_weak_family,
        ),
        VetoModule(
            "awake_gs_child_family_boundary",
            "Earth/G/S awake child zero-spend boundary where HGB/LGB are only borderline and Cat is negative.",
            veto_awake_gs_child_family_boundary,
        ),
        VetoModule(
            "cryo_gs_adult_hgb_override",
            "Single adult Earth/G/S cryo-zero leaf where HGB is high but Cat stays negative and LGB is only weakly positive.",
            veto_cryo_gs_adult_hgb_override,
        ),
        VetoModule(
            "gp_midspend_weak_family",
            "Earth/G/P awake mid-spend row with no group or surname support despite positive-looking tree probabilities.",
            veto_gp_midspend_weak_family,
        ),
        VetoModule(
            "cryo_gp_large_surname_child",
            "Earth/G/P cryo-zero child inside a large surname family, restricted to the weak-HGB boundary band.",
            veto_cryo_gp_large_surname_child,
        ),
        VetoModule(
            "awake_gp_pso_large_group_zero",
            "Earth/G/P PSO zero-spend child in a large group; a narrow veto for route-family conflict.",
            veto_awake_gp_pso_large_group_zero,
        ),
    ]


PRESETS = [
    ("01_veto_cryo_gs_young", ["cryo_gs_young_weak_family"]),
    ("02_veto_plus_awake_child", ["cryo_gs_young_weak_family", "awake_gs_child_family_boundary"]),
    (
        "03_veto_plus_adult_hgb",
        ["cryo_gs_young_weak_family", "awake_gs_child_family_boundary", "cryo_gs_adult_hgb_override"],
    ),
    (
        "04_veto_plus_gp_midspend",
        [
            "cryo_gs_young_weak_family",
            "awake_gs_child_family_boundary",
            "cryo_gs_adult_hgb_override",
            "gp_midspend_weak_family",
        ],
    ),
    (
        "05_veto_plus_large_surname_child",
        [
            "cryo_gs_young_weak_family",
            "awake_gs_child_family_boundary",
            "cryo_gs_adult_hgb_override",
            "gp_midspend_weak_family",
            "cryo_gp_large_surname_child",
        ],
    ),
    (
        "06_veto_all_ultra_strict",
        [
            "cryo_gs_young_weak_family",
            "awake_gs_child_family_boundary",
            "cryo_gs_adult_hgb_override",
            "gp_midspend_weak_family",
            "cryo_gp_large_surname_child",
            "awake_gp_pso_large_group_zero",
        ],
    ),
    (
        "07_drop_cryo_gs_young",
        [
            "awake_gs_child_family_boundary",
            "cryo_gs_adult_hgb_override",
            "gp_midspend_weak_family",
            "cryo_gp_large_surname_child",
            "awake_gp_pso_large_group_zero",
        ],
    ),
    (
        "08_drop_awake_child",
        [
            "cryo_gs_young_weak_family",
            "cryo_gs_adult_hgb_override",
            "gp_midspend_weak_family",
            "cryo_gp_large_surname_child",
            "awake_gp_pso_large_group_zero",
        ],
    ),
    (
        "09_drop_adult_hgb",
        [
            "cryo_gs_young_weak_family",
            "awake_gs_child_family_boundary",
            "gp_midspend_weak_family",
            "cryo_gp_large_surname_child",
            "awake_gp_pso_large_group_zero",
        ],
    ),
    (
        "10_drop_gp_midspend",
        [
            "cryo_gs_young_weak_family",
            "awake_gs_child_family_boundary",
            "cryo_gs_adult_hgb_override",
            "cryo_gp_large_surname_child",
            "awake_gp_pso_large_group_zero",
        ],
    ),
    (
        "11_drop_large_surname_child",
        [
            "cryo_gs_young_weak_family",
            "awake_gs_child_family_boundary",
            "cryo_gs_adult_hgb_override",
            "gp_midspend_weak_family",
            "awake_gp_pso_large_group_zero",
        ],
    ),
    (
        "12_drop_pso_large_group",
        [
            "cryo_gs_young_weak_family",
            "awake_gs_child_family_boundary",
            "cryo_gs_adult_hgb_override",
            "gp_midspend_weak_family",
            "cryo_gp_large_surname_child",
        ],
    ),
]


def apply_vetoes(
    stage1: pd.DataFrame,
    feat: pd.DataFrame,
    enabled: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = stage1.set_index(ID_COL).copy()
    before = out[TARGET].astype(bool).copy()
    audit_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for module in modules():
        if module.name not in enabled:
            continue
        feat["stage_pred"] = out[TARGET].astype(bool).reindex(feat.index)
        selected = module.selector(feat).fillna(False)
        selected_ids = selected[selected].index
        changed_ids: list[str] = []
        for pid in selected_ids:
            old = bool(out.loc[pid, TARGET])
            out.loc[pid, TARGET] = False
            new = bool(out.loc[pid, TARGET])
            changed = old != new
            if changed:
                changed_ids.append(pid)
            row = {
                ID_COL: pid,
                "rule": module.name,
                "before": old,
                "after": new,
                "changed": changed,
                "rationale": module.rationale,
            }
            row.update(feat.loc[pid, AUDIT_FEATURES].to_dict())
            audit_rows.append(row)
        summary_rows.append(
            {
                "rule": module.name,
                "selected": int(len(selected_ids)),
                "changed": int(len(changed_ids)),
                "rationale": module.rationale,
            }
        )

    final = out.reset_index()
    final_s = final.set_index(ID_COL)[TARGET].astype(bool)
    changed = before.ne(final_s)
    summary_rows.append(
        {
            "rule": "__TOTAL__",
            "selected": int(changed.sum()),
            "changed": int(changed.sum()),
            "rationale": f"ultra_strict_T_to_F={int(changed.sum())}; true_rate={final_s.mean():.6f}",
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
    audit_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []

    for order, (preset_name, enabled_names) in enumerate(PRESETS, start=1):
        final, audit, summary = apply_vetoes(stage1, feat.copy(), set(enabled_names))
        final_s = final.set_index(ID_COL)[TARGET].astype(bool)
        changed_vs_stage1 = stage1_s.ne(final_s)
        changed_vs_anchor = anchor_s.ne(final_s)

        filename = f"{order:02d}_{preset_name}.csv"
        path = SUB_DIR / filename
        final.to_csv(path, index=False)

        audit.insert(0, "preset", preset_name)
        summary.insert(0, "preset", preset_name)
        audit_frames.append(audit)
        summary_frames.append(summary)

        manifest_rows.append(
            {
                "file": filename,
                "preset": preset_name,
                "path": str(path),
                "ultra_T_to_F_vs_82043": int(changed_vs_stage1.sum()),
                "n_changed_vs_81201": int(changed_vs_anchor.sum()),
                "true_rate": float(final_s.mean()),
            }
        )

    stage1_audit.to_csv(OUT_DIR / "stage1_82043_audit.csv", index=False)
    stage1_summary.to_csv(OUT_DIR / "stage1_82043_module_summary.csv", index=False)
    pd.concat(audit_frames, ignore_index=True).to_csv(OUT_DIR / "ultra_strict_audit.csv", index=False)
    pd.concat(summary_frames, ignore_index=True).to_csv(OUT_DIR / "ultra_strict_module_summary.csv", index=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "submission_manifest.csv", index=False)

    print("[81201 ultra strict veto]")
    print(manifest.to_string(index=False))
    print()
    print(f"manifest: {OUT_DIR / 'submission_manifest.csv'}")
    print(f"submissions: {SUB_DIR}")


if __name__ == "__main__":
    main()
