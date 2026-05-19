from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder


ID_COL = "PassengerId"
TARGET_COL = "Transported"
SEED = 42
SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)


def mode_or_nan(s: pd.Series) -> Any:
    ss = s.dropna()
    if ss.empty:
        return np.nan
    m = ss.mode(dropna=True)
    return m.iloc[0] if len(m) else ss.iloc[0]


def normalize_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    out = s.copy()
    if out.dtype == object or str(out.dtype).startswith("string"):
        low = out.astype("string").str.lower()
        out = low.map({"true": True, "false": False, "1": True, "0": False})
    return out


def safe_group_map(df: pd.DataFrame, keys: List[str], value: str, agg: str) -> Dict[Tuple[Any, ...], float]:
    work = df[keys + [value]].copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna(subset=[value])
    if work.empty:
        return {}
    if agg == "median":
        grouped = work.groupby(keys, dropna=False)[value].median()
    elif agg == "mean":
        grouped = work.groupby(keys, dropna=False)[value].mean()
    else:
        raise ValueError(agg)
    return {tuple(k if isinstance(k, tuple) else (k,)): float(v) for k, v in grouped.items()}


def learn_mode_map(
    df: pd.DataFrame,
    key_col: str,
    value_col: str,
    min_count: int = 2,
    min_conf: float = 0.80,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    mapping: Dict[str, Any] = {}
    confs: Dict[str, float] = {}
    tmp = df[[key_col, value_col]].dropna().copy()
    if tmp.empty:
        return mapping, confs
    tmp[key_col] = tmp[key_col].astype(str)
    for key, sub in tmp.groupby(key_col, dropna=False):
        vc = sub[value_col].value_counts(dropna=True, normalize=False)
        n = int(vc.sum())
        if n < min_count or vc.empty:
            continue
        conf = float(vc.iloc[0] / n)
        if conf >= min_conf:
            mapping[str(key)] = vc.index[0]
            confs[str(key)] = conf
    return mapping, confs


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 3 or np.nanstd(x) < 1e-9:
        return 0.0, float(np.nanmedian(y)), 0.0
    slope, intercept = np.polyfit(x.astype(float), y.astype(float), deg=1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot <= 1e-9 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(slope), float(intercept), float(r2)


@dataclass
class CabinLine:
    slope: float
    intercept: float
    r2: float
    n: int
    lo: float
    hi: float

    def predict(self, group_num: np.ndarray) -> np.ndarray:
        pred = self.slope * group_num.astype(float) + self.intercept
        return np.clip(pred, self.lo, self.hi)


class FoldSafeFeatureBuilderV2:
    """
    Fold-safe feature builder for local direction checks.

    fit() learns only from the training portion of a fold. transform() may use
    feature-only structure inside the dataframe being transformed, such as group
    consistency within validation/test, but it never mixes validation rows into
    training-fold statistics.
    """

    def __init__(self, surname_min_count: int = 2, surname_min_conf: float = 0.80):
        self.surname_min_count = surname_min_count
        self.surname_min_conf = surname_min_conf

        self.global_modes: Dict[str, Any] = {}
        self.global_medians: Dict[str, float] = {}
        self.surname_maps: Dict[str, Dict[str, Any]] = {}
        self.surname_confs: Dict[str, Dict[str, float]] = {}
        self.age_map_lvl1: Dict[Tuple[Any, ...], float] = {}
        self.age_map_lvl2: Dict[Tuple[Any, ...], float] = {}
        self.spend_maps: Dict[str, Dict[Tuple[Any, ...], float]] = {}
        self.cabin_median_maps: Dict[str, Dict[Tuple[Any, ...], float]] = {}
        self.cabin_lines: Dict[Tuple[str, str], CabinLine] = {}
        self.freq_maps: Dict[str, Dict[Any, int]] = {}

        self.raw_missing_cols = [
            "HomePlanet", "CryoSleep", "Cabin", "Destination", "Age", "VIP",
            "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Name",
        ]

        self.cat_cols = [
            "HomePlanet", "CryoSleep", "Destination", "VIP",
            "GroupId", "Deck", "Side", "DeckSide", "Surname",
            "AgeBin", "CabinNumBin", "CabinRegion", "DeckRegion",
            "HomePlanet_Destination", "HomePlanet_Deck", "Destination_Deck",
            "CryoSleep_NoSpend", "VIP_Luxury", "SurnameHomePlanet",
            "SurnameDeckSide",
        ]

        self.num_cols = [
            "GroupNum", "GroupMemberNo", "GroupSize", "IsSolo",
            "CabinNum", "CabinNumWasMissing", "CabinNumFilledByGroup",
            "CabinNumFilledByLine", "CabinNumLinePred", "CabinNumLineResidual",
            "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
            "SpendMissingCountRaw", "TotalSpend", "LuxurySpend", "BasicSpend",
            "NoSpend", "AnySpend", "LogTotalSpend", "SpendNonZeroCount",
            "MeanSpendPerActive", "SpendPerGroupMember", "RowMissingCount",
            "SurnameFreq", "DeckFreq", "HomePlanetFreq", "DestinationFreq",
            "DeckSideFreq", "CabinRegionFreq", "SurnameHomePlanetConf",
            "SurnameDeckConf", "SurnameSideConf", "Group_TotalSpend",
            "Group_MeanSpend", "Group_MaxSpend", "Group_StdSpend",
            "Group_ZeroSpendRate", "Group_CryoRate", "Group_AgeMean",
            "Group_AgeStd", "Deck_CabinNum_Mean", "Deck_CabinNum_Std",
            "AgeMinusGroupMean", "DeckNumDelta", "CryoSpendConflict",
            "Cryo_x_NoSpend", "VIP_x_LogTotalSpend", "GroupSize_x_LogTotalSpend",
        ] + [f"{c}_missing" for c in self.raw_missing_cols]

        self.feature_cols = self.cat_cols + self.num_cols

    def _base_parse(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()

        pid = x[ID_COL].astype("string").str.split("_", expand=True)
        x["GroupId"] = pid[0].fillna("UNK").astype(str)
        x["GroupNum"] = pd.to_numeric(pid[0], errors="coerce").astype(float)
        x["GroupMemberNo"] = pd.to_numeric(pid[1], errors="coerce").astype(float)
        x["GroupSize"] = x.groupby("GroupId")[ID_COL].transform("size").astype(float)
        x["IsSolo"] = (x["GroupSize"] == 1).astype(float)

        cabin = x["Cabin"].astype("string").str.split("/", expand=True)
        x["Deck"] = cabin[0].replace("", np.nan).astype("string")
        x["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce").astype(float)
        x["Side"] = cabin[2].replace("", np.nan).astype("string")
        x["DeckSide"] = (x["Deck"].fillna("MissingDeck").astype(str) + "_" + x["Side"].fillna("MissingSide").astype(str))

        x["Surname"] = x["Name"].astype("string").str.split().str[-1]
        x.loc[x["Name"].isna(), "Surname"] = pd.NA

        for c in self.raw_missing_cols:
            x[f"{c}_missing"] = x[c].isna().astype(float)
        x["RowMissingCount"] = x[self.raw_missing_cols].isna().sum(axis=1).astype(float)

        for c in SPEND_COLS + ["Age"]:
            x[c] = pd.to_numeric(x[c], errors="coerce").astype(float)

        x["SpendMissingCountRaw"] = x[SPEND_COLS].isna().sum(axis=1).astype(float)
        x["CabinNumWasMissing"] = x["CabinNum"].isna().astype(float)
        x["CabinNumFilledByGroup"] = 0.0
        x["CabinNumFilledByLine"] = 0.0
        x["CabinNumLinePred"] = np.nan

        x["CryoSleep"] = normalize_bool_series(x["CryoSleep"])
        x["VIP"] = normalize_bool_series(x["VIP"])
        return x

    def _apply_within_dataset_repairs(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()

        # CryoSleep/spend consistency before group fills.
        cryo_true = x["CryoSleep"] == True
        x.loc[cryo_true, SPEND_COLS] = x.loc[cryo_true, SPEND_COLS].fillna(0.0)

        spend_zero = x[SPEND_COLS].fillna(0.0).sum(axis=1) == 0
        spend_positive = x[SPEND_COLS].fillna(0.0).sum(axis=1) > 0
        missing_cryo = x["CryoSleep"].isna()
        x.loc[missing_cryo & spend_positive, "CryoSleep"] = False
        x.loc[missing_cryo & spend_zero, "CryoSleep"] = True
        x.loc[(x["CryoSleep"] == True) & spend_positive, "CryoSleep"] = False

        # Validation/test-internal feature-only consistency. This is allowed for
        # OOF because it does not use labels or training-fold rows.
        for c in ["HomePlanet", "Destination", "VIP", "CryoSleep", "Surname", "Deck", "Side"]:
            grp_mode = x.groupby("GroupId", dropna=False)[c].transform(mode_or_nan)
            x[c] = x[c].fillna(grp_mode)

        group_cabin = x.groupby("GroupId", dropna=False)["CabinNum"].transform("median")
        cabin_missing = x["CabinNum"].isna() & group_cabin.notna()
        x.loc[cabin_missing, "CabinNum"] = group_cabin.loc[cabin_missing]
        x.loc[cabin_missing, "CabinNumFilledByGroup"] = 1.0

        group_age = x.groupby("GroupId", dropna=False)["Age"].transform("median")
        x["Age"] = x["Age"].fillna(group_age)

        x["DeckSide"] = (x["Deck"].fillna("MissingDeck").astype(str) + "_" + x["Side"].fillna("MissingSide").astype(str))
        return x

    def _apply_surname_maps(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        surname = x["Surname"].astype("string").astype(str)
        for col in ["HomePlanet", "Destination", "Deck", "Side"]:
            mapping = self.surname_maps.get(col, {})
            if not mapping:
                continue
            mask = x[col].isna()
            fill = surname.map(mapping)
            x.loc[mask & fill.notna(), col] = fill.loc[mask & fill.notna()]

        hp_map = self.surname_maps.get("HomePlanet", {})
        deck_map = self.surname_maps.get("Deck", {})
        side_map = self.surname_maps.get("Side", {})
        x["SurnameHomePlanet"] = surname.map(hp_map).fillna("UnknownSurnameHP").astype(str)
        deck_part = surname.map(deck_map).fillna("UnknownDeck").astype(str)
        side_part = surname.map(side_map).fillna("UnknownSide").astype(str)
        x["SurnameDeckSide"] = deck_part + "_" + side_part
        x["SurnameHomePlanetConf"] = surname.map(self.surname_confs.get("HomePlanet", {})).fillna(0.0).astype(float)
        x["SurnameDeckConf"] = surname.map(self.surname_confs.get("Deck", {})).fillna(0.0).astype(float)
        x["SurnameSideConf"] = surname.map(self.surname_confs.get("Side", {})).fillna(0.0).astype(float)
        return x

    def _apply_fixed_rules(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        hp_missing = x["HomePlanet"].isna()
        x.loc[hp_missing & x["Deck"].isin(["A", "B", "C", "T"]), "HomePlanet"] = "Europa"
        x.loc[x["HomePlanet"].isna() & x["Deck"].isin(["G"]), "HomePlanet"] = "Earth"
        x.loc[x["VIP"].isna() & (x["HomePlanet"] == "Earth"), "VIP"] = False
        return x

    def _apply_global_fills(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for c in ["HomePlanet", "Destination", "Deck", "Side", "Surname"]:
            x[c] = x[c].fillna(self.global_modes.get(c, f"Missing{c}"))
        x["VIP"] = x["VIP"].fillna(self.global_modes.get("VIP", False))
        x["CryoSleep"] = x["CryoSleep"].fillna(self.global_modes.get("CryoSleep", False))
        x["DeckSide"] = (x["Deck"].fillna("MissingDeck").astype(str) + "_" + x["Side"].fillna("MissingSide").astype(str))
        return x

    def _learn_cabin_lines(self, x: pd.DataFrame) -> None:
        self.cabin_lines = {}
        known = x.dropna(subset=["CabinNum", "GroupNum", "Deck", "Side"]).copy()
        for (deck, side), sub in known.groupby(["Deck", "Side"], dropna=False):
            if len(sub) < 25:
                continue
            slope, intercept, r2 = fit_line(sub["GroupNum"].to_numpy(), sub["CabinNum"].to_numpy())
            if r2 < 0.60:
                continue
            self.cabin_lines[(str(deck), str(side))] = CabinLine(
                slope=slope,
                intercept=intercept,
                r2=r2,
                n=int(len(sub)),
                lo=float(sub["CabinNum"].min()),
                hi=float(sub["CabinNum"].max()),
            )

    def _apply_cabin_fills(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        pred = np.full(len(x), np.nan, dtype=float)
        for key, line in self.cabin_lines.items():
            deck, side = key
            mask = (
                x["CabinNum"].isna()
                & (x["Deck"].astype(str) == deck)
                & (x["Side"].astype(str) == side)
                & x["GroupNum"].notna()
            )
            if mask.any():
                pred[mask.to_numpy()] = line.predict(x.loc[mask, "GroupNum"].to_numpy())
        x["CabinNumLinePred"] = pred

        use_line = x["CabinNum"].isna() & pd.Series(pred, index=x.index).notna()
        x.loc[use_line, "CabinNum"] = pd.Series(pred, index=x.index).loc[use_line]
        x.loc[use_line, "CabinNumFilledByLine"] = 1.0

        if x["CabinNum"].isna().any():
            for keys, map_name in [
                (["Deck", "Side"], "deck_side"),
                (["Deck"], "deck"),
            ]:
                mapping = self.cabin_median_maps.get(map_name, {})
                mask = x["CabinNum"].isna()
                if not mask.any():
                    break
                vals = x.loc[mask].apply(
                    lambda r: mapping.get(tuple(r[k] for k in keys), np.nan),
                    axis=1,
                )
                good = vals.notna()
                x.loc[vals.index[good], "CabinNum"] = vals.loc[good].astype(float)

        x["CabinNum"] = x["CabinNum"].fillna(self.global_medians.get("CabinNum", 0.0)).astype(float)
        raw_pred = x["CabinNumLinePred"].fillna(x["CabinNum"])
        x["CabinNumLineResidual"] = (x["CabinNum"] - raw_pred).astype(float)
        return x

    def _apply_age_spend_fills(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()

        def lookup(row: pd.Series, mapping: Dict[Tuple[Any, ...], float], keys: List[str], fallback: float) -> float:
            val = mapping.get(tuple(row[k] for k in keys), np.nan)
            return float(fallback if pd.isna(val) else val)

        need_age = x["Age"].isna()
        if need_age.any():
            x.loc[need_age, "Age"] = x.loc[need_age].apply(
                lambda r: lookup(r, self.age_map_lvl1, ["HomePlanet", "Deck", "VIP", "CryoSleep"], np.nan),
                axis=1,
            )
        need_age = x["Age"].isna()
        if need_age.any():
            x.loc[need_age, "Age"] = x.loc[need_age].apply(
                lambda r: lookup(r, self.age_map_lvl2, ["HomePlanet", "Deck"], self.global_medians["Age"]),
                axis=1,
            )
        x["Age"] = x["Age"].fillna(self.global_medians["Age"]).astype(float)

        cryo_true = x["CryoSleep"] == True
        x.loc[cryo_true, SPEND_COLS] = x.loc[cryo_true, SPEND_COLS].fillna(0.0)
        for c in SPEND_COLS:
            mask = x[c].isna()
            if mask.any():
                x.loc[mask, c] = x.loc[mask].apply(
                    lambda r, col=c: lookup(r, self.spend_maps[col], ["HomePlanet", "CryoSleep", "Deck"], 0.0),
                    axis=1,
                )
            x[c] = x[c].fillna(0.0).astype(float)
        return x

    def _add_final_features(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        x["DeckSide"] = (x["Deck"].astype(str) + "_" + x["Side"].astype(str))

        x["TotalSpend"] = x[SPEND_COLS].sum(axis=1).astype(float)
        x["LuxurySpend"] = x[["Spa", "VRDeck"]].sum(axis=1).astype(float)
        x["BasicSpend"] = x[["RoomService", "FoodCourt", "ShoppingMall"]].sum(axis=1).astype(float)
        x["NoSpend"] = (x["TotalSpend"] == 0).astype(float)
        x["AnySpend"] = (x["TotalSpend"] > 0).astype(float)
        x["LogTotalSpend"] = np.log1p(x["TotalSpend"]).astype(float)
        x["SpendNonZeroCount"] = (x[SPEND_COLS] > 0).sum(axis=1).astype(float)
        x["MeanSpendPerActive"] = x["TotalSpend"] / np.maximum(x["SpendNonZeroCount"], 1.0)
        x["SpendPerGroupMember"] = x["TotalSpend"] / x["GroupSize"].clip(lower=1.0)
        x["CryoSpendConflict"] = ((x["CryoSleep"] == True) & (x["TotalSpend"] > 0)).astype(float)

        gb = x.groupby("GroupId", dropna=False)
        x["Group_TotalSpend"] = gb["TotalSpend"].transform("sum").astype(float)
        x["Group_MeanSpend"] = gb["TotalSpend"].transform("mean").astype(float)
        x["Group_MaxSpend"] = gb["TotalSpend"].transform("max").astype(float)
        x["Group_StdSpend"] = gb["TotalSpend"].transform("std").fillna(0.0).astype(float)
        x["Group_ZeroSpendRate"] = gb["NoSpend"].transform("mean").astype(float)
        x["Group_CryoRate"] = gb["CryoSleep"].transform(lambda s: pd.Series(s).astype(bool).mean()).astype(float)
        x["Group_AgeMean"] = gb["Age"].transform("mean").astype(float)
        x["Group_AgeStd"] = gb["Age"].transform("std").fillna(0.0).astype(float)

        deck_gb = x.groupby("Deck", dropna=False)
        x["Deck_CabinNum_Mean"] = deck_gb["CabinNum"].transform("mean").astype(float)
        x["Deck_CabinNum_Std"] = deck_gb["CabinNum"].transform("std").fillna(0.0).astype(float)
        x["AgeMinusGroupMean"] = (x["Age"] - x["Group_AgeMean"]).astype(float)
        x["DeckNumDelta"] = (x["CabinNum"] - x["Deck_CabinNum_Mean"]).astype(float)

        x["Cryo_x_NoSpend"] = (x["CryoSleep"].astype(bool).astype(float) * x["NoSpend"]).astype(float)
        x["VIP_x_LogTotalSpend"] = (x["VIP"].astype(bool).astype(float) * x["LogTotalSpend"]).astype(float)
        x["GroupSize_x_LogTotalSpend"] = (x["GroupSize"] * x["LogTotalSpend"]).astype(float)

        x["AgeBin"] = pd.cut(
            x["Age"],
            bins=[-1, 12, 18, 25, 40, 60, 120],
            labels=["u13", "13_18", "19_25", "26_40", "41_60", "61p"],
        ).astype(str)
        x["CabinNumBin"] = pd.cut(
            x["CabinNum"],
            bins=[-1, 100, 300, 600, 900, 1200, 2000],
            labels=["c0", "c1", "c2", "c3", "c4", "c5"],
        ).astype(str)
        x["CabinRegion"] = x["Deck"].astype(str) + "_" + x["CabinNumBin"].astype(str)
        x["DeckRegion"] = x["Deck"].astype(str) + "_" + x["Side"].astype(str) + "_" + x["CabinNumBin"].astype(str)
        x["HomePlanet_Destination"] = x["HomePlanet"].astype(str) + "_" + x["Destination"].astype(str)
        x["HomePlanet_Deck"] = x["HomePlanet"].astype(str) + "_" + x["Deck"].astype(str)
        x["Destination_Deck"] = x["Destination"].astype(str) + "_" + x["Deck"].astype(str)
        x["CryoSleep_NoSpend"] = x["CryoSleep"].astype(str) + "_" + x["NoSpend"].astype(int).astype(str)
        x["VIP_Luxury"] = x["VIP"].astype(str) + "_" + pd.cut(
            x["LuxurySpend"],
            bins=[-1, 0, 100, 1000, 10000, 100000],
            labels=["0", "low", "mid", "high", "ultra"],
        ).astype(str)

        for new_col, source_col in [
            ("SurnameFreq", "Surname"),
            ("DeckFreq", "Deck"),
            ("HomePlanetFreq", "HomePlanet"),
            ("DestinationFreq", "Destination"),
            ("DeckSideFreq", "DeckSide"),
            ("CabinRegionFreq", "CabinRegion"),
        ]:
            x[new_col] = x[source_col].map(self.freq_maps.get(new_col, {})).fillna(1).astype(float)

        for c in self.cat_cols:
            x[c] = x[c].fillna(f"Missing{c}").astype(str)
        for c in self.num_cols:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0).astype(float)
        return x[self.feature_cols].copy()

    def fit(self, train_df: pd.DataFrame) -> "FoldSafeFeatureBuilderV2":
        x = self._base_parse(train_df)
        x = self._apply_within_dataset_repairs(x)

        for col in ["HomePlanet", "Destination", "Deck", "Side"]:
            mapping, confs = learn_mode_map(
                x, "Surname", col, min_count=self.surname_min_count, min_conf=self.surname_min_conf
            )
            self.surname_maps[col] = mapping
            self.surname_confs[col] = confs

        x = self._apply_surname_maps(x)
        x = self._apply_fixed_rules(x)

        for c, fallback in [
            ("HomePlanet", "Earth"),
            ("Destination", "TRAPPIST-1e"),
            ("Deck", "MissingDeck"),
            ("Side", "MissingSide"),
            ("Surname", "MissingSurname"),
        ]:
            m = x[c].dropna().mode(dropna=True)
            self.global_modes[c] = m.iloc[0] if len(m) else fallback
        self.global_modes["VIP"] = False
        self.global_modes["CryoSleep"] = False

        x = self._apply_global_fills(x)
        self._learn_cabin_lines(x)
        self.cabin_median_maps["deck_side"] = safe_group_map(x, ["Deck", "Side"], "CabinNum", "median")
        self.cabin_median_maps["deck"] = safe_group_map(x, ["Deck"], "CabinNum", "median")
        self.global_medians["CabinNum"] = float(pd.to_numeric(x["CabinNum"], errors="coerce").median())
        x = self._apply_cabin_fills(x)

        self.global_medians["Age"] = float(pd.to_numeric(x["Age"], errors="coerce").median())
        self.age_map_lvl1 = safe_group_map(x, ["HomePlanet", "Deck", "VIP", "CryoSleep"], "Age", "median")
        self.age_map_lvl2 = safe_group_map(x, ["HomePlanet", "Deck"], "Age", "median")
        for c in SPEND_COLS:
            self.spend_maps[c] = safe_group_map(x, ["HomePlanet", "CryoSleep", "Deck"], c, "median")

        x = self._apply_age_spend_fills(x)
        x_tmp = self._add_final_features(x)
        for new_col, source_col in [
            ("SurnameFreq", "Surname"),
            ("DeckFreq", "Deck"),
            ("HomePlanetFreq", "HomePlanet"),
            ("DestinationFreq", "Destination"),
            ("DeckSideFreq", "DeckSide"),
            ("CabinRegionFreq", "CabinRegion"),
        ]:
            self.freq_maps[new_col] = x_tmp[source_col].value_counts(dropna=False).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = self._base_parse(df)
        x = self._apply_within_dataset_repairs(x)
        x = self._apply_surname_maps(x)
        x = self._apply_fixed_rules(x)
        x = self._apply_global_fills(x)
        x = self._apply_cabin_fills(x)
        x = self._apply_age_spend_fills(x)
        return self._add_final_features(x)


def encode_for_tree_models(
    x_tr: pd.DataFrame,
    x_va: pd.DataFrame,
    x_te: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    tr = x_tr.copy()
    va = x_va.copy()
    te = x_te.copy()
    tr[cat_cols] = enc.fit_transform(tr[cat_cols].astype(str))
    va[cat_cols] = enc.transform(va[cat_cols].astype(str))
    te[cat_cols] = enc.transform(te[cat_cols].astype(str))
    return tr, va, te


def train_cat(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray, x_te: pd.DataFrame, cat_cols: List[str], preset: str) -> Tuple[np.ndarray, np.ndarray]:
    from catboost import CatBoostClassifier

    iterations = 900 if preset == "quick" else 1800
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": iterations,
        "learning_rate": 0.035 if preset == "quick" else 0.025,
        "depth": 6,
        "l2_leaf_reg": 5.0,
        "random_seed": SEED,
        "verbose": False,
        "od_type": "Iter",
        "od_wait": 120,
        "allow_writing_files": False,
    }
    cat_idx = [x_tr.columns.get_loc(c) for c in cat_cols]
    model = CatBoostClassifier(**params)
    model.fit(x_tr, y_tr, eval_set=(x_va, y_va), cat_features=cat_idx, use_best_model=True, verbose=False)
    return model.predict_proba(x_va)[:, 1].astype(float), model.predict_proba(x_te)[:, 1].astype(float)


def train_lgb(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray, x_te: pd.DataFrame, cat_cols: List[str], preset: str) -> Tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb

    tr, va, te = encode_for_tree_models(x_tr, x_va, x_te, cat_cols)
    n_estimators = 900 if preset == "quick" else 1800
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=n_estimators,
        learning_rate=0.035 if preset == "quick" else 0.025,
        num_leaves=63,
        max_depth=6,
        min_child_samples=25,
        subsample=0.88,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        random_state=SEED,
        verbosity=-1,
        n_jobs=-1,
    )
    model.fit(
        tr, y_tr,
        eval_set=[(va, y_va)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=False)],
        categorical_feature=[tr.columns.get_loc(c) for c in cat_cols],
    )
    return model.predict_proba(va)[:, 1].astype(float), model.predict_proba(te)[:, 1].astype(float)


def train_xgb(x_tr: pd.DataFrame, y_tr: np.ndarray, x_va: pd.DataFrame, y_va: np.ndarray, x_te: pd.DataFrame, cat_cols: List[str], preset: str) -> Tuple[np.ndarray, np.ndarray]:
    import xgboost as xgb

    tr, va, te = encode_for_tree_models(x_tr, x_va, x_te, cat_cols)
    n_estimators = 700 if preset == "quick" else 1400
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        learning_rate=0.035 if preset == "quick" else 0.025,
        max_depth=4,
        min_child_weight=3.0,
        subsample=0.90,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        reg_alpha=0.01,
        gamma=1.0,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(tr, y_tr, eval_set=[(va, y_va)], verbose=False)
    return model.predict_proba(va)[:, 1].astype(float), model.predict_proba(te)[:, 1].astype(float)


MODEL_FNS = {
    "cat": train_cat,
    "lgb": train_lgb,
    "xgb": train_xgb,
}


def best_threshold(y: np.ndarray, p: np.ndarray, lo: float = 0.40, hi: float = 0.60, step: float = 0.001) -> Tuple[float, float]:
    best_acc = -1.0
    best_thr = 0.5
    for thr in np.arange(lo, hi + 1e-12, step):
        acc = float(accuracy_score(y, (p >= thr).astype(int)))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_acc, best_thr


def group_smooth(p: np.ndarray, groups: pd.Series, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return p.copy()
    tmp = pd.DataFrame({"g": groups.astype(str).to_numpy(), "p": p})
    gmean = tmp.groupby("g")["p"].transform("mean").to_numpy()
    gsize = groups.astype(str).map(groups.astype(str).value_counts()).to_numpy(dtype=float)
    a = np.where(gsize >= 4, alpha * 1.15, np.where(gsize >= 2, alpha, alpha * 0.5))
    return (1.0 - a) * p + a * gmean


def make_bool_submission(test_ids: pd.Series, p: np.ndarray, thr: float, path: Path) -> None:
    sub = pd.DataFrame({ID_COL: test_ids.to_numpy(), TARGET_COL: (p >= thr)})
    sub.to_csv(path, index=False)


def read_anchor(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    s = pd.read_csv(path)[TARGET_COL]
    if s.dtype == bool:
        return s.astype(int).to_numpy()
    if s.dtype == object:
        return s.astype(str).str.lower().map({"true": 1, "false": 0}).astype(int).to_numpy()
    return s.astype(int).to_numpy()


def rank_average(preds: List[np.ndarray]) -> np.ndarray:
    ranks = []
    for p in preds:
        ranks.append(pd.Series(p).rank(method="average").to_numpy() / len(p))
    return np.mean(ranks, axis=0)


def train_oof_models(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    model_names: List[str],
    folds: int,
    preset: str,
    out_dir: Path,
    surname_min_count: int,
    surname_min_conf: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = train_raw[TARGET_COL].astype(int).to_numpy()
    groups = train_raw[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str)
    test_groups = test_raw[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str)

    oof_df = pd.DataFrame({ID_COL: train_raw[ID_COL].values, TARGET_COL: y})
    test_df = pd.DataFrame({ID_COL: test_raw[ID_COL].values})
    metric_rows = []

    split_iter = list(StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, groups))

    for model_name in model_names:
        if model_name not in MODEL_FNS:
            print(f"[skip] unknown model {model_name}")
            continue
        try:
            __import__({"cat": "catboost", "lgb": "lightgbm", "xgb": "xgboost"}[model_name])
        except Exception as e:
            print(f"[skip] {model_name}: import failed: {e}")
            continue

        print(f"[model] {model_name}")
        oof = np.zeros(len(train_raw), dtype=float)
        test_pred = np.zeros(len(test_raw), dtype=float)
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(split_iter):
            print(f"  fold {fold + 1}/{folds}")
            raw_tr = train_raw.iloc[tr_idx].reset_index(drop=True)
            raw_va = train_raw.iloc[va_idx].reset_index(drop=True)
            y_tr = y[tr_idx]
            y_va = y[va_idx]

            builder = FoldSafeFeatureBuilderV2(
                surname_min_count=surname_min_count,
                surname_min_conf=surname_min_conf,
            ).fit(raw_tr)
            x_tr = builder.transform(raw_tr)
            x_va = builder.transform(raw_va)
            x_te = builder.transform(test_raw.reset_index(drop=True))

            va_pred, te_pred = MODEL_FNS[model_name](x_tr, y_tr, x_va, y_va, x_te, builder.cat_cols, preset)
            oof[va_idx] = va_pred
            test_pred += te_pred / folds
            fold_scores.append(float(accuracy_score(y_va, (va_pred >= 0.5).astype(int))))

        acc_05 = float(accuracy_score(y, (oof >= 0.5).astype(int)))
        bacc, bthr = best_threshold(y, oof)
        auc = float(roc_auc_score(y, oof))
        ll = float(log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6), labels=[0, 1]))
        print(f"  {model_name}: acc@0.5={acc_05:.6f} best={bacc:.6f}@{bthr:.3f} auc={auc:.6f} logloss={ll:.5f}")

        oof_df[model_name] = oof
        test_df[model_name] = test_pred
        metric_rows.append({
            "name": model_name,
            "acc@0.5": acc_05,
            "best_acc": bacc,
            "best_thr": bthr,
            "auc": auc,
            "logloss": ll,
            "fold_scores": json.dumps(fold_scores),
        })

    metrics = pd.DataFrame(metric_rows).sort_values("best_acc", ascending=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    test_df.to_csv(out_dir / "test_predictions.csv", index=False)
    metrics.to_csv(out_dir / "model_metrics.csv", index=False)
    return oof_df, test_df, metrics


def blend_search(
    oof_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    out_dir: Path,
    top_n: int,
    anchor_path: Path,
) -> pd.DataFrame:
    y = oof_df[TARGET_COL].astype(int).to_numpy()
    model_cols = [c for c in test_df.columns if c != ID_COL and c in oof_df.columns]
    train_groups = train_raw[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str)
    test_groups = test_raw[ID_COL].astype("string").str.split("_", expand=True)[0].astype(str)
    anchor = read_anchor(anchor_path)

    rows = []
    candidates: List[Tuple[str, Dict[str, float], np.ndarray, np.ndarray]] = []

    for c in model_cols:
        candidates.append((c, {c: 1.0}, oof_df[c].to_numpy(dtype=float), test_df[c].to_numpy(dtype=float)))

    if len(model_cols) >= 2:
        for a, b in itertools_combinations(model_cols, 2):
            for w in [0.35, 0.45, 0.50, 0.55, 0.65]:
                weights = {a: w, b: 1 - w}
                po = w * oof_df[a].to_numpy(dtype=float) + (1 - w) * oof_df[b].to_numpy(dtype=float)
                pt = w * test_df[a].to_numpy(dtype=float) + (1 - w) * test_df[b].to_numpy(dtype=float)
                candidates.append((f"{a}{int(w*100)}_{b}{int((1-w)*100)}", weights, po, pt))

    if len(model_cols) >= 3:
        triples = [model_cols[:3]]
        for cols in triples:
            for w1 in np.arange(0.2, 0.71, 0.1):
                for w2 in np.arange(0.1, 0.71, 0.1):
                    w3 = 1.0 - w1 - w2
                    if w3 < 0.05 or w3 > 0.70:
                        continue
                    weights = {cols[0]: float(w1), cols[1]: float(w2), cols[2]: float(w3)}
                    po = sum(oof_df[k].to_numpy(dtype=float) * v for k, v in weights.items())
                    pt = sum(test_df[k].to_numpy(dtype=float) * v for k, v in weights.items())
                    tag = "_".join(f"{k}{int(round(v*100)):02d}" for k, v in weights.items())
                    candidates.append((tag, weights, po, pt))

        po_rank = rank_average([oof_df[c].to_numpy(dtype=float) for c in model_cols])
        pt_rank = rank_average([test_df[c].to_numpy(dtype=float) for c in model_cols])
        candidates.append(("rankavg_" + "_".join(model_cols), {c: 1 / len(model_cols) for c in model_cols}, po_rank, pt_rank))

    sub_dir = out_dir / "submissions"
    ensure_dir(sub_dir)
    alphas = [0.0, 0.05, 0.10, 0.15]
    thresholds = np.arange(0.445, 0.506, 0.0025)

    for name, weights, po, pt in candidates:
        for alpha in alphas:
            p_o = group_smooth(po, train_groups, alpha)
            p_t = group_smooth(pt, test_groups, alpha)
            for thr in thresholds:
                pred = (p_o >= thr).astype(int)
                acc = float(accuracy_score(y, pred))
                test_bool = (p_t >= thr).astype(int)
                diff_rate = np.nan
                n_diff = np.nan
                if anchor is not None and len(anchor) == len(test_bool):
                    diff = test_bool != anchor
                    diff_rate = float(diff.mean())
                    n_diff = int(diff.sum())
                rows.append({
                    "name": name,
                    "weights": json.dumps(weights, sort_keys=True),
                    "alpha": alpha,
                    "thr": float(thr),
                    "oof_acc": acc,
                    "test_true_rate": float(test_bool.mean()),
                    "diff_vs_anchor": diff_rate,
                    "n_diff_vs_anchor": n_diff,
                    "pred_oof_mean": float(p_o.mean()),
                    "pred_test_mean": float(p_t.mean()),
                })

    res = pd.DataFrame(rows).sort_values(["oof_acc", "diff_vs_anchor"], ascending=[False, True]).reset_index(drop=True)
    res.to_csv(out_dir / "blend_results_all.csv", index=False)

    manifest_rows = []
    for i, row in res.head(top_n).iterrows():
        name = str(row["name"])
        weights = json.loads(row["weights"])
        alpha = float(row["alpha"])
        thr = float(row["thr"])
        po = sum(oof_df[k].to_numpy(dtype=float) * v for k, v in weights.items())
        pt = sum(test_df[k].to_numpy(dtype=float) * v for k, v in weights.items())
        pt = group_smooth(pt, test_groups, alpha)
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:120]
        fname = f"foldsafev2_{i+1:02d}_{safe}_a{int(round(alpha*100)):02d}_thr{int(round(thr*1000)):03d}.csv"
        make_bool_submission(test_df[ID_COL], pt, thr, sub_dir / fname)
        item = row.to_dict()
        item["file"] = fname
        manifest_rows.append(item)

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(out_dir / "submission_manifest.csv", index=False)
    return manifest


def itertools_combinations(items: List[str], r: int) -> Iterable[Tuple[str, ...]]:
    if r == 2:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                yield items[i], items[j]
    else:
        raise NotImplementedError


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default="train.csv")
    ap.add_argument("--test", type=str, default="test.csv")
    ap.add_argument("--out-dir", type=str, default="foldsafe_v2_local_probe_outputs")
    ap.add_argument("--models", type=str, default="cat,lgb,xgb", help="comma-separated: cat,lgb,xgb")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--preset", type=str, default="quick", choices=["quick", "medium"])
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--surname-min-count", type=int, default=2)
    ap.add_argument("--surname-min-conf", type=float, default=0.80)
    ap.add_argument(
        "--anchor",
        type=str,
        default="local_reuse_outputs/submissions/groupaware_xgb_F1F2_lgb_F1F2_xgb_F1F2F3_a15_thr470.csv",
    )
    args = ap.parse_args()

    set_seed(SEED)
    root = Path(".").resolve()
    out_dir = root / args.out_dir
    ensure_dir(out_dir)

    train_raw = pd.read_csv(args.train)
    test_raw = pd.read_csv(args.test)
    train_raw[TARGET_COL] = train_raw[TARGET_COL].astype(int)

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"[start] models={model_names} folds={args.folds} preset={args.preset}")
    print(f"[out] {out_dir}")

    oof_df, test_df, metrics = train_oof_models(
        train_raw=train_raw,
        test_raw=test_raw,
        model_names=model_names,
        folds=args.folds,
        preset=args.preset,
        out_dir=out_dir,
        surname_min_count=args.surname_min_count,
        surname_min_conf=args.surname_min_conf,
    )

    print("\n[model metrics]")
    print(metrics.to_string(index=False))

    manifest = blend_search(
        oof_df=oof_df,
        test_df=test_df,
        train_raw=train_raw,
        test_raw=test_raw,
        out_dir=out_dir,
        top_n=args.top_n,
        anchor_path=Path(args.anchor),
    )

    print("\n[top submissions]")
    cols = ["file", "oof_acc", "alpha", "thr", "test_true_rate", "diff_vs_anchor", "n_diff_vs_anchor", "weights"]
    print(manifest[cols].to_string(index=False))
    print(f"\nDone. Manifest: {out_dir / 'submission_manifest.csv'}")


if __name__ == "__main__":
    main()
