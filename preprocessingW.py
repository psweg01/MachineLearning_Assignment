# preprocessing.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

NUMERIC_COLS = ["age", "prestg10", "childs", "year"]  # include year so it's scaled
CAT_COLS = ["educcat", "wrkstat", "maritalcat", "occrecode", "gender"]

# For edu ordinal mapping (0..4). Adjust names if your CSV uses slightly different strings.
EDU_ORDER = [
    "Less Than High School",   # 0
    "High School",             # 1
    "Junior College",          # 2
    "Bachelor",                # 3
    "Graduate"                 # 4
]
EDU_TO_ORD = {name: idx for idx, name in enumerate(EDU_ORDER)}

# Define which raw wrkstat values count as "working"
WORKING_VALUES = {"Full-Time", "Part-Time"}

# FittedPreprocessor class
# bundles learned parameters with the exact transformation logic (prevents discrepancies between train/test preprocessing)
# guarantees identical columns and order between train and test sets
class FittedPreprocessor:
    # creates instance of class which holds the learned parameters
    # and can transform new dataframes in the same way
    def __init__(
            self, 
            means: Dict[str, float], 
            stds: Dict[str, float],
            medians: Dict[str, float],                # train medians for NUMERIC_COLS
            cat_levels: Dict[str, List[str]],
            ohe_cols: List[str],
            edu_ord_median: float,                    # train median for educcat_ordinal
            final_cols: Optional[List[str]] = None,   # frozen final feature order
            ohe_drop_first: bool = False,             # for linear baselines you can set True
            include_is_working: bool = True           # turn off for GLM if you keep wrkstat_*
        ):
        self.means  = means
        self.stds   = stds
        self.medians = medians
        self.cat_levels = cat_levels
        self.ohe_cols   = ohe_cols
        self.edu_ord_median = edu_ord_median
        self.final_cols = final_cols or []
        self.ohe_drop_first = ohe_drop_first
        self.include_is_working = include_is_working

    # --------- RAW-DERIVED FEATURES (run before scaling/OHE) ----------
    def _derive_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # educcat_ordinal (0..4); add a missing flag; impute with TRAIN median
        edu_raw = out["educcat"].astype("object")
        edu_ord = edu_raw.map(EDU_TO_ORD)
        out["educcat_ordinal_missing"] = edu_ord.isna().astype(int)
        out["educcat_ordinal"] = edu_ord.fillna(self.edu_ord_median).astype(int)

        # is_working (optional for GLM if wrkstat_* kept)
        wrk_raw = out["wrkstat"].astype("object").fillna("Unknown")
        if self.include_is_working:
            out["is_working"] = wrk_raw.isin(WORKING_VALUES).astype(int)

        # child bins (mutually exclusive)
        ch = out["childs"].fillna(0)
        out["childs_0"]     = (ch == 0).astype(int)
        out["childs_1_2"]   = ch.isin([1, 2]).astype(int)
        out["childs_3plus"] = (ch >= 3).astype(int)

        return out

    # Private helpers (_ prefix)
    # standardizes numeric columns (median impute first to avoid NaNs)
    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in NUMERIC_COLS:
            out[c] = out[c].fillna(self.medians[c])  # use TRAIN medians → no leakage
            out[c] = (out[c] - self.means[c]) / (self.stds[c] + 1e-12)
        return out

    # One-hot encodes categorical columns
    def _one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Ensure all categorical columns exist and are length-matched
        for c in CAT_COLS:
            if c not in out:
                out[c] = "Unknown"  # broadcast to full length (to prevent length-0 case)
            vals = out[c].astype("object").fillna("Unknown")
            mask = ~vals.isin(self.cat_levels[c])
            if mask.any():
                vals = vals.where(~mask, "Unknown")
            out[c] = pd.Categorical(vals, categories=self.cat_levels[c])

        ohe = pd.get_dummies(out[CAT_COLS], prefix=CAT_COLS, drop_first=self.ohe_drop_first)

        # align to training columns (add missing, drop extra)
        for col in self.ohe_cols:
            if col not in ohe.columns:
                ohe[col] = 0
        ohe = ohe[self.ohe_cols]

        # drop original cats and concat
        out = out.drop(columns=CAT_COLS)
        out = pd.concat([out, ohe], axis=1)
        return out

    # builds interaction features as specified
    def _interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # prestige × education (education OHE columns start with 'educcat_')
        for col in [c for c in out.columns if c.startswith("educcat_")]:
            out[f"{col}_x_prestg10"] = out[col] * out["prestg10"]
        # age × workstatus
        for col in [c for c in out.columns if c.startswith("wrkstat_")]:
            out[f"age_x_{col}"] = out["age"] * out[col]

        # age × prestige & nonlinearities
        out["age_x_prestg10"] = out["age"] * out["prestg10"]
        out["age_squared"] = out["age"] ** 2
        out["prestg10_squared"] = out["prestg10"] ** 2

        # gender x year, prestige, workstatus
        for gcol in [c for c in out.columns if c.startswith("gender_")]:
            out[f"{gcol}_x_year"] = out[gcol] * out["year"]
            out[f"{gcol}_x_prestg10"] = out[gcol] * out["prestg10"]
            for wcol in [c for c in out.columns if c.startswith("wrkstat_")]:
                out[f"{gcol}_x_{wcol}"] = out[gcol] * out[wcol]
    
        # children × marital status
        for col in [c for c in out.columns if c.startswith("maritalcat_")]:
            out[f"childs_x_{col}"] = out["childs"] * out[col]

        return out

    # Public API method
    # runs the full transformation pipeline on a new dataframe
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # 1) derive raw features
        # 2) scale numerics (using train stats)
        # 3) one-hot with locked levels
        # 4) interactions

        X = X.copy()
        X = self._derive_raw(X)
        X = self._scale(X)
        X = self._one_hot(X)
        X = self._interactions(X)

        # Last line of defense: enforce final column order from TRAIN
        if self.final_cols:
            for c in self.final_cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[self.final_cols]

        return X

# Perform the fit phase once on the training data and store everything needed to transform any future dataset the same way
def fit_preprocessor(X_train: pd.DataFrame, *, ohe_drop_first: bool = False, include_is_working: bool = True) -> FittedPreprocessor:
    X = X_train.copy()
    # --- set up category levels from TRAIN ---
    cat_levels: Dict[str, List[str]] = {}
    for c in CAT_COLS:
        col = X[c].astype("object").fillna("Unknown")
        cat_levels[c] = ["Unknown"] + sorted([v for v in pd.Series(col.unique()) if v != "Unknown"])
        X[c] = col

    # --- compute numeric medians/means/stds from TRAIN ---
    medians, means, stds = {}, {}, {}
    for c in NUMERIC_COLS:
        med = X[c].median()
        X[c] = X[c].fillna(med)
        medians[c] = med
        means[c]   = X[c].mean()
        stds[c]    = X[c].std(ddof=0)

    # TRAIN median for educcat_ordinal
    edu_ord = X["educcat"].astype("object").map(EDU_TO_ORD)
    edu_ord_median = float(edu_ord.dropna().median()) if not edu_ord.dropna().empty else 1.0  # default ~ High School

    # --- build OHE column index from TRAIN ---
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    ohe = pd.get_dummies(X[CAT_COLS], prefix=CAT_COLS, drop_first=ohe_drop_first)
    ohe_cols = list(ohe.columns)

    # Build final column order by actually running the pipeline ON TRAIN ONCE
    temp = X_train.copy()
    temp_prep = FittedPreprocessor(means, stds, medians, cat_levels, ohe_cols, edu_ord_median, final_cols=None, ohe_drop_first=ohe_drop_first, include_is_working=include_is_working)
    temp = temp_prep.transform(temp)
    final_cols = list(temp.columns)

    return FittedPreprocessor(means, stds, medians, cat_levels, ohe_cols, edu_ord_median, final_cols=final_cols, ohe_drop_first=ohe_drop_first, include_is_working=include_is_working)


if __name__ == "__main__":
    # paths
    TR_PATH = "../MachineLearning_Assignment/Data/X_trn.csv"
    TE_PATH = "../MachineLearning_Assignment/Data/X_test.csv"
    OUT_TR = "../MachineLearning_Assignment/Data/X_trn_preprocessed.csv"
    OUT_TE = "../MachineLearning_Assignment/Data/X_test_preprocessed.csv"

    Xtr = pd.read_csv(TR_PATH)
    Xte = pd.read_csv(TE_PATH)

    prep = fit_preprocessor(Xtr, ohe_drop_first=False, include_is_working=True) # drop_first=True for linear baselines and include_is_working=False for GLM
    Xtr_p = prep.transform(Xtr)
    Xte_p = prep.transform(Xte)

    # sanity: identical columns/order
    assert list(Xtr_p.columns) == list(Xte_p.columns)

    Xtr_p.to_csv(OUT_TR, index=False)
    Xte_p.to_csv(OUT_TE, index=False)
    print("Done. Shapes:", Xtr_p.shape, Xte_p.shape)
