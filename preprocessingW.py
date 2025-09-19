# preprocessing.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

NUMERIC_COLS = ["age", "prestg10", "childs", "year"]  # include year so it's scaled
CAT_COLS = ["educcat", "wrkstat", "maritalcat", "occrecode", "gender"]

# FittedPreprocessor class
# bundles learned parameters with the exact transformation logic (prevents discrepancies between train/test preprocessing)
# guarantees identical columns and order between train and test sets
class FittedPreprocessor:
    # creates instance of class which holds the learned parameters
    # and can transform new dataframes in the same way
    def __init__(self, means: Dict[str, float], stds: Dict[str, float],
                 cat_levels: Dict[str, List], ohe_cols: List[str]):
        self.means = means
        self.stds = stds
        self.cat_levels = cat_levels   # fixed category sets from train
        self.ohe_cols = ohe_cols       # full ohe column order after train

    # Private helpers (_ prefix)
    # standardizes numeric columns (median impute first to avoid NaNs)
    def _scale(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in NUMERIC_COLS:
            # simple median impute first
            med = out[c].median() if c in out else np.nan
            out[c] = out[c].fillna(med)
            out[c] = (out[c] - self.means[c]) / (self.stds[c] + 1e-12)
        return out

    # One-hot encodes categorical columns
    def _one_hot(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # lock categories using pd.Categorical
        for c in CAT_COLS:
            if c not in out:
                out[c] = pd.Series(pd.Categorical([], categories=self.cat_levels[c]))
            # impute missing as "Unknown"
            out[c] = out[c].astype("object").fillna("Unknown")
            out[c] = pd.Categorical(out[c], categories=self.cat_levels[c])
        # build dummies with locked categories (ensures same levels as train)
        ohe = pd.get_dummies(out[CAT_COLS], prefix=CAT_COLS, drop_first=False)
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
        # age × prestige
        out["age_x_prestg10"] = out["age"] * out["prestg10"]
        # age squared
        out["age_squared"] = out["age"] ** 2
        # gender x year
        for gcol in [c for c in out.columns if c.startswith("gender_")]:
            out[f"{gcol}_x_year"] = out[gcol] * out["year"]
        # gender × workstatus
        for gcol in [c for c in out.columns if c.startswith("gender_")]:
            for wcol in [c for c in out.columns if c.startswith("wrkstat_")]:
                out[f"{gcol}_x_{wcol}"] = out[gcol] * out[wcol]
        # gender × prestige
        for gcol in [c for c in out.columns if c.startswith("gender_")]:
            out[f"{gcol}_x_prestg10"] = out[gcol] * out["prestg10"]
        # children × marital status
        for col in [c for c in out.columns if c.startswith("maritalcat_")]:
            out[f"childs_x_{col}"] = out["childs"] * out[col]
        return out

    # Public method
    # runs the full transformation pipeline on a new dataframe
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # birth_year removed to avoid collinearity
        # 1) scale numerics (using train stats)
        X = self._scale(X)
        # 2) one-hot with locked levels
        X = self._one_hot(X)
        # 3) interactions
        X = self._interactions(X)
        return X

# Perform the fit phase once on the training data and store everything needed to transform any future dataset the same way
def fit_preprocessor(X_train: pd.DataFrame) -> FittedPreprocessor:
    X = X_train.copy()
    # --- set up category levels from TRAIN ---
    cat_levels = {}
    for c in CAT_COLS:
        col = X[c].astype("object").fillna("Unknown")
        # group rare levels for high-card columns (e.g., occrecode)
        if c == "occrecode":
            counts = col.value_counts()
            keep = counts[counts >= 20].index  # tweak threshold as you wish
            col = col.where(col.isin(keep), other="Other")
        cat_levels[c] = ["Unknown"] + sorted([v for v in pd.Series(col.unique()) if v != "Unknown"])
        X[c] = col

    # --- compute numeric means/stds from TRAIN ---
    means = {}
    stds = {}
    for c in NUMERIC_COLS:
        med = X[c].median()
        X[c] = X[c].fillna(med)
        means[c] = X[c].mean()
        stds[c] = X[c].std(ddof=0)

    # --- build OHE column index from TRAIN ---
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=cat_levels[c])
    ohe = pd.get_dummies(X[CAT_COLS], prefix=CAT_COLS, drop_first=False)
    ohe_cols = list(ohe.columns)

    return FittedPreprocessor(means, stds, cat_levels, ohe_cols)


if __name__ == "__main__":
    # paths
    TR_PATH = "../MachineLearning_Assignment/Data/X_trn.csv"
    TE_PATH = "../MachineLearning_Assignment/Data/X_test.csv"
    OUT_TR = "../MachineLearning_Assignment/Data/X_trn_preprocessed.csv"
    OUT_TE = "../MachineLearning_Assignment/Data/X_test_preprocessed.csv"

    Xtr = pd.read_csv(TR_PATH)
    Xte = pd.read_csv(TE_PATH)

    prep = fit_preprocessor(Xtr)
    Xtr_p = prep.transform(Xtr)
    Xte_p = prep.transform(Xte)

    # sanity: identical columns/order
    assert list(Xtr_p.columns) == list(Xte_p.columns)

    Xtr_p.to_csv(OUT_TR, index=False)
    Xte_p.to_csv(OUT_TE, index=False)
    print("Done. Shapes:", Xtr_p.shape, Xte_p.shape)
