# drf.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass
from math import ceil

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# ====== import your fitted preprocessor ======
from preprocessingW import fit_preprocessor  # adjust name if needed

RNG_SEED = 42
N_FOLDS = 5
N_SAMPLES_CV = 256
N_SAMPLES_TEST = 1000

TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "drf_predictions.npy"

# ---------- CRPS for ensemble (same API as your baseline) ----------
def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    y = y.reshape(-1, 1)
    n, m = samples.shape
    term1 = np.mean(np.abs(samples - y), axis=1)

    term2 = np.empty(n)
    for i in range(n):
        s = np.sort(samples[i])
        k = np.arange(1, m + 1)
        coeff = (2 * k - m - 1)
        sum_abs = np.dot(coeff, s)  # equals sum_{j<k} (s_k - s_j)
        e_abs = (2.0 / (m * m)) * sum_abs
        term2[i] = 0.5 * e_abs

    return term1 - term2

# ---------- DRF helper ----------
@dataclass
class LeafIndex:
    # maps leaf id -> array of training indices in that leaf
    leaf_to_idx: Dict[int, np.ndarray]
    # per-leaf std on z-scale for jitter
    leaf_std: Dict[int, float]

class DistributionalForest:
    def __init__(self, rf: RandomForestRegressor, *, use_log1p: bool = True, tau: float = 0.0):
        self.rf = rf
        self.use_log1p = use_log1p
        self.tau = float(tau)
        self._z_train: Optional[np.ndarray] = None
        self._per_tree_leafindex: List[LeafIndex] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DistributionalForest":
        if self.use_log1p:
            z = np.log1p(y.clip(min=0.0))
        else:
            z = y.copy()
        self._z_train = z

        self.rf.fit(X, z)
        self._per_tree_leafindex = self._build_leaf_indices(X, z)
        return self

    def _build_leaf_indices(self, X: np.ndarray, z: np.ndarray) -> List[LeafIndex]:
        out: List[LeafIndex] = []
        for est in self.rf.estimators_:
            leaf_ids = est.apply(X)  # (n_train,)
            # group indices by leaf id
            leaf_to_idx: Dict[int, List[int]] = {}
            for i, lid in enumerate(leaf_ids):
                leaf_to_idx.setdefault(int(lid), []).append(i)
            # convert to numpy + compute per-leaf std (z-scale)
            leaf_to_idx_np: Dict[int, np.ndarray] = {}
            leaf_std: Dict[int, float] = {}
            for lid, idxs in leaf_to_idx.items():
                idxs_np = np.asarray(idxs, dtype=int)
                leaf_to_idx_np[lid] = idxs_np
                if len(idxs_np) > 1:
                    s = float(np.std(z[idxs_np], ddof=1))
                else:
                    s = 0.0
                leaf_std[lid] = s
            out.append(LeafIndex(leaf_to_idx_np, leaf_std))
        return out

    def sample(self, X_new: np.ndarray, n_samples: int, *, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Returns samples on the ORIGINAL scale (n_new, n_samples).
        Strategy: K = ceil(n_samples / n_trees) draws per tree in batch, then concat and truncate.
        """
        assert self._z_train is not None, "Call fit first."
        rng = rng or np.random.default_rng(RNG_SEED)

        T = len(self.rf.estimators_)
        K = int(ceil(n_samples / T))
        n_new = X_new.shape[0]
        z_draws_accum = []

        for est, leafdex in zip(self.rf.estimators_, self._per_tree_leafindex):
            leaves_new = est.apply(X_new)  # (n_new,)
            # Vectorized sampling per unique leaf
            z_draws = np.empty((n_new, K), dtype=float)

            # compute small jitter per leaf on z-scale
            # use max with tiny floor to avoid all-zero jitter (optional)
            for lid in np.unique(leaves_new):
                rows = np.where(leaves_new == lid)[0]
                pool_idx = leafdex.leaf_to_idx.get(int(lid), None)
                if pool_idx is None or len(pool_idx) == 0:
                    # Fallback: sample from global z_train
                    chosen = rng.integers(0, len(self._z_train), size=(len(rows), K))
                    z_base = self._z_train[chosen]
                    leaf_s = float(np.std(self._z_train, ddof=1))
                else:
                    chosen = rng.integers(0, len(pool_idx), size=(len(rows), K))
                    z_base = self._z_train[pool_idx[chosen]]
                    leaf_s = leafdex.leaf_std.get(int(lid), 0.0)

                if self.tau > 0.0 and leaf_s > 0.0:
                    noise = rng.normal(loc=0.0, scale=self.tau * leaf_s, size=(len(rows), K))
                    z_base = z_base + noise

                z_draws[rows, :] = z_base

            z_draws_accum.append(z_draws)

        # concat across trees, then back-transform & truncate
        z_all = np.concatenate(z_draws_accum, axis=1)[:, :n_samples]
        y_all = np.expm1(z_all) if self.use_log1p else z_all
        # clip to non-negative incomes
        np.maximum(y_all, 0.0, out=y_all)
        return y_all

# ---------- Training / CV ----------
def main():
    # 1) load
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel()
    X_test_df = pd.read_csv(TST_X_PATH)

    # 2) fit preprocessor on TRAIN ONLY, then transform
    # For trees: prefer ohe_drop_first=False; include_is_working=True is fine.
    prep = fit_preprocessor(X_df, ohe_drop_first=False, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=float)
    X_test = prep.transform(X_test_df).to_numpy(dtype=float)

    # 3) CV grid (keep it small/targeted)
    param_grid = [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "max_features": "sqrt", "tau": 0.0},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "max_features": 0.5, "tau": 0.05},
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "max_features": 0.5, "tau": 0.1},
        {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 5, "max_features": 0.5, "tau": 0.0}, # beste
    ]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    results: List[Tuple[dict, float]] = []

    rng_master = np.random.default_rng(RNG_SEED)

    for pg in param_grid:
        fold_crps = []
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            rf = RandomForestRegressor(
                n_estimators=pg["n_estimators"],
                max_depth=pg["max_depth"],
                min_samples_leaf=pg["min_samples_leaf"],
                max_features=pg["max_features"],
                bootstrap=True,
                n_jobs=-1,
                random_state=RNG_SEED + fold  # vary seed across folds a bit
            )
            drf = DistributionalForest(rf, use_log1p=True, tau=pg["tau"]).fit(X_tr, y_tr)

            samples_val = drf.sample(X_val, n_samples=N_SAMPLES_CV, rng=rng_master)
            fold_crps.append(crps_ensemble(y_val, samples_val).mean())

        mean_crps = float(np.mean(fold_crps))
        results.append((pg, mean_crps))
        print(f"Params: {pg}  |  mean CV-CRPS: {mean_crps:.6f}")

    # pick best
    best_pg, best_crps = min(results, key=lambda t: t[1])
    print("\nSelected params:", best_pg, "| mean CV-CRPS:", best_crps)

    # 4) refit on FULL train with chosen params
    rf_full = RandomForestRegressor(
        n_estimators=best_pg["n_estimators"],
        max_depth=best_pg["max_depth"],
        min_samples_leaf=best_pg["min_samples_leaf"],
        max_features=best_pg["max_features"],
        bootstrap=True,
        n_jobs=-1,
        random_state=RNG_SEED
    )
    drf_full = DistributionalForest(rf_full, use_log1p=True, tau=best_pg["tau"]).fit(X, y)

    # 5) generate TEST samples (original scale), save predictions.npy
    samples_test = drf_full.sample(X_test, n_samples=N_SAMPLES_TEST, rng=rng_master)
    print("Test samples shape:", samples_test.shape)  # should be (n_test, 1000)
    np.save(OUT_PATH, samples_test.astype(np.float32))
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
