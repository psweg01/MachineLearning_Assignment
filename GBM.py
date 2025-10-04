# gbm_two_stage.py
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, List, Dict
from dataclasses import dataclass

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR

# ===== import your preprocessor =====
from preprocessingW import fit_preprocessor  # adjust module name if needed

RNG_SEED = 42
N_FOLDS = 5
N_SAMPLES_CV = 256
N_SAMPLES_TEST = 1000
EPS_VAR = 1e-6
BIAS = 1.27036   # -(psi(1/2)+log 2), bias fix for E[log(r^2)] under Normal

TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "ngb_two_stage_predictions.npy"

# ---------- CRPS (ensemble version) ----------
def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    y = y.reshape(-1, 1)
    n, m = samples.shape
    term1 = np.mean(np.abs(samples - y), axis=1)

    term2 = np.empty(n)
    for i in range(n):
        s = np.sort(samples[i])
        k = np.arange(1, m + 1)
        coeff = (2 * k - m - 1)
        sum_abs = np.dot(coeff, s)
        e_abs = (2.0 / (m * m)) * sum_abs
        term2[i] = 0.5 * e_abs
    return term1 - term2

# ---------- OOF predictions for a regressor ----------
def oof_predict(model_params: dict, X: np.ndarray, z: np.ndarray, n_splits: int) -> Tuple[np.ndarray, List[HGBR]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RNG_SEED)
    z_oof = np.empty_like(z, dtype=float)
    models = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        m = HGBR(random_state=RNG_SEED + fold, **model_params)
        m.fit(X[tr_idx], z[tr_idx])
        z_oof[val_idx] = m.predict(X[val_idx])
        models.append(m)
    return z_oof, models

# ---------- GBM configs (small, robust grids) ----------
MU_GRID = [
    {"max_iter": 600, "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 40, "l2_regularization": 0.0, "early_stopping": False},
    {"max_iter": 800, "learning_rate": 0.05, "max_depth": 7, "min_samples_leaf": 60, "l2_regularization": 0.0, "early_stopping": False},
    {"max_iter": 800, "learning_rate": 0.1,  "max_depth": 5, "min_samples_leaf": 60, "l2_regularization": 0.0, "early_stopping": False},
]
SIG_GRID = [
    {"max_iter": 400, "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 60, "l2_regularization": 0.0, "early_stopping": False},
    {"max_iter": 600, "learning_rate": 0.05, "max_depth": 7, "min_samples_leaf": 80, "l2_regularization": 0.0, "early_stopping": False},
    {"max_iter": 600, "learning_rate": 0.1,  "max_depth": 5, "min_samples_leaf": 80, "l2_regularization": 0.0, "early_stopping": False},
]

def main():
    # 1) load
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel()
    X_test_df = pd.read_csv(TST_X_PATH)

    # 2) preprocess (fit on TRAIN only)
    prep = fit_preprocessor(X_df, ohe_drop_first=False, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=float)
    X_test = prep.transform(X_test_df).to_numpy(dtype=float)

    # target on log1p scale
    z = np.log1p(np.clip(y, a_min=0.0, a_max=None))

    # 3) joint CV over (MU_GRID × SIG_GRID) using OOF residuals for variance targets
    kf_outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    results = []  # (mu_params, sig_params, mean_cv_crps)

    for mu_params in MU_GRID:
        # OOF mu once per mu_params (avoids leakage and keeps it fast)
        z_oof, _ = oof_predict(mu_params, X, z, n_splits=N_FOLDS)
        r_oof = z - z_oof
        t_all = np.log(r_oof * r_oof + EPS_VAR)  # variance target for sigma-model

        for sig_params in SIG_GRID:
            fold_scores = []
            for fold, (tr_idx, val_idx) in enumerate(kf_outer.split(X), 1):
                # fit mu on TRAIN fold
                mu_model = HGBR(random_state=RNG_SEED + fold, **mu_params)
                mu_model.fit(X[tr_idx], z[tr_idx])
                mu_val = mu_model.predict(X[val_idx])

                # fit sigma-model on TRAIN fold using OOF-based t (restricted to TRAIN indices)
                sig_model = HGBR(random_state=RNG_SEED + 100 + fold, **sig_params)
                sig_model.fit(X[tr_idx], t_all[tr_idx])

                # predict sigma on VAL
                t_val = sig_model.predict(X[val_idx]) + BIAS
                sigma_val = np.sqrt(np.exp(np.clip(t_val, -10.0, 5.0)))

                # sample on original scale
                rng = np.random.default_rng(RNG_SEED + fold)
                z_samp = mu_val[:, None] + sigma_val[:, None] * rng.standard_normal((len(val_idx), N_SAMPLES_CV))
                y_samp = np.expm1(z_samp)
                np.maximum(y_samp, 0.0, out=y_samp)

                fold_scores.append(crps_ensemble(y[val_idx], y_samp).mean())

            mean_crps = float(np.mean(fold_scores))
            results.append((mu_params, sig_params, mean_crps))
            print(f"MU {mu_params} | SIG {sig_params} -> mean CV-CRPS: {mean_crps:.6f}")

    # pick best
    best_mu, best_sig, best_cv = min(results, key=lambda t: t[2])
    print("\nSelected:")
    print("  MU params :", best_mu)
    print("  SIG params:", best_sig)
    print(f"  mean CV-CRPS: {best_cv:.6f}\n")

    # 4) refit on FULL train with best params
    mu_full = HGBR(random_state=RNG_SEED, **best_mu).fit(X, z)

    # build OOF residuals again with best_mu (for sigma target on FULL)
    z_oof_best, _ = oof_predict(best_mu, X, z, n_splits=N_FOLDS)
    r_oof_best = z - z_oof_best
    t_full = np.log(r_oof_best * r_oof_best + EPS_VAR)

    sig_full = HGBR(random_state=RNG_SEED + 999, **best_sig).fit(X, t_full)

    # 5) TEST sampling
    mu_test = mu_full.predict(X_test)
    t_test = sig_full.predict(X_test) + BIAS
    sigma_test = np.sqrt(np.exp(np.clip(t_test, -10.0, 5.0)))

    rng = np.random.default_rng(RNG_SEED)
    z_draws = mu_test[:, None] + sigma_test[:, None] * rng.standard_normal((len(mu_test), N_SAMPLES_TEST))
    y_draws = np.expm1(z_draws)
    np.maximum(y_draws, 0.0, out=y_draws)

    print("Test samples shape:", y_draws.shape)  # (n_test, 1000)
    np.save(OUT_PATH, y_draws.astype(np.float32))
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
