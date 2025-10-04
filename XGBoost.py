# xgb_lognormal2stage.py
import numpy as np
import pandas as pd
from typing import Iterable, Tuple
from dataclasses import dataclass

from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping

# ====== import your fitted preprocessor ======
from preprocessingW import fit_preprocessor  # adjust if needed

RNG_SEED = 42
N_FOLDS = 5
N_SAMPLES_CV = 256
N_SAMPLES_TEST = 1000

TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "xgb_lognormal2stage_predictions.npy"

# ---------- CRPS for ensemble (same as your baseline) ----------
def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    y: (n,), samples: (n, m) -> CRPS per row
    """
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

# ---------- helpers ----------
def make_xgb(params_overrides=None) -> XGBRegressor:
    p = dict(
        n_estimators=2000,            # with early stopping; train will stop much earlier
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        tree_method="hist",           # FAST
        n_jobs=-1,
        random_state=RNG_SEED
    )
    if params_overrides:
        p.update(params_overrides)
    return XGBRegressor(**p)

def fit_mu_model(X_tr, z_tr, X_val, z_val, params) -> XGBRegressor:
    model = make_xgb(params)
    model.fit(
        X_tr, z_tr,
        eval_set=[(X_val, z_val)],
        verbose=False
        #callbacks=[EarlyStopping(rounds=100, save_best=True)]
    )
    return model

def fit_sigma_model(X_tr, z_tr, mu_tr_pred, X_val, z_val, mu_val_pred, params) -> XGBRegressor:
    # target: log-variance; stabilize with epsilon
    eps = 1e-6
    log_var_tr = np.log((z_tr - mu_tr_pred) ** 2 + eps)
    log_var_val = np.log((z_val - mu_val_pred) ** 2 + eps)

    model = make_xgb(params)
    model.fit(
        X_tr, log_var_tr,
        eval_set=[(X_val, log_var_val)],
        verbose=False
        #callbacks=[EarlyStopping(rounds=100, save_best=True)]
    )
    return model

def predict_sigma_from_logvar(model: XGBRegressor, X: np.ndarray) -> np.ndarray:
    log_var = model.predict(X)
    var = np.exp(log_var)
    sigma = np.sqrt(np.maximum(var, 1e-8))
    return sigma

def sample_y(mu: np.ndarray, sigma: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Sample on original scale using z ~ N(mu, sigma^2), y = exp(z) - 1
    Returns (n, n_samples)
    """
    rng = np.random.default_rng(RNG_SEED)
    n = len(mu)
    Z = mu[:, None] + sigma[:, None] * rng.standard_normal((n, n_samples))
    Y = np.expm1(Z)
    np.maximum(Y, 0.0, out=Y)
    return Y

# ---------- main ----------
def main():
    # 1) load
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel().astype(float)
    X_test_df = pd.read_csv(TST_X_PATH)

    # 2) preprocess (fit on TRAIN ONLY)
    # For trees: prefer ohe_drop_first=False; keep is_working
    prep = fit_preprocessor(X_df, ohe_drop_first=False, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=float)
    X_test = prep.transform(X_test_df).to_numpy(dtype=float)

    # work on z = log1p(y)
    z = np.log1p(np.maximum(y, 0.0))

    # 3) small hyperparam grid (fast)
    param_grid = [
        dict(max_depth=4, min_child_weight=2, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9),
        dict(max_depth=5, min_child_weight=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
        dict(max_depth=6, min_child_weight=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8),
        dict(max_depth=5, min_child_weight=10,learning_rate=0.10,subsample=0.8, colsample_bytree=0.8),
    ]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    results = []

    for params in param_grid:
        fold_scores = []
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            z_tr, z_val = z[tr_idx], z[val_idx]
            y_val = y[val_idx]

            # --- fit μ on z-scale ---
            mu_model = fit_mu_model(X_tr, z_tr, X_val, z_val, params)
            mu_val = mu_model.predict(X_val)
            mu_tr  = mu_model.predict(X_tr)  # in-sample; fast approximation for variance training

            # --- fit σ via log-variance regression ---
            sigma_model = fit_sigma_model(X_tr, z_tr, mu_tr, X_val, z_val, mu_val, params)
            sigma_val = predict_sigma_from_logvar(sigma_model, X_val)

            # --- CRPS on original scale via sampling ---
            samples_val = sample_y(mu_val, sigma_val, N_SAMPLES_CV)
            fold_scores.append(crps_ensemble(y_val, samples_val).mean())

        results.append((params, float(np.mean(fold_scores))))
        print(f"params={params} | mean CV-CRPS={np.mean(fold_scores):.6f}")

    # pick best
    best_params, best_cv = min(results, key=lambda t: t[1])
    print(f"\nSelected params: {best_params} | mean CV-CRPS={best_cv:.6f}")

    # 4) refit on FULL TRAIN
    mu_model_full = fit_mu_model(X, z, X, z, best_params)  # early stopping will use the same set; fine here
    mu_full = mu_model_full.predict(X)

    sigma_model_full = fit_sigma_model(X, z, mu_full, X, z, mu_full, best_params)
    # (Yes, this is in-sample for sigma targets; acceptable for speed. If you want extra rigor,
    #  compute OOF μ predictions with an inner CV when building targets.)

    # 5) predict on TEST, sample, save
    mu_test = mu_model_full.predict(X_test)
    sigma_test = predict_sigma_from_logvar(sigma_model_full, X_test)

    samples_test = sample_y(mu_test, sigma_test, N_SAMPLES_TEST).astype(np.float32)
    print("Test samples shape:", samples_test.shape)  # expect (n_test, 1000)
    np.save(OUT_PATH, samples_test)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
