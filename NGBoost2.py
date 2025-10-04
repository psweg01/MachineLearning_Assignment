# ngboost_like_xgb.py
import os, numpy as np, pandas as pd
from typing import Tuple
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

try:
    from scipy.special import digamma
except ImportError:
    # tiny fallback for a few ν values you might sweep
    def digamma(x):
        # very rough: asymptotic for x>2 (good enough for ν>=4); else precompute a dict
        import math
        y = x
        r = 0.0
        while y < 5:
            r -= 1.0 / y
            y += 1.0
        f = 1.0/(y*y)
        return r + math.log(y) - 0.5/y - f*(1/12 - f*(1/120 - f*(1/252)))

def t_bias_constant(nu: float) -> float:
    return float(digamma(nu/2.0) - digamma(0.5) - np.log(nu))


# --- import your preprocessor ---
from preprocessingW import fit_preprocessor  # uses your code exactly

# ---------------- Utilities ----------------
CHI2_LOG_BIAS = 1.27036  # E[log eps^2] = log sigma^2 - 1.27036 for eps~N(0,1)
nu = 8  # start here; we’ll tune this

# ---------- CRPS for ensemble (vectorized) ----------
def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    y: (n,); samples: (n, m)  -> returns (n,)
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

def _to_numpy(df_or_arr):
    return df_or_arr.values if isinstance(df_or_arr, pd.DataFrame) else df_or_arr

# ------------- Core training -------------
def fit_ngboost_like_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    use_log1p: bool = True,
    mu_params: dict = None,
    sig_params: dict = None,
) -> Tuple[XGBRegressor, XGBRegressor, dict]:
    """
    Two-stage 'NGBoost-like' fit with XGBoost:
      Stage 1: mu(x) via MSE
      Stage 2: log sigma^2(x) via MSE on log r^2 + bias correction
    Returns: (mu_model, sigma_model, meta)
    """
    X_np = _to_numpy(X).astype(np.float32)
    y = y.astype(np.float64)
    if use_log1p:
        y_t = np.log1p(y)
    else:
        y_t = y.copy()

    n = X_np.shape[0]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # sensible defaults for tabular data
    if mu_params is None:
        mu_params = dict(
            n_estimators=1200, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            reg_lambda=1.0, random_state=seed, n_jobs=0, tree_method="hist"
        )
    if sig_params is None:
        sig_params = dict(
            n_estimators=800, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, random_state=seed+1, n_jobs=0, tree_method="hist"
        )

    # -------- Stage 1: OOF mu for residuals --------
    oof_mu = np.zeros(n, dtype=np.float64)
    for tr, va in kf.split(X_np):
        m = XGBRegressor(**mu_params, early_stopping_rounds=50)
        m.fit(X_np[tr], y_t[tr],
              eval_set=[(X_np[va], y_t[va])],
              verbose=False)
        oof_mu[va] = m.predict(X_np[va])

    # -------- Stage 2: target for sigma model --------
    # residuals in the *training* target space
    r = y_t - oof_mu

    t = np.log(np.maximum(r**2, 1e-12))          # log r^2
    t_bc = t + CHI2_LOG_BIAS                     # unbiased log sigma^2

    b_nu = t_bias_constant(nu)
    t_bc = np.log(np.maximum(r**2, 1e-12)) + b_nu

    # (optional) small clipping to avoid numerical extremes
    t_bc = np.clip(t_bc, -20.0, 20.0)

    # OOF for sigma (not strictly needed for sampling, but useful to check calibration)
    oof_t_hat = np.zeros(n, dtype=np.float64)
    for tr, va in kf.split(X_np):
        m2 = XGBRegressor(**sig_params, early_stopping_rounds=50)
        m2.fit(X_np[tr], t_bc[tr],
               eval_set=[(X_np[va], t_bc[va])],
               verbose=False)
        oof_t_hat[va] = m2.predict(X_np[va])

    # -------- Fit final models on full data --------
    mu_final = XGBRegressor(**mu_params)
    mu_final.fit(X_np, y_t, verbose=False)

    sig_final = XGBRegressor(**sig_params)
    sig_final.fit(X_np, t_bc, verbose=False)

    meta = dict(use_log1p=use_log1p, mu_params=mu_params, sig_params=sig_params)
    return mu_final, sig_final, meta

def sample_predictions(mu_model, sig_model, X, n_samples=1000, *, use_log1p=True, seed=123):
    rng = np.random.default_rng(seed)
    X_np = _to_numpy(X).astype(np.float32)
    mu = mu_model.predict(X_np).astype(np.float64)              # (n,)
    log_sig2 = sig_model.predict(X_np).astype(np.float64)       # (n,)
    sigma = np.sqrt(np.exp(np.clip(log_sig2, -20.0, 20.0)))     # (n,)

    n = X_np.shape[0]
    eps = rng.standard_normal(size=(n, n_samples))
    draws = mu[:, None] + sigma[:, None] * eps
    if use_log1p:
        draws = np.expm1(draws)
        draws = np.maximum(draws, 0.0)  # income can't be negative
    return draws  # shape (n, n_samples)

def sample_predictions_t(mu_model, sig_model, X, n_samples=1000, *, nu=8, use_log1p=True, seed=123):
    rng = np.random.default_rng(seed)
    X_np = X.values.astype(np.float32) if hasattr(X, "values") else X.astype(np.float32)
    mu = mu_model.predict(X_np).astype(np.float64)
    log_sig2 = sig_model.predict(X_np).astype(np.float64)
    sigma = np.sqrt(np.exp(np.clip(log_sig2, -20.0, 20.0)))  # scale of the t

    n = X_np.shape[0]
    eps = rng.standard_t(df=nu, size=(n, n_samples))
    draws = mu[:, None] + sigma[:, None] * eps
    if use_log1p:
        draws = np.expm1(draws)
        draws = np.maximum(draws, 0.0)
    return draws


# ------------- End-to-end runner -------------
def main():
    # === paths (adjust to your local layout if needed) ===
    X_TRN = "MachineLearning_Assignment/data/X_trn.csv"
    Y_TRN = "MachineLearning_Assignment/data/y_trn.csv"
    X_TST = "MachineLearning_Assignment/data/X_test.csv"
    OUT   = "predictions.npy"

    # 1) Load raw
    Xtr_raw = pd.read_csv(X_TRN)
    ytr = pd.read_csv(Y_TRN).values.ravel().astype(float)
    Xte_raw = pd.read_csv(X_TST)

    # 2) Fit your preprocessor on TRAIN and transform both
    prep = fit_preprocessor(Xtr_raw, ohe_drop_first=False, include_is_working=True)
    Xtr = prep.transform(Xtr_raw)
    Xte = prep.transform(Xte_raw)
    assert list(Xtr.columns) == list(Xte.columns)

    # 3) Train "NGBoost-like" with XGBoost
    mu_model, sig_model, meta = fit_ngboost_like_xgb(
        Xtr, ytr,
        n_splits=5, seed=42,
        use_log1p=True,  # toggle False to work on original scale
    )

    # 4) Quick internal check: CRPS on a validation split (OPTIONAL)
    #   (Here we just do a small holdout for sanity; for full rigor use k-fold CV outside this runner.)
    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(Xtr, ytr, test_size=0.2, random_state=42)
    mu_tmp, sig_tmp, _ = fit_ngboost_like_xgb(X_tr, y_tr, n_splits=5, seed=123, use_log1p=True)
    va_draws = sample_predictions(mu_tmp, sig_tmp, X_va, n_samples=400, use_log1p=True, seed=7)
    print("Val CRPS (mean):", crps_ensemble(y_va, va_draws).mean())

    # 5) Final test draws (exactly 1000 per row) and save
    #preds = sample_predictions(mu_model, sig_model, Xte, n_samples=1000, use_log1p=meta["use_log1p"], seed=777)
    preds = sample_predictions_t(mu_model, sig_model, Xte, n_samples=1000, nu=nu, use_log1p=True, seed=777)

    print("predictions shape:", preds.shape)
    np.save(OUT, preds)

if __name__ == "__main__":
    main()
