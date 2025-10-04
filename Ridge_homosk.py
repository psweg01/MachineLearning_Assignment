import numpy as np
import pandas as pd
from typing import Iterable, Tuple
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# ====== import your fitted preprocessor ======
from preprocessingW import fit_preprocessor

RNG_SEED = 42
N_FOLDS = 5
N_SAMPLES_CV = 256     # fewer samples for speed during CV
N_SAMPLES_TEST = 1000  # assignment requirement

TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "ridge_homosk_predictions.npy"

# ---------- CRPS for ensemble (vectorized, O(n*m log m)) ----------
def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    y: (n,), samples: (n, m)
    returns CRPS per row, shape (n,)
    """
    y = y.reshape(-1, 1)
    n, m = samples.shape

    # term1 = E|X - y| ≈ (1/m) * sum_j |s_ij - y_i|
    term1 = np.mean(np.abs(samples - y), axis=1)

    # term2 = 0.5 * E|X - X'| ≈ (1/(2 m^2)) * sum_{j,k} |s_ij - s_ik|
    # Efficient computation via sorting: sum_{j<k} (s_k - s_j) = sum_k (2k - m - 1) s_(k)
    term2 = np.empty(n)
    for i in range(n):
        s = np.sort(samples[i])
        k = np.arange(1, m + 1)
        coeff = (2 * k - m - 1)
        sum_abs = np.dot(coeff, s)  # equals sum_{j<k} (s_k - s_j)
        e_abs = (2.0 / (m * m)) * sum_abs
        term2[i] = 0.5 * e_abs

    return term1 - term2

# ---------- helpers ----------
def fit_ridge_on_log(X: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[Ridge, float]:
    """Fit Ridge on z=log1p(y), return (model, sigma_z)."""
    z = np.log1p(y)
    model = Ridge(alpha=alpha, random_state=RNG_SEED)
    model.fit(X, z)
    z_hat = model.predict(X)
    sigma = float(np.sqrt(mean_squared_error(z, z_hat)))
    return model, sigma

def sample_from_model(model: Ridge, X: np.ndarray, sigma: float, n_samples: int, *, clip_nonneg=True) -> np.ndarray:
    """Return samples on original scale (n, n_samples) from LogNormal induced by z ~ N(mu, sigma^2)."""
    mu = model.predict(X)  # on z-scale
    n = len(mu)
    z_draws = mu[:, None] + sigma * np.random.default_rng(RNG_SEED).standard_normal((n, n_samples))
    y_draws = np.expm1(z_draws)
    if clip_nonneg:
        np.maximum(y_draws, 0.0, out=y_draws)
    return y_draws

# ---------- main ----------
def main():
    # 1) load
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel()
    X_test_df = pd.read_csv(TST_X_PATH)

    # 2) fit preprocessor on TRAIN ONLY, then transform
    prep = fit_preprocessor(X_df, ohe_drop_first=True, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=float)
    X_test = prep.transform(X_test_df).to_numpy(dtype=float)
    feature_names = list(prep.final_cols)

    # 3) CV over Ridge alphas using CRPS on original scale
    alphas: Iterable[float] = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

    alpha_scores = []
    for alpha in alphas:
        fold_crps = []
        for tr_idx, val_idx in kf.split(X):
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # fit on z-scale, estimate sigma on TRAIN fold
            model, sigma = fit_ridge_on_log(X_tr, y_tr, alpha=alpha)

            # sample on VAL fold and compute CRPS (original scale)
            samples_val = sample_from_model(model, X_val, sigma, n_samples=N_SAMPLES_CV)
            fold_crps.append(crps_ensemble(y_val, samples_val).mean())

        alpha_scores.append((alpha, float(np.mean(fold_crps))))

    alpha_star, cv_crps = min(alpha_scores, key=lambda t: t[1])
    print("CV results (alpha -> mean CRPS):")
    for a, s in alpha_scores:
        print(f"{a:>7}: {s:.6f}")
    print(f"\nSelected alpha: {alpha_star}  |  mean CV-CRPS: {cv_crps:.6f}")

    # 4) refit on FULL train with chosen alpha, estimate sigma on FULL train
    model, sigma_full = fit_ridge_on_log(X, y, alpha=alpha_star)
    print(f"Full-train sigma (log-scale): {sigma_full:.6f}")

    # 5) (optional) OOF-style sanity CRPS: reuse the CV model selection CRPS as your report sanity check

    # 6) generate TEST samples (original scale), save predictions.npy
    samples_test = sample_from_model(model, X_test, sigma_full, n_samples=N_SAMPLES_TEST)
    print("Test samples shape:", samples_test.shape)  # should be (n_test, 1000)
    np.save(OUT_PATH, samples_test.astype(np.float32))
    print(f"Saved: {OUT_PATH}")

    # 7) print top coefficients by absolute value
    coef = model.coef_.ravel()
    topk = np.argsort(np.abs(coef))[-20:][::-1]
    print("\nTop 20 coefficients (|weight|):")
    for idx in topk:
        print(f"{feature_names[idx]:40s} {coef[idx]: .6f}")
    print(f"\nIntercept (z-scale): {model.intercept_:.6f}")

if __name__ == "__main__":
    main()
