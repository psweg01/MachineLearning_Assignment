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
EPS_VAR = 1e-6         # for log(r^2 + eps)
CLIP_LOGSIG2 = (-10.0, 5.0)  # optional numeric safety when exponentiating

TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "ridge_heterosk_predictions.npy"

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
# ---------- mean stage ----------
def fit_mean_model(X: np.ndarray, y: np.ndarray, alpha_mu: float) -> Tuple[Ridge, np.ndarray, np.ndarray]:
    """
    Fit Ridge on z = log1p(y). Return model and (z, z_hat) on the FIT data (for residuals).
    """
    z = np.log1p(y)
    model_mu = Ridge(alpha=alpha_mu, random_state=RNG_SEED)
    model_mu.fit(X, z)
    z_hat = model_mu.predict(X)
    return model_mu, z, z_hat

# ---------- variance stage ----------
def fit_sigma_model(X: np.ndarray, z: np.ndarray, z_hat: np.ndarray, alpha_sigma: float) -> Ridge:
    """
    Fit Ridge to predict t = log(r^2 + eps), where r = z - z_hat (on the SAME data used to fit mu).
    """
    r = z - z_hat
    t = np.log(r * r + EPS_VAR)
    model_sig = Ridge(alpha=alpha_sigma, random_state=RNG_SEED)
    model_sig.fit(X, t)
    return model_sig

def predict_sigma(model_sig: Ridge, X: np.ndarray) -> np.ndarray:
    """
    Predict sigma(x) from t_pred = model_sig(X) with numeric safety.
    """
    # t_pred = model_sig.predict(X)
    # t_pred = np.clip(t_pred, CLIP_LOGSIG2[0], CLIP_LOGSIG2[1])  # optional, prevents overflow/underflow
    # sigma = np.sqrt(np.exp(t_pred))
    BIAS = 1.27036  # -(psi(0.5)+log(2))
    t_pred = model_sig.predict(X) + BIAS
    sigma = np.sqrt(np.exp(np.clip(t_pred, *CLIP_LOGSIG2)))

    return sigma

# ---------- sampling ----------
def sample_from_models(model_mu: Ridge, model_sig: Ridge, X: np.ndarray, n_samples: int, *, clip_nonneg=True) -> np.ndarray:
    """
    Return samples on original scale (n, n_samples), with
    z ~ N(mu(x), sigma(x)^2), y = exp(z) - 1.
    """
    rng = np.random.default_rng(RNG_SEED)
    mu = model_mu.predict(X)                   # (n,)
    sigma = predict_sigma(model_sig, X)        # (n,)
    n = len(mu)
    z_draws = mu[:, None] + sigma[:, None] * rng.standard_normal((n, n_samples))
    y_draws = np.expm1(z_draws)
    if clip_nonneg:
        np.maximum(y_draws, 0.0, out=y_draws)
    return y_draws

# ---------- CV loop ----------
def cv_select_alphas(
    X: np.ndarray, y: np.ndarray,
    alphas_mu: Iterable[float], alphas_sigma: Iterable[float],
    n_folds: int = N_FOLDS, n_samples_cv: int = N_SAMPLES_CV
) -> Tuple[float, float, float]:
    """
    Jointly select (alpha_mu, alpha_sigma) by minimizing mean CV-CRPS.
    Returns (alpha_mu*, alpha_sigma*, best_mean_crps).
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RNG_SEED)
    results = []

    for a_mu in alphas_mu:
        for a_sig in alphas_sigma:
            fold_scores = []
            for tr_idx, val_idx in kf.split(X):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr, y_val = y[tr_idx], y[val_idx]

                # Fit mean on TRAIN fold
                model_mu, z_tr, zhat_tr = fit_mean_model(X_tr, y_tr, alpha_mu=a_mu)

                # Fit sigma on TRAIN fold (uses TRAIN residuals only)
                model_sig = fit_sigma_model(X_tr, z_tr, zhat_tr, alpha_sigma=a_sig)

                # Sample on VAL and compute CRPS on original scale
                samples_val = sample_from_models(model_mu, model_sig, X_val, n_samples_cv)
                fold_scores.append(crps_ensemble(y_val, samples_val).mean())

            results.append((a_mu, a_sig, float(np.mean(fold_scores))))

    # pick best pair
    alpha_mu_star, alpha_sig_star, best = min(results, key=lambda t: t[2])

    # pretty print table
    print("CV (alpha_mu, alpha_sigma) -> mean CRPS")
    for a_mu, a_sig, s in sorted(results, key=lambda t: t[2]):
        print(f"({a_mu:>7g}, {a_sig:>7g}) -> {s:.6f}")
    print(f"\nSelected: alpha_mu={alpha_mu_star}, alpha_sigma={alpha_sig_star} | mean CV-CRPS={best:.6f}\n")

    return alpha_mu_star, alpha_sig_star, best

# ---------- main ----------
def main():
    # 1) load
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel()
    X_test_df = pd.read_csv(TST_X_PATH)

    # 2) preprocessor (fit on TRAIN ONLY)
    prep = fit_preprocessor(X_df, ohe_drop_first=True, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=float)
    X_test = prep.transform(X_test_df).to_numpy(dtype=float)
    feature_names = list(prep.final_cols)

    # 3) CV grids
    alphas_mu   = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alphas_sigma= [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]

    a_mu, a_sig, cv_crps = cv_select_alphas(X, y, alphas_mu, alphas_sigma)

    # 4) refit on FULL TRAIN
    model_mu, z_full, zhat_full = fit_mean_model(X, y, alpha_mu=a_mu)
    model_sig = fit_sigma_model(X, z_full, zhat_full, alpha_sigma=a_sig)

    # quick diagnostics on train (optional)
    sigma_full = predict_sigma(model_sig, X)
    rmse_z = np.sqrt(mean_squared_error(z_full, zhat_full))
    print(f"Train diagnostics: RMSE(z)={rmse_z:.6f} | median sigma(z)={np.median(sigma_full):.6f}")

    # 5) TEST sampling & save
    samples_test = sample_from_models(model_mu, model_sig, X_test, n_samples=N_SAMPLES_TEST)
    print("Test samples shape:", samples_test.shape)  # should be (n_test, 1000)
    np.save(OUT_PATH, samples_test.astype(np.float32))
    print(f"Saved: {OUT_PATH}")

    # 6) Feature importance printouts
    coef_mu = model_mu.coef_.ravel()
    coef_sig = model_sig.coef_.ravel()

    def topk(coef, k=20):
        idx = np.argsort(np.abs(coef))[-k:][::-1]
        return idx

    print("\nTop 20 |coeff| for mean model (mu on z-scale):")
    for idx in topk(coef_mu, 20):
        print(f"{feature_names[idx]:40s} {coef_mu[idx]: .6f}")
    print(f"Intercept(mu): {model_mu.intercept_:.6f}")

    print("\nTop 20 |coeff| for variance model (log sigma^2 on z-scale):")
    for idx in topk(coef_sig, 20):
        print(f"{feature_names[idx]:40s} {coef_sig[idx]: .6f}")
    print(f"Intercept(log sigma^2): {model_sig.intercept_:.6f}")

if __name__ == "__main__":
    main()