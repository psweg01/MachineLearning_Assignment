# ngboost_lognormal1p.py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Tuple

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold

# NGBoost
from ngboost import NGBRegressor
from ngboost.scores import LogScore
from ngboost.distns.distn import RegressionDistn

# ====== import your fitted preprocessor ======
from preprocessingW import fit_preprocessor  # adjust if different

RNG_SEED = 42
N_FOLDS = 5
N_SAMPLES_CV = 256
N_SAMPLES_TEST = 1000

TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "ngboost_lognormal1p_predictions.npy"

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

# ---------- Custom LogNormal1p distribution for NGBoost ----------
# We implement our own log-likelihood and gradient to comply with the assignment rule.

class LogNormal1pLogScore(LogScore):
    """
    Negative log-likelihood for:  z = log(1+y) ~ Normal(loc=mu, scale=sigma)
    Internal params: [mu, log_sigma]; sigma = exp(log_sigma)
    S(θ; y) = 0.5*((z-mu)^2/sigma^2) + log_sigma + 0.5*log(2π) + log(1+y)
    Gradients w.r.t. [mu, log_sigma]:
        dS/dmu        = -(z - mu) / sigma^2
        dS/dlog_sigma = 1 - ((z - mu)**2 / sigma**2)
    """
    def score(self, Y):
        z = np.log1p(np.maximum(Y, 0.0))
        a = z - self.loc
        return 0.5*(a*a)/(self.scale*self.scale) + self.logscale + 0.5*np.log(2*np.pi) + np.log1p(np.maximum(Y, 0.0))

    def d_score(self, Y):
        z = np.log1p(np.maximum(Y, 0.0))
        a = z - self.loc
        sig2 = self.scale*self.scale
        D = np.zeros((len(Y), 2))
        D[:, 0] = -a / sig2
        D[:, 1] = 1.0 - (a*a)/sig2
        return D
    # metric(): we rely on NGBoost's default MC Fisher for LogScore (OK but slower).

class LogNormal1p(RegressionDistn):
    """
    Custom RegressionDistn with internal params [mu, log_sigma].
    """
    n_params = 2
    scores = [LogNormal1pLogScore]

    def __init__(self, params):
        self._params = params
        self.loc = params[0]
        self.logscale = params[1]
        self.scale = np.exp(self.logscale)

    # class-level fit to marginal (used for initialization)
    @staticmethod
    def fit(Y):
        z = np.log1p(np.maximum(Y, 0.0))
        mu0 = float(np.mean(z))
        sig0 = float(np.std(z, ddof=0))
        sig0 = max(sig0, 1e-6)  # numerical safety
        return np.array([mu0, np.log(sig0)])

    # NGBoost expects a sample method that returns m draws (m, n)
    def sample(self, m):
        n = len(self.loc)
        rng = np.random.default_rng(RNG_SEED)
        Z = self.loc[None, :] + self.scale[None, :] * rng.standard_normal((m, n))
        Y = np.expm1(Z)
        np.maximum(Y, 0.0, out=Y)
        return Y

    # required by RegressionDistn for .predict()
    def mean(self):
        # E[exp(Z)-1] when Z~N(mu,sigma^2) = exp(mu + 0.5*sigma^2) - 1
        return np.exp(self.loc + 0.5*(self.scale**2)) - 1.0

    @property
    def params(self):
        return {"loc": self.loc, "scale": self.scale}

# ---------- main ----------
def main():
    # 1) load
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel().astype(float)
    X_test_df = pd.read_csv(TST_X_PATH)

    # 2) fit preprocessor on TRAIN ONLY, then transform
    # For trees: ohe_drop_first=False, keep is_working
    prep = fit_preprocessor(X_df, ohe_drop_first=False, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=float)
    X_test = prep.transform(X_test_df).to_numpy(dtype=float)

    # 3) hyperparameter grid (modest; expand if time permits)
    learners = [
        DecisionTreeRegressor(max_depth=2, min_samples_leaf=20, random_state=RNG_SEED),
        DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=RNG_SEED),
    ]
    learning_rates = [0.05, 0.1]
    n_estimators_list = [600, 1000]

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

    def val_crps(ngb, X_tr, y_tr, X_val, y_val):
        ngb.fit(X_tr, y_tr)
        dist_val = ngb.pred_dist(X_val)            # our custom dist with vectorized params
        S = dist_val.sample(N_SAMPLES_CV)          # shape (m, n)
        samples_val = S.T.astype(float)            # (n, m)
        return crps_ensemble(y_val, samples_val).mean()

    results = []
    for base in learners:
        for lr in learning_rates:
            for ne in n_estimators_list:
                fold_scores = []
                for tr_idx, val_idx in kf.split(X):
                    X_tr, X_val = X[tr_idx], X[val_idx]
                    y_tr, y_val = y[tr_idx], y[val_idx]
                    ngb = NGBRegressor(
                        Dist=LogNormal1p,
                        Score=LogScore,
                        Base=base,
                        natural_gradient=True,
                        n_estimators=ne,
                        learning_rate=lr,
                        random_state=RNG_SEED,
                        verbose=False
                    )
                    fold_scores.append(val_crps(ngb, X_tr, y_tr, X_val, y_val))
                results.append(((base.get_params(), lr, ne), float(np.mean(fold_scores))))
                print(f"Base={base.get_params()}, lr={lr}, n_estimators={ne} | mean CV-CRPS={np.mean(fold_scores):.6f}")

    # pick best
    (best_base_params, best_lr, best_ne), best_cv = min(results, key=lambda t: t[1])
    print(f"\nSelected: base={best_base_params}, lr={best_lr}, n_estimators={best_ne} | mean CV-CRPS={best_cv:.6f}")

    # 4) refit on FULL TRAIN
    best_base = DecisionTreeRegressor(**best_base_params)
    ngb_full = NGBRegressor(
        Dist=LogNormal1p,
        Score=LogScore,
        Base=best_base,
        natural_gradient=True,
        n_estimators=best_ne,
        learning_rate=best_lr,
        random_state=RNG_SEED,
        verbose=False
    ).fit(X, y)

    # 5) generate TEST samples (original scale), save predictions.npy
    dist_test = ngb_full.pred_dist(X_test)
    S_test = dist_test.sample(N_SAMPLES_TEST)  # (m, n)
    samples_test = S_test.T.astype(np.float32) # (n, m) = (5578, 1000) expected
    print("Test samples shape:", samples_test.shape)
    np.save(OUT_PATH, samples_test)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
