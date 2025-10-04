# mdn_gauss_log1p.py
import os, math, time
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, Dict
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== import your fitted preprocessor ======
from preprocessingW import fit_preprocessor  # adjust if your file is named differently

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

N_FOLDS = 5
N_SAMPLES_CV = 256     # fewer for CV speed
N_SAMPLES_TEST = 1000  # assignment requirement

TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "mdn_mog_log1p_predictions.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- CRPS for ensemble (vectorized, matches your baseline) ----------
def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    y: (n,), samples: (n, m)
    returns CRPS per row, shape (n,)
    """
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

# ---------- MDN model ----------
class MDN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, K: int, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # heads: logits (K), means (K), scales raw (K)
        self.head_logits = nn.Linear(hidden, K)
        self.head_mu     = nn.Linear(hidden, K)
        self.head_s      = nn.Linear(hidden, K)

    def forward(self, x):
        h = self.net(x)
        logits = self.head_logits(h)            # (n, K)
        mu     = self.head_mu(h)                # (n, K)
        # softplus for strictly positive sigma; add tiny epsilon for stability
        sigma  = F.softplus(self.head_s(h)) + 1e-4  # (n, K)
        return logits, mu, sigma

# ---------- mixture NLL (Gaussian on z = log1p(y)), manual, stable ----------
def mdn_nll(z: torch.Tensor, logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    z: (n,)
    logits, mu, sigma: (n, K)
    returns mean NLL over batch
    """
    n, K = logits.shape
    z = z.view(-1, 1).expand(-1, K)  # (n, K)

    log_pi = F.log_softmax(logits, dim=1)      # (n, K)
    # log N(z | mu, sigma^2)
    log_norm = -0.5 * ((z - mu) / sigma)**2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    # log-sum-exp over components of log(pi_k) + log_norm_k
    log_mix = torch.logsumexp(log_pi + log_norm, dim=1)  # (n,)
    nll = -log_mix.mean()
    return nll

# ---------- sampling from the fitted MDN (vectorized) ----------
@torch.no_grad()
def mdn_sample(logits: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Returns samples on z-scale, shape (n, n_samples)
    """
    n, K = logits.shape
    pi = F.softmax(logits, dim=1)  # (n, K)

    # sample component indices for each row and sample
    comp_idx = torch.multinomial(pi, num_samples=n_samples, replacement=True)  # (n, n_samples) in [0..K-1]

    # gather means/scales for chosen components
    arange_n = torch.arange(n, device=logits.device).unsqueeze(1).expand(n, n_samples)
    mu_sel    = mu[arange_n, comp_idx]     # (n, n_samples)
    sigma_sel = sigma[arange_n, comp_idx]  # (n, n_samples)

    eps = torch.randn(n, n_samples, device=logits.device)
    z_samples = mu_sel + sigma_sel * eps
    return z_samples  # z-scale

# ---------- training helpers ----------
def train_one(
    X_tr: np.ndarray, z_tr: np.ndarray,
    X_va: np.ndarray, z_va: np.ndarray,
    in_dim: int, K: int, hidden: int, weight_decay: float,
    max_epochs: int = 200, batch_size: int = 512, patience: int = 20, lr: float = 1e-3
) -> Tuple[MDN, int, float, float]:
    """
    Train with early stopping on val NLL, return (best_model, best_epoch, best_val_nll, wall_time)
    """
    model = MDN(in_dim, hidden, K).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    z_tr_t = torch.tensor(z_tr, dtype=torch.float32, device=DEVICE)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    z_va_t = torch.tensor(z_va, dtype=torch.float32, device=DEVICE)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    wait = 0
    t0 = time.time()

    n = X_tr_t.shape[0]
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_tr_t[idx]; zb = z_tr_t[idx]
            logits, mu, sigma = model(xb)
            loss = mdn_nll(zb, logits, mu, sigma)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total += float(loss.item()) * len(idx)

        # validation
        model.eval()
        with torch.no_grad():
            logits_v, mu_v, sigma_v = model(X_va_t)
            val_loss = mdn_nll(z_va_t, logits_v, mu_v, sigma_v).item()

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    wall = time.time() - t0
    # restore
    model.load_state_dict(best_state)
    return model, best_epoch, best_val, wall

def val_crps_from_samples(model: MDN, X_va: np.ndarray, y_va: np.ndarray, n_samples: int = N_SAMPLES_CV) -> float:
    model.eval()
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        logits, mu, sigma = model(X_va_t)
        z_samps = mdn_sample(logits, mu, sigma, n_samples)           # (n, S) on z-scale
        y_samps = torch.expm1(z_samps).clamp_min(0.0).cpu().numpy()  # back to original scale
    return float(crps_ensemble(y_va, y_samps).mean())

# ---------- main ----------
def main():
    # 1) load
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel()
    X_test_df = pd.read_csv(TST_X_PATH)

    # 2) fit preprocessor on TRAIN ONLY, then transform
    prep = fit_preprocessor(X_df, ohe_drop_first=False, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=float)
    X_test = prep.transform(X_test_df).to_numpy(dtype=float)
    in_dim = X.shape[1]

    # target transform
    z = np.log1p(y).astype(np.float32)

    # 3) CV over MDN configs, select by mean VAL-CRPS
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)

    grid = []
    for K in [2, 3, 5]:
        for hidden in [128, 256]:
            for wd in [0.0, 1e-4]:
                grid.append({"K": K, "hidden": hidden, "weight_decay": wd})

    scores: Dict[Tuple[int, int, float], float] = {}
    times: Dict[Tuple[int, int, float], float] = {}

    for cfg in grid:
        fold_crps = []
        fold_time = []
        for tr_idx, va_idx in kf.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            z_tr, z_va = z[tr_idx], z[va_idx]
            y_va = y[va_idx]

            # split a small holdout from the training fold for early stopping on NLL
            X_tr_in, X_es, z_tr_in, z_es = train_test_split(
                X_tr, z_tr, test_size=0.1, random_state=RNG_SEED
            )

            model, best_epoch, best_val, wall = train_one(
                X_tr_in, z_tr_in, X_es, z_es,
                in_dim=in_dim,
                K=cfg["K"], hidden=cfg["hidden"], weight_decay=cfg["weight_decay"],
                max_epochs=200, batch_size=512, patience=20, lr=1e-3
            )
            # CRPS on validation fold (original scale)
            crps_val = val_crps_from_samples(model, X_va, y_va, n_samples=N_SAMPLES_CV)
            fold_crps.append(crps_val)
            fold_time.append(wall)

        key = (cfg["K"], cfg["hidden"], cfg["weight_decay"])
        scores[key] = float(np.mean(fold_crps))
        times[key]  = float(np.sum(fold_time))
        print(f"CFG {key}  |  mean VAL-CRPS: {scores[key]:.6f}  | total time ~ {times[key]:.1f}s")

    best_key = min(scores.items(), key=lambda kv: kv[1])[0]
    K_star, hidden_star, wd_star = best_key
    print(f"\nSelected MDN config: K={K_star}, hidden={hidden_star}, weight_decay={wd_star}  |  mean VAL-CRPS: {scores[best_key]:.6f}")

    # 4) Refit on FULL training with small early-stopping split (still no test leakage)
    X_tr_in, X_es, z_tr_in, z_es = train_test_split(X, z, test_size=0.1, random_state=RNG_SEED)
    model, best_epoch, best_val, wall = train_one(
        X_tr_in, z_tr_in, X_es, z_es,
        in_dim=in_dim, K=K_star, hidden=hidden_star, weight_decay=wd_star,
        max_epochs=300, batch_size=512, patience=30, lr=1e-3
    )
    print(f"Full-train finished. Best epoch: {best_epoch} | best val NLL: {best_val:.6f}")

    # 5) Generate TEST samples (original scale) and save predictions.npy
    model.eval()
    with torch.no_grad():
        logits_te, mu_te, sigma_te = model(torch.tensor(X_test, dtype=torch.float32, device=DEVICE))
        z_samps = mdn_sample(logits_te, mu_te, sigma_te, N_SAMPLES_TEST)      # (n_test, 1000)
        y_samps = torch.expm1(z_samps).clamp_min(0.0).cpu().numpy().astype(np.float32)

    print("Test samples shape:", y_samps.shape)  # should be (n_test, 1000)
    np.save(OUT_PATH, y_samps)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
