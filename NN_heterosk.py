# nn_hetero_log1p.py
import os, math, time, gc
import numpy as np
import pandas as pd
from typing import Tuple, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

# ====== import your fitted preprocessor ======
# If your file is named preprocessingW.py, change this import accordingly.
from preprocessingW import fit_preprocessor

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------- paths ----------
TRN_X_PATH = "MachineLearning_Assignment/data/X_trn.csv"
TRN_Y_PATH = "MachineLearning_Assignment/data/y_trn.csv"
TST_X_PATH = "MachineLearning_Assignment/data/X_test.csv"
OUT_PATH   = "nn_hetero_log1p_predictions.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- CRPS for ensemble (vectorized) ----------
def crps_ensemble(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    y: (n,), samples: (n, m)
    returns CRPS per row, shape (n,)
    """
    y = y.reshape(-1, 1)
    n, m = samples.shape
    term1 = np.mean(np.abs(samples - y), axis=1)
    term2 = np.empty(n)
    # efficient double-absolute via sorting
    for i in range(n):
        s = np.sort(samples[i])
        k = np.arange(1, m + 1)
        coeff = (2 * k - m - 1)
        sum_abs = np.dot(coeff, s)
        e_abs = (2.0 / (m * m)) * sum_abs
        term2[i] = 0.5 * e_abs
    return term1 - term2

# ---------- model ----------
class MLPBlock(nn.Module):
    def __init__(self, d_in, d_hidden, p_drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.ln2 = nn.LayerNorm(d_hidden)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x):
        h = F.gelu(self.ln1(self.fc1(x)))
        h = self.drop(h)
        h = F.gelu(self.ln2(self.fc2(h)))
        return self.drop(h)

class HeteroGaussianNet(nn.Module):
    def __init__(self, d_in: int, widths=(256, 128), p_drop=0.1):
        super().__init__()
        layers = []
        d = d_in
        for w in widths:
            layers += [MLPBlock(d, w, p_drop), nn.Linear(w, w)]
            d = w
        self.backbone = nn.Sequential(*layers)
        self.head_mu = nn.Linear(d, 1)
        self.head_raw = nn.Linear(d, 1)  # raw scale param; we'll map to sigma>0 with softplus

    def forward(self, x):
        h = self.backbone(x)
        mu = self.head_mu(h).squeeze(-1)        # (B,)
        raw = self.head_raw(h).squeeze(-1)      # (B,)
        return mu, raw

def gaussian_nll(z_true: torch.Tensor, mu: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for Normal on z-scale with per-sample sigma(x).
    We ensure sigma>0 via softplus; we compute log_sigma stably.
    """
    sigma = F.softplus(raw) + 1e-6
    log_sigma = torch.log(sigma)
    resid2 = (z_true - mu) ** 2
    return 0.5 * (resid2 / (sigma ** 2) + 2.0 * log_sigma)

# Optional: switch to Student-t if needed (robust to heavy tails)
def studentt_nll(z_true, mu, raw_scale, raw_nu):
    """
    Student-t NLL with df = softplus(raw_nu)+2 (to keep variance finite),
    sigma = softplus(raw_scale)+1e-6. Fully implemented (no torch.distributions).
    """
    sigma = F.softplus(raw_scale) + 1e-6
    nu = F.softplus(raw_nu) + 2.0
    # log likelihood (negative)
    # const = lgamma((nu+1)/2) - lgamma(nu/2) - 0.5*log(nu*pi) - log(sigma)
    # nll  = -const + ((nu+1)/2)*log(1 + ((z-mu)^2)/(nu*sigma^2))
    t = (z_true - mu) / sigma
    const = torch.lgamma((nu + 1.0) / 2.0) - torch.lgamma(nu / 2.0) - 0.5 * torch.log(nu * math.pi) - torch.log(sigma)
    nll = -const + 0.5 * (nu + 1.0) * torch.log1p((t * t) / nu)
    return nll

# ---------- utils ----------
def set_torch_deterministic(seed=RNG_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32))

def make_loader(X: np.ndarray, z: Optional[np.ndarray], batch_size=1024, shuffle=True):
    if z is None:
        ds = TensorDataset(to_tensor(X))
    else:
        ds = TensorDataset(to_tensor(X), to_tensor(z))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

@torch.no_grad()
def predict_mu_sigma(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    mus, sigmas = [], []
    for batch in loader:
        if len(batch) == 2:
            xb, _ = batch
        else:
            xb = batch[0]
        xb = xb.to(DEVICE)
        mu, raw = model(xb)
        sigma = F.softplus(raw) + 1e-6
        mus.append(mu.detach().cpu().numpy())
        sigmas.append(sigma.detach().cpu().numpy())
    return np.concatenate(mus), np.concatenate(sigmas)

def sample_original_scale(mu: np.ndarray, sigma: np.ndarray, n_samples: int, seed=RNG_SEED) -> np.ndarray:
    """
    Given mu,sigma on z-scale, sample z~N(mu,sigma^2) then y=expm1(z). Returns (n, n_samples).
    """
    rng = np.random.default_rng(seed)
    n = mu.shape[0]
    z_draws = mu[:, None] + sigma[:, None] * rng.standard_normal((n, n_samples))
    y_draws = np.expm1(z_draws)
    np.maximum(y_draws, 0.0, out=y_draws)  # clip negatives (numerical safety)
    return y_draws

def train_one_fold(X_tr, z_tr, X_val, z_val, cfg):
    set_torch_deterministic(RNG_SEED)
    model = HeteroGaussianNet(d_in=X_tr.shape[1], widths=cfg["widths"], p_drop=cfg["dropout"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    train_loader = make_loader(X_tr, z_tr, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = make_loader(X_val, z_val, batch_size=4096, shuffle=False)

    best_crps = float("inf")
    best_state = None
    patience = cfg["patience"]
    stale = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, zb in train_loader:
            xb = xb.to(DEVICE); zb = zb.to(DEVICE)
            mu, raw = model(xb)
            loss = gaussian_nll(zb, mu, raw).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # ---- validation CRPS (on original scale) ----
        with torch.no_grad():
            mu_val, sigma_val = predict_mu_sigma(model, val_loader)
        y_val = np.expm1(z_val)  # original scale
        samples_val = sample_original_scale(mu_val, sigma_val, n_samples=cfg["n_samples_cv"])
        val_crps = float(crps_ensemble(y_val, samples_val).mean())

        if val_crps + 1e-6 < best_crps:
            best_crps = val_crps
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[epoch {epoch:03d}] loss ~ {float(loss):.4f} | val CRPS = {val_crps:.6f} | best = {best_crps:.6f}")
        if stale >= patience:
            print(f"Early stopping at epoch {epoch}. Best val CRPS={best_crps:.6f}")
            break

    model.load_state_dict(best_state)
    return model, best_crps

def main():
    # ----- load raw -----
    X_df = pd.read_csv(TRN_X_PATH)
    y = pd.read_csv(TRN_Y_PATH).values.ravel().astype(np.float64)
    X_test_df = pd.read_csv(TST_X_PATH)

    # ----- fit preprocessor on TRAIN ONLY -----
    # For NN: keep full OHE basis.
    prep = fit_preprocessor(X_df, ohe_drop_first=False, include_is_working=True)
    X = prep.transform(X_df).to_numpy(dtype=np.float32)
    X_test = prep.transform(X_test_df).to_numpy(dtype=np.float32)

    # ----- target transform -----
    z = np.log1p(y).astype(np.float32)

    # ----- CV config -----
    cfg = dict(
        widths=(256, 128),
        dropout=0.15,
        lr=1e-3,
        wd=1e-4,
        epochs=200,
        batch_size=1024,
        patience=20,
        n_samples_cv=256,   # speed for CV
        n_samples_test=1000 # assignment requirement
    )

    # ----- 5-fold CV -----
    kf = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    fold_scores = []
    fold_states = []

    for fi, (tr_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n==== Fold {fi}/5 ====")
        X_tr, X_val = X[tr_idx], X[val_idx]
        z_tr, z_val = z[tr_idx], z[val_idx]

        model, val_crps = train_one_fold(X_tr, z_tr, X_val, z_val, cfg)
        fold_scores.append(val_crps)
        fold_states.append({k: v.cpu() for k, v in model.state_dict().items()})
        del model
        gc.collect(); torch.cuda.empty_cache()

    print("\nCV CRPS per fold:", [f"{s:.6f}" for s in fold_scores])
    print("Mean CV-CRPS:", float(np.mean(fold_scores)))

    # ----- pick the best fold's weights as a starting point -----
    best_fold = int(np.argmin(fold_scores))
    best_state = fold_states[best_fold]

    # ----- Refit on FULL train for best-epoch-style duration -----
    # We'll reuse the same training loop but without a validation stop; here we do a short fine-tune.
    model_full = HeteroGaussianNet(d_in=X.shape[1], widths=cfg["widths"], p_drop=cfg["dropout"]).to(DEVICE)
    model_full.load_state_dict(best_state, strict=False)
    opt = torch.optim.AdamW(model_full.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    full_loader = make_loader(X, z, batch_size=cfg["batch_size"], shuffle=True)

    model_full.train()
    for e in range(50):
        for xb, zb in full_loader:
            xb = xb.to(DEVICE); zb = zb.to(DEVICE)
            mu, raw = model_full(xb)
            loss = gaussian_nll(zb, mu, raw).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
            opt.step()
        sched.step()
    model_full.eval()

    # ----- TEST predictions: 1000 samples per row on original scale -----
    test_loader = make_loader(X_test, None, batch_size=4096, shuffle=False)
    with torch.no_grad():
        mu_test, sigma_test = predict_mu_sigma(model_full, test_loader)
    samples_test = sample_original_scale(mu_test, sigma_test, n_samples=cfg["n_samples_test"])
    print("Test samples shape:", samples_test.shape)  # (n_test, 1000)

    # Save exactly as required
    np.save(OUT_PATH, samples_test.astype(np.float32))
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
