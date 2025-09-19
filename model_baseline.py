# Baseline model: simple feedforward net with 2 hidden layers
# Outputs parameters of Gaussian distribution (mu, sigma)
# Trained by maximizing log-likelihood (minimizing NLL loss)
# Uses Adam optimizer with weight decay (L2 regularization)

import numpy as np, torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add preprocessingW for consistent preprocessing
X_train = np.load("../MachineLearning_Assignment/Data/X_train_preprocessed.npy")
X_test = np.load("../MachineLearning_Assignment/Data/X_test_preprocessed.npy")
y_train = np.load("../MachineLearning_Assignment/Data/y_train.npy")

# validation split
n = X_train.shape[0]
n_val = int(0.2 * n)
X_val = X_train[-n_val:]
y_val = y_train[-n_val:]
X_train = X_train[:-n_val]
y_train = y_train[:-n_val]

# X_num, X_cat_onehot prepared already; y is (n,)
Xtr = t.tensor(X_train.values, dtype=t.float32)
ytr = t.tensor(y_train.values, dtype=t.float32).view(-1,1)
Xva = t.tensor(X_val.values, dtype=t.float32)
yva = t.tensor(y_val.values, dtype=t.float32).view(-1,1)

d_in = Xtr.shape[1]
model = nn.Sequential(
    nn.Linear(d_in, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 2)   # outputs [mu, s] where sigma = softplus(s)
)

opt = t.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

def nll_gauss(y, mu, sigma):
    # sigma > 0
    return 0.5*((y-mu)**2 / (sigma**2) + 2*t.log(sigma) + np.log(2*np.pi))

def step(X, y):
    out = model(X)
    mu = out[:, :1]
    sigma = nn.functional.softplus(out[:, 1:]) + 1e-6
    loss = nll_gauss(y, mu, sigma).mean()
    return loss, mu, sigma

for epoch in range(200):
    opt.zero_grad()
    loss, _, _ = step(Xtr, ytr)
    loss.backward(); opt.step()
    if (epoch+1) % 20 == 0:
        with t.no_grad():
            lval, _, _ = step(Xva, yva)
        print(epoch+1, float(loss), float(lval))

# Sampling for test
Xt = t.tensor(X_test.values, dtype=t.float32)
with t.no_grad():
    out = model(Xt)
    mu = out[:, :1]
    sigma = nn.functional.softplus(out[:, 1:]) + 1e-6
n_test = Xt.shape[0]
S = 1000
eps = t.randn(n_test, S)
y_samps = (mu + sigma*eps).cpu().numpy()  # shape (n_test, 1000)
np.save("predictions.npy", y_samps)
