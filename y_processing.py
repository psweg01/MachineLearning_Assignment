import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load (uses your path)
y = pd.read_csv("MachineLearning_Assignment/data/y_trn.csv").values.ravel()
y = y[np.isfinite(y)]  # drop NaN/inf just in case

# Basic derived arrays for specific views
y_pos = y[y > 0]                  # needed for log-x histogram
y_log1p = np.log1p(np.clip(y, 0, None))  # safe log for nonnegative values
p99 = np.percentile(y, 99)        # bulk range

# 2) Plot
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
(ax1, ax2), (ax3, ax4) = axs

# (a) Raw values (full range)
ax1.hist(y, bins='fd')
ax1.set_title("Income (raw)")
ax1.set_xlabel("income")
ax1.set_ylabel("count")

# (b) Raw values (clipped to 99th percentile)
ax2.hist(y, bins='fd', range=(0, p99))
ax2.set_title("Income (raw, 0–99th pct)")
ax2.set_xlabel("income")
ax2.set_ylabel("count")

# (c) Log-transformed values
ax3.hist(y_log1p, bins='fd')
ax3.set_title("Income transformed: log1p(y)")
ax3.set_xlabel("log1p(income)")
ax3.set_ylabel("count")

# (d) Log-scaled x-axis (positive incomes only)
if y_pos.size > 0:
    bins = np.logspace(np.log10(max(y_pos.min(), 1e-8)), np.log10(y_pos.max()), 60)
    ax4.hist(y_pos, bins=bins)
    ax4.set_xscale('log')
    ax4.set_title("Income (x-axis log scale)")
    ax4.set_xlabel("income (log scale)")
    ax4.set_ylabel("count")
else:
    ax4.text(0.5, 0.5, "No positive values for log-x", ha="center", va="center")
    ax4.axis("off")

plt.tight_layout()
plt.show()
