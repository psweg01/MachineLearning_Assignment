# Exploratory Data Analysis (EDA) starter for the assignment
# Assumes files in current dir: X_trn.csv, y_trn.csv
# Uses matplotlib only for plots (no seaborn)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 0) Config
# ----------------------------
FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# Set your categorical columns here (adjust if names differ)
CATEGORICAL_COLS = ["gender", "maritalcat", "educcat", "wrkstat", "occrecode"]
ORDERED_CATEGORICAL_COLS = []  # e.g., if educcat has a known order
YEAR_COL = "year"
TARGET_NAME = "realrinc"

# ----------------------------
# 1) Load training data
# ----------------------------
X = pd.read_csv("Data/X_trn.csv")
y = pd.read_csv("Data/y_trn.csv").squeeze("columns")  # becomes a Series if single column
if y.name is None:
    y.name = TARGET_NAME

# Merge for EDA convenience (train only)
df = X.copy()
df[TARGET_NAME] = y.values

# ----------------------------
# 2) Dtypes & basic structure
# ----------------------------
# Cast categoricals
for c in CATEGORICAL_COLS:
    if c in df.columns:
        df[c] = df[c].astype("category")

for c in ORDERED_CATEGORICAL_COLS:
    if c in df.columns:
        df[c] = df[c].cat.as_ordered()

print("\n--- Shape ---")
print(df.shape)

print("\n--- Head ---")
print(df.head(3))

print("\n--- dtypes ---")
print(df.dtypes)

# ----------------------------
# 3) Missingness overview
# ----------------------------
missing = df.isna().mean().sort_values(ascending=False).to_frame("missing_frac")
missing["missing_pct"] = (100 * missing["missing_frac"]).round(2)
print("\n--- Missingness (% of rows) ---")
print(missing.head(20))

missing.to_csv("missingness_summary.csv", index=True)

# ----------------------------
# 4) Cardinality (unique counts)
# ----------------------------
card = df.nunique(dropna=False).sort_values(ascending=False).to_frame("n_unique")
card.to_csv("cardinality_summary.csv")
print("\n--- Cardinality (unique values per column) ---")
print(card.head(20))

# ----------------------------
# 5) Descriptive stats
# ----------------------------
num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != TARGET_NAME]
cat_cols = [c for c in df.columns if pd.api.types.is_categorical_dtype(df[c])]

desc_num = df[num_cols + [TARGET_NAME]].describe().T
desc_num.to_csv("numeric_descriptives.csv")
print("\n--- Numeric descriptives ---")
print(desc_num)

# Categorical value counts (top 20; full saved to CSVs)
for c in cat_cols:
    vc = df[c].value_counts(dropna=False)
    vc.to_csv(f"value_counts__{c}.csv")
    print(f"\n--- Value counts: {c} (top 10) ---")
    print(vc.head(10))

# ----------------------------
# 6) Distributions
# ----------------------------
# Numeric histograms
for c in num_cols + [TARGET_NAME]:
    plt.figure()
    df[c].hist(bins=50)
    plt.title(f"Histogram: {c}")
    plt.xlabel(c); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f"hist__{c}.png"))
    plt.close()

# Categorical bar charts (top 20 categories)
for c in cat_cols:
    counts = df[c].value_counts(dropna=False).head(20)
    plt.figure()
    counts.plot(kind="bar")
    plt.title(f"Bar chart (top 20): {c}")
    plt.xlabel(c); plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f"bar__{c}.png"))
    plt.close()

# ----------------------------
# 7) Correlations among numeric features (Pearson & Spearman)
# ----------------------------
def corr_heatmap(data, title, fname):
    corr = data.corr()
    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, fname))
    plt.close()
    return corr

pearson_corr = corr_heatmap(df[num_cols], "Pearson correlation (numeric features)", "corr_numeric_pearson.png")
spearman_corr = df[num_cols].corr(method="spearman")
spearman_corr.to_csv("spearman_numeric_corr.csv")
pearson_corr.to_csv("pearson_numeric_corr.csv")

# ----------------------------
# 8) Target relationships (train only)
# ----------------------------
# Correlation with target (numeric)
targ_corr = df[num_cols + [TARGET_NAME]].corr()[TARGET_NAME].sort_values(ascending=False)
targ_corr.to_csv("target_corr_numeric.csv")
print("\n--- Correlation with target (numeric, Pearson) ---")
print(targ_corr)

# Binned boxplots: target vs selected numeric features
SEL_NUM_FOR_BOX = num_cols[:6]  # adjust if you want more/less
for c in SEL_NUM_FOR_BOX:
    # Bin feature into deciles
    try:
        dec = pd.qcut(df[c], q=10, duplicates="drop")
    except ValueError:
        continue
    tmp = pd.DataFrame({c: dec, TARGET_NAME: df[TARGET_NAME]})
    plt.figure()
    tmp.boxplot(column=TARGET_NAME, by=c, grid=False, rot=90)
    plt.suptitle("")
    plt.title(f"{TARGET_NAME} by deciles of {c}")
    plt.xlabel(c); plt.ylabel(TARGET_NAME)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f"box__{TARGET_NAME}__by__{c}.png"))
    plt.close()

# Target by category (top levels only to keep readable)
for c in cat_cols:
    means = df.groupby(c, observed=True)[TARGET_NAME].mean().sort_values(ascending=False)
    counts = df[c].value_counts()
    top = means.index[:20]
    plt.figure()
    means.loc[top].plot(kind="bar")
    plt.title(f"Mean {TARGET_NAME} by {c} (top 20 levels)")
    plt.xlabel(c); plt.ylabel(f"Mean {TARGET_NAME}")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, f"mean_target_by__{c}.png"))
    plt.close()

# ----------------------------
# 9) Temporal stability (by year)
# ----------------------------
if YEAR_COL in df.columns:
    # Target distribution by year (faceted as separate saves)
    years = np.sort(df[YEAR_COL].dropna().unique())
    for yv in years:
        sub = df.loc[df[YEAR_COL] == yv, TARGET_NAME]
        if len(sub) == 0: 
            continue
        plt.figure()
        sub.hist(bins=50)
        plt.title(f"{TARGET_NAME} histogram in year={yv}")
        plt.xlabel(TARGET_NAME); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGDIR, f"hist__{TARGET_NAME}__year_{int(yv)}.png"))
        plt.close()

    # Mean/median target by year
    year_stats = df.groupby(YEAR_COL)[TARGET_NAME].agg(["count", "mean", "median", "std"])
    year_stats.to_csv("target_by_year.csv")
    print("\n--- Target by year (count/mean/median/std) ---")
    print(year_stats.head())

# ----------------------------
# 10) Outlier snapshot (numeric)
# ----------------------------
def iqr_outlier_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k*iqr, q3 + k*iqr

outlier_report = []
for c in [*num_cols, TARGET_NAME]:
    lo, hi = iqr_outlier_bounds(df[c].dropna())
    frac = ((df[c] < lo) | (df[c] > hi)).mean()
    outlier_report.append((c, lo, hi, round(100*frac, 2)))
outlier_df = pd.DataFrame(outlier_report, columns=["column", "low_bound", "high_bound", "pct_outliers"])
outlier_df.sort_values("pct_outliers", ascending=False).to_csv("outlier_report.csv", index=False)
print("\n--- Outlier report (IQR rule) ---")
print(outlier_df.head(10))

print("\nEDA complete. Tables saved next to script; figures in ./figures")
