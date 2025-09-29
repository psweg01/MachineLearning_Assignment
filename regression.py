import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import properscoring as ps

# ----------------------------
# 1. Load data
# ----------------------------
X = pd.read_csv("Data/X_trn.csv")
y = pd.read_csv("Data/Y_trn.csv").values.ravel()  # flatten here

# ----------------------------
# 2. Identify numeric & categorical
# ----------------------------
num_cols = ["year", "age", "prestg10", "childs"]
cat_cols = ["occrecode", "wrkstat", "gender", "educcat", "maritalcat"]

preprocessor = ColumnTransformer(
    [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols)],
    remainder="passthrough"
)

# ----------------------------
# 3. Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 4. Fit linear regression
# ----------------------------
model = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])
model.fit(X_train, y_train)

# ----------------------------
# 5. Predict and compute σ
# ----------------------------
y_pred = model.predict(X_test)
sigma_hat = np.sqrt(mean_squared_error(y_test, y_pred))
print("Estimated σ:", sigma_hat)

# ----------------------------
# 6. Generate fewer predictive samples to save memory
# ----------------------------
n_samp = 100  # smaller number
samples_test = y_pred[:, None] + sigma_hat * np.random.randn(len(y_test), n_samp)

# ----------------------------
# 7. Compute CRPS
# ----------------------------
mean_crps = np.mean(ps.crps_ensemble(y_test, samples_test))
print("Mean CRPS:", mean_crps)

# ----------------------------
# 8. Other performance metrics
# ----------------------------
print("R²  :", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# ----------------------------
# 9. Predict new test data
# ----------------------------
X_new = pd.read_csv("Data/X_test.csv")
mu_hat = model.predict(X_new)
samples_new = mu_hat[:, None] + sigma_hat * np.random.randn(len(mu_hat), n_samp)
np.save("predictions.npy", samples_new.astype(np.float32))

# ----------------------------
# 10. Feature names & coefficients
# ----------------------------
linreg = model.named_steps["reg"]
prep = model.named_steps["prep"]
all_feature_names = prep.get_feature_names_out()

print("\nIntercept:", linreg.intercept_)
for name, coef in zip(all_feature_names, linreg.coef_.ravel()):
    print(f"{name:40s} {coef: .4f}")
