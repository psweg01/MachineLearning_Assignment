import pandas as pd

def preprocess_data(path):
    # --- Read the CSV file ---
    data = pd.read_csv(path)

    # --- Add a column with the birth year ---
    data["birth_year"] = data["year"] - data["age"]

    numeric_cols = ["age", "prestg10", "childs"]
    for col in numeric_cols:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = (data[col] - mean) / std

    # --- One-hot encode categorical variables ---
    data = pd.get_dummies(data, columns=["educcat"], prefix="edu")
    data = pd.get_dummies(data, columns=["wrkstat"], prefix="wrk")
    data = pd.get_dummies(data, columns=["maritalcat"], prefix="mar")
    data = pd.get_dummies(data, columns=["occrecode"], prefix="occ")

    # --- One-hot encode gender separately ---
    gender_dummies = pd.get_dummies(data["gender"], prefix="gender")
    data = pd.concat([data, gender_dummies], axis=1)

    # --- Drop the original 'gender' column ---
    data = data.drop(columns=["gender"])

    # ---------------------------------------------------------------------
    #                   Cross-term feature engineering
    # ---------------------------------------------------------------------

    # 1. educcat × prestige (prestg10)
    for col in [c for c in data.columns if c.startswith("edu_")]:
        data[f"{col}_x_prestg10"] = data[col] * data["prestg10"]

    # 2. age × workstatus
    for col in [c for c in data.columns if c.startswith("wrk_")]:
        data[f"age_x_{col}"] = data["age"] * data[col]

    # 3. age × prestige
    data["age_x_prestg10"] = data["age"] * data["prestg10"]

    # 4. gender × workstatus
    # (gender is still a single column with strings; convert to dummies first)
    for gcol in [c for c in data.columns if c.startswith("gender_")]:
        for wcol in [c for c in data.columns if c.startswith("wrk_")]:
            data[f"{gcol}_x_{wcol}"] = data[gcol] * data[wcol]

    # 5. gender × prestige
    for gcol in [c for c in data.columns if c.startswith("gender_")]:
        data[f"{gcol}_x_prestg10"] = data[gcol] * data["prestg10"]

    # 6. children × marital status
    for col in [c for c in data.columns if c.startswith("mar_")]:
        data[f"childs_x_{col}"] = data["childs"] * data[col]

    # ---------------------------------------------------------------------
    print(data.head())

    # Save the preprocessed data to a new CSV file
    data.to_csv(f"{path}_preprocessed.csv", index=False)


if __name__ == "__main__":
    preprocess_data("../MachineLearning_Assignment/Data/X_test.csv")
