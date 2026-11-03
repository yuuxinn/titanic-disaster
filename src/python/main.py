"""
Titanic survival prediction - Python version.

Loads train.csv / test.csv, cleans and engineers features, fits a
logistic regression model, and reports accuracy. Every step prints
what it did so the whole pipeline can be followed from the terminal.
"""

import os
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def find_data_dir():
    """Locate the data folder.

    Inside Docker the data is mounted at /app/data (next to this script).
    When running locally from the repo, the data lives in src/data
    (one level up from this script). DATA_DIR overrides both.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("DATA_DIR"),
        os.path.join(script_dir, "data"),
        os.path.join(script_dir, "..", "data"),
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "train.csv")):
            return os.path.abspath(path)
    print("ERROR: could not find train.csv.")
    print("Put train.csv and test.csv in src/data/ (see README.md).")
    sys.exit(1)


def extract_title(name_series):
    """Pull the title (Mr, Mrs, Miss, ...) out of the Name column."""
    titles = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = ["Mr", "Mrs", "Miss", "Master"]
    return titles.where(titles.isin(common), "Rare")


def learn_fill_values(train_df):
    """Compute imputation values from the TRAINING set only."""
    titles = extract_title(train_df["Name"])
    return {
        "age_by_title": train_df.groupby(titles)["Age"].median().to_dict(),
        "age_overall": train_df["Age"].median(),
        "fare": train_df["Fare"].median(),
        "embarked": train_df["Embarked"].mode()[0],
    }


def prepare(df, fill, name):
    """Clean the data and build model features. Prints every change."""
    df = df.copy()
    print(f"\n--- Preparing {name} set ---")

    df["Title"] = extract_title(df["Name"])
    print(f"[add] Title extracted from Name: {df['Title'].value_counts().to_dict()}")

    n_missing = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(fill["age_by_title"]))
    df["Age"] = df["Age"].fillna(fill["age_overall"])
    print(f"[fill] Age: {n_missing} missing values filled with median age per Title")

    n_missing = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fill["fare"])
    print(f"[fill] Fare: {n_missing} missing values filled with median {fill['fare']:.2f}")

    n_missing = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fill["embarked"])
    print(f"[fill] Embarked: {n_missing} missing values filled with mode '{fill['embarked']}'")

    df["HasCabin"] = df["Cabin"].notna().astype(int)
    print(f"[add] HasCabin (1 if a cabin is listed): {df['HasCabin'].sum()} passengers have a cabin")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    print(f"[add] FamilySize = SibSp + Parch + 1 (mean {df['FamilySize'].mean():.2f})")

    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print(f"[add] IsAlone: {df['IsAlone'].sum()} passengers travelling alone")

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("[encode] Sex: male -> 0, female -> 1")

    df = pd.get_dummies(df, columns=["Pclass", "Embarked", "Title"], dtype=int)
    print("[encode] One-hot encoded Pclass, Embarked, Title")

    dropped = ["PassengerId", "Name", "Ticket", "Cabin"]
    print(f"[drop] Columns not used as features: {dropped}")
    return df


def main():
    data_dir = find_data_dir()
    banner("STEP 1: Load training data")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    print(f"Loaded train.csv from {data_dir}")
    print(f"Shape: {train.shape[0]} rows x {train.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(train.head().to_string())

    banner("STEP 2: Explore training data")
    print("Missing values per column:")
    print(train.isna().sum().to_string())
    print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
    print("\nSurvival rate by Sex:")
    print(train.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:")
    print(train.groupby("Pclass")["Survived"].mean().round(3).to_string())

    banner("STEP 3: Clean data and engineer features")
    fill = learn_fill_values(train)
    train_prep = prepare(train, fill, "training")

    features = [c for c in train_prep.columns
                if c not in ("PassengerId", "Survived", "Name", "Ticket", "Cabin")]
    X = train_prep[features]
    y = train_prep["Survived"]
    print(f"\nFinal feature list ({len(features)}): {features}")

    banner("STEP 4: Logistic regression - accuracy on training data")
    # Hold out 20% of train.csv so we also get an honest estimate on unseen data.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    val_model = LogisticRegression(max_iter=1000)
    val_model.fit(X_tr, y_tr)
    print(f"Hold-out validation (80/20 split of train.csv): "
          f"{accuracy_score(y_val, val_model.predict(X_val)):.4f}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Model refit on all {len(X)} training rows.")
    print(f"Training accuracy: {train_acc:.4f}")
    print("\nModel coefficients:")
    coefs = pd.Series(model.coef_[0], index=features).sort_values()
    print(coefs.round(3).to_string())

    banner("STEP 5: Load test data and predict")
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    print(f"Loaded test.csv: {test.shape[0]} rows x {test.shape[1]} columns")
    test_prep = prepare(test, fill, "test")
    # Make sure test has exactly the same columns as train (missing dummies -> 0).
    X_test = test_prep.reindex(columns=features, fill_value=0)
    preds = model.predict(X_test)
    print(f"\nPredicted survivors: {preds.sum()} of {len(preds)} "
          f"({preds.mean():.3f} survival rate)")

    out_path = os.path.join(data_dir, "predictions_python.csv")
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(
        out_path, index=False)
    print(f"Predictions saved to {out_path}")

    banner("STEP 6: Accuracy on test data")
    if "Survived" in test.columns:
        print(f"Test accuracy: {accuracy_score(test['Survived'], preds):.4f}")
    else:
        print("test.csv has no 'Survived' column - Kaggle keeps the true labels "
              "private, so test accuracy cannot be computed locally.")
        print("Upload predictions_python.csv to the Kaggle competition to get the "
              "official test accuracy.")
        print("The hold-out validation accuracy in STEP 4 is the best local "
              "estimate of test performance.")

    banner("Done")


if __name__ == "__main__":
    main()
