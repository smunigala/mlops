"""Loads the raw dataset, cleans it, and creates train/test splits."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
DROP_COLUMNS = ["CustomerID", "Unnamed: 0"]
TARGET_COLUMN = "ProdTaken"


def load_and_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop identifier columns that carry no predictive signal
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    # Fix known inconsistent category labels
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace("Fe Male", "Female")
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

    # Drop rows with a missing target, impute numeric columns with the median
    df = df.dropna(subset=[TARGET_COLUMN])
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def main():
    df = load_and_clean_data()

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Train shape:", Xtrain.shape, "Test shape:", Xtest.shape)
    print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
