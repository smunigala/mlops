"""Registers the tourism dataset: validates schema and prints a summary."""
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]


def register_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    print(f"Dataset registered successfully from: {path}")
    print(f"Shape: {df.shape}")
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nTarget distribution (ProdTaken):\n", df["ProdTaken"].value_counts())

    return df


if __name__ == "__main__":
    register_dataset()
