
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "engine_data.csv"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"

FEATURE_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]
TARGET_COLUMN = "Engine Condition"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def prepare_data():
    """Create and validate reproducible train/test CSV artifacts."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Registered dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    if df.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected dataset schema. Expected {expected_columns}; "
            f"received {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if df.isnull().any().any():
        raise ValueError("Missing values detected in the dataset.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    if set(y.unique()) != {0, 1}:
        raise ValueError(
            f"Expected target classes 0 and 1; got {sorted(y.unique())}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "X_train.csv": X_train,
        "X_test.csv": X_test,
        "y_train.csv": y_train.rename(TARGET_COLUMN),
        "y_test.csv": y_test.rename(TARGET_COLUMN),
    }

    for filename, frame in outputs.items():
        frame.to_csv(ARTIFACTS_DIR / filename, index=False)

    # Reload artifacts and validate dimensions and feature ordering.
    X_train_check = pd.read_csv(ARTIFACTS_DIR / "X_train.csv")
    X_test_check = pd.read_csv(ARTIFACTS_DIR / "X_test.csv")
    y_train_check = pd.read_csv(ARTIFACTS_DIR / "y_train.csv")
    y_test_check = pd.read_csv(ARTIFACTS_DIR / "y_test.csv")

    assert X_train_check.columns.tolist() == FEATURE_COLUMNS
    assert X_test_check.columns.tolist() == FEATURE_COLUMNS
    assert len(X_train_check) == len(y_train_check)
    assert len(X_test_check) == len(y_test_check)
    assert len(X_train_check) + len(X_test_check) == len(df)
    assert set(y_train_check[TARGET_COLUMN].unique()) == {0, 1}
    assert set(y_test_check[TARGET_COLUMN].unique()) == {0, 1}

    report = {
        "status": "passed",
        "source_dataset": "data/engine_data.csv",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "stratified": True,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "train_rows": int(len(X_train_check)),
        "test_rows": int(len(X_test_check)),
        "artifacts": list(outputs.keys()),
    }

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "data_preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DATA PREPARATION SUCCESSFUL")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    prepare_data()
