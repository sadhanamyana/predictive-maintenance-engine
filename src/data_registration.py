from pathlib import Path
import json
import hashlib
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "engine_data.csv"

EXPECTED_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
    "Engine Condition",
]

FEATURE_COLUMNS = EXPECTED_COLUMNS[:-1]
TARGET_COLUMN = "Engine Condition"


def register_and_validate_data():
    """Validate the registered dataset and save a data-quality report."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Validate schema.
    if df.columns.tolist() != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema.\nExpected: {EXPECTED_COLUMNS}"
            f"\nReceived: {df.columns.tolist()}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    # Validate numeric values.
    for column in EXPECTED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.isnull().any().any():
        raise ValueError("Missing values detected.")

    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values detected.")

    observed_classes = set(df[TARGET_COLUMN].unique())
    if observed_classes != {0, 1}:
        raise ValueError(
            f"Expected binary target values 0 and 1; got {observed_classes}"
        )

    # Generate a fingerprint for traceability.
    file_hash = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    report = {
        "dataset_path": str(DATA_PATH.relative_to(PROJECT_DIR)),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": {
            str(k): int(v)
            for k, v in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "sha256": file_hash,
        "validation_status": "passed",
    }

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_path = reports_dir / "data_registration_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DATA REGISTRATION SUCCESSFUL")
    print(json.dumps(report, indent=2))

    return df


if __name__ == "__main__":
    register_and_validate_data()
