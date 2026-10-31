from pathlib import Path
import itertools
import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_validate


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
REPORTS_DIR = PROJECT_DIR / "reports"

FEATURE_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]
TARGET_COLUMN = "Engine Condition"

PARAMETER_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [2],
    "n_estimators": [100, 200],
    "subsample": [0.8],
}

RANDOM_STATE = 42
CV_FOLDS = 3


def load_and_validate_artifacts():
    X_train = pd.read_csv(ARTIFACTS_DIR / "X_train.csv")
    X_test = pd.read_csv(ARTIFACTS_DIR / "X_test.csv")
    y_train = pd.read_csv(ARTIFACTS_DIR / "y_train.csv")[TARGET_COLUMN]
    y_test = pd.read_csv(ARTIFACTS_DIR / "y_test.csv")[TARGET_COLUMN]

    if X_train.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Unexpected training feature columns.")
    if X_test.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Unexpected test feature columns.")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and target row counts do not match.")
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Missing values found in feature artifacts.")
    if set(pd.concat([y_train, y_test]).unique()) != {0, 1}:
        raise ValueError("Expected target classes 0 and 1.")

    return X_train, X_test, y_train, y_test


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_validate_artifacts()

    tracking_db = PROJECT_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.resolve()}")
    mlflow.set_experiment("engine-condition-classification")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    combinations = list(itertools.product(
        PARAMETER_GRID["learning_rate"],
        PARAMETER_GRID["max_depth"],
        PARAMETER_GRID["n_estimators"],
        PARAMETER_GRID["subsample"],
    ))

    experiment_results = []
    best_auc = -np.inf
    best_params = None

    for number, values in enumerate(combinations, start=1):
        params = dict(zip(PARAMETER_GRID.keys(), values))
        model = GradientBoostingClassifier(
            **params, random_state=RANDOM_STATE
        )

        with mlflow.start_run(run_name=f"gradient_boosting_{number:02d}"):
            mlflow.log_params({
                **params,
                "model_type": "GradientBoostingClassifier",
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
            })

            scores = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring={
                    "accuracy": "accuracy",
                    "precision": "precision",
                    "recall": "recall",
                    "f1": "f1",
                    "roc_auc": "roc_auc",
                },
                n_jobs=-1,
                error_score="raise",
            )

            metrics = {}
            for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                metrics[f"cv_{metric}_mean"] = float(
                    np.mean(scores[f"test_{metric}"])
                )
                metrics[f"cv_{metric}_std"] = float(
                    np.std(scores[f"test_{metric}"])
                )

            mlflow.log_metrics(metrics)

            experiment_results.append({
                "combination": number,
                **params,
                **metrics,
            })

            print(
                f"Combination {number}/{len(combinations)} | "
                f"CV ROC-AUC: {metrics['cv_roc_auc_mean']:.4f} | "
                f"Parameters: {params}"
            )

            if metrics["cv_roc_auc_mean"] > best_auc:
                best_auc = metrics["cv_roc_auc_mean"]
                best_params = params.copy()

    best_model = GradientBoostingClassifier(
        **best_params, random_state=RANDOM_STATE
    )
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
    }

    model_path = ARTIFACTS_DIR / "best_engine_condition_model.joblib"
    joblib.dump(best_model, model_path)

    results_path = REPORTS_DIR / "hyperparameter_experiments.csv"
    pd.DataFrame(experiment_results).to_csv(results_path, index=False)

    report = {
        "status": "passed",
        "model_type": "GradientBoostingClassifier",
        "selection_metric": "mean cross-validation ROC-AUC",
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "parameter_grid": PARAMETER_GRID,
        "combinations_evaluated": len(combinations),
        "best_parameters": best_params,
        "best_cv_roc_auc": float(best_auc),
        "test_metrics": test_metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "model_path": str(model_path.relative_to(PROJECT_DIR)),
        "experiment_results_path": str(results_path.relative_to(PROJECT_DIR)),
    }

    report_path = REPORTS_DIR / "model_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMODEL TRAINING COMPLETED")
    print(json.dumps(report, indent=2))
    print(f"\nSaved model: {model_path}")
    print(f"Saved experiments: {results_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
