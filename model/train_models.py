"""
train_models.py
----------------
Trains 5 classification models on the German Credit dataset, evaluates each
using Accuracy, AUC, Precision, Recall, F1 and MCC, saves the fitted
pipelines (model/*.joblib), writes a held-out test split to
test_data.csv (used by the Streamlit demo app), and writes the metrics
comparison table to model/metrics_comparison.csv.

Models:
    1. Logistic Regression
    2. Decision Tree Classifier
    3. K-Nearest Neighbors Classifier
    4. Gaussian Naive Bayes
    5. Random Forest (Ensemble)
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "german_credit.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics_comparison.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2  # held out for evaluation + Streamlit demo


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y_raw = df["target"]

    label_encoder = LabelEncoder()
    # "bad" -> 0, "good" -> 1 (sorted alphabetically); "bad" is the positive
    # class of interest in credit risk, so we flip afterwards for clarity.
    y = label_encoder.fit_transform(y_raw)
    # Ensure "good"=1 is the positive class for AUC/precision/recall semantics.
    classes = list(label_encoder.classes_)
    if classes[1] != "good":
        y = 1 - y
        classes = classes[::-1]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

    return X, y, numeric_cols, categorical_cols, classes


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE
        ),
    }


def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, numeric_cols, categorical_cols, classes = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save the held-out test split (features + true label) for the Streamlit
    # demo app and as the required test_data.csv submission artifact.
    test_df = X_test.copy()
    test_df["target"] = np.where(y_test == 1, "good", "bad")
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved test data ({test_df.shape[0]} rows) to {TEST_DATA_PATH}")

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    results = []

    for name, model in get_models().items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        metrics["ML Model Name"] = name
        results.append(metrics)

        file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{file_name}.joblib"))
        print(f"{name:28s} -> " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k != 'ML Model Name'))

    results_df = pd.DataFrame(results)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    results_df.to_csv(METRICS_PATH, index=False)
    print(f"\nSaved comparison table to {METRICS_PATH}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
