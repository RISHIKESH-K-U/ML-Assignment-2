"""
app.py
------
Streamlit demo app for the ML Assignment 2 - German Credit Risk Classification.

Features implemented (per assignment Step 6):
    a. Dataset upload option (CSV) - upload test data only
    b. Model selection dropdown (choose among the 5 trained models)
    c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
    d. Confusion matrix + classification report

Run locally with:  streamlit run app.py
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
SAMPLE_TEST_DATA_PATH = os.path.join(BASE_DIR, "test_data.csv")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

st.set_page_config(page_title="German Credit Risk Classifier", layout="wide")


@st.cache_resource
def load_model(model_file_name: str):
    return joblib.load(os.path.join(MODEL_DIR, model_file_name))


@st.cache_data
def load_sample_test_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_TEST_DATA_PATH)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, pos_label="good"),
        "Recall": recall_score(y_true, y_pred, pos_label="good"),
        "F1": f1_score(y_true, y_pred, pos_label="good"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("🏦 German Credit Risk Classification")
    st.caption(
        "ML Assignment 2 — Compare Logistic Regression, Decision Tree, kNN, "
        "Naive Bayes and Random Forest on the UCI Statlog German Credit dataset."
    )

    # --- Sidebar: model selection ---
    st.sidebar.header("⚙️ Configuration")
    model_name = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))
    pipeline = load_model(MODEL_FILES[model_name])

    # --- Dataset upload (test data only) ---
    st.sidebar.header("📁 Test Data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload test_data.csv (must include a 'target' column with good/bad)",
        type=["csv"],
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded uploaded file: {df.shape[0]} rows")
    else:
        df = load_sample_test_data()
        st.sidebar.info("No file uploaded — using bundled sample test_data.csv")

    st.subheader("📊 Test Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if "target" not in df.columns:
        st.error(
            "Uploaded CSV must contain a 'target' column (values: 'good' / 'bad') "
            "to compute evaluation metrics."
        )
        return

    X = df.drop(columns=["target"])
    y_true = df["target"]

    y_pred = pipeline.predict(X)
    y_pred = np.where(y_pred == 1, "good", "bad") if set(np.unique(y_pred)) <= {0, 1} else y_pred

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        classes = pipeline.classes_
        good_idx = list(classes).index(1) if 1 in classes else list(classes).index("good")
        y_proba = proba[:, good_idx]
    else:
        y_proba = np.where(y_pred == "good", 1, 0)

    metrics = compute_metrics(y_true, y_pred, y_proba)

    st.subheader(f"📈 Evaluation Metrics — {model_name}")
    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.3f}")

    st.subheader("🧮 Confusion Matrix & Classification Report")
    col1, col2 = st.columns(2)

    with col1:
        cm = confusion_matrix(y_true, y_pred, labels=["good", "bad"])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["good", "bad"],
            yticklabels=["good", "bad"],
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    with col2:
        report = classification_report(y_true, y_pred, labels=["good", "bad"], output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, use_container_width=True)

    st.divider()
    st.caption(
        "Dataset: UCI Statlog (German Credit Data) | "
        "Models trained on an 80/20 stratified split | "
        "Built with scikit-learn + Streamlit"
    )


if __name__ == "__main__":
    main()
