# German Credit Risk Classification — ML Assignment 2

## a. Problem Statement

Banks need to decide whether to extend credit to a loan applicant. Approving a
risky applicant leads to defaults and financial loss, while rejecting a
creditworthy applicant means lost business. This project builds and compares
five supervised classification models that predict whether a loan applicant
represents a **good** or **bad** credit risk based on their financial and
personal attributes, and exposes the models through an interactive Streamlit
web application for evaluation and demonstration.

## b. Dataset Description

- **Name:** Statlog (German Credit Data)
- **Source:** UCI Machine Learning Repository —
  https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
- **Instances:** 1000
- **Features:** 20 (7 numeric, 13 categorical) covering checking/savings
  account status, credit history, loan purpose and amount, employment,
  installment rate, personal status, property, housing, job, age, existing
  credits, dependents, telephone ownership and foreign-worker status.
- **Target:** Binary — `good` (700 instances) vs `bad` (300 instances) credit
  risk.
- **Preprocessing:** Coded categorical values (e.g. `A11`, `A34`) were decoded
  to human-readable labels (see `model/prepare_data.py`). Numeric features are
  standard-scaled and categorical features one-hot encoded inside a scikit-learn
  `Pipeline`/`ColumnTransformer`, so raw CSV rows can be fed directly to every
  saved model.
- **Split:** 80% train / 20% test, stratified on the target (`random_state=42`).
  The 200-row test split is saved as `test_data.csv` and used by the Streamlit
  app.

## c. GitHub Repository Link

> https://github.com/RISHIKESH-K-U/ML-Assignment-2

## d. Models Used

All 5 models were trained on the identical train/test split of the dataset
described above.

### Comparison Table

| ML Model Name             | Accuracy | AUC   | Precision | Recall | F1    | MCC   |
|----------------------------|---------|-------|-----------|--------|-------|-------|
| Logistic Regression        | 0.705   | 0.759 | 0.776     | 0.814  | 0.794 | 0.274 |
| Decision Tree               | 0.680   | 0.629 | 0.779     | 0.757  | 0.768 | 0.253 |
| kNN                         | 0.705   | 0.693 | 0.743     | 0.886  | 0.808 | 0.209 |
| Naive Bayes                 | 0.680   | 0.711 | 0.833     | 0.679  | 0.748 | 0.335 |
| Random Forest (Ensemble)    | 0.750   | 0.795 | 0.781     | 0.893  | 0.833 | 0.355 |

*(Metrics are also written to `model/metrics_comparison.csv` when
`model/train_models.py` is run.)*

### Observations

| ML Model Name            | Observation about model performance |
|---------------------------|--------------------------------------|
| Logistic Regression       | Solid, well-balanced baseline. Good AUC (0.759) shows it separates the two classes reasonably well despite being a simple linear model, suggesting the decision boundary in the (scaled/encoded) feature space is close to linear. |
| Decision Tree              | Weakest AUC (0.629) among all models — a single unpruned tree overfits the training data and generalizes poorly, capturing noise rather than a stable decision boundary. Accuracy and F1 are also the lowest along with Naive Bayes. |
| kNN                        | Highest recall (0.886) but the lowest MCC (0.209), meaning it aggressively predicts the majority "good" class and catches most good applicants at the cost of misclassifying many bad-risk applicants as good — risky for a bank use case where missing a "bad" applicant is costly. |
| Naive Bayes                | Highest precision (0.833) but lowest recall (0.679) — conservative, only labels an applicant "good" when very confident, missing many actual good applicants. The independence assumption between the 20 correlated features (e.g. credit amount vs. duration) likely limits its ceiling, but it still achieves a respectable MCC (0.335), the 2nd best. |
| Random Forest (Ensemble)   | Best across nearly every metric — highest Accuracy (0.750), AUC (0.795), F1 (0.833) and MCC (0.355). By averaging many decorrelated decision trees it reduces the overfitting seen in the single Decision Tree while still capturing non-linear feature interactions, giving the most balanced and reliable predictions. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — it dominates on Accuracy, AUC, F1 and MCC (the most reliable metric for imbalanced binary classification), making it the most trustworthy model for this credit-risk dataset. |

## Repository Structure

```
project-folder/
├── app.py                     # Streamlit app (main entry point)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── test_data.csv              # Held-out test set used by the Streamlit app
├── data/
│   └── german_credit.csv      # Full cleaned dataset
└── model/
    ├── prepare_data.py        # Downloads & cleans the UCI dataset
    ├── train_models.py        # Trains all 5 models, computes metrics, saves pipelines
    ├── metrics_comparison.csv # Generated comparison table
    └── *.joblib               # Saved trained model pipelines
```

## How to Run Locally

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Regenerate data/models if needed:
python model/prepare_data.py
python model/train_models.py

# Launch the app:
streamlit run app.py
```

## Live Streamlit App

> https://ml-assignment-2-german-credit-risk.streamlit.app/
>
> *(Update this link once deployed on Streamlit Community Cloud.)*
