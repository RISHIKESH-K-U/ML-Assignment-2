"""
prepare_data.py
----------------
Downloads the UCI Statlog (German Credit Data) dataset and produces a clean,
human-readable CSV (data/german_credit.csv) used for training and for the
Streamlit demo (a held-out slice becomes test_data.csv).

Dataset source: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
1000 instances, 20 features, binary target (credit risk: good/bad).
"""

import os
import urllib.request

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "german.data")
CLEAN_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "german_credit.csv")

# Column names as documented in german.doc
COLUMNS = [
    "checking_account_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits_count",
    "job",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "target",
]

# Human-readable decodings for the coded categorical values (from german.doc)
DECODE_MAPS = {
    "checking_account_status": {
        "A11": "< 0 DM", "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM", "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken", "A31": "all credits paid back duly",
        "A32": "existing credits paid back duly", "A33": "delay in past payments",
        "A34": "critical account",
    },
    "purpose": {
        "A40": "new car", "A41": "used car", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "other",
    },
    "savings_account": {
        "A61": "< 100 DM", "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM", "A64": ">= 1000 DM", "A65": "unknown/none",
    },
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years", "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single",
    },
    "other_debtors_guarantors": {
        "A101": "none", "A102": "co-applicant", "A103": "guarantor",
    },
    "property": {
        "A121": "real estate", "A122": "building society savings/life insurance",
        "A123": "car or other", "A124": "unknown/none",
    },
    "other_installment_plans": {
        "A141": "bank", "A142": "stores", "A143": "none",
    },
    "housing": {
        "A151": "rent", "A152": "own", "A153": "for free",
    },
    "job": {
        "A171": "unemployed/unskilled non-resident", "A172": "unskilled resident",
        "A173": "skilled employee/official", "A174": "management/self-employed/highly qualified",
    },
    "telephone": {
        "A191": "none", "A192": "yes, registered",
    },
    "foreign_worker": {
        "A201": "yes", "A202": "no",
    },
    # target: 1 = Good credit risk, 2 = Bad credit risk (per german.doc)
    "target": {1: "good", 2: "bad"},
}


def download_raw() -> None:
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    if not os.path.exists(RAW_PATH):
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    else:
        print("Raw data already present, skipping download.")


def build_clean_csv() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, sep=" ", header=None, names=COLUMNS)

    for col, mapping in DECODE_MAPS.items():
        df[col] = df[col].map(mapping)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved clean dataset to {CLEAN_PATH}  shape={df.shape}")
    return df


if __name__ == "__main__":
    download_raw()
    build_clean_csv()
