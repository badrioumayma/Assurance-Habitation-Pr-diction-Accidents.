"""Train the insurance claim model without data leakage.

The data is split first. All preprocessing (imputation, outlier capping,
encoding, scaling, SMOTE) is fit on the training data only, inside a pipeline.

Usage:  python src/train.py
Output: models/model.joblib, models/metrics.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "train_Insurance.csv"
MODELS = ROOT / "models"

NUMERIC = ["Year_Of_Observation", "Insured_Period", "Residential",
           "Building_Dimension", "Number_Of_Windows"]
CATEGORICAL = ["Building_Painted", "Building_Fenced", "Garden",
               "Settlement", "Building_Type"]
FEATURES = NUMERIC + CATEGORICAL


class IQRCapper(BaseEstimator, TransformerMixin):
    """Clip values to [Q1 - 1.5*IQR, Q3 + 1.5*IQR], learned on training data."""

    def fit(self, X, y=None):
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.low_, self.high_ = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(X, self.low_, self.high_)


def load_data():
    df = pd.read_csv(DATA)
    df = df.rename(columns={"YearOfObservation": "Year_Of_Observation",
                            "Building Dimension": "Building_Dimension",
                            "NumberOfWindows": "Number_Of_Windows"})
    df = df.drop(columns=["Customer Id"]).drop_duplicates(ignore_index=True)
    df["Number_Of_Windows"] = (df["Number_Of_Windows"]
                               .replace({"without": "0", ">=10": "10"}).astype(int))
    y = (df["Claim"] == "oui").astype(int)
    # Geo_Code is dropped: 1,100+ codes, most seen only a few times, and not
    # something a user of the app could reasonably enter.
    return df[FEATURES], y


def make_pipeline(model):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("cap", IQRCapper()),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                      unknown_value=-1))])
    prep = ColumnTransformer([("num", numeric, NUMERIC),
                              ("cat", categorical, CATEGORICAL)])
    return Pipeline([("prep", prep), ("smote", SMOTE(random_state=42)), ("model", model)])


CANDIDATES = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(max_depth=4, random_state=42),
}


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in CANDIDATES.items():
        scores = cross_val_score(make_pipeline(model), X_train, y_train,
                                 cv=cv, scoring="roc_auc")
        results[name] = {"cv_roc_auc": round(float(scores.mean()), 4),
                         "cv_roc_auc_std": round(float(scores.std()), 4)}
        print(f"{name:20s} CV ROC-AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    best_name = max(results, key=lambda n: results[n]["cv_roc_auc"])
    best = make_pipeline(CANDIDATES[best_name]).fit(X_train, y_train)

    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    test = {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "f1_claim": round(float(f1_score(y_test, pred)), 4),
        "baseline_claim_rate": round(float(y_test.mean()), 4),
    }
    print(f"\nBest model: {best_name}\nHeld-out test: {test}")
    print(classification_report(y_test, pred, target_names=["No claim", "Claim"]))

    MODELS.mkdir(exist_ok=True)
    joblib.dump(best, MODELS / "model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(
        {"best_model": best_name, "cv": results, "test": test}, indent=2))


if __name__ == "__main__":
    main()
