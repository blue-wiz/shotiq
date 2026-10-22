"""Train and evaluate xG models with a held-out league."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

TEST_LEAGUE = "La Liga 2015/16"  # never seen in training


def split(df: pd.DataFrame):
    test = df.competition == TEST_LEAGUE
    return df[~test], df[test]


def make_models() -> dict:
    return {
        "Baseline: distance + angle (logistic)": (
            ["distance", "angle"],
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        ),
        "Full features (logistic)": (
            FEATURES,
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        ),
        "Full features (gradient boosting)": (
            FEATURES,
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                           min_samples_leaf=80, l2_regularization=1.0,
                                           random_state=42),
        ),
    }


def evaluate(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "roc_auc": roc_auc_score(y, p), "pred_goals": float(p.sum()),
            "actual_goals": int(np.sum(y))}


def train_and_evaluate(df: pd.DataFrame):
    train, test = split(df)
    results, fitted, preds = [], {}, {}
    base_rate = train.goal.mean()
    r = evaluate(test.goal, np.full(len(test), base_rate)); r["model"] = "Constant (league average)"
    results.append(r)
    for name, (cols, model) in make_models().items():
        model.fit(train[cols], train.goal)
        p = model.predict_proba(test[cols])[:, 1]
        r = evaluate(test.goal, p); r["model"] = name
        results.append(r); fitted[name] = (cols, model); preds[name] = p
    sb = test.statsbomb_xg.to_numpy()
    r = evaluate(test.goal, sb); r["model"] = "StatsBomb's own xG (benchmark)"
    results.append(r)
    return pd.DataFrame(results).set_index("model"), fitted, preds, train, test


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean, "observed_rate": frac})
