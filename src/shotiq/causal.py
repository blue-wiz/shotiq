"""Estimate the effect of the match situation on the quality of shots a team takes.

Estimand: among shots that are taken, how much does the expected-goals value of a shot
change when the shooting team is trailing (or leading) rather than level, holding fixed
team strength, opponent strength, home advantage, match minute, red cards and league?

Shot location, defenders and shot type are NOT controlled for. They are the channels
through which the match situation could change shot quality (mediators), so controlling
for them would hide the effect we want to measure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from doubleml import DoubleMLData, DoubleMLPLR
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

CONTROLS = ["team_ppg", "opp_ppg", "team_gdpg", "opp_gdpg", "is_home", "minute_c", "red_diff",
            "second_half"]
LEAGUES = ["Premier League 2015/16", "La Liga 2015/16", "Serie A 2015/16", "Ligue 1 2015/16",
           "1. Bundesliga 2015/16"]


def contrast_sample(df: pd.DataFrame, treated: str) -> pd.DataFrame:
    """Shots in state `treated` (D=1) or level (D=0), with league dummies added."""
    s = df[df.state.isin([treated, "level"])].copy()
    s["D"] = (s.state == treated).astype(int)
    s["second_half"] = (s.period >= 2).astype(int)
    for i, lg in enumerate(LEAGUES[1:]):
        s[f"lg{i}"] = (s.competition == lg).astype(int)
    return s


def x_cols(s: pd.DataFrame) -> list[str]:
    return CONTROLS + [c for c in s.columns if c.startswith("lg")]


def naive(s: pd.DataFrame, y: str) -> dict:
    m = smf.ols(f"{y} ~ D", s).fit(cov_type="cluster", cov_kwds={"groups": s.match_id})
    return _row("Raw difference", m, "D")


def ols(s: pd.DataFrame, y: str) -> dict:
    f = f"{y} ~ D + " + " + ".join(x_cols(s))
    m = smf.ols(f, s).fit(cov_type="cluster", cov_kwds={"groups": s.match_id})
    return _row("Regression with controls", m, "D")


def _row(name, m, term):
    ci = m.conf_int().loc[term]
    return {"method": name, "estimate": m.params[term], "se": m.bse[term],
            "ci_low": ci[0], "ci_high": ci[1], "p": m.pvalues[term]}


def dml(s: pd.DataFrame, y: str, seed: int = 42, n_folds: int = 5, n_rep: int = 3):
    """Double machine learning (partially linear model), standard errors clustered by match."""
    data = DoubleMLData(s[[y, "D", "match_id"] + x_cols(s)].reset_index(drop=True),
                        y_col=y, d_cols="D", x_cols=x_cols(s), cluster_cols="match_id")
    ml_l = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3,
                                         min_samples_leaf=100, random_state=seed)
    ml_m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=100, random_state=seed)
    np.random.seed(seed)
    model = DoubleMLPLR(data, ml_l=ml_l, ml_m=ml_m, n_folds=n_folds, n_rep=n_rep)
    model.fit()
    ci = model.confint().iloc[0]
    row = {"method": "Double machine learning", "estimate": float(model.coef[0]),
           "se": float(model.se[0]), "ci_low": float(ci.iloc[0]), "ci_high": float(ci.iloc[1]),
           "p": float(model.pval[0])}
    return row, model


def sensitivity(model: DoubleMLPLR, cf_y: float = 0.03, cf_d: float = 0.03) -> dict:
    """How strong would an unmeasured confounder have to be to change the conclusion?

    cf_y / cf_d: share of remaining variation in the outcome / treatment explained by
    the hypothetical confounder (Chernozhukov et al., 2022).
    """
    model.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d)
    p = model.sensitivity_params
    return {"cf_y": cf_y, "cf_d": cf_d,
            "theta_lower": float(np.ravel(p["theta"]["lower"])[0]),
            "theta_upper": float(np.ravel(p["theta"]["upper"])[0]),
            "ci_lower": float(np.ravel(p["ci"]["lower"])[0]),
            "ci_upper": float(np.ravel(p["ci"]["upper"])[0]),
            "robustness_value": float(np.ravel(p["rv"])[0]),
            "robustness_value_ci": float(np.ravel(p["rva"])[0])}
