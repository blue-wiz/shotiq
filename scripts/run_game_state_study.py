"""Research extension: does the match situation change the chances a team creates?

Run after download_data.py, run_analysis.py and download_game_state.py.
Writes reports/game_state/ (tables, summary.json) and reports/figures/gs_*.png.
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotiq import causal as CS
from shotiq.game_state import build_analysis_table, exposure, team_strength

OUT = ROOT / "reports" / "game_state"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "reports" / "figures"
NAVY, TERRA, TEAL, GREY = "#1f3a5f", "#c8553d", "#2a9d8f", "#8a8f98"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})

results = pd.read_csv(ROOT / "data" / "raw" / "match_results.csv")
states = pd.read_parquet(ROOT / "data" / "raw" / "game_states.parquet")
shots = pd.read_csv(ROOT / "data" / "processed" / "shots_with_xg.csv")
df = build_analysis_table(shots, states, results)
summary = {"n_shots": int(len(df)), "n_matches": int(df.match_id.nunique()),
           "state_share": df.state.value_counts(normalize=True).round(4).to_dict(),
           "mean_xg_by_state": df.groupby("state").xg.mean().round(4).to_dict()}

# ---------------------------------------------------------------- 1. Main estimates
main, models = [], {}
for t in ["trailing", "leading"]:
    s = CS.contrast_sample(df, t)
    base = s[s.D == 0].xg.mean()
    for row in [CS.naive(s, "xg"), CS.ols(s, "xg")]:
        main.append({"contrast": f"{t} vs level", **row})
    row, m = CS.dml(s, "xg", n_rep=5)
    models[t] = m
    main.append({"contrast": f"{t} vs level", **row})
    for r in main[-3:]:
        r["level_mean_xg"] = base
        r["pct_of_level"] = 100 * r["estimate"] / base
main = pd.DataFrame(main)
main.round(5).to_csv(OUT / "main_estimates.csv", index=False)
print(main.round(4).to_string())

# ---------------------------------------------------------------- 2. Sensitivity to unmeasured confounding
sens = []
for t, m in models.items():
    for cf in [0.01, 0.03, 0.05]:
        sens.append({"contrast": f"{t} vs level", **CS.sensitivity(m, cf, cf)})
sens = pd.DataFrame(sens)
sens.round(5).to_csv(OUT / "sensitivity.csv", index=False)
# Benchmark: how much of the treatment and outcome does team strength explain on its own?
bench = {}
for t, m in models.items():
    try:
        b = m.sensitivity_benchmark(benchmarking_set=["team_ppg", "team_gdpg"])
        bench[t] = {k: float(np.ravel(v)[0]) for k, v in b.iloc[0].items()}
    except Exception as e:  # benchmarking is optional
        bench[t] = {"error": str(e)}
print(sens.round(4).to_string())

# ---------------------------------------------------------------- 3. Robustness: other outcomes
rob = []
for t in ["trailing", "leading"]:
    s = CS.contrast_sample(df, t)
    for y, label in [("statsbomb_xg", "StatsBomb's xG instead of ShotIQ's"),
                     ("goal", "Actual goals (1/0) instead of xG")]:
        row, _ = CS.dml(s, y, n_rep=3)
        rob.append({"contrast": f"{t} vs level", "outcome": label, **row})
rob = pd.DataFrame(rob)
rob.round(5).to_csv(OUT / "robustness.csv", index=False)

# ---------------------------------------------------------------- 4. Mechanisms: how does shot quality change?
mech_vars = {"distance": "Distance to goal (yards)", "n_blockers": "Defenders blocking the goal",
             "one_on_one": "One-on-one with keeper (share)", "from_counter": "From a counter-attack (share)",
             "outside_box": "From outside the box (share)", "from_set_piece": "Possession began at a set piece or throw-in (share)"}
mech = []
for t in ["trailing", "leading"]:
    s = CS.contrast_sample(df, t)
    for y, label in mech_vars.items():
        row, _ = CS.dml(s, y, n_rep=2)
        mech.append({"contrast": f"{t} vs level", "variable": y, "label": label,
                     "level_mean": s[s.D == 0][y].mean(), **row})
mech = pd.DataFrame(mech)
mech.round(5).to_csv(OUT / "mechanisms.csv", index=False)

# ---------------------------------------------------------------- 5. Heterogeneity by league
het = []
for t in ["trailing", "leading"]:
    for lg in CS.LEAGUES:
        sub = df[df.competition == lg]
        if len(sub) < 2000:      # Bundesliga has only a handful of matches in the open data
            continue
        s = CS.contrast_sample(sub, t)
        s = s[[c for c in s.columns if not c.startswith("lg")]]
        row, _ = CS.dml(s, "xg", n_rep=2)
        het.append({"contrast": f"{t} vs level", "league": lg.replace(" 2015/16", ""),
                    "n_shots": len(s), **row})
het = pd.DataFrame(het)
het.round(5).to_csv(OUT / "by_league.csv", index=False)

# ---------------------------------------------------------------- 6. Shot volume: shots and xG per 90 minutes
exp_ = exposure(results)
agg = df.groupby(["match_id", "team", "state"]).agg(shots=("xg", "size"), xg=("xg", "sum")).reset_index()
rate = exp_.merge(agg, on=["match_id", "team", "state"], how="left").fillna({"shots": 0, "xg": 0})
rate = rate[rate.minutes >= 1]
st = team_strength(results)
rate = rate.merge(st.rename(columns={"ppg_loo": "team_ppg", "gdpg_loo": "team_gdpg"}), on=["match_id", "team"])
opp = results.melt(id_vars="match_id", value_vars=["home", "away"], value_name="team")
opp = opp.merge(opp, on="match_id", suffixes=("", "_o"))
opp = opp[opp.team != opp.team_o][["match_id", "team", "team_o", "variable"]]
rate = rate.merge(opp, on=["match_id", "team"])
rate["is_home"] = (rate.variable == "home").astype(int)
rate = rate.merge(st.rename(columns={"team": "team_o", "ppg_loo": "opp_ppg", "gdpg_loo": "opp_gdpg"}),
                  on=["match_id", "team_o"])
rate = rate.merge(df[["match_id", "competition"]].drop_duplicates(), on="match_id")
rate["log_min"] = np.log(rate.minutes)
pois = smf.glm("shots ~ C(state, Treatment('level')) + team_ppg + opp_ppg + is_home + C(competition)",
               data=rate, family=__import__("statsmodels.api").api.families.Poisson(),
               offset=rate.log_min).fit(cov_type="cluster", cov_kwds={"groups": rate.match_id})
raw_rate = rate.groupby("state")[["shots", "xg", "minutes"]].sum()
raw_rate["shots_per90"] = 90 * raw_rate.shots / raw_rate.minutes
raw_rate["xg_per90"] = 90 * raw_rate.xg / raw_rate.minutes
raw_rate.round(3).to_csv(OUT / "rates_by_state.csv")
rr = {}
for t in ["trailing", "leading"]:
    k = f"C(state, Treatment('level'))[T.{t}]"
    lo, hi = pois.conf_int().loc[k]
    rr[t] = {"rate_ratio": float(np.exp(pois.params[k])), "ci_low": float(np.exp(lo)),
             "ci_high": float(np.exp(hi)), "p": float(pois.pvalues[k])}
print(raw_rate.round(2)); print(rr)

# ---------------------------------------------------------------- Figures
# Fig A: estimates by method
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
for ax, t, col in zip(axes, ["trailing", "leading"], [TERRA, TEAL]):
    sub = main[main.contrast == f"{t} vs level"].iloc[::-1]
    y = np.arange(len(sub))
    ax.errorbar(sub.estimate, y, xerr=[sub.estimate - sub.ci_low, sub.ci_high - sub.estimate],
                fmt="o", color=col, ecolor=col, capsize=4, ms=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y, sub.method); ax.set_ylim(-0.6, len(sub) - 0.4)
    ax.set_title(f"{t.capitalize()} vs level", pad=10)
    for yy, (e, hi, pc) in enumerate(zip(sub.estimate, sub.ci_high, sub["pct_of_level"])):
        ax.text(hi + 0.0008, yy, f"{e:+.3f} ({pc:+.0f}%)", va="center", fontsize=8)
    ax.set_xlim(-0.007, 0.034); ax.set_xticks([0, 0.01, 0.02, 0.03])
    ax.set_xlabel("Change in xG per shot (95% CI)")
fig.suptitle("Effect of the match situation on shot quality", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "gs_main_estimates.png", dpi=160); plt.close(fig)

# Fig B: mechanisms (leading vs level), standardised as % of the level mean
lm = mech[mech.contrast == "leading vs level"].copy()
lm["pct"] = 100 * lm.estimate / lm.level_mean
lm["label"] = lm["label"].str.replace(" or throw-in", "/throw-in")
lm["lo"] = 100 * lm.ci_low / lm.level_mean; lm["hi"] = 100 * lm.ci_high / lm.level_mean
lm = lm.sort_values("pct")
fig, ax = plt.subplots(figsize=(8, 3.8))
y = np.arange(len(lm))
ax.errorbar(lm.pct, y, xerr=[lm.pct - lm.lo, lm.hi - lm.pct], fmt="o", color=TEAL, capsize=4)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y, lm.label)
ax.set_xlabel("Change when leading vs level, % of the level average (95% CI)")
ax.set_title("How leading teams' chances differ")
fig.tight_layout(); fig.savefig(FIG / "gs_mechanisms.png", dpi=160); plt.close(fig)

# Fig C: shots and xG per 90 by state
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
order = ["trailing", "level", "leading"]; cols = [TERRA, GREY, TEAL]
for ax, col, lab in zip(axes, ["shots_per90", "xg_per90"], ["Shots per 90 minutes", "xG per 90 minutes"]):
    v = raw_rate.loc[order, col]
    ax.bar([o.capitalize() for o in order], v, color=cols)
    for i, x in enumerate(v):
        ax.text(i, x * 1.01, f"{x:.2f}", ha="center", fontsize=9)
    ax.set_title(lab)
fig.suptitle("Shot volume by match situation (raw, all teams)", fontweight="bold")
fig.tight_layout(); fig.savefig(FIG / "gs_rates.png", dpi=160); plt.close(fig)

# Fig D: by league
fig, ax = plt.subplots(figsize=(8, 3.8))
lab = het.contrast.str.split(" ").str[0].str.capitalize() + " · " + het.league
for i, r in enumerate(het.itertuples()):
    c = TERRA if r.contrast.startswith("trailing") else TEAL
    ax.errorbar(r.estimate, i, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt="o",
                color=c, ecolor=c, capsize=3)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(np.arange(len(het)), lab)
ax.set_xlabel("Change in xG per shot vs level (DML, 95% CI)"); ax.set_title("Results by league")
fig.tight_layout(); fig.savefig(FIG / "gs_by_league.png", dpi=160); plt.close(fig)

summary.update({"main": main.round(5).to_dict(orient="records"),
                "sensitivity": sens.round(5).to_dict(orient="records"), "benchmark": bench,
                "robustness": rob.round(5).to_dict(orient="records"),
                "mechanisms": mech.round(5).to_dict(orient="records"),
                "by_league": het.round(5).to_dict(orient="records"),
                "rates": raw_rate.round(4).reset_index().to_dict(orient="records"), "rate_ratios": rr})
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print("done")
