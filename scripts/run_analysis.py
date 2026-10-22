"""Step 2: build features, train models, evaluate on the held-out league, make figures."""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotiq.features import build_features, FEATURES
from shotiq.model import train_and_evaluate, calibration_table, TEST_LEAGUE

FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
NAVY, TERRA, TEAL, GREY = "#1f3a5f", "#c8553d", "#2a9d8f", "#8a8f98"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})

shots = pd.read_parquet(ROOT / "data" / "raw" / "shots_raw.parquet")
df = build_features(shots)
results, fitted, preds, train, test = train_and_evaluate(df)
best = "Full features (gradient boosting)"
test = test.assign(xg=preds[best])
cols, model = fitted[best]
# Out-of-fold xG for every shot (5-fold, grouped by match) so player finishing is judged
# on predictions the model made without seeing that shot.
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
df["xg"] = np.nan
for tr, te in GroupKFold(n_splits=5).split(df, df.goal, df.match_id):
    m = clone(model).fit(df.iloc[tr][cols], df.iloc[tr].goal)
    df.loc[df.index[te], "xg"] = m.predict_proba(df.iloc[te][cols])[:, 1]
df.drop(columns=["freeze_frame"]).to_csv(ROOT / "data" / "processed" / "shots_with_xg.csv", index=False)

print(results.round(4).to_string())
results.round(4).to_csv(ROOT / "reports" / "model_comparison.csv")

# --- Figure 1: model comparison (log loss, lower is better)
fig, ax = plt.subplots(figsize=(8, 3.6))
order = results.sort_values("log_loss", ascending=False)
colors = [TERRA if m == best else (TEAL if "StatsBomb" in m else GREY) for m in order.index]
ax.barh(order.index, order.log_loss, color=colors)
for i, v in enumerate(order.log_loss):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
ax.set_xlim(order.log_loss.min() * 0.9, order.log_loss.max() * 1.04)
ax.set_xlabel("Log loss on unseen league (La Liga 2015/16) — lower is better")
ax.set_title("Model comparison")
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=160); fig.savefig(FIG / "model_comparison.svg"); plt.close(fig)

# --- Figure 2: calibration
cal = calibration_table(test.goal, test.xg)
cal_sb = calibration_table(test.goal, test.statsbomb_xg)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 0.6], [0, 0.6], ls="--", color=GREY, lw=1, label="Perfect calibration")
ax.plot(cal_sb.mean_predicted, cal_sb.observed_rate, "s-", color=TEAL, label="StatsBomb xG")
ax.plot(cal.mean_predicted, cal.observed_rate, "o-", color=TERRA, label="ShotIQ xG")
ax.set_xlabel("Predicted goal probability (decile mean)"); ax.set_ylabel("Observed goal rate")
ax.set_title("Calibration on La Liga 2015/16"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=160); fig.savefig(FIG / "calibration.svg"); plt.close(fig)

# --- Figure 3: xG surface (grid, average open-play right-foot shot, 1 blocker)
xs = np.linspace(80, 119.5, 80); ys = np.linspace(18, 62, 90)
gx, gy = np.meshgrid(xs, ys)
from shotiq.features import distance, shot_angle
grid = pd.DataFrame({"distance": distance(gx.ravel(), gy.ravel()), "angle": shot_angle(gx.ravel(), gy.ravel())})
for c in FEATURES:
    if c not in grid:
        grid[c] = 0
grid["n_blockers"] = 1; grid["keeper_dist_line"] = 2.0; grid["keeper_off_centre"] = 0.5
surface = model.predict_proba(grid[cols])[:, 1].reshape(gx.shape)
fig, ax = plt.subplots(figsize=(6.4, 5))
im = ax.contourf(gy, gx, surface, levels=[0,.02,.05,.1,.15,.2,.3,.4,.5,.6,.8,1.0], cmap="YlOrRd")
ax.plot([36, 44], [120, 120], color="k", lw=4)
ax.plot([18, 18, 62, 62], [120, 102, 102, 120], color="k", lw=1)
ax.plot([30, 30, 50, 50], [120, 114, 114, 120], color="k", lw=1)
ax.set_xlim(18, 62); ax.set_ylim(80, 121); ax.set_aspect("equal"); ax.axis("off")
fig.colorbar(im, ax=ax, label="xG", shrink=0.8)
ax.set_title("Where goals come from: modelled xG by shot location")
fig.tight_layout(); fig.savefig(FIG / "xg_surface.png", dpi=160); fig.savefig(FIG / "xg_surface.svg"); plt.close(fig)

# --- Figure 4: finishing — goals minus xG (all leagues, non-penalty, min 60 shots)
p = df.groupby("player").agg(shots=("goal", "size"), goals=("goal", "sum"), xg=("xg", "sum"),
                             var=("xg", lambda s: float(np.sum(s * (1 - s)))))
p = p[p.shots >= 60].copy()
p["g_minus_xg"] = p.goals - p.xg
p["z"] = p.g_minus_xg / np.sqrt(p["var"])
p.sort_values("g_minus_xg", ascending=False).round(2).to_csv(ROOT / "reports" / "player_finishing.csv")
top = pd.concat([p.nlargest(8, "g_minus_xg"), p.nsmallest(5, "g_minus_xg")]).sort_values("g_minus_xg")
fig, ax = plt.subplots(figsize=(8, 5.2))
ax.barh(top.index, top.g_minus_xg, color=[TEAL if v > 0 else TERRA for v in top.g_minus_xg])
ax.axvline(0, color="k", lw=0.8)
ax.set_xlim(top.g_minus_xg.min() - 2.5, top.g_minus_xg.max() + 2.5)
for i, (v, z) in enumerate(zip(top.g_minus_xg, top.z)):
    ax.text(v + (0.3 if v > 0 else -0.3), i, f"z={z:+.1f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
ax.set_xlabel("Non-penalty goals minus xG, 2015/16 (min 60 shots)")
ax.set_title("Finishing over and under expectation")
fig.tight_layout(); fig.savefig(FIG / "finishing.png", dpi=160); fig.savefig(FIG / "finishing.svg"); plt.close(fig)

summary = {
    "n_shots_total": int(len(shots)), "n_shots_model": int(len(df)),
    "n_matches": int(shots.match_id.nunique()), "test_league": TEST_LEAGUE,
    "n_test_shots": int(len(test)), "conversion_rate": float(df.goal.mean()),
    "results": results.round(4).reset_index().to_dict(orient="records"),
    "top_finishers": p.nlargest(5, "g_minus_xg")[["shots", "goals", "xg", "g_minus_xg", "z"]].round(2).reset_index().to_dict(orient="records"),
    "feature_effect_header_vs_foot": None,
}
(ROOT / "reports" / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary["top_finishers"], indent=1))
print("shots model:", len(df), "test:", len(test), "conv", round(df.goal.mean(),4))
