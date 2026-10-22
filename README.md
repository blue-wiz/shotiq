# ShotIQ — an expected-goals (xG) model on StatsBomb data

**How likely was that shot to be a goal?** ShotIQ answers that question for 38,318 non-penalty shots from 1,551 matches across the 2015/16 big-five European leagues. It is then tested on a league it has never seen.

![Modelled xG by shot location](reports/figures/xg_surface.png)

## Results

The model was trained on the Premier League, Serie A, Ligue 1 and Bundesliga, and tested on **La Liga 2015/16** (9,071 shots it never saw).

| Model | Log loss ↓ | Brier ↓ | ROC AUC ↑ | Predicted goals (actual 945) |
|---|---|---|---|---|
| Constant (league average) | 0.335 | 0.094 | 0.500 | 839 |
| Distance + angle (logistic) | 0.291 | 0.084 | 0.762 | 881 |
| Full features (logistic) | 0.264 | 0.077 | 0.820 | 937 |
| **Full features (gradient boosting)** | **0.262** | **0.076** | **0.824** | **929** |
| StatsBomb's own xG (benchmark) | 0.256 | 0.074 | 0.833 | 904 |

- **Context beats location alone.** Adding freeze-frame features (where defenders and the goalkeeper were at the moment of the shot) and shot context to distance and angle cuts log loss by 10%.
- **The model gets within 3% of StatsBomb's commercial xG** (log loss 0.262 vs 0.256) on a league it never saw.
- **It is well calibrated.** Predicted 929 goals versus 945 actual in La Liga, and the calibration curve sits close to the diagonal.
- **Finishing skill is visible but rare.** Using out-of-fold xG, Luis Suárez scored 12.4 more non-penalty goals than expected (z = +3.1). Most players sit within normal random variation.

![Model comparison](reports/figures/model_comparison.png)
![Calibration](reports/figures/calibration.png)
![Finishing](reports/figures/finishing.png)

## How it works

1. **Download** (`scripts/download_data.py`): fetches every 2015/16 big-five league match from StatsBomb Open Data and keeps one row per shot, including the freeze frame.
2. **Features** (`src/shotiq/features.py`):
   - distance and angle to goal
   - number of defenders in the triangle between the ball and the posts
   - defenders within 3 yards
   - goalkeeper position
   - header, first-time, volley, under pressure, one-on-one, set piece, counter-attack and direct free kick
3. **Models** (`src/shotiq/model.py`): a constant baseline, a two-feature logistic regression, a full logistic regression and a gradient-boosted tree model. All are evaluated on a held-out league.
4. **Analysis** (`scripts/run_analysis.py`):
   - model comparison and calibration
   - xG surface
   - player finishing, using out-of-fold xG from 5-fold cross-validation grouped by match, so no shot is scored by a model that trained on it

## Why each tool

| Tool | Why |
|---|---|
| **StatsBomb Open Data** | Professional-grade event data, free for non-commercial use. It is the only free source with freeze frames (player positions at the moment of each shot). |
| **Python + pandas** | Flattening nested JSON events into a tidy one-row-per-shot table. |
| **scikit-learn** | Logistic regression is the transparent, explainable baseline. `HistGradientBoostingClassifier` captures interactions (for example, a header from close range) without manual feature crosses. |
| **Log loss, Brier and calibration** | xG is a *probability*, so it must be judged on probability quality, not accuracy. A model that always says "no goal" is 90% accurate and useless. |
| **Held-out league + GroupKFold** | Tests the model on unseen data and stops shots from the same match leaking between train and test. |
| **pytest** | Checks the geometry (the penalty spot is 12 yards), the parsing, the exclusion of penalties and the absence of leakage. |
| **matplotlib** | Static, reproducible charts that render on GitHub. |

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py    # ~2 minutes, ~38k shots
python scripts/run_analysis.py     # trains models, writes reports/ and figures
pytest -q                          # 6 tests
```

## Project layout

```
src/shotiq/      data.py (download + parse), features.py, model.py
scripts/         download_data.py, run_analysis.py
tests/           test_shotiq.py
reports/         model_comparison.csv, player_finishing.csv, summary.json, figures/
data/processed/  shots_with_xg.csv (raw downloads are git-ignored)
```

## Limitations

- The data covers a single season (2015/16).
- There is no information on shot height or the player's weaker foot.
- The freeze frame only covers players visible on the broadcast.
- Player finishing results are noisy, so treat z-scores below about 2 as luck rather than skill.

## Data and credit

Data provided by **StatsBomb** (now Hudl StatsBomb) via [StatsBomb Open Data](https://github.com/statsbomb/open-data), used under its non-commercial licence. All analysis and modelling in this repository are my own.
