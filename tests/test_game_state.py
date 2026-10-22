import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotiq.game_state import exposure, states_from_events, team_strength


def ev(i, typ, team_id, minute, outcome=None, card=None, period=1):
    e = {"index": i, "type": {"name": typ}, "team": {"id": team_id, "name": {1: "Home", 2: "Away"}[team_id]},
         "minute": minute, "second": 0, "period": period, "id": f"e{i}"}
    if typ == "Shot":
        e["shot"] = {"outcome": {"name": outcome}}
    if card:
        e["foul_committed"] = {"card": {"name": card}}
    return e


EVENTS = [ev(1, "Shot", 1, 10, "Goal"),        # Home score: shot sees 0-0
          ev(2, "Shot", 2, 20, "Saved"),       # Away shot: trailing 0-1
          ev(3, "Foul Committed", 2, 30, card="Red Card"),
          ev(4, "Own Goal For", 2, 40),        # own goal levels it 1-1
          ev(5, "Shot", 1, 50, "Off T"),       # Home shot: level, opponent down to 10
          ev(6, "Pass", 1, 93, period=2)]


def test_score_is_recorded_before_each_shot():
    rows, result = states_from_events(EVENTS, 1, home_id=1, away_id=2)
    assert [(r["goals_for"], r["goals_against"]) for r in rows] == [(0, 0), (0, 1), (1, 1)]
    assert rows[2]["reds_against"] == 1 and rows[2]["reds_for"] == 0
    assert (result["home_score"], result["away_score"]) == (1, 1)


def test_exposure_splits_the_match_correctly():
    _, result = states_from_events(EVENTS, 1, 1, 2)
    e = exposure(pd.DataFrame([result])).set_index(["team", "state"]).minutes
    assert e[("Home", "leading")] == pytest.approx(30)   # 10' to 40'
    assert e[("Away", "trailing")] == pytest.approx(30)
    assert e[("Home", "level")] == pytest.approx(10 + 53)  # 0-10' and 40'-93'
    assert e.groupby(level="team").sum().tolist() == pytest.approx([93, 93])


def test_team_strength_leaves_out_the_current_match():
    m = pd.DataFrame({"match_id": [1, 2], "home": ["A", "A"], "away": ["B", "C"],
                      "home_score": [3, 0], "away_score": [0, 0]})
    st = team_strength(m).set_index(["match_id", "team"])
    assert st.loc[(1, "A"), "ppg_loo"] == 1.0   # only the 0-0 draw counts
    assert st.loc[(2, "A"), "ppg_loo"] == 3.0   # only the 3-0 win counts


RES = ROOT / "data" / "raw" / "match_results.csv"


@pytest.mark.skipif(not RES.exists(), reason="run scripts/download_game_state.py first")
def test_rebuilt_final_scores_match_official_results():
    r = pd.read_csv(RES)
    m = pd.read_csv(ROOT / "data" / "raw" / "matches.csv")
    j = r.merge(m, on="match_id", suffixes=("", "_official"))
    assert len(j) == len(m)
    assert (j.home_score == j.home_score_official).all() and (j.away_score == j.away_score_official).all()
