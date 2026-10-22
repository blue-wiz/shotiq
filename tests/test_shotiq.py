import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotiq.data import shots_from_events
from shotiq.features import blockers, build_features, distance, shot_angle, FEATURES
from shotiq.model import TEST_LEAGUE, split


def test_penalty_spot_distance_is_12_yards():
    assert distance(108, 40) == pytest.approx(12.0)


def test_angle_is_symmetric_and_shrinks_with_distance():
    assert shot_angle(110, 30) == pytest.approx(shot_angle(110, 50))
    assert shot_angle(100, 40) > shot_angle(80, 40)


def test_blocker_inside_triangle_is_counted_and_teammates_are_not():
    ff = [{"x": 110, "y": 40, "teammate": False, "keeper": False},
          {"x": 110, "y": 40.5, "teammate": True, "keeper": False},
          {"x": 119, "y": 40, "teammate": False, "keeper": True},
          {"x": 110, "y": 10, "teammate": False, "keeper": False}]
    assert blockers(100, 40, ff) == 1


def _event(outcome="Goal", shot_type="Open Play"):
    return {"type": {"name": "Shot"}, "id": "e1", "period": 1, "minute": 10,
            "team": {"name": "A"}, "player": {"name": "P"}, "location": [108, 40],
            "play_pattern": {"name": "Regular Play"},
            "shot": {"body_part": {"name": "Right Foot"}, "type": {"name": shot_type},
                     "technique": {"name": "Normal"}, "outcome": {"name": outcome},
                     "statsbomb_xg": 0.3, "freeze_frame": []}}


def test_event_parsing_and_penalties_are_excluded():
    rows = shots_from_events([_event(), _event("Saved", "Penalty"), {"type": {"name": "Pass"}}], 1, "X")
    assert len(rows) == 2 and rows[0]["goal"] == 1
    feats = build_features(pd.DataFrame(rows))
    assert len(feats) == 1
    assert set(FEATURES).issubset(feats.columns)


DATA = ROOT / "data" / "raw" / "shots_raw.parquet"


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/download_data.py first")
def test_held_out_league_never_in_training():
    df = build_features(pd.read_parquet(DATA))
    train, test = split(df)
    assert TEST_LEAGUE not in set(train.competition)
    assert set(test.competition) == {TEST_LEAGUE}
    assert not set(train.match_id) & set(test.match_id)


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/download_data.py first")
def test_real_data_sanity():
    df = build_features(pd.read_parquet(DATA))
    assert len(df) > 30_000
    assert 0.07 < df.goal.mean() < 0.13          # typical non-penalty conversion rate
    assert df.distance.between(0, 130).all()
    assert df.angle.between(0, np.pi).all()
