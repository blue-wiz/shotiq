"""Feature engineering for shots.

StatsBomb pitch: 120 x 80 yards, attacking towards x = 120, goal mouth from
y = 36 to y = 44 (centre y = 40).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GOAL_X, GOAL_Y = 120.0, 40.0
POST_LOW, POST_HIGH = 36.0, 44.0

NUMERIC = ["distance", "angle", "n_blockers", "keeper_dist_line", "keeper_off_centre",
           "n_defenders_close"]
BINARY = ["is_header", "first_time", "under_pressure", "one_on_one", "from_set_piece",
          "from_counter", "direct_free_kick", "volley"]
FEATURES = NUMERIC + BINARY


def distance(x, y):
    return np.hypot(GOAL_X - np.asarray(x, float), GOAL_Y - np.asarray(y, float))


def shot_angle(x, y):
    """Angle (radians) subtended by the goal mouth from the shot location."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    a1 = np.arctan2(POST_LOW - y, GOAL_X - x)
    a2 = np.arctan2(POST_HIGH - y, GOAL_X - x)
    return np.abs(a2 - a1)


def _in_triangle(px, py, ax, ay, bx, by, cx, cy):
    d1 = (px - bx) * (ay - by) - (ax - bx) * (py - by)
    d2 = (px - cx) * (by - cy) - (bx - cx) * (py - cy)
    d3 = (px - ax) * (cy - ay) - (cx - ax) * (py - ay)
    neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (neg and pos)


def blockers(x, y, freeze_frame) -> int:
    """Opposition outfield players inside the triangle between ball and the two posts."""
    if freeze_frame is None:
        return 0
    n = 0
    for p in freeze_frame:
        if p["teammate"] or p["keeper"]:
            continue
        if _in_triangle(p["x"], p["y"], x, y, GOAL_X, POST_LOW, GOAL_X, POST_HIGH):
            n += 1
    return n


def close_defenders(x, y, freeze_frame, radius: float = 3.0) -> int:
    if freeze_frame is None:
        return 0
    return sum(1 for p in freeze_frame
               if not p["teammate"] and not p["keeper"]
               and np.hypot(p["x"] - x, p["y"] - y) <= radius)


def build_features(shots: pd.DataFrame) -> pd.DataFrame:
    """Return open-play + set-piece shots (no penalties) with model features added."""
    df = shots[shots["shot_type"] != "Penalty"].copy()
    df["distance"] = distance(df.x, df.y)
    df["angle"] = shot_angle(df.x, df.y)
    df["n_blockers"] = [blockers(x, y, ff) for x, y, ff in zip(df.x, df.y, df.freeze_frame)]
    df["n_defenders_close"] = [close_defenders(x, y, ff) for x, y, ff in zip(df.x, df.y, df.freeze_frame)]
    # Keeper position: distance off the goal line and sideways from centre. Missing keeper
    # (empty net) -> keeper far away, which the tree model learns as a big chance.
    df["keeper_dist_line"] = (GOAL_X - df["keeper_x"]).fillna(30.0)
    df["keeper_off_centre"] = (df["keeper_y"] - GOAL_Y).abs().fillna(20.0)
    df["is_header"] = (df.body_part == "Head").astype(int)
    df["from_set_piece"] = df.play_pattern.isin(["From Corner", "From Free Kick", "From Throw In"]).astype(int)
    df["from_counter"] = (df.play_pattern == "From Counter").astype(int)
    df["direct_free_kick"] = (df.shot_type == "Free Kick").astype(int)
    df["volley"] = df.technique.isin(["Volley", "Half Volley", "Overhead Kick"]).astype(int)
    for c in ["first_time", "under_pressure", "one_on_one"]:
        df[c] = df[c].astype(int)
    return df.reset_index(drop=True)
