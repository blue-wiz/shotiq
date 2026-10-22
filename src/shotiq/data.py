"""Download StatsBomb open data and extract one row per shot.

Data: StatsBomb Open Data (https://github.com/statsbomb/open-data), free for
non-commercial use with attribution to StatsBomb.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# 2015/16 is the one season where StatsBomb released all five big European leagues.
COMPETITIONS = {
    "Premier League 2015/16": (2, 27),
    "1. Bundesliga 2015/16": (9, 27),
    "Serie A 2015/16": (12, 27),
    "Ligue 1 2015/16": (7, 27),
    "La Liga 2015/16": (11, 27),
}


def _get_json(url: str, retries: int = 5):
    """Fetch JSON with simple exponential back-off (GitHub raw drops some connections)."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def list_matches(competition_id: int, season_id: int) -> list[dict]:
    return _get_json(f"{BASE}/matches/{competition_id}/{season_id}.json")


def shots_from_events(events: list[dict], match_id: int, competition: str) -> list[dict]:
    """Turn a match's event list into flat shot records (penalties excluded later)."""
    rows = []
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        s = e["shot"]
        x, y = e["location"][:2]
        ff = s.get("freeze_frame", []) or []
        defenders = [p for p in ff if not p["teammate"] and p.get("position", {}).get("name") != "Goalkeeper"]
        keeper = [p for p in ff if not p["teammate"] and p.get("position", {}).get("name") == "Goalkeeper"]
        rows.append({
            "match_id": match_id,
            "competition": competition,
            "event_id": e["id"],
            "period": e["period"],
            "minute": e["minute"],
            "team": e["team"]["name"],
            "player": e["player"]["name"],
            "x": x,
            "y": y,
            "body_part": s["body_part"]["name"],
            "shot_type": s["type"]["name"],
            "technique": s["technique"]["name"],
            "play_pattern": e["play_pattern"]["name"],
            "under_pressure": bool(e.get("under_pressure", False)),
            "first_time": bool(s.get("first_time", False)),
            "one_on_one": bool(s.get("one_on_one", False)),
            "freeze_frame": [
                {"x": p["location"][0], "y": p["location"][1], "teammate": p["teammate"],
                 "keeper": p.get("position", {}).get("name") == "Goalkeeper"}
                for p in ff
            ],
            "n_defenders_ff": len(defenders),
            "keeper_x": keeper[0]["location"][0] if keeper else None,
            "keeper_y": keeper[0]["location"][1] if keeper else None,
            "statsbomb_xg": s.get("statsbomb_xg"),
            "outcome": s["outcome"]["name"],
            "goal": int(s["outcome"]["name"] == "Goal"),
        })
    return rows


def _match_shots(args):
    match_id, competition = args
    events = _get_json(f"{BASE}/events/{match_id}.json")
    return shots_from_events(events, match_id, competition)


def download_shots(out_dir: Path, workers: int = 8) -> pd.DataFrame:
    """Download every match in COMPETITIONS and save shots (and match list) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs, matches = [], []
    for name, (cid, sid) in COMPETITIONS.items():
        for m in list_matches(cid, sid):
            jobs.append((m["match_id"], name))
            matches.append({"match_id": m["match_id"], "competition": name,
                            "match_date": m["match_date"],
                            "home": m["home_team"]["home_team_name"],
                            "away": m["away_team"]["away_team_name"],
                            "home_score": m["home_score"], "away_score": m["away_score"]})
    rows = []
    with ThreadPoolExecutor(workers) as pool:
        for r in pool.map(_match_shots, jobs):
            rows.extend(r)
    shots = pd.DataFrame(rows)
    pd.DataFrame(matches).to_csv(out_dir / "matches.csv", index=False)
    shots.to_parquet(out_dir / "shots_raw.parquet", index=False)
    return shots
