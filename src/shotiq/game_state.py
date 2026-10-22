"""Reconstruct the match situation (score, red cards) at the moment of every shot.

For each match we read the full StatsBomb event stream in order and keep a running
score and a running count of red cards for both teams. Every shot then gets the state
*before* it was taken, so the shot cannot influence its own treatment.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BASE, COMPETITIONS, _get_json, list_matches

RED = {"Red Card", "Second Yellow"}


def _card(e: dict) -> str | None:
    for key in ("foul_committed", "bad_behaviour"):
        c = e.get(key, {}).get("card", {}).get("name")
        if c:
            return c
    return None


def states_from_events(events: list[dict], match_id: int, home_id: int, away_id: int) -> list[dict]:
    """One row per shot with the score and red cards *before* that shot, plus the final score.

    Teams are matched by StatsBomb team id, because display names can differ between
    the match list and the event files (e.g. "Olympique Marseille" vs "Marseille").
    """
    names = {}
    for e in events:
        if "team" in e:
            names[e["team"]["id"]] = e["team"]["name"]
    home, away = names[home_id], names[away_id]
    goals = {home: 0, away: 0}
    reds = {home: 0, away: 0}
    rows, timeline = [], []          # timeline: (second, scoring team) for every goal
    end_second = 0
    for e in sorted(events, key=lambda ev: ev["index"]):
        t = e.get("type", {}).get("name")
        team = e.get("team", {}).get("name")
        if e.get("period", 1) <= 2:
            end_second = max(end_second, e["minute"] * 60 + e["second"])
        if t == "Shot":
            opp = away if team == home else home
            rows.append({
                "match_id": match_id, "event_id": e["id"], "team": team, "opponent": opp,
                "is_home": int(team == home),
                "goals_for": goals[team], "goals_against": goals[opp],
                "reds_for": reds[team], "reds_against": reds[opp],
                "second": e["minute"] * 60 + e["second"],
            })
            if e["shot"]["outcome"]["name"] == "Goal":
                goals[team] += 1
                timeline.append((e["minute"] * 60 + e["second"], team))
        elif t == "Own Goal For":
            goals[team] += 1
            timeline.append((e["minute"] * 60 + e["second"], team))
        if _card(e) in RED and team in reds:
            reds[team] += 1
    result = {"match_id": match_id, "home": home, "away": away,
              "home_score": goals[home], "away_score": goals[away], "end_second": end_second,
              "goal_timeline": ";".join(f"{s}|{tm}" for s, tm in timeline)}
    return rows, result


def _job(args):
    match_id, home_id, away_id = args
    return states_from_events(_get_json(f"{BASE}/events/{match_id}.json"), match_id, home_id, away_id)


def download_states(out: Path, workers: int = 8) -> pd.DataFrame:
    jobs = [(m["match_id"], m["home_team"]["home_team_id"], m["away_team"]["away_team_id"])
            for cid, sid in COMPETITIONS.values() for m in list_matches(cid, sid)]
    rows, results = [], []
    with ThreadPoolExecutor(workers) as pool:
        for r, res in pool.map(_job, jobs):
            rows.extend(r)
            results.append(res)
    df = pd.DataFrame(rows)
    df.to_parquet(out, index=False)
    # Final scores rebuilt from events, with the same team names the shots use.
    pd.DataFrame(results).to_csv(out.with_name("match_results.csv"), index=False)
    return df


def team_strength(matches: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-match-out points per game for each team in each match.

    Using the season *excluding the current match* stops the match being analysed from
    feeding into its own control variable.
    """
    long = pd.concat([
        pd.DataFrame({"match_id": matches.match_id, "team": matches.home,
                      "gf": matches.home_score, "ga": matches.away_score}),
        pd.DataFrame({"match_id": matches.match_id, "team": matches.away,
                      "gf": matches.away_score, "ga": matches.home_score}),
    ])
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    long["gd"] = long.gf - long.ga
    tot = long.groupby("team")[["pts", "gd"]].sum()
    n = long.groupby("team").size()
    long = long.join(tot, on="team", rsuffix="_tot").join(n.rename("n"), on="team")
    long["ppg_loo"] = (long.pts_tot - long.pts) / (long.n - 1)
    long["gdpg_loo"] = (long.gd_tot - long.gd) / (long.n - 1)
    return long[["match_id", "team", "ppg_loo", "gdpg_loo"]]


def build_analysis_table(shots: pd.DataFrame, states: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Join shots (with xG) to their game state and to both teams' strength."""
    df = shots.merge(states.drop(columns=["team"]), on=["match_id", "event_id"], how="inner")
    st = team_strength(matches)
    df = df.merge(st.rename(columns={"ppg_loo": "team_ppg", "gdpg_loo": "team_gdpg"}),
                  on=["match_id", "team"], how="left")
    df = df.merge(st.rename(columns={"team": "opponent", "ppg_loo": "opp_ppg", "gdpg_loo": "opp_gdpg"}),
                  on=["match_id", "opponent"], how="left")
    diff = df.goals_for - df.goals_against
    df["score_diff"] = diff
    df["state"] = np.select([diff < 0, diff == 0], ["trailing", "level"], "leading")
    df["red_diff"] = df.reds_against - df.reds_for          # +1 = opponent has one fewer player
    df["minute_c"] = df.second / 60
    df["outside_box"] = ((df.x < 102) | (df.y < 18) | (df.y > 62)).astype(int)
    return df


def exposure(results: pd.DataFrame) -> pd.DataFrame:
    """Minutes each team spent trailing, level and leading in each match (regulation time)."""
    out = []
    for r in results.itertuples(index=False):
        events = []
        if isinstance(r.goal_timeline, str) and r.goal_timeline:
            for item in r.goal_timeline.split(";"):
                s, tm = item.split("|", 1)
                events.append((int(s), tm))
        events.sort()
        score = {r.home: 0, r.away: 0}
        last = 0
        mins = {(tm, st): 0.0 for tm in (r.home, r.away) for st in ("trailing", "level", "leading")}
        for s, tm in events + [(r.end_second, None)]:
            s = min(s, r.end_second)
            for me, opp in ((r.home, r.away), (r.away, r.home)):
                d = score[me] - score[opp]
                st = "trailing" if d < 0 else ("level" if d == 0 else "leading")
                mins[(me, st)] += (s - last) / 60
            last = s
            if tm is not None:
                score[tm] += 1
        for (tm, st), m in mins.items():
            out.append({"match_id": r.match_id, "team": tm, "state": st, "minutes": m})
    return pd.DataFrame(out)
