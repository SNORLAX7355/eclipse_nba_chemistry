"""Collect NBA data that complements (does not duplicate) the committed datasets.

Outputs
-------
basketball_data/lineups_pairs_<season>.csv , lineups_pairs_five_seasons.csv
    LeagueDashLineups with group_quantity=2: *measured* two-man on-court
    advanced stats (OFF/DEF/NET rating, POSS). Ground truth for pair play.
basketball_data/player_vs_player_onoff_2025-26.csv
    PlayerVsPlayer (measure_type="Advanced") teammate on/off splits for a
    sample of high-minute pairs. Documents what the endpoint actually returns.

Run from the repository root:  python scripts/collect_pair_data.py
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
DATA = Path("basketball_data")
REQUEST_PAUSE = 2.0          # be polite to stats.nba.com
MAX_ATTEMPTS = 3             # bounded retries, never infinite


def _retry(call, label):
    """Call `call()` with bounded retries. Returns None after MAX_ATTEMPTS."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:                      # noqa: BLE001 - log and move on
            print(f"  [{label}] attempt {attempt}/{MAX_ATTEMPTS} failed: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            if attempt == MAX_ATTEMPTS:
                return None
            time.sleep(REQUEST_PAUSE * 2 * attempt)
    return None


def fetch_two_man_lineups(season: str) -> pd.DataFrame | None:
    """Measured two-player combination stats for one season."""
    from nba_api.stats.endpoints import leaguedashlineups

    def call():
        return leaguedashlineups.LeagueDashLineups(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
            group_quantity=2,
            timeout=90,
        ).get_data_frames()[0]

    df = _retry(call, f"pairs {season}")
    if df is None:
        return None
    df["SEASON"] = season
    return df


def fetch_player_vs_player(player_id: int, vs_player_id: int, season: str):
    """Advanced on/off split of `player_id` by whether `vs_player_id` is on court."""
    from nba_api.stats.endpoints import playervsplayer

    def call():
        resp = playervsplayer.PlayerVsPlayer(
            player_id=int(player_id),
            vs_player_id=int(vs_player_id),
            season=season,
            season_type_playoffs="Regular Season",
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="Per100Possessions",
            timeout=90,
        )
        # resultSets: [0] Overall, [1] OnOffCourt, [2:8] shot splits, [8,9] player info
        return resp.get_data_frames()[1]

    return _retry(call, f"pvp {player_id}/{vs_player_id}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--pvp-sample", type=int, default=15,
                    help="number of teammate pairs to query with PlayerVsPlayer")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"error: run from the repository root ({DATA} not found)", file=sys.stderr)
        return 2

    # ---------------- two-man lineups ----------------
    if not args.skip_pairs:
        frames, failed = [], []
        for season in SEASONS:
            print(f"Fetching two-man lineups for {season} ...")
            df = fetch_two_man_lineups(season)
            if df is None:
                failed.append(season)
                continue
            df.to_csv(DATA / f"lineups_pairs_{season}.csv", index=False)
            print(f"  rows={len(df)} cols={len(df.columns)}")
            frames.append(df)
            time.sleep(REQUEST_PAUSE)
        if frames:
            allp = pd.concat(frames, ignore_index=True)
            allp.to_csv(DATA / "lineups_pairs_five_seasons.csv", index=False)
            print(f"Saved lineups_pairs_five_seasons.csv rows={len(allp)}")
        if failed:
            print(f"WARNING: two-man fetch failed for: {failed}", file=sys.stderr)

    # ---------------- PlayerVsPlayer sample ----------------
    season = SEASONS[-1]
    pair_path = DATA / f"lineups_pairs_{season}.csv"
    if not pair_path.exists():
        print(f"skipping PlayerVsPlayer sample: {pair_path} missing", file=sys.stderr)
        return 0

    pairs = pd.read_csv(pair_path).sort_values("MIN", ascending=False)
    rows = []
    for rec in pairs.head(args.pvp_sample).itertuples(index=False):
        ids = [int(s) for s in str(rec.GROUP_ID).strip("-").split("-") if s.isdigit()]
        if len(ids) != 2:
            continue
        a, b = ids
        print(f"PlayerVsPlayer {rec.GROUP_NAME} ({a} vs {b}) ...")
        df = fetch_player_vs_player(a, b, season)
        time.sleep(REQUEST_PAUSE)
        if df is None or df.empty:
            continue
        keep = ["PLAYER_ID", "PLAYER_NAME", "VS_PLAYER_ID", "VS_PLAYER_NAME",
                "COURT_STATUS", "GP", "MIN", "POSS", "OFF_RATING", "DEF_RATING",
                "NET_RATING", "TS_PCT", "USG_PCT", "AST_PCT", "PIE"]
        sub = df[[c for c in keep if c in df.columns]].copy()
        sub["SEASON"] = season
        sub["PAIR_GROUP_ID"] = rec.GROUP_ID
        sub["PAIR_LINEUP_MIN"] = rec.MIN
        sub["PAIR_LINEUP_NET_RATING"] = rec.NET_RATING
        rows.append(sub)

    if rows:
        pvp = pd.concat(rows, ignore_index=True)
        pvp.to_csv(DATA / f"player_vs_player_onoff_{season}.csv", index=False)
        print(f"Saved player_vs_player_onoff_{season}.csv rows={len(pvp)}")
    else:
        print("WARNING: no PlayerVsPlayer rows collected", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
