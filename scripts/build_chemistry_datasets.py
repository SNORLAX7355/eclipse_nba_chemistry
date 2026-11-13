"""Build the derived lineup-chemistry datasets from the committed CSVs.

Reads only files already in ``basketball_data/`` (no network access) and writes:

* ``pair_co_play_proxies_five_seasons.csv``   - one row per player pair / team /
  season, with the possession-weighted co-play proxy, sample-size columns, a
  shrunk estimate, and the *measured* two-man net rating where available.
* ``lineup_player_features_five_seasons.csv`` - one row per five-man lineup,
  with member-attribute composition features and Eliot's archetype counts.

Run from the repository root:  python scripts/build_chemistry_datasets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineup_chemistry as lc  # noqa: E402


def main() -> int:
    data = lc.data_dir(Path(__file__).resolve().parents[1])

    lineups = pd.read_csv(data / "lineups_five_seasons.csv")
    players = pd.read_csv(data / "player_stats_five_seasons.csv")
    clusters = pd.read_csv(data / "player_clusters_five_seasons.csv")
    print(f"lineups={lineups.shape} players={players.shape} clusters={clusters.shape}")

    lineups = lc.add_player_ids(lineups)
    bad = int((lineups.N_UNIQUE_PLAYERS != 5).sum())
    print(f"malformed five-man groups: {bad}")

    # ---------------- pair co-play proxies ----------------
    pair_lineups = lc.explode_lineup_pairs(lineups)
    pairs = lc.aggregate_pair_proxies(pair_lineups)
    print(f"pair/lineup rows={len(pair_lineups):,} -> unique pair-team-seasons={len(pairs):,}")

    # Shrink using the possession count where an observation earns equal
    # weight with the prior, estimated from the data rather than guessed.
    rel = lc.estimate_reliability(lineups, value_col="NET_RATING", weight_col="POSS")
    prior_weight = min(max(rel["equal_weight_poss"], 50.0), 20_000.0)
    print(
        f"reliability: sd_true={rel['sd_true']:.2f} "
        f"noise_k={rel['var_noise_coefficient']:.0f} "
        f"equal_weight_poss={rel['equal_weight_poss']:.0f} "
        f"reliability@median({rel['median_poss']:.0f} poss)={rel['reliability_at_median_poss']:.3f}"
    )
    pairs["PAIR_NET_PROXY_SHRUNK"] = lc.shrink_to_prior(
        pairs["PAIR_NET_PROXY"], pairs["SUM_LINEUP_POSS"], prior_weight=prior_weight
    )
    pairs["SHRINK_PRIOR_WEIGHT_POSS"] = prior_weight

    # Measured two-man ratings, when the collection script has been run.
    two_man_path = data / "lineups_pairs_five_seasons.csv"
    if two_man_path.exists():
        two_man = pd.read_csv(two_man_path)
        pairs = lc.attach_measured_pair_ratings(pairs, two_man)
        matched = int(pairs["MEASURED_PAIR_NET_RATING"].notna().sum())
        print(f"measured two-man ratings joined for {matched:,}/{len(pairs):,} pairs "
              f"({100 * matched / len(pairs):.1f}%)")
    else:
        print(f"note: {two_man_path.name} absent - run scripts/collect_pair_data.py "
              "to add measured two-man ratings")

    names = (
        players.drop_duplicates(["PLAYER_ID", "SEASON"])
        .set_index(["PLAYER_ID", "SEASON"])["PLAYER_NAME"]
    )
    pairs["PLAYER_A"] = pd.MultiIndex.from_arrays(
        [pairs.PLAYER_A_ID, pairs.SEASON]
    ).map(names)
    pairs["PLAYER_B"] = pd.MultiIndex.from_arrays(
        [pairs.PLAYER_B_ID, pairs.SEASON]
    ).map(names)

    out_pairs = data / "pair_co_play_proxies_five_seasons.csv"
    pairs.to_csv(out_pairs, index=False)
    print(f"wrote {out_pairs.name} rows={len(pairs):,} cols={len(pairs.columns)}")

    # ---------------- lineup composition features ----------------
    features = lc.build_lineup_player_features(lineups, players)
    print(f"lineups with all five members matched: {len(features):,}/{len(lineups):,}")
    features = lc.add_archetype_composition(features, lineups, clusters)
    labelled = int(features["ARCHETYPE_MIX"].notna().sum())
    print(f"lineups with all five archetype labels: {labelled:,} "
          f"({100 * labelled / len(features):.1f}%)")

    out_feats = data / "lineup_player_features_five_seasons.csv"
    features.to_csv(out_feats, index=False)
    print(f"wrote {out_feats.name} rows={len(features):,} cols={len(features.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
