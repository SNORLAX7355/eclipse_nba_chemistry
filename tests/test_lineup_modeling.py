"""Tests for lineup_modeling.

Synthetic fixtures with hand-computed expected values, plus integrity checks
against the committed `using_dataset/` tables (skipped if absent).

Run from the repository root:  python -m pytest tests -q
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lineup_modeling as lm

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def make_lineups() -> pd.DataFrame:
    """Three lineups: two above the possession threshold, one below."""
    return pd.DataFrame(
        {
            "GROUP_ID": ["-1-2-3-4-5-", "-1-2-3-4-6-", "-7-8-9-10-11-"],
            "TEAM_ABBREVIATION": ["XXX", "XXX", "YYY"],
            "SEASON": ["2024-25"] * 3,
            "GP": [40, 30, 10],
            "MIN": [500.0, 300.0, 40.0],
            "POSS": [1000, 600, 80],
            "OFF_RATING": [118.0, 110.0, 99.0],
            "DEF_RATING": [108.0, 112.0, 115.0],
            "NET_RATING": [10.0, -2.0, -16.0],
        }
    )


def make_players() -> pd.DataFrame:
    """Eleven players. Members 1-5 have USG_PCT 0.10..0.50 for hand checks.

    Player 11 carries an all-zero offensive synergy block, i.e. missing data.
    """
    pids = list(range(1, 12))
    n = len(pids)
    df = pd.DataFrame(
        {
            "PLAYER_ID": pids,
            "SEASON": ["2024-25"] * n,
            "USG_PCT": [0.10, 0.20, 0.30, 0.40, 0.50] + [0.20] * 6,
            "AST_PCT": [0.15] * n,
            "REB_PCT": [0.10] * n,
            "OREB_PCT": [0.05] * n,
            "DREB_PCT": [0.15] * n,
            "TM_TOV_PCT": [0.12] * n,
            "TS_PCT": [0.55] * n,
            "EFG_PCT": [0.52] * n,
            "FG3A": [4.0] * n,
            "FG3_PCT": [0.36] * n,
            "STL": [1.0] * n,
            "BLK": [0.5] * n,
            "CLUSTER": [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5],
            "SYNERGY_OFFENSIVE_SPOTUP_PCT": [0.6] * n,
            "SYNERGY_OFFENSIVE_ISOLATION_PCT": [0.4] * n,
            "SYNERGY_DEFENSIVE_SPOTUP_PCT": [0.7] * n,
            "SYNERGY_DEFENSIVE_POSTUP_PCT": [0.3] * n,
        }
    )
    # Player 11: no play-type data at all, encoded by the source as zeros.
    df.loc[df.PLAYER_ID == 11,
           ["SYNERGY_OFFENSIVE_SPOTUP_PCT", "SYNERGY_OFFENSIVE_ISOLATION_PCT"]] = 0.0
    return df


# --------------------------------------------------------------------------
# parse_group_id
# --------------------------------------------------------------------------
class TestParseGroupId:
    def test_valid_five_man_group(self):
        assert lm.parse_group_id("-1-2-3-4-5-") == [1, 2, 3, 4, 5]

    def test_realistic_ids(self):
        assert lm.parse_group_id("-201145-203115-203932-203999-1628420-") == [
            201145, 203115, 203932, 203999, 1628420
        ]

    def test_order_is_preserved(self):
        # Unlike lineup_chemistry.parse_group_ids, slot order is kept here.
        assert lm.parse_group_id("-5-4-3-2-1-") == [5, 4, 3, 2, 1]

    @pytest.mark.parametrize("bad", ["-1-2-3-4-", "-1-2-3-4-5-6-", "-1-1-2-3-4-",
                                     "", "-", None, float("nan"), "-a-b-c-d-e-"])
    def test_malformed_groups_return_none(self, bad):
        assert lm.parse_group_id(bad) is None

    def test_custom_size(self):
        assert lm.parse_group_id("-1-2-", size=2) == [1, 2]
        assert lm.parse_group_id("-1-2-", size=5) is None


class TestExplodeLineupMembers:
    def test_five_rows_per_lineup(self):
        out = lm.explode_lineup_members(make_lineups())
        assert len(out) == 15
        assert out.groupby("GROUP_ID").size().unique().tolist() == [5]

    def test_player_ids_and_slots(self):
        out = lm.explode_lineup_members(make_lineups())
        first = out[out.GROUP_ID == "-1-2-3-4-5-"].sort_values("SLOT")
        assert first.PLAYER_ID.tolist() == [1, 2, 3, 4, 5]
        assert first.SLOT.tolist() == [0, 1, 2, 3, 4]

    def test_malformed_group_dropped(self):
        bad = make_lineups()
        bad.loc[0, "GROUP_ID"] = "-1-2-3-4-"
        out = lm.explode_lineup_members(bad)
        assert "-1-2-3-4-" not in set(out.GROUP_ID)
        assert len(out) == 10


# --------------------------------------------------------------------------
# build_lineup_dataset
# --------------------------------------------------------------------------
class TestBuildLineupDataset:
    def test_possession_threshold_filters_rows(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        assert set(out.GROUP_ID) == {"-1-2-3-4-5-", "-1-2-3-4-6-"}
        low = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=50)
        assert len(low) == 3

    def test_avg_and_std_match_hand_calculation(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        vals = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        assert row.AVG_USG_PCT == pytest.approx(vals.mean())
        assert row.STD_USG_PCT == pytest.approx(vals.std(ddof=0))

    def test_std_is_population_not_sample(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        vals = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        assert row.STD_USG_PCT != pytest.approx(vals.std(ddof=1))

    def test_identical_members_give_zero_spread(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        assert (out.STD_TS_PCT.abs() < 1e-12).all()

    def test_targets_and_metadata_carried_through(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        assert row.OFF_RATING == pytest.approx(118.0)
        assert row.DEF_RATING == pytest.approx(108.0)
        assert row.POSS == 1000
        assert row.TEAM == "XXX"
        assert row.SEASON == "2024-25"

    def test_cluster_counts_sum_to_five_and_match_labels(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        # players 1,2 -> 0 ; 3,4 -> 1 ; 5 -> 2
        assert (row.CLUSTER_0, row.CLUSTER_1, row.CLUSTER_2) == (2, 2, 1)
        cl = [c for c in out.columns if c.startswith("CLUSTER_")]
        assert (out[cl].sum(axis=1) == 5).all()

    def test_all_six_cluster_columns_exist_even_if_unused(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        for k in range(6):
            assert f"CLUSTER_{k}" in out.columns

    def test_missing_tov_pct_name_would_raise(self):
        """The notebook prototype silently dropped TOV_PCT; we fail loudly."""
        with pytest.raises(KeyError):
            lm.build_lineup_dataset(make_lineups(), make_players(),
                                    min_poss=100, box_cols=["TOV_PCT"])

    def test_zero_synergy_row_treated_as_missing(self):
        """Player 11's all-zero block must not drag the lineup average to 0.48."""
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=50,
                                      zero_synergy_as_missing=True)
        row = out[out.GROUP_ID == "-7-8-9-10-11-"].iloc[0]
        assert row.AVG_SYNERGY_OFFENSIVE_SPOTUP_PCT == pytest.approx(0.6)
        assert row.N_PLAYTYPE_MISSING == 1

    def test_zero_synergy_kept_when_disabled(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=50,
                                      zero_synergy_as_missing=False)
        row = out[out.GROUP_ID == "-7-8-9-10-11-"].iloc[0]
        assert row.AVG_SYNERGY_OFFENSIVE_SPOTUP_PCT == pytest.approx(0.48)

    def test_incomplete_lineups_dropped_when_required(self):
        players = make_players()
        out = lm.build_lineup_dataset(
            make_lineups(), players[players.PLAYER_ID != 5], min_poss=100,
            require_complete=True,
        )
        assert "-1-2-3-4-5-" not in set(out.GROUP_ID)

    def test_incomplete_lineups_kept_and_flagged_when_allowed(self):
        players = make_players()
        out = lm.build_lineup_dataset(
            make_lineups(), players[players.PLAYER_ID != 5], min_poss=100,
            require_complete=False,
        )
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        assert row.MATCHED_PLAYERS == 4
        assert row.AVG_USG_PCT == pytest.approx(np.mean([0.10, 0.20, 0.30, 0.40]))

    def test_player_profiles_matched_within_season(self):
        players = make_players()
        stale = players.copy()
        stale["SEASON"] = "2023-24"
        stale["USG_PCT"] = 0.99
        out = lm.build_lineup_dataset(
            make_lineups(), pd.concat([players, stale], ignore_index=True),
            min_poss=100,
        )
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        assert row.AVG_USG_PCT == pytest.approx(0.30)

    def test_one_row_per_lineup_season(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=50)
        assert not out.duplicated(["GROUP_ID", "SEASON"]).any()

    def test_no_nan_features_on_complete_data(self):
        out = lm.build_lineup_dataset(make_lineups(), make_players(), min_poss=100)
        feats = [c for c in out.columns if c.startswith(("AVG_", "STD_"))]
        assert out[feats].isna().sum().sum() == 0


# --------------------------------------------------------------------------
# feature_sets
# --------------------------------------------------------------------------
class TestFeatureSets:
    def _cols(self):
        return lm.build_lineup_dataset(
            make_lineups(), make_players(), min_poss=100
        ).columns

    def test_sets_are_nested_and_increasing(self):
        fs = lm.feature_sets(self._cols())
        order = ["avg_box", "avg_box_playtype", "plus_std", "plus_clusters", "plus_usage"]
        for a, b in zip(order, order[1:]):
            assert set(fs[a]).issubset(set(fs[b]))
            assert len(fs[a]) < len(fs[b])

    def test_avg_box_excludes_playtypes_and_std(self):
        fs = lm.feature_sets(self._cols())
        assert all("SYNERGY" not in c for c in fs["avg_box"])
        assert all(c.startswith("AVG_") for c in fs["avg_box"])

    def test_usage_columns_only_in_the_last_set(self):
        fs = lm.feature_sets(self._cols())
        for name in ["avg_box", "avg_box_playtype", "plus_std", "plus_clusters"]:
            assert not set(lm.LINEUP_USAGE_COLS) & set(fs[name])
        assert set(lm.LINEUP_USAGE_COLS).issubset(set(fs["plus_usage"]))

    def test_no_target_leaks_into_any_feature_set(self):
        fs = lm.feature_sets(self._cols())
        banned = set(lm.TARGETS) | {"NET_RATING"}
        for name, cols in fs.items():
            assert not banned & set(cols), name


# --------------------------------------------------------------------------
# reliability and scoring
# --------------------------------------------------------------------------
class TestRatingReliability:
    def test_recovers_known_variance_components(self):
        rng = np.random.default_rng(0)
        n = 60_000
        poss = rng.integers(25, 1500, n)
        k = 20_000.0
        obs = rng.normal(0, 5.0, n) + rng.normal(0, np.sqrt(k / poss))
        out = lm.rating_reliability(
            pd.DataFrame({"OFF_RATING": obs, "POSS": poss}),
            value_col="OFF_RATING", n_bins=15,
        )
        assert out["var_true"] == pytest.approx(25.0, rel=0.25)
        assert out["noise_k"] == pytest.approx(k, rel=0.25)

    def test_pure_noise_reports_low_reliability(self):
        rng = np.random.default_rng(1)
        n = 40_000
        poss = rng.integers(25, 1500, n)
        out = lm.rating_reliability(
            pd.DataFrame({"OFF_RATING": rng.normal(0, np.sqrt(20_000.0 / poss)),
                          "POSS": poss}),
            value_col="OFF_RATING", n_bins=15,
        )
        assert out["reliability"] < 0.1

    def test_reliability_increases_with_possessions(self):
        rng = np.random.default_rng(3)
        n = 40_000
        k = 20_000.0
        low = rng.integers(25, 120, n)
        high = rng.integers(600, 2500, n)
        res = []
        for poss in (low, high):
            obs = rng.normal(0, 5.0, n) + rng.normal(0, np.sqrt(k / poss))
            res.append(lm.rating_reliability(
                pd.DataFrame({"OFF_RATING": obs, "POSS": poss}),
                value_col="OFF_RATING", n_bins=12)["reliability"])
        assert res[1] > res[0]

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError):
            lm.rating_reliability(
                pd.DataFrame({"OFF_RATING": [1.0, 2.0], "POSS": [10, 20]}),
                value_col="OFF_RATING",
            )


class TestWeightedRmse:
    def test_perfect_prediction_is_zero(self):
        assert lm.weighted_rmse([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)

    def test_known_unweighted_value(self):
        # sqrt((3^2 + 4^2) / 2) = sqrt(12.5)
        assert lm.weighted_rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(math.sqrt(12.5))

    def test_weighting_changes_the_score(self):
        a = lm.weighted_rmse([0.0, 0.0], [1.0, 10.0], [1, 1])
        b = lm.weighted_rmse([0.0, 0.0], [1.0, 10.0], [1000, 1])
        assert b < a


# --------------------------------------------------------------------------
# Integrity of the committed modelling tables
# --------------------------------------------------------------------------
def _paths():
    try:
        return lm.data_paths(REPO_ROOT)
    except FileNotFoundError:
        return None, None


LINEUP_PATH, PLAYER_PATH = _paths()


@pytest.mark.skipif(LINEUP_PATH is None, reason="using_dataset/ not present")
class TestCommittedModellingData:
    def test_player_table_has_one_row_per_player_season(self):
        p = pd.read_csv(PLAYER_PATH)
        assert not p.duplicated(["PLAYER_ID", "SEASON"]).any()

    def test_every_cluster_label_is_present_and_integral(self):
        p = pd.read_csv(PLAYER_PATH)
        assert p.CLUSTER.notna().all()
        assert sorted(p.CLUSTER.unique()) == list(range(6))

    def test_synergy_blocks_sum_to_one_where_present(self):
        p = pd.read_csv(PLAYER_PATH)
        for block in ("OFFENSIVE", "DEFENSIVE"):
            cols = [c for c in p.columns
                    if c.startswith(f"SYNERGY_{block}") and c.endswith("_PCT")]
            sums = p[cols].sum(axis=1)
            nonzero = sums[sums > 0]
            assert nonzero.between(0.97, 1.03).all()

    def test_builder_reproduces_the_teams_1476_lineups(self):
        lineups = pd.read_csv(LINEUP_PATH)
        players = pd.read_csv(PLAYER_PATH)
        out = lm.build_lineup_dataset(lineups, players, min_poss=100)
        assert len(out) == 1476
        assert (out.MATCHED_PLAYERS == 5).all()
        assert not out.duplicated(["GROUP_ID", "SEASON"]).any()

    def test_lower_threshold_strictly_grows_the_sample(self):
        lineups = pd.read_csv(LINEUP_PATH)
        players = pd.read_csv(PLAYER_PATH)
        sizes = [len(lm.build_lineup_dataset(lineups, players, min_poss=t))
                 for t in (200, 100, 50)]
        assert sizes[0] < sizes[1] < sizes[2]

    def test_targets_are_in_a_plausible_range(self):
        lineups = pd.read_csv(LINEUP_PATH)
        players = pd.read_csv(PLAYER_PATH)
        out = lm.build_lineup_dataset(lineups, players, min_poss=100)
        for col in lm.TARGETS:
            assert out[col].between(60, 160).all()

    def test_net_rating_equals_off_minus_def(self):
        lineups = pd.read_csv(LINEUP_PATH)
        diff = (lineups.NET_RATING - (lineups.OFF_RATING - lineups.DEF_RATING)).abs()
        assert diff.max() < 0.2
