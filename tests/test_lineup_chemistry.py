"""Tests for lineup_chemistry.

Most cases use tiny synthetic fixtures whose expected values are worked out by
hand, so a regression in the weighting or pair logic fails loudly.  The final
class runs light integrity checks against the committed CSVs; those are skipped
automatically if the data files are absent.

Run from the repository root:  python -m pytest tests -q
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import lineup_chemistry as lc

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "basketball_data"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
def make_lineups() -> pd.DataFrame:
    """Three five-man lineups on one team sharing players 1 and 2.

    Lineups A and B both contain the pair (1, 2):
        A: net +10 over 100 possessions
        B: net  -5 over  50 possessions
    so the (1, 2) proxy must be (10*100 + -5*50) / 150 = +5.0 exactly.
    Lineup C contains neither, and is there to prove it is excluded.
    """
    return pd.DataFrame(
        {
            "GROUP_ID": ["-1-2-3-4-5-", "-1-2-3-4-6-", "-7-8-9-10-11-"],
            "GROUP_NAME": ["A", "B", "C"],
            "TEAM_ABBREVIATION": ["XXX", "XXX", "YYY"],
            "TEAM_ID": [1, 1, 2],
            "GP": [10, 5, 8],
            "MIN": [40.0, 20.0, 30.0],
            "POSS": [100, 50, 80],
            "OFF_RATING": [115.0, 100.0, 110.0],
            "DEF_RATING": [105.0, 105.0, 108.0],
            "NET_RATING": [10.0, -5.0, 2.0],
            "SEASON": ["2024-25", "2024-25", "2024-25"],
        }
    )


def make_players() -> pd.DataFrame:
    """Eleven players with hand-chosen metric values."""
    pids = list(range(1, 12))
    return pd.DataFrame(
        {
            "PLAYER_ID": pids,
            "PLAYER_NAME": [f"Player {p}" for p in pids],
            "SEASON": ["2024-25"] * 11,
            # lineup A members (1-5) are 0.10, 0.20, 0.30, 0.40, 0.50
            "USG_PCT": [0.10, 0.20, 0.30, 0.40, 0.50] + [0.20] * 6,
            "TS_PCT": [0.55] * 11,
            "AST_PCT": [0.15] * 11,
            "OREB_PCT": [0.05] * 11,
            "DREB_PCT": [0.15] * 11,
            "TM_TOV_PCT": [0.12] * 11,
            "FG3A": [3.0] * 11,
            "FGA": [10.0] * 11,
            "AST": [4.0] * 11,
            "TOV": [2.0] * 11,
        }
    )


def make_clusters() -> pd.DataFrame:
    """Labels for players 1-10 only: player 11 is deliberately unlabelled."""
    return pd.DataFrame(
        {
            "PLAYER_ID": list(range(1, 11)),
            "SEASON": ["2024-25"] * 10,
            "CLUSTER": [0, 0, 1, 1, 2, 2, 0, 1, 2, 1],
        }
    )


# --------------------------------------------------------------------------
# parse_group_ids
# --------------------------------------------------------------------------
class TestParseGroupIds:
    def test_standard_five_man_group(self):
        assert lc.parse_group_ids("-201145-203115-203932-203999-1628420-") == (
            201145, 203115, 203932, 203999, 1628420,
        )

    def test_result_is_sorted_regardless_of_input_order(self):
        assert lc.parse_group_ids("-203999-201142-") == (201142, 203999)
        assert lc.parse_group_ids("-201142-203999-") == (201142, 203999)

    def test_two_man_group(self):
        assert lc.parse_group_ids("-201142-1641708-") == (201142, 1641708)

    @pytest.mark.parametrize("bad", [None, float("nan"), "", "-", "--", "-abc-"])
    def test_unparseable_inputs_give_empty_tuple(self, bad):
        assert lc.parse_group_ids(bad) == ()

    def test_no_whitespace_or_padding_artefacts(self):
        assert lc.parse_group_ids("  -1-2-  ") == (1, 2)

    def test_duplicate_ids_are_preserved_not_silently_deduped(self):
        # add_player_ids relies on this to flag malformed groups.
        assert lc.parse_group_ids("-5-5-") == (5, 5)


class TestAddPlayerIds:
    def test_counts_and_uniqueness_flags(self):
        df = pd.DataFrame({"GROUP_ID": ["-1-2-3-4-5-", "-1-1-2-3-4-", "-1-2-"]})
        out = lc.add_player_ids(df)
        assert out["N_PLAYERS"].tolist() == [5, 5, 2]
        assert out["N_UNIQUE_PLAYERS"].tolist() == [5, 4, 2]

    def test_input_is_not_mutated(self):
        df = pd.DataFrame({"GROUP_ID": ["-1-2-3-4-5-"]})
        lc.add_player_ids(df)
        assert list(df.columns) == ["GROUP_ID"]


# --------------------------------------------------------------------------
# weighted_mean
# --------------------------------------------------------------------------
class TestWeightedMean:
    def test_known_value(self):
        assert lc.weighted_mean([10.0, 0.0], [3, 1]) == pytest.approx(7.5)

    def test_matches_the_documented_pair_formula(self):
        # (10*100 + -5*50) / 150
        assert lc.weighted_mean([10.0, -5.0], [100, 50]) == pytest.approx(5.0)

    def test_equal_weights_reduce_to_simple_mean(self):
        vals = [1.0, 2.0, 6.0]
        assert lc.weighted_mean(vals, [2, 2, 2]) == pytest.approx(np.mean(vals))

    def test_zero_weight_rows_are_ignored(self):
        assert lc.weighted_mean([10.0, 999.0], [5, 0]) == pytest.approx(10.0)

    def test_nan_value_is_ignored_rather_than_poisoning_the_result(self):
        assert lc.weighted_mean([10.0, float("nan")], [5, 5]) == pytest.approx(10.0)

    def test_all_weights_zero_returns_nan(self):
        assert math.isnan(lc.weighted_mean([1.0, 2.0], [0, 0]))

    def test_empty_input_returns_nan(self):
        assert math.isnan(lc.weighted_mean([], []))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            lc.weighted_mean([1.0, 2.0], [1])

    def test_negative_weights_are_excluded(self):
        assert lc.weighted_mean([10.0, 100.0], [5, -5]) == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Pair extraction
# --------------------------------------------------------------------------
class TestExplodeLineupPairs:
    def test_each_five_man_lineup_yields_exactly_ten_pairs(self):
        pairs = lc.explode_lineup_pairs(make_lineups())
        assert len(pairs) == 30  # 3 lineups x 10
        assert pairs.groupby("LINEUP_GROUP_ID").size().unique().tolist() == [10]

    def test_pairs_match_itertools_combinations(self):
        pairs = lc.explode_lineup_pairs(make_lineups())
        got = set(
            pairs.loc[pairs.LINEUP_GROUP_ID == "-1-2-3-4-5-", ["PLAYER_A_ID", "PLAYER_B_ID"]]
            .itertuples(index=False, name=None)
        )
        assert got == set(itertools.combinations([1, 2, 3, 4, 5], 2))

    def test_pair_key_is_canonically_ordered(self):
        pairs = lc.explode_lineup_pairs(make_lineups())
        assert (pairs.PLAYER_A_ID < pairs.PLAYER_B_ID).all()

    def test_lineup_attributes_are_carried_through(self):
        pairs = lc.explode_lineup_pairs(make_lineups())
        row = pairs[
            (pairs.PLAYER_A_ID == 1) & (pairs.PLAYER_B_ID == 2)
            & (pairs.LINEUP_GROUP_ID == "-1-2-3-4-5-")
        ].iloc[0]
        assert row.LINEUP_POSS == 100
        assert row.LINEUP_NET_RATING == 10.0
        assert row.TEAM == "XXX"
        assert row.SEASON == "2024-25"

    def test_malformed_groups_are_dropped(self):
        bad = pd.DataFrame(
            {
                "GROUP_ID": ["-1-2-3-4-", "-1-1-2-3-4-", "-1-2-3-4-5-"],
                "TEAM_ABBREVIATION": ["X"] * 3,
                "MIN": [10.0] * 3,
                "POSS": [20] * 3,
                "NET_RATING": [1.0] * 3,
                "SEASON": ["2024-25"] * 3,
            }
        )
        pairs = lc.explode_lineup_pairs(bad)
        assert pairs.LINEUP_GROUP_ID.unique().tolist() == ["-1-2-3-4-5-"]

    def test_empty_input_gives_empty_frame_with_schema(self):
        empty = make_lineups().iloc[0:0]
        pairs = lc.explode_lineup_pairs(empty)
        assert pairs.empty
        assert "PLAYER_A_ID" in pairs.columns


# --------------------------------------------------------------------------
# Pair aggregation - the headline arithmetic
# --------------------------------------------------------------------------
class TestAggregatePairProxies:
    def test_possession_weighted_proxy_matches_hand_calculation(self):
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        row = proxies[(proxies.PLAYER_A_ID == 1) & (proxies.PLAYER_B_ID == 2)].iloc[0]
        assert row.PAIR_NET_PROXY == pytest.approx(5.0)
        assert row.SAMPLED_LINEUPS == 2
        assert row.SUM_LINEUP_POSS == 150
        assert row.SUM_LINEUP_MIN == pytest.approx(60.0)

    def test_weighting_is_by_possessions_not_minutes_and_not_unweighted(self):
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        row = proxies[(proxies.PLAYER_A_ID == 1) & (proxies.PLAYER_B_ID == 2)].iloc[0]
        unweighted = (10.0 + -5.0) / 2          # 2.5
        minute_weighted = (10.0 * 40 + -5.0 * 20) / 60  # 5.0 here too
        assert row.PAIR_NET_PROXY != pytest.approx(unweighted)
        # Guard the formula itself rather than relying on this fixture's tie:
        tweaked = lc.explode_lineup_pairs(make_lineups())
        tweaked.loc[tweaked.LINEUP_GROUP_ID == "-1-2-3-4-6-", "LINEUP_MIN"] = 400.0
        row2 = lc.aggregate_pair_proxies(tweaked)
        row2 = row2[(row2.PLAYER_A_ID == 1) & (row2.PLAYER_B_ID == 2)].iloc[0]
        assert row2.PAIR_NET_PROXY == pytest.approx(5.0)  # unchanged by MIN
        assert minute_weighted == pytest.approx(5.0)

    def test_pair_appearing_in_one_lineup_equals_that_lineups_rating(self):
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        row = proxies[(proxies.PLAYER_A_ID == 7) & (proxies.PLAYER_B_ID == 8)].iloc[0]
        assert row.PAIR_NET_PROXY == pytest.approx(2.0)
        assert row.SAMPLED_LINEUPS == 1

    def test_pair_count_is_correct(self):
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        # lineups A+B share players 1-4 -> C(4,2)=6 shared pairs, each lineup
        # adds 4 pairs with its unique fifth member; lineup C adds 10.
        assert len(proxies) == 6 + 4 + 4 + 10

    def test_same_pair_on_two_teams_is_not_pooled(self):
        lineups = make_lineups()
        moved = lineups.iloc[[1]].copy()
        moved["TEAM_ABBREVIATION"] = "ZZZ"
        moved["GROUP_ID"] = "-1-2-3-4-99-"
        proxies = lc.aggregate_pair_proxies(
            lc.explode_lineup_pairs(pd.concat([lineups, moved], ignore_index=True))
        )
        rows = proxies[(proxies.PLAYER_A_ID == 1) & (proxies.PLAYER_B_ID == 2)]
        assert set(rows.TEAM) == {"XXX", "ZZZ"}
        assert rows[rows.TEAM == "XXX"].iloc[0].PAIR_NET_PROXY == pytest.approx(5.0)
        assert rows[rows.TEAM == "ZZZ"].iloc[0].PAIR_NET_PROXY == pytest.approx(-5.0)

    def test_seasons_are_not_pooled(self):
        lineups = make_lineups()
        other = lineups.iloc[[0]].copy()
        other["SEASON"] = "2023-24"
        other["NET_RATING"] = -20.0
        proxies = lc.aggregate_pair_proxies(
            lc.explode_lineup_pairs(pd.concat([lineups, other], ignore_index=True))
        )
        rows = proxies[(proxies.PLAYER_A_ID == 1) & (proxies.PLAYER_B_ID == 2)]
        assert set(rows.SEASON) == {"2024-25", "2023-24"}
        assert rows[rows.SEASON == "2023-24"].iloc[0].PAIR_NET_PROXY == pytest.approx(-20.0)

    def test_zero_possession_and_missing_rating_lineups_are_excluded(self):
        pairs = lc.explode_lineup_pairs(make_lineups())
        pairs.loc[pairs.LINEUP_GROUP_ID == "-1-2-3-4-6-", "LINEUP_POSS"] = 0
        proxies = lc.aggregate_pair_proxies(pairs)
        row = proxies[(proxies.PLAYER_A_ID == 1) & (proxies.PLAYER_B_ID == 2)].iloc[0]
        assert row.PAIR_NET_PROXY == pytest.approx(10.0)  # only lineup A survives
        assert row.SUM_LINEUP_POSS == 100

    def test_empty_input_returns_empty_frame_with_schema(self):
        out = lc.aggregate_pair_proxies(pd.DataFrame())
        assert out.empty
        assert "PAIR_NET_PROXY" in out.columns

    def test_no_duplicate_pair_keys_in_output(self):
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        keys = ["SEASON", "TEAM", "PLAYER_A_ID", "PLAYER_B_ID"]
        assert not proxies.duplicated(keys).any()


# --------------------------------------------------------------------------
# Shrinkage
# --------------------------------------------------------------------------
class TestShrinkToPrior:
    def test_known_value(self):
        out = lc.shrink_to_prior(
            pd.Series([20.0]), pd.Series([100.0]), prior=0.0, prior_weight=100.0
        )
        assert out.iloc[0] == pytest.approx(10.0)

    def test_large_sample_barely_moves_small_sample_moves_a_lot(self):
        vals = pd.Series([20.0, 20.0])
        out = lc.shrink_to_prior(vals, pd.Series([5000.0, 10.0]), prior=0.0, prior_weight=200.0)
        assert out.iloc[0] > 19.0
        assert abs(out.iloc[1]) < 2.0

    def test_shrinkage_never_overshoots_the_prior(self):
        out = lc.shrink_to_prior(
            pd.Series([50.0, -50.0]), pd.Series([30.0, 30.0]), prior=0.0, prior_weight=200.0
        )
        assert 0.0 < out.iloc[0] < 50.0
        assert -50.0 < out.iloc[1] < 0.0

    def test_default_prior_is_the_weighted_mean(self):
        vals, wts = pd.Series([10.0, 0.0]), pd.Series([3.0, 1.0])
        out = lc.shrink_to_prior(vals, wts, prior_weight=1e9)
        assert out.iloc[0] == pytest.approx(7.5, abs=1e-4)

    def test_non_positive_prior_weight_raises(self):
        with pytest.raises(ValueError):
            lc.shrink_to_prior(pd.Series([1.0]), pd.Series([1.0]), prior_weight=0.0)


# --------------------------------------------------------------------------
# Measured two-man join
# --------------------------------------------------------------------------
class TestAttachMeasuredPairRatings:
    def _two_man(self):
        return pd.DataFrame(
            {
                "GROUP_ID": ["-1-2-", "-3-4-"],
                "TEAM_ABBREVIATION": ["XXX", "XXX"],
                "SEASON": ["2024-25", "2024-25"],
                "NET_RATING": [4.0, -1.0],
                "POSS": [900, 700],
                "MIN": [400.0, 320.0],
            }
        )

    def test_measured_values_land_on_the_right_pair(self):
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        out = lc.attach_measured_pair_ratings(proxies, self._two_man())
        row = out[(out.PLAYER_A_ID == 1) & (out.PLAYER_B_ID == 2)].iloc[0]
        assert row.MEASURED_PAIR_NET_RATING == pytest.approx(4.0)
        assert row.MEASURED_PAIR_POSS == 900

    def test_row_count_is_preserved_and_unmatched_pairs_are_nan(self):
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        out = lc.attach_measured_pair_ratings(proxies, self._two_man())
        assert len(out) == len(proxies)
        row = out[(out.PLAYER_A_ID == 7) & (out.PLAYER_B_ID == 8)].iloc[0]
        assert pd.isna(row.MEASURED_PAIR_NET_RATING)

    def test_duplicate_measured_keys_do_not_inflate_the_join(self):
        dup = pd.concat([self._two_man(), self._two_man().iloc[[0]]], ignore_index=True)
        proxies = lc.aggregate_pair_proxies(lc.explode_lineup_pairs(make_lineups()))
        out = lc.attach_measured_pair_ratings(proxies, dup)
        assert len(out) == len(proxies)


# --------------------------------------------------------------------------
# Lineup features
# --------------------------------------------------------------------------
class TestBuildLineupPlayerFeatures:
    def test_mean_and_sd_match_hand_calculation(self):
        feats = lc.build_lineup_player_features(
            make_lineups(), make_players(), metrics=["USG_PCT"]
        )
        row = feats[feats.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        vals = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        assert row.MEAN_USG_PCT == pytest.approx(vals.mean())
        assert row.SD_USG_PCT == pytest.approx(vals.std(ddof=0))  # population SD
        assert row.MAX_USG_PCT == pytest.approx(0.50)
        assert row.MIN_USG_PCT == pytest.approx(0.10)

    def test_sd_is_population_not_sample(self):
        feats = lc.build_lineup_player_features(
            make_lineups(), make_players(), metrics=["USG_PCT"]
        )
        row = feats[feats.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        vals = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
        assert row.SD_USG_PCT != pytest.approx(vals.std(ddof=1))

    def test_identical_members_give_zero_spread(self):
        feats = lc.build_lineup_player_features(
            make_lineups(), make_players(), metrics=["TS_PCT"]
        )
        assert (feats.SD_TS_PCT.abs() < 1e-12).all()

    def test_outcome_columns_are_carried_through(self):
        feats = lc.build_lineup_player_features(
            make_lineups(), make_players(), metrics=["USG_PCT"]
        )
        row = feats[feats.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        assert row.NET_RATING == pytest.approx(10.0)
        assert row.POSS == 100
        assert row.SEASON == "2024-25"

    def test_derived_ratios_are_computed_when_absent(self):
        feats = lc.build_lineup_player_features(
            make_lineups(), make_players(),
            metrics=["THREE_POINT_ATTEMPT_RATE", "AST_TO_TOV"],
        )
        row = feats.iloc[0]
        assert row.MEAN_THREE_POINT_ATTEMPT_RATE == pytest.approx(0.3)  # 3/10
        assert row.MEAN_AST_TO_TOV == pytest.approx(2.0)                # 4/2

    def test_incomplete_lineups_are_dropped_when_required(self):
        players = make_players()
        partial = players[players.PLAYER_ID != 5]        # break lineup A
        feats = lc.build_lineup_player_features(
            make_lineups(), partial, metrics=["USG_PCT"], require_complete=True
        )
        assert "-1-2-3-4-5-" not in set(feats.GROUP_ID)
        assert (feats.MATCHED_PLAYERS == 5).all()

    def test_incomplete_lineups_are_kept_and_flagged_when_allowed(self):
        players = make_players()
        partial = players[players.PLAYER_ID != 5]
        feats = lc.build_lineup_player_features(
            make_lineups(), partial, metrics=["USG_PCT"], require_complete=False
        )
        row = feats[feats.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        assert row.MATCHED_PLAYERS == 4
        assert row.MEAN_USG_PCT == pytest.approx(np.mean([0.10, 0.20, 0.30, 0.40]))

    def test_player_attributes_are_matched_within_season(self):
        """A player's 2023-24 row must not be attached to a 2024-25 lineup."""
        players = make_players()
        stale = players.copy()
        stale["SEASON"] = "2023-24"
        stale["USG_PCT"] = 0.99
        feats = lc.build_lineup_player_features(
            make_lineups(), pd.concat([players, stale], ignore_index=True),
            metrics=["USG_PCT"],
        )
        row = feats[feats.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        assert row.MEAN_USG_PCT == pytest.approx(0.30)  # not 0.99

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError):
            lc.build_lineup_player_features(
                make_lineups(), make_players(), metrics=["NOT_A_COLUMN"]
            )

    def test_one_row_per_lineup(self):
        feats = lc.build_lineup_player_features(
            make_lineups(), make_players(), metrics=["USG_PCT"]
        )
        assert not feats.duplicated(["GROUP_ID", "SEASON"]).any()


# --------------------------------------------------------------------------
# Archetype composition
# --------------------------------------------------------------------------
class TestArchetypeComposition:
    def _features(self):
        return lc.build_lineup_player_features(
            make_lineups(), make_players(), metrics=["USG_PCT"]
        )

    def test_counts_sum_to_five_and_match_labels(self):
        out = lc.add_archetype_composition(
            self._features(), make_lineups(), make_clusters()
        )
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        # players 1,2 -> 0 ; 3,4 -> 1 ; 5 -> 2
        assert (row.N_CLUSTER_0, row.N_CLUSTER_1, row.N_CLUSTER_2) == (2, 2, 1)
        assert row.ARCHETYPE_MIX == "2-2-1"
        assert row.N_ARCHETYPES_PRESENT == 3

    def test_partially_labelled_lineup_is_nan_not_wrong(self):
        out = lc.add_archetype_composition(
            self._features(), make_lineups(), make_clusters()
        )
        # lineup C contains player 11, which has no cluster label
        row = out[out.GROUP_ID == "-7-8-9-10-11-"].iloc[0]
        assert row.CLUSTERS_MATCHED == 4
        assert pd.isna(row.N_CLUSTER_0)
        assert pd.isna(row.ARCHETYPE_MIX)

    def test_row_count_preserved(self):
        feats = self._features()
        out = lc.add_archetype_composition(feats, make_lineups(), make_clusters())
        assert len(out) == len(feats)

    def test_labels_are_matched_within_season(self):
        clusters = make_clusters()
        stale = clusters.copy()
        stale["SEASON"] = "2023-24"
        stale["CLUSTER"] = 2
        out = lc.add_archetype_composition(
            self._features(), make_lineups(),
            pd.concat([clusters, stale], ignore_index=True),
        )
        row = out[out.GROUP_ID == "-1-2-3-4-5-"].iloc[0]
        assert row.ARCHETYPE_MIX == "2-2-1"


# --------------------------------------------------------------------------
# Reliability and scoring
# --------------------------------------------------------------------------
class TestEstimateReliability:
    def test_recovers_known_variance_components(self):
        """Simulate true spread SD=4 plus sampling noise k/POSS with k=24000."""
        rng = np.random.default_rng(0)
        n = 60_000
        poss = rng.integers(25, 1500, n)
        true = rng.normal(0, 4.0, n)
        k = 24_000.0
        obs = true + rng.normal(0, np.sqrt(k / poss))
        out = lc.estimate_reliability(
            pd.DataFrame({"NET_RATING": obs, "POSS": poss}), n_bins=15
        )
        assert out["var_true"] == pytest.approx(16.0, rel=0.25)
        assert out["var_noise_coefficient"] == pytest.approx(k, rel=0.25)

    def test_pure_noise_data_reports_near_zero_true_variance(self):
        rng = np.random.default_rng(1)
        n = 40_000
        poss = rng.integers(25, 1500, n)
        obs = rng.normal(0, np.sqrt(24_000.0 / poss))
        out = lc.estimate_reliability(
            pd.DataFrame({"NET_RATING": obs, "POSS": poss}), n_bins=15
        )
        assert out["var_true"] < 1.0
        assert out["reliability_at_median_poss"] < 0.1

    def test_var_true_is_never_negative(self):
        rng = np.random.default_rng(2)
        n = 20_000
        poss = rng.integers(25, 1500, n)
        out = lc.estimate_reliability(
            pd.DataFrame(
                {"NET_RATING": rng.normal(0, np.sqrt(50_000.0 / poss)), "POSS": poss}
            ),
            n_bins=12,
        )
        assert out["var_true"] >= 0.0

    def test_too_few_rows_raises(self):
        with pytest.raises(ValueError):
            lc.estimate_reliability(
                pd.DataFrame({"NET_RATING": [1.0, 2.0], "POSS": [10, 20]})
            )


class TestPossessionWeightedR2:
    def test_perfect_prediction_is_one(self):
        y = [1.0, 5.0, 9.0]
        assert lc.possession_weighted_r2(y, y, [10, 20, 30]) == pytest.approx(1.0)

    def test_predicting_the_weighted_mean_is_zero(self):
        y, w = np.array([1.0, 5.0, 9.0]), np.array([10.0, 20.0, 30.0])
        mean = np.sum(y * w) / np.sum(w)
        assert lc.possession_weighted_r2(y, np.full(3, mean), w) == pytest.approx(0.0)

    def test_worse_than_baseline_is_negative(self):
        assert lc.possession_weighted_r2([1.0, 5.0, 9.0], [9.0, 5.0, 1.0], [1, 1, 1]) < 0

    def test_external_baseline_is_honoured(self):
        y = [10.0, 10.0]
        # Predicting 10 exactly, but baseline also 10 -> ss_tot 0 -> nan
        assert math.isnan(lc.possession_weighted_r2(y, y, [1, 1], baseline=10.0))

    def test_weights_actually_change_the_score(self):
        y, p = [0.0, 10.0], [0.0, 0.0]
        a = lc.possession_weighted_r2(y, p, [1, 1])
        b = lc.possession_weighted_r2(y, p, [1000, 1])
        assert a != pytest.approx(b)


# --------------------------------------------------------------------------
# Integrity of the committed datasets (skipped if the files are missing)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not DATA.is_dir(), reason="basketball_data/ not present")
class TestCommittedData:
    def test_five_man_lineups_all_parse_to_five_distinct_players(self):
        lineups = lc.add_player_ids(pd.read_csv(DATA / "lineups_five_seasons.csv"))
        assert (lineups.N_PLAYERS == 5).all()
        assert (lineups.N_UNIQUE_PLAYERS == 5).all()

    def test_lineup_keys_are_unique(self):
        lineups = pd.read_csv(DATA / "lineups_five_seasons.csv")
        assert not lineups.duplicated(["GROUP_ID", "SEASON"]).any()

    def test_every_lineup_member_exists_in_player_stats(self):
        lineups = lc.add_player_ids(pd.read_csv(DATA / "lineups_five_seasons.csv"))
        players = pd.read_csv(DATA / "player_stats_five_seasons.csv")
        known = set(zip(players.PLAYER_ID, players.SEASON))
        missing = sum(
            (pid, row.SEASON) not in known
            for row in lineups.itertuples(index=False)
            for pid in row.PLAYER_IDS
        )
        assert missing == 0

    def test_net_rating_equals_off_minus_def(self):
        lineups = pd.read_csv(DATA / "lineups_five_seasons.csv")
        diff = (lineups.NET_RATING - (lineups.OFF_RATING - lineups.DEF_RATING)).abs()
        assert diff.max() < 0.2  # endpoint rounds each rating to 0.1

    def test_player_stats_have_one_row_per_player_season(self):
        players = pd.read_csv(DATA / "player_stats_five_seasons.csv")
        assert not players.duplicated(["PLAYER_ID", "SEASON"]).any()

    @pytest.mark.skipif(
        not (DATA / "lineups_pairs_five_seasons.csv").exists(),
        reason="two-man lineup data not collected",
    )
    def test_two_man_lineups_parse_to_two_distinct_players(self):
        pairs = lc.add_player_ids(pd.read_csv(DATA / "lineups_pairs_five_seasons.csv"))
        assert (pairs.N_UNIQUE_PLAYERS == 2).all()

    @pytest.mark.skipif(
        not (DATA / "pair_co_play_proxies_five_seasons.csv").exists(),
        reason="derived pair file not built",
    )
    def test_derived_pair_output_structure(self):
        pairs = pd.read_csv(DATA / "pair_co_play_proxies_five_seasons.csv")
        for col in [
            "SEASON", "TEAM", "PLAYER_A_ID", "PLAYER_B_ID", "SAMPLED_LINEUPS",
            "SUM_LINEUP_POSS", "PAIR_NET_PROXY", "PLAYER_A", "PLAYER_B",
        ]:
            assert col in pairs.columns, col
        assert (pairs.PLAYER_A_ID < pairs.PLAYER_B_ID).all()
        assert not pairs.duplicated(
            ["SEASON", "TEAM", "PLAYER_A_ID", "PLAYER_B_ID"]
        ).any()
        assert (pairs.SUM_LINEUP_POSS > 0).all()
        assert pairs.PAIR_NET_PROXY.notna().all()
        # The proxy is a weighted average of observed lineup net ratings, so it
        # cannot fall outside their range.
        assert pairs.PAIR_NET_PROXY.between(-120, 120).all()

    @pytest.mark.skipif(
        not (DATA / "lineup_player_features_five_seasons.csv").exists(),
        reason="derived feature file not built",
    )
    def test_derived_feature_output_structure(self):
        feats = pd.read_csv(DATA / "lineup_player_features_five_seasons.csv")
        assert not feats.duplicated(["GROUP_ID", "SEASON"]).any()
        assert (feats.MATCHED_PLAYERS == 5).all()
        assert (feats.SD_USG_PCT >= 0).all()
        assert (feats.MAX_USG_PCT >= feats.MEAN_USG_PCT - 1e-9).all()
        assert (feats.MIN_USG_PCT <= feats.MEAN_USG_PCT + 1e-9).all()
        labelled = feats[feats.ARCHETYPE_MIX.notna()]
        counts = labelled[[c for c in feats.columns if c.startswith("N_CLUSTER_")]]
        assert (counts.sum(axis=1) == 5).all()
