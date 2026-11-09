"""Lineup chemistry and player-compatibility helpers for Eclipse Analytics.

This module holds the reusable transformations behind
``data_exploration_lineup_chemistry.ipynb``.  Everything here is a pure
function over DataFrames so that it can be unit tested with small synthetic
fixtures (see ``tests/test_lineup_chemistry.py``).

Vocabulary used consistently throughout the project
---------------------------------------------------
observed lineup performance
    ``NET_RATING`` measured by the NBA for a specific five-player group.  A
    description of what happened, not an estimate of ability.
measured pair performance
    ``NET_RATING`` for a two-player group, straight from
    ``LeagueDashLineups(group_quantity=2)``.  Also observed, but on a much
    larger sample than any single five-man lineup.
pair co-play proxy
    A possession-weighted average of the net ratings of the five-man lineups a
    pair appears in.  **An estimate, not a chemistry score**: it is confounded
    by the other three teammates, opponents, coaching and usage patterns, and
    the contributing lineups overlap heavily.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "SEASONS",
    "CLUSTER_LABELS",
    "parse_group_ids",
    "add_player_ids",
    "weighted_mean",
    "explode_lineup_pairs",
    "aggregate_pair_proxies",
    "shrink_to_prior",
    "attach_measured_pair_ratings",
    "build_lineup_player_features",
    "add_archetype_composition",
    "estimate_reliability",
    "possession_weighted_r2",
]

SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]

#: Plain-language names for Eliot's K=3 K-means clusters, read off
#: ``basketball_data/cluster_profiles_five_seasons.csv``.  Cluster 0 rebounds
#: and finishes inside with a low three-point rate; cluster 1 is a low-usage
#: perimeter group with by far the highest three-point attempt rate; cluster 2
#: has the highest usage, assist rate and PIE.
CLUSTER_LABELS = {
    0: "Interior finisher",
    1: "Perimeter role player",
    2: "High-usage creator",
}

LINEUP_SIZE = 5


# --------------------------------------------------------------------------
# Parsing and joining
# --------------------------------------------------------------------------
def parse_group_ids(group_id: object) -> tuple[int, ...]:
    """Parse a ``LeagueDashLineups`` ``GROUP_ID`` into sorted player IDs.

    The endpoint formats the field as ``"-201142-1641708-"``.  IDs are
    returned sorted so that a group has one canonical representation
    regardless of the order the API happens to use.

    >>> parse_group_ids("-203999-201142-")
    (201142, 203999)
    >>> parse_group_ids(float("nan"))
    ()
    """
    if group_id is None or (isinstance(group_id, float) and np.isnan(group_id)):
        return ()
    parts = str(group_id).strip().strip("-").split("-")
    ids = [int(p) for p in parts if p.strip().isdigit()]
    return tuple(sorted(ids))


def add_player_ids(
    lineups: pd.DataFrame,
    *,
    group_col: str = "GROUP_ID",
    out_col: str = "PLAYER_IDS",
) -> pd.DataFrame:
    """Return a copy of ``lineups`` with a parsed tuple-of-IDs column plus
    ``N_PLAYERS`` (parsed count) and ``N_UNIQUE_PLAYERS`` (distinct count).

    Keeping both counts makes a malformed group visible instead of silently
    producing a short or duplicated player list.
    """
    out = lineups.copy()
    out[out_col] = out[group_col].map(parse_group_ids)
    out["N_PLAYERS"] = out[out_col].map(len)
    out["N_UNIQUE_PLAYERS"] = out[out_col].map(lambda ids: len(set(ids)))
    return out


# --------------------------------------------------------------------------
# Weighted statistics
# --------------------------------------------------------------------------
def weighted_mean(values, weights) -> float:
    """Weighted mean that ignores pairs where either side is missing.

    Returns ``nan`` when no usable weight remains, rather than raising or
    silently falling back to an unweighted mean.

    >>> weighted_mean([10.0, 0.0], [3, 1])
    7.5
    >>> weighted_mean([1.0, 2.0], [0, 0])
    nan
    """
    v = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(pd.Series(list(weights)), errors="coerce").to_numpy(dtype=float)
    if v.size != w.size:
        raise ValueError(f"values/weights length mismatch: {v.size} vs {w.size}")
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------
def explode_lineup_pairs(
    lineups: pd.DataFrame,
    *,
    lineup_size: int = LINEUP_SIZE,
    team_col: str = "TEAM_ABBREVIATION",
    season_col: str = "SEASON",
) -> pd.DataFrame:
    """One row per (player pair, contributing five-man lineup).

    A group of ``lineup_size`` distinct players yields
    ``C(lineup_size, 2)`` = 10 unordered pairs.  Groups that do not parse to
    exactly ``lineup_size`` distinct IDs are dropped (they cannot be trusted),
    and ``PLAYER_A_ID < PLAYER_B_ID`` always holds so a pair has a single key.

    The returned frame is intentionally *long*: aggregation is a separate,
    separately testable step.
    """
    frame = lineups if "PLAYER_IDS" in lineups.columns else add_player_ids(lineups)
    has_season = season_col in frame.columns

    records: list[dict] = []
    for row in frame.itertuples(index=False):
        ids = tuple(row.PLAYER_IDS)
        if len(ids) != lineup_size or len(set(ids)) != lineup_size:
            continue
        for a, b in itertools.combinations(sorted(ids), 2):
            records.append(
                {
                    "SEASON": getattr(row, season_col) if has_season else pd.NA,
                    "TEAM": getattr(row, team_col),
                    "PLAYER_A_ID": a,
                    "PLAYER_B_ID": b,
                    "LINEUP_GROUP_ID": row.GROUP_ID,
                    "LINEUP_MIN": row.MIN,
                    "LINEUP_POSS": row.POSS,
                    "LINEUP_NET_RATING": row.NET_RATING,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[
            "SEASON", "TEAM", "PLAYER_A_ID", "PLAYER_B_ID", "LINEUP_GROUP_ID",
            "LINEUP_MIN", "LINEUP_POSS", "LINEUP_NET_RATING",
        ],
    )


def aggregate_pair_proxies(
    pair_lineups: pd.DataFrame,
    *,
    keys: tuple[str, ...] = ("SEASON", "TEAM", "PLAYER_A_ID", "PLAYER_B_ID"),
) -> pd.DataFrame:
    """Collapse the long pair/lineup table into one row per pair.

    ``PAIR_NET_PROXY`` is the possession-weighted mean of the contributing
    lineups' net ratings::

        PAIR_NET_PROXY = sum(NET_RATING_i * POSS_i) / sum(POSS_i)

    Grouping includes ``TEAM`` so that a pair traded apart and reunited on a
    different roster is not pooled across teams.

    ``SAMPLED_LINEUPS`` counts *distinct* contributing five-man lineups, and
    ``SUM_LINEUP_POSS`` deliberately double-counts possessions across
    overlapping lineups — it measures how much evidence feeds the proxy, not
    how many possessions the pair actually shared.  Use
    ``attach_measured_pair_ratings`` for the real shared-possession count.
    """
    if pair_lineups.empty:
        return pd.DataFrame(
            columns=[*keys, "SAMPLED_LINEUPS", "SUM_LINEUP_MIN",
                     "SUM_LINEUP_POSS", "PAIR_NET_PROXY"]
        )

    df = pair_lineups.copy()
    df["LINEUP_POSS"] = pd.to_numeric(df["LINEUP_POSS"], errors="coerce")
    df["LINEUP_NET_RATING"] = pd.to_numeric(df["LINEUP_NET_RATING"], errors="coerce")
    # A lineup with no possessions or no rating carries no information.
    df = df[df["LINEUP_POSS"].gt(0) & df["LINEUP_NET_RATING"].notna()]
    if df.empty:
        return pd.DataFrame(
            columns=[*keys, "SAMPLED_LINEUPS", "SUM_LINEUP_MIN",
                     "SUM_LINEUP_POSS", "PAIR_NET_PROXY"]
        )

    df["_WEIGHTED_NET"] = df["LINEUP_NET_RATING"] * df["LINEUP_POSS"]
    grouped = df.groupby(list(keys), as_index=False, dropna=False).agg(
        SAMPLED_LINEUPS=("LINEUP_GROUP_ID", "nunique"),
        SUM_LINEUP_MIN=("LINEUP_MIN", "sum"),
        SUM_LINEUP_POSS=("LINEUP_POSS", "sum"),
        _WEIGHTED_NET=("_WEIGHTED_NET", "sum"),
    )
    grouped["PAIR_NET_PROXY"] = (
        grouped["_WEIGHTED_NET"] / grouped["SUM_LINEUP_POSS"].replace(0, np.nan)
    )
    return grouped.drop(columns="_WEIGHTED_NET")


def shrink_to_prior(
    values: pd.Series,
    weights: pd.Series,
    *,
    prior: float | None = None,
    prior_weight: float = 200.0,
) -> pd.Series:
    """Shrink noisy rate estimates toward a prior mean.

    ``(value * weight + prior * prior_weight) / (weight + prior_weight)``

    Small-sample net ratings are mostly noise, so an unshrunk leaderboard is
    dominated by lineups with ~30 possessions.  ``prior_weight`` is the
    possession count at which an observation gets equal weight with the prior;
    it should be set from ``estimate_reliability``, not guessed.

    ``prior`` defaults to the weighted mean of ``values``.
    """
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").clip(lower=0).fillna(0)
    if prior is None:
        prior = weighted_mean(v, w)
    if prior_weight <= 0:
        raise ValueError("prior_weight must be positive")
    return (v.fillna(prior) * w + prior * prior_weight) / (w + prior_weight)


def attach_measured_pair_ratings(
    pair_proxies: pd.DataFrame,
    two_man_lineups: pd.DataFrame,
    *,
    team_col: str = "TEAM_ABBREVIATION",
) -> pd.DataFrame:
    """Left-join measured two-man ratings onto the five-man-derived proxies.

    ``two_man_lineups`` is ``LeagueDashLineups(group_quantity=2)`` output.  The
    result adds ``MEASURED_PAIR_NET_RATING``, ``MEASURED_PAIR_POSS`` and
    ``MEASURED_PAIR_MIN``, giving a direct check of whether the proxy tracks
    what the pair actually did on the floor.
    """
    two = add_player_ids(two_man_lineups)
    two = two[two["N_UNIQUE_PLAYERS"].eq(2)].copy()
    two["PLAYER_A_ID"] = two["PLAYER_IDS"].map(lambda ids: ids[0])
    two["PLAYER_B_ID"] = two["PLAYER_IDS"].map(lambda ids: ids[1])
    two["TEAM"] = two[team_col]

    keys = ["SEASON", "TEAM", "PLAYER_A_ID", "PLAYER_B_ID"]
    keys = [k for k in keys if k in two.columns and k in pair_proxies.columns]
    measured = (
        two[keys + ["NET_RATING", "POSS", "MIN"]]
        .rename(
            columns={
                "NET_RATING": "MEASURED_PAIR_NET_RATING",
                "POSS": "MEASURED_PAIR_POSS",
                "MIN": "MEASURED_PAIR_MIN",
            }
        )
        # Guard the join: a duplicate key would silently inflate the output.
        .drop_duplicates(subset=keys, keep="first")
    )
    return pair_proxies.merge(measured, on=keys, how="left", validate="one_to_one")


# --------------------------------------------------------------------------
# Lineup-level features
# --------------------------------------------------------------------------
#: Per-player inputs for lineup composition features.  Rate and efficiency
#: statistics are preferred over raw counting stats because a lineup's mean
#: ``AST`` mostly reflects how many minutes its members play.
DEFAULT_PLAYER_METRICS = [
    "USG_PCT",
    "TS_PCT",
    "AST_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "TM_TOV_PCT",
    "THREE_POINT_ATTEMPT_RATE",
    "AST_TO_TOV",
]


def _ensure_derived_player_metrics(players: pd.DataFrame) -> pd.DataFrame:
    """Add the two ratios Eliot's clustering uses, if absent."""
    out = players.copy()
    if "THREE_POINT_ATTEMPT_RATE" not in out.columns:
        out["THREE_POINT_ATTEMPT_RATE"] = out["FG3A"] / out["FGA"].replace(0, np.nan)
    if "AST_TO_TOV" not in out.columns:
        out["AST_TO_TOV"] = out["AST"] / out["TOV"].replace(0, np.nan)
    return out


def build_lineup_player_features(
    lineups: pd.DataFrame,
    players: pd.DataFrame,
    *,
    metrics: list[str] | None = None,
    lineup_size: int = LINEUP_SIZE,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Aggregate member attributes into lineup-level composition features.

    For each metric the output carries ``MEAN_`` (overall level), ``SD_``
    (population SD, ``ddof=0`` — a fixed five-player roster, not a sample),
    ``MAX_`` and ``MIN_``.  Spread and extremes are what encode *role
    diversity*; a mean alone cannot distinguish five average-usage players from
    one star plus four low-usage teammates.

    ``players`` is joined on ``(PLAYER_ID, SEASON)`` when both frames carry a
    ``SEASON`` column, so a player's 2021-22 attributes are never attached to a
    2024-25 lineup.  With ``require_complete=True`` a lineup is dropped unless
    every member is found, instead of averaging over whoever happened to match.
    """
    metrics = list(metrics or DEFAULT_PLAYER_METRICS)
    frame = lineups if "PLAYER_IDS" in lineups.columns else add_player_ids(lineups)
    plr = _ensure_derived_player_metrics(players)

    missing = [m for m in metrics if m not in plr.columns]
    if missing:
        raise KeyError(f"player metrics not found: {missing}")

    use_season = "SEASON" in frame.columns and "SEASON" in plr.columns
    index_cols = ["PLAYER_ID", "SEASON"] if use_season else ["PLAYER_ID"]
    lookup = (
        plr.drop_duplicates(subset=index_cols)
        .set_index(index_cols)[metrics]
        .apply(pd.to_numeric, errors="coerce")
    )

    records: list[dict] = []
    for row in frame.itertuples(index=False):
        ids = tuple(row.PLAYER_IDS)
        if len(ids) != lineup_size or len(set(ids)) != lineup_size:
            continue
        keys = [(pid, row.SEASON) for pid in ids] if use_season else list(ids)
        found = [k for k in keys if k in lookup.index]
        if require_complete and len(found) != lineup_size:
            continue
        if not found:
            continue
        vals = lookup.loc[found]

        record: dict = {
            "GROUP_ID": row.GROUP_ID,
            "GROUP_NAME": getattr(row, "GROUP_NAME", None),
            "TEAM": getattr(row, "TEAM_ABBREVIATION", None),
            "MIN": row.MIN,
            "POSS": row.POSS,
            "NET_RATING": row.NET_RATING,
            "OFF_RATING": getattr(row, "OFF_RATING", np.nan),
            "DEF_RATING": getattr(row, "DEF_RATING", np.nan),
            "GP": getattr(row, "GP", np.nan),
            "MATCHED_PLAYERS": len(found),
        }
        if use_season:
            record["SEASON"] = row.SEASON
        for metric in metrics:
            col = vals[metric]
            record[f"MEAN_{metric}"] = col.mean()
            record[f"SD_{metric}"] = col.std(ddof=0)
            record[f"MAX_{metric}"] = col.max()
            record[f"MIN_{metric}"] = col.min()
        records.append(record)

    return pd.DataFrame.from_records(records)


def add_archetype_composition(
    features: pd.DataFrame,
    lineups: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    cluster_col: str = "CLUSTER",
    require_complete: bool = True,
) -> pd.DataFrame:
    """Add archetype counts and a readable composition string to ``features``.

    ``clusters`` is Eliot's ``player_clusters_five_seasons.csv``; its labels are
    reused rather than refitted so the two analyses cannot drift apart.  Adds
    ``N_CLUSTER_<k>`` counts, ``ARCHETYPE_MIX`` (e.g. ``"1-3-1"``),
    ``N_ARCHETYPES_PRESENT`` and ``CLUSTERS_MATCHED``.

    With ``require_complete=True``, lineups where any member lacks a label get
    ``NaN`` counts, keeping partially-labelled lineups out of archetype
    comparisons without dropping them from the feature table.
    """
    frame = lineups if "PLAYER_IDS" in lineups.columns else add_player_ids(lineups)
    use_season = "SEASON" in frame.columns and "SEASON" in clusters.columns
    index_cols = ["PLAYER_ID", "SEASON"] if use_season else ["PLAYER_ID"]
    label = (
        clusters.drop_duplicates(subset=index_cols)
        .set_index(index_cols)[cluster_col]
    )
    cluster_ids = sorted(pd.unique(label.dropna()).tolist())

    join_cols = ["GROUP_ID", "SEASON"] if use_season else ["GROUP_ID"]
    records: list[dict] = []
    for row in frame.itertuples(index=False):
        ids = tuple(row.PLAYER_IDS)
        keys = [(pid, row.SEASON) for pid in ids] if use_season else list(ids)
        labels = [label[k] for k in keys if k in label.index]
        record: dict = {"GROUP_ID": row.GROUP_ID, "CLUSTERS_MATCHED": len(labels)}
        if use_season:
            record["SEASON"] = row.SEASON
        complete = len(labels) == len(ids)
        counts = pd.Series(labels).value_counts() if labels else pd.Series(dtype=int)
        for cid in cluster_ids:
            record[f"N_CLUSTER_{int(cid)}"] = (
                int(counts.get(cid, 0)) if (complete or not require_complete) else np.nan
            )
        if complete or not require_complete:
            record["ARCHETYPE_MIX"] = "-".join(
                str(int(counts.get(cid, 0))) for cid in cluster_ids
            )
            record["N_ARCHETYPES_PRESENT"] = int(
                sum(counts.get(cid, 0) > 0 for cid in cluster_ids)
            )
        else:
            record["ARCHETYPE_MIX"] = None
            record["N_ARCHETYPES_PRESENT"] = np.nan
        records.append(record)

    comp = pd.DataFrame.from_records(records).drop_duplicates(subset=join_cols)
    return features.merge(comp, on=join_cols, how="left", validate="one_to_one")


# --------------------------------------------------------------------------
# How much signal is even there?
# --------------------------------------------------------------------------
def estimate_reliability(
    df: pd.DataFrame,
    *,
    value_col: str = "NET_RATING",
    weight_col: str = "POSS",
    n_bins: int = 12,
) -> dict:
    """Split observed net-rating variance into true spread and sampling noise.

    A lineup's net rating is a mean over ``POSS`` possessions, so its sampling
    variance falls roughly as ``k / POSS``.  Across possession bins::

        Var(observed | POSS) = Var(true) + k / POSS

    Fitting that line by OLS on ``1 / POSS`` gives an intercept (the spread in
    genuine lineup quality) and a slope ``k`` (noise scale).  The returned
    ``equal_weight_poss = k / Var(true)`` is the possession count at which an
    observation deserves equal weight with the prior, i.e. the principled
    ``prior_weight`` for :func:`shrink_to_prior`.

    Returns a dict with ``var_true``, ``var_noise_coefficient``,
    ``equal_weight_poss``, ``reliability_at_median_poss``, ``n_bins_used`` and
    the per-bin table.  ``var_true`` is clipped at 0: a negative intercept
    means the data are consistent with no detectable true spread at all.
    """
    data = df[[value_col, weight_col]].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[data[weight_col] > 0]
    if len(data) < n_bins * 3:
        raise ValueError(f"need at least {n_bins * 3} rows, got {len(data)}")

    data = data.assign(
        _bin=pd.qcut(data[weight_col], n_bins, labels=False, duplicates="drop")
    )
    grp = data.groupby("_bin").agg(
        n=(value_col, "size"),
        var_obs=(value_col, lambda s: s.var(ddof=1)),
        mean_poss=(weight_col, "mean"),
        inv_poss=(weight_col, lambda s: (1.0 / s).mean()),
    )
    grp = grp[grp["n"] >= 3].dropna()
    if len(grp) < 3:
        raise ValueError("not enough usable possession bins")

    slope, intercept = np.polyfit(grp["inv_poss"], grp["var_obs"], 1)
    var_true = max(float(intercept), 0.0)
    k = float(slope)
    median_poss = float(data[weight_col].median())
    noise_at_median = k / median_poss if median_poss > 0 else np.nan
    denom = var_true + noise_at_median
    return {
        "var_true": var_true,
        "sd_true": float(np.sqrt(var_true)),
        "var_noise_coefficient": k,
        "equal_weight_poss": float(k / var_true) if var_true > 0 else float("inf"),
        "median_poss": median_poss,
        "reliability_at_median_poss": float(var_true / denom) if denom > 0 else 0.0,
        "n_bins_used": int(len(grp)),
        "bins": grp.reset_index(),
    }


def possession_weighted_r2(y_true, y_pred, weights, *, baseline=None) -> float:
    """Possession-weighted R^2 against a constant baseline.

    ``baseline`` defaults to the weighted mean of ``y_true``.  When evaluating a
    held-out season, pass the *training* mean so the score does not borrow the
    test set's own average.  A negative value means the model is worse than
    that constant.
    """
    y = pd.to_numeric(pd.Series(np.asarray(y_true, dtype=float)), errors="coerce")
    p = pd.to_numeric(pd.Series(np.asarray(y_pred, dtype=float)), errors="coerce")
    w = pd.to_numeric(pd.Series(np.asarray(weights, dtype=float)), errors="coerce")
    ok = y.notna() & p.notna() & w.notna() & (w > 0)
    y, p, w = y[ok].to_numpy(), p[ok].to_numpy(), w[ok].to_numpy()
    if y.size == 0:
        return float("nan")
    if baseline is None:
        baseline = float(np.sum(y * w) / np.sum(w))
    ss_res = float(np.sum(w * (y - p) ** 2))
    ss_tot = float(np.sum(w * (y - baseline) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------
# Convenience loaders
# --------------------------------------------------------------------------
def data_dir(start: Path | str = ".") -> Path:
    """Locate ``basketball_data`` from the repo root or a subdirectory."""
    start = Path(start)
    for candidate in (start / "basketball_data", start / ".." / "basketball_data"):
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "basketball_data not found - run from the repository root"
    )
