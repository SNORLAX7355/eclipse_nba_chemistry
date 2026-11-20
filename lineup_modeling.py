"""Build the lineup modelling table and evaluate ORTG / DRTG models.

This is the modelling counterpart to ``lineup_chemistry.py``. It reimplements
the dataset builder prototyped in ``modeling.ipynb`` as a vectorised, tested
function, then adds the feature-set, normalisation, model and
possession-threshold experiments that `modeling.ipynb` left open.

Three correctness notes carried over from auditing that prototype:

* ``TOV_PCT`` is not a column in ``using_dataset/full_data.csv`` (the real name
  is ``TM_TOV_PCT``). The prototype filtered feature lists to existing columns,
  so the turnover feature was silently dropped. :data:`BOX_SCORE_COLS` uses the
  real name.
* ~1% of players carry an **all-zero** synergy row, which means "no play-type
  data", not "never ran this play". Averaging those zeros drags a lineup's
  play-type profile toward zero. :func:`build_lineup_dataset` converts all-zero
  synergy rows to ``NaN`` by default and reports how many members were affected.
* The synergy percentages are **compositional** - each player's offensive shares
  sum to 1 - so one column per block is redundant. Harmless for trees, but it
  makes linear-model coefficients unidentifiable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "BOX_SCORE_COLS",
    "LINEUP_USAGE_COLS",
    "TARGETS",
    "CLUSTER_LABELS_PLAYSTYLE",
    "parse_group_id",
    "explode_lineup_members",
    "build_lineup_dataset",
    "feature_sets",
    "rating_reliability",
    "weighted_rmse",
]

#: Per-player box-score / advanced columns. ``TM_TOV_PCT`` is the real column
#: name for team turnover percentage while a player is on the floor.
BOX_SCORE_COLS = [
    "USG_PCT",
    "AST_PCT",
    "REB_PCT",
    "OREB_PCT",
    "DREB_PCT",
    "TM_TOV_PCT",
    "TS_PCT",
    "EFG_PCT",
    "FG3A",
    "FG3_PCT",
    "STL",
    "BLK",
]

#: Lineup-level usage columns. These describe *how much the coach played the
#: lineup*, which is a consequence of the lineup performing well, not a cause.
#: Kept separate so their effect can be measured rather than assumed.
LINEUP_USAGE_COLS = ["MIN", "POSS", "GP"]

TARGETS = ["OFF_RATING", "DEF_RATING"]

#: Play-style clusters from the team's K=6 clustering, as described in
#: ``cluster_descriptions_and_players.txt`` on the ``past-5-seasons`` branch.
#: Clusters 0 and 2 are noted there as heavily overlapping.
CLUSTER_LABELS_PLAYSTYLE = {
    0: "Interior / two-way big",
    1: "Primary ball handler",
    2: "Interior big (less defensive)",
    3: "Off-ball role player / defensive wing",
    4: "Spot-up specialist",
    5: "Off-ball shooter",
}


# --------------------------------------------------------------------------
# Lineup -> member expansion
# --------------------------------------------------------------------------
def parse_group_id(group_id: object, *, size: int = 5) -> list[int] | None:
    """Parse ``"-201145-203115-..."`` into a list of ``size`` player IDs.

    Returns ``None`` when the group does not contain exactly ``size`` distinct
    integer IDs, so a malformed group is dropped rather than silently producing
    a short lineup.

    >>> parse_group_id("-1-2-3-4-5-")
    [1, 2, 3, 4, 5]
    >>> parse_group_id("-1-2-3-4-") is None
    True
    """
    if group_id is None or (isinstance(group_id, float) and np.isnan(group_id)):
        return None
    parts = str(group_id).strip().strip("-").split("-")
    ids = [int(p) for p in parts if p.strip().lstrip("+").isdigit()]
    if len(ids) != size or len(set(ids)) != size:
        return None
    return ids


def explode_lineup_members(
    lineups: pd.DataFrame, *, size: int = 5, group_col: str = "GROUP_ID"
) -> pd.DataFrame:
    """Long table with one row per (lineup, member slot).

    Columns: ``GROUP_ID``, ``SEASON`` (when present), ``SLOT``, ``PLAYER_ID``.
    Malformed groups are dropped.
    """
    parsed = lineups[group_col].map(lambda g: parse_group_id(g, size=size))
    keep = parsed.notna()
    base_cols = [group_col] + (["SEASON"] if "SEASON" in lineups.columns else [])
    out = lineups.loc[keep, base_cols].copy()
    out["_ids"] = parsed[keep].to_numpy()
    out = out.explode("_ids", ignore_index=True)
    out["PLAYER_ID"] = out.pop("_ids").astype("int64")
    out["SLOT"] = out.groupby([group_col] + (["SEASON"] if "SEASON" in out else []),
                              dropna=False).cumcount()
    return out


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
def build_lineup_dataset(
    lineups: pd.DataFrame,
    players: pd.DataFrame,
    *,
    min_poss: int = 100,
    box_cols: list[str] | None = None,
    playtype_prefix: str = "SYNERGY_",
    cluster_col: str = "CLUSTER",
    zero_synergy_as_missing: bool = True,
    require_complete: bool = True,
    size: int = 5,
) -> pd.DataFrame:
    """Aggregate member attributes into one row per five-man lineup.

    Produces, for every box-score and play-type column, an ``AVG_`` mean and a
    ``STD_`` spread (population SD, ``ddof=0`` - a fixed five-player group is
    not a sample). Adds ``CLUSTER_<k>`` counts, the lineup's ``OFF_RATING`` /
    ``DEF_RATING`` / ``NET_RATING`` targets, and ``MIN`` / ``POSS`` / ``GP``.

    Joins are on ``(PLAYER_ID, SEASON)`` so a player's 2021-22 profile can never
    be attached to a 2024-25 lineup. With ``require_complete=True`` a lineup is
    kept only when all five members are found.

    Unlike the notebook prototype this is vectorised (merge + groupby rather
    than a per-row Python loop), which matters once the possession threshold is
    lowered and the table grows to ~10,000 rows.
    """
    box_cols = list(box_cols if box_cols is not None else BOX_SCORE_COLS)
    lineups = lineups.copy()
    if min_poss is not None:
        lineups = lineups[lineups["POSS"] >= min_poss]

    play_cols = sorted(c for c in players.columns
                       if c.startswith(playtype_prefix) and c.endswith("_PCT"))
    missing_box = [c for c in box_cols if c not in players.columns]
    if missing_box:
        raise KeyError(f"box-score columns not in player table: {missing_box}")

    plr = players.copy()
    if zero_synergy_as_missing and play_cols:
        for block in ("OFFENSIVE", "DEFENSIVE"):
            block_cols = [c for c in play_cols if block in c]
            if not block_cols:
                continue
            # An all-zero block means the endpoint returned no play-type data
            # for that player, not that he never ran any of these plays.
            blank = plr[block_cols].fillna(0).sum(axis=1).eq(0)
            plr.loc[blank, block_cols] = np.nan

    use_season = "SEASON" in lineups.columns and "SEASON" in plr.columns
    keys = ["PLAYER_ID", "SEASON"] if use_season else ["PLAYER_ID"]
    value_cols = box_cols + play_cols
    lookup = plr.drop_duplicates(subset=keys)[keys + value_cols + [cluster_col]]

    members = explode_lineup_members(lineups, size=size)
    merged = members.merge(lookup, on=keys, how="left", validate="many_to_one")

    group_keys = ["GROUP_ID", "SEASON"] if use_season else ["GROUP_ID"]
    grouped = merged.groupby(group_keys, dropna=False)

    found = grouped[cluster_col].size().rename("MEMBERS")
    matched = grouped[box_cols[0]].count().rename("MATCHED_PLAYERS")

    agg = grouped[value_cols].agg(["mean", lambda s: s.std(ddof=0)])
    agg.columns = [f"{'AVG' if stat == 'mean' else 'STD'}_{col}"
                   for col, stat in
                   [(c, ("mean" if s == "mean" else "std")) for c, s in agg.columns]]

    counts = (
        merged.pivot_table(index=group_keys, columns=cluster_col,
                           values="PLAYER_ID", aggfunc="count", fill_value=0)
        .rename(columns=lambda k: f"CLUSTER_{int(k)}")
    )
    all_clusters = sorted(pd.unique(lookup[cluster_col].dropna()).tolist())
    for k in all_clusters:
        name = f"CLUSTER_{int(k)}"
        if name not in counts.columns:
            counts[name] = 0
    counts = counts[[f"CLUSTER_{int(k)}" for k in all_clusters]]

    # How many members lacked play-type data, kept as a diagnostic so a lineup
    # built from a thin profile is identifiable rather than invisible.
    if play_cols:
        # A member counts as missing if *any* play-type column is absent -
        # probing a single column would miss a player whose offensive block is
        # blank but whose defensive block is intact.
        n_missing = (
            merged.assign(_m=merged[play_cols].isna().any(axis=1))
            .groupby(group_keys, dropna=False)["_m"].sum()
            .rename("N_PLAYTYPE_MISSING")
        )
    else:
        n_missing = pd.Series(0, index=agg.index, name="N_PLAYTYPE_MISSING")

    meta_cols = [c for c in ["MIN", "POSS", "GP", "TEAM_ABBREVIATION",
                             "OFF_RATING", "DEF_RATING", "NET_RATING"]
                 if c in lineups.columns]
    meta = lineups.set_index(group_keys)[meta_cols]
    meta = meta[~meta.index.duplicated(keep="first")]

    out = (
        meta.join(agg, how="inner")
        .join(counts, how="left")
        .join(found).join(matched).join(n_missing)
        .reset_index()
    )
    out[[c for c in out.columns if c.startswith("CLUSTER_")]] = (
        out[[c for c in out.columns if c.startswith("CLUSTER_")]].fillna(0).astype(int)
    )
    if require_complete:
        out = out[out["MATCHED_PLAYERS"].eq(size)].reset_index(drop=True)
    if "TEAM_ABBREVIATION" in out.columns:
        out = out.rename(columns={"TEAM_ABBREVIATION": "TEAM"})
    return out


# --------------------------------------------------------------------------
# Feature sets - the question "which columns are actually worth using?"
# --------------------------------------------------------------------------
def feature_sets(columns) -> dict[str, list[str]]:
    """Named, nested feature sets for the ablation.

    Each set answers one of the open questions from the task:

    ``avg_box``
        Average box-score profile only - the conservative baseline.
    ``avg_box_playtype``
        Plus average play-type shares. The combination suggested by the team.
    ``plus_std``
        Plus the within-lineup spread of every feature - tests whether role
        *diversity* adds anything over role *level*.
    ``plus_clusters``
        Plus archetype counts.
    ``plus_usage``
        Plus ``MIN`` / ``POSS`` / ``GP``. **Expected to look good and be
        useless**: a lineup plays more because it performed well, so these
        encode the coach's hindsight, not a property of the five players.
    """
    cols = list(columns)
    avg_box = [c for c in cols if c.startswith("AVG_") and "SYNERGY" not in c]
    avg_play = [c for c in cols if c.startswith("AVG_") and "SYNERGY" in c]
    std_all = [c for c in cols if c.startswith("STD_")]
    clusters = [c for c in cols if c.startswith("CLUSTER_")]
    usage = [c for c in LINEUP_USAGE_COLS if c in cols]

    sets = {
        "avg_box": avg_box,
        "avg_box_playtype": avg_box + avg_play,
        "plus_std": avg_box + avg_play + std_all,
        "plus_clusters": avg_box + avg_play + std_all + clusters,
        "plus_usage": avg_box + avg_play + std_all + clusters + usage,
    }
    return {k: v for k, v in sets.items() if v}


# --------------------------------------------------------------------------
# How much of the target is even learnable?
# --------------------------------------------------------------------------
def rating_reliability(
    df: pd.DataFrame, *, value_col: str, weight_col: str = "POSS", n_bins: int = 10
) -> dict:
    """Split observed variance of a rating into true spread and sampling noise.

    ``Var(observed | POSS) = Var(true) + k / POSS``, fitted by OLS across
    possession bins. ``reliability`` is the share of observed variance that is
    real at the median possession count - an upper bound on any model's R^2.
    """
    data = df[[value_col, weight_col]].apply(pd.to_numeric, errors="coerce").dropna()
    data = data[data[weight_col] > 0]
    if len(data) < n_bins * 3:
        raise ValueError(f"need >= {n_bins * 3} rows, got {len(data)}")
    data = data.assign(
        _bin=pd.qcut(data[weight_col], n_bins, labels=False, duplicates="drop")
    )
    grp = data.groupby("_bin").agg(
        n=(value_col, "size"),
        var_obs=(value_col, lambda s: s.var(ddof=1)),
        inv_poss=(weight_col, lambda s: (1.0 / s).mean()),
    )
    grp = grp[grp["n"] >= 3].dropna()
    if len(grp) < 3:
        raise ValueError("not enough usable possession bins")
    slope, intercept = np.polyfit(grp["inv_poss"], grp["var_obs"], 1)
    var_true = max(float(intercept), 0.0)
    median_poss = float(data[weight_col].median())
    noise = float(slope) / median_poss
    denom = var_true + noise
    return {
        "value_col": value_col,
        "var_true": var_true,
        "sd_true": float(np.sqrt(var_true)),
        "noise_k": float(slope),
        "median_poss": median_poss,
        "reliability": float(var_true / denom) if denom > 0 else 0.0,
        "equal_weight_poss": float(slope / var_true) if var_true > 0 else float("inf"),
    }


def weighted_rmse(y_true, y_pred, weights=None) -> float:
    """RMSE, optionally possession-weighted."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if weights is None:
        return float(np.sqrt(np.mean((y - p) ** 2)))
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.average((y - p) ** 2, weights=w)))


def data_paths(root: Path | str = ".") -> tuple[Path, Path]:
    """Locate the lineup and player tables.

    ``lineups_five_seasons.csv`` moved from ``basketball_data/`` to
    ``using_dataset/`` in commit 6333f18, so both locations are checked.
    """
    root = Path(root)
    lineup_candidates = [
        root / "using_dataset" / "lineups_five_seasons.csv",
        root / "basketball_data" / "lineups_five_seasons.csv",
    ]
    player_candidates = [root / "using_dataset" / "full_data.csv"]
    lineup = next((p for p in lineup_candidates if p.exists()), None)
    player = next((p for p in player_candidates if p.exists()), None)
    if lineup is None or player is None:
        raise FileNotFoundError(
            "could not find lineups_five_seasons.csv / full_data.csv - "
            "run from the repository root"
        )
    return lineup, player
