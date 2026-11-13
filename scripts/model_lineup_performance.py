"""Baseline models for five-man lineup net rating, with honest validation.

Design decisions that matter more than the model choice
-------------------------------------------------------
1. **Prior-season features only.** A lineup's outcome in season *t* is
   predicted from what its members did in season *t-1*.  Same-season player
   stats are computed from the very possessions the lineup outcome is measured
   on, so using them leaks the answer.  ``--demo-leakage`` quantifies how much.
2. **Temporal holdout.** Train on 2022-23 .. 2024-25, test on 2025-26.  No
   lineup, team or season appears in both.
3. **Possession weighting everywhere.** A 1,500-possession lineup carries far
   more information than a 25-possession one.
4. **Baseline is the training mean**, not the test mean, so the score cannot
   borrow the test set's own average.
5. **Compared against the signal ceiling.** Most of the variance in a single
   lineup's net rating is sampling noise; ``estimate_reliability`` says how
   much is left to explain.

Run from the repository root:
    python scripts/model_lineup_performance.py [--demo-leakage]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineup_chemistry as lc  # noqa: E402

TEST_SEASON = "2025-26"
RANDOM_STATE = 42


def shift_season_forward(df: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Relabel each row's SEASON as the *following* season.

    After this, joining on ``SEASON`` attaches season *t-1* attributes to a
    season *t* lineup.  Rows from the last season have no following season and
    are dropped.
    """
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}
    out = df.copy()
    out["SEASON"] = out["SEASON"].map(nxt)
    return out.dropna(subset=["SEASON"])


def build_design_matrix(data: Path, *, same_season: bool = False):
    """Lineup rows with composition features and archetype counts."""
    lineups = lc.add_player_ids(pd.read_csv(data / "lineups_five_seasons.csv"))
    players = pd.read_csv(data / "player_stats_five_seasons.csv")
    clusters = pd.read_csv(data / "player_clusters_five_seasons.csv")

    metrics = list(lc.DEFAULT_PLAYER_METRICS)
    if same_season:
        # Deliberately leaky: these are derived from the same possessions as
        # the target.  Only used by --demo-leakage.
        metrics += ["NET_RATING", "PIE", "PLUS_MINUS"]
    else:
        players = shift_season_forward(players, lc.SEASONS)
        clusters = shift_season_forward(clusters, lc.SEASONS)

    X = lc.build_lineup_player_features(lineups, players, metrics=metrics)
    X = lc.add_archetype_composition(X, lineups, clusters)

    features = [
        c for c in X.columns
        if c.startswith(("MEAN_", "SD_", "MAX_", "MIN_", "N_CLUSTER_")) and c != "MIN"
    ] + ["N_ARCHETYPES_PRESENT"]
    X = X.dropna(subset=features + ["NET_RATING", "POSS"])
    return X, features


def score(name: str, frame: pd.DataFrame, pred: np.ndarray, baseline: float) -> dict:
    return {
        "model": name,
        "n": len(frame),
        "wtd_R2_vs_baseline": lc.possession_weighted_r2(
            frame.NET_RATING, pred, frame.POSS, baseline=baseline
        ),
        "wtd_RMSE": float(
            np.sqrt(np.average((frame.NET_RATING - pred) ** 2, weights=frame.POSS))
        ),
        "pearson_r": float(np.corrcoef(frame.NET_RATING, pred)[0, 1])
        if len(frame) > 2 else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-leakage", action="store_true",
                    help="also fit with same-season features to show the inflation")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = lc.data_dir(root)

    X, features = build_design_matrix(data)
    print(f"modelling rows={len(X):,}  features={len(features)}")
    print("rows by season:", X.SEASON.value_counts().sort_index().to_dict())

    train = X[X.SEASON != TEST_SEASON]
    test = X[X.SEASON == TEST_SEASON]
    print(f"train={len(train):,} {sorted(train.SEASON.unique())}  "
          f"test={len(test):,} [{TEST_SEASON}]")

    baseline = lc.weighted_mean(train.NET_RATING, train.POSS)
    print(f"baseline (training possession-weighted mean NET_RATING) = {baseline:.3f}\n")

    models = {
        "Ridge (alpha=100)": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "Ridge (alpha=1000)": make_pipeline(StandardScaler(), Ridge(alpha=1000.0)),
        "RandomForest (300 trees, depth 6)": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=40,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    fitted = {}
    for name, model in models.items():
        kw = ({"ridge__sample_weight": train.POSS} if "Ridge" in name
              else {"sample_weight": train.POSS})
        model.fit(train[features], train.NET_RATING, **kw)
        fitted[name] = model

    for label, subset in (
        (f"held-out {TEST_SEASON}, all lineups", test),
        (f"held-out {TEST_SEASON}, POSS>=100", test[test.POSS >= 100]),
    ):
        rows = [score("Baseline (training mean)", subset,
                      np.full(len(subset), baseline), baseline)]
        for name, model in fitted.items():
            rows.append(score(name, subset, model.predict(subset[features]), baseline))
        print(f"=== {label} (n={len(subset)}) ===")
        print(pd.DataFrame(rows).round(4).to_string(index=False), "\n")

    # Group-aware CV inside the training seasons: a team-season never spans
    # folds, so repeated/overlapping lineups from one roster cannot leak.
    grouped = train.assign(G=train.TEAM + "_" + train.SEASON)
    folds = []
    for tr_idx, va_idx in GroupKFold(n_splits=5).split(
        grouped[features], grouped.NET_RATING, groups=grouped.G
    ):
        a, b = grouped.iloc[tr_idx], grouped.iloc[va_idx]
        m = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        m.fit(a[features], a.NET_RATING, ridge__sample_weight=a.POSS)
        folds.append(
            lc.possession_weighted_r2(
                b.NET_RATING, m.predict(b[features]), b.POSS,
                baseline=lc.weighted_mean(a.NET_RATING, a.POSS),
            )
        )
    print("=== GroupKFold CV on training seasons (groups = team-season), Ridge a=100 ===")
    print("fold weighted R2:", [round(f, 4) for f in folds],
          " mean:", round(float(np.mean(folds)), 4), "\n")

    # Signal ceiling from the full five-season sample (the per-season estimate
    # is unstable on a few hundred rows).
    rel = lc.estimate_reliability(
        pd.read_csv(data / "lineups_five_seasons.csv"),
        value_col="NET_RATING", weight_col="POSS",
    )
    print("=== signal ceiling (all 10,000 lineups) ===")
    print(f"sd of true lineup quality  = {rel['sd_true']:.2f} net-rating points")
    print(f"possessions for equal weight with prior = {rel['equal_weight_poss']:.0f}")
    print(f"reliability at median ({rel['median_poss']:.0f} poss) = "
          f"{rel['reliability_at_median_poss']:.3f}")
    print("  -> only that fraction of a single lineup's net-rating variance is "
          "real; the rest is sampling noise.\n")

    if args.demo_leakage:
        XL, fl = build_design_matrix(data, same_season=True)
        # Restrict to exactly the lineups the prior-season model could use, so
        # the only thing that changes is which season the features come from.
        same_rows = set(zip(X.GROUP_ID, X.SEASON))
        XL = XL[[k in same_rows for k in zip(XL.GROUP_ID, XL.SEASON)]]
        trl, tel = XL[XL.SEASON != TEST_SEASON], XL[XL.SEASON == TEST_SEASON]
        bl = lc.weighted_mean(trl.NET_RATING, trl.POSS)
        m = make_pipeline(StandardScaler(), Ridge(alpha=100.0))
        m.fit(trl[fl], trl.NET_RATING, ridge__sample_weight=trl.POSS)
        print("=== leakage demonstration: SAME-season features ===")
        print("(includes each member's same-season NET_RATING / PIE / PLUS_MINUS, "
              "computed from the very possessions being predicted)")
        rows = [
            score("same-season, all lineups", tel, m.predict(tel[fl]), bl),
            score("same-season, POSS>=100", tel[tel.POSS >= 100],
                  m.predict(tel[tel.POSS >= 100][fl]), bl),
        ]
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        print("\nCompare with the prior-season numbers above: the same model looks "
              "several times better purely because of leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
