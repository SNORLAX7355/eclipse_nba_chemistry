"""Predict five-man lineup OFF_RATING and DEF_RATING from member profiles.

Answers the open questions from `modeling.ipynb`:

* which features are worth using (avg box scores, avg play types, the STD
  columns, cluster counts, and GP/POSS/MIN),
* whether normalising helps,
* which model and hyper-parameters work best,
* whether 1,476 lineups is enough, or the possession threshold should drop.

The validation choice matters more than any of those. Lineups from one roster
overlap heavily - two lineups can share four of five players - so a random
train/test split puts near-duplicates on both sides and reports a score the
model cannot reproduce on a new team. Everything here uses **GroupKFold with
team-season as the group**, plus a temporal holdout on 2025-26.

Run from the repository root:
    python scripts/model_ortg_drtg.py [--quick] [--threshold N]
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineup_modeling as lm  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
# Nested n_jobs (GridSearchCV around a parallel forest) triggers a joblib notice
# on every fit and oversubscribes cores; the grid does the parallelising instead.
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SEASON = "2025-26"
THRESHOLDS = [200, 150, 100, 75, 50, 25]


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def load(root: Path, min_poss: int) -> pd.DataFrame:
    lineup_path, player_path = lm.data_paths(root)
    lineups = pd.read_csv(lineup_path)
    players = pd.read_csv(player_path)
    data = lm.build_lineup_dataset(lineups, players, min_poss=min_poss)
    data["GROUP"] = data.TEAM.astype(str) + "_" + data.SEASON.astype(str)
    return data


def cv_r2(model, X, y, groups, *, grouped: bool = True) -> float:
    """Mean cross-validated R^2.

    ``grouped=True`` keeps every lineup from one team-season on the same side
    of the split; ``grouped=False`` is the naive random split, kept only to
    show how much it overstates performance.
    """
    cv = (GroupKFold(n_splits=N_SPLITS) if grouped
          else KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE))
    scores = cross_val_score(
        model, X, y, groups=groups if grouped else None, cv=cv, scoring="r2"
    )
    return float(np.mean(scores))


def models(scaled: bool) -> dict:
    """Candidate models. Scaling only matters for Ridge and kNN."""
    def wrap(est):
        return make_pipeline(StandardScaler(), est) if scaled else est

    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "DecisionTree (default)": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "DecisionTree (depth 4)": DecisionTreeRegressor(
            max_depth=4, min_samples_leaf=20, random_state=RANDOM_STATE),
        "Ridge (alpha=10)": wrap(Ridge(alpha=10.0)),
        "kNN (k=25)": wrap(KNeighborsRegressor(n_neighbors=25)),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=10,
            random_state=RANDOM_STATE, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300,
            random_state=RANDOM_STATE),
    }


# --------------------------------------------------------------------------
def experiment_reliability(root: Path) -> None:
    banner("1. HOW MUCH OF ORTG / DRTG IS EVEN LEARNABLE?")
    print("Var(observed | POSS) = Var(true) + k/POSS, fitted across possession bins.")
    print("'reliability' is the share of observed variance that is real at the")
    print("median lineup - an approximate ceiling on any model's R^2.\n")
    rows = []
    for thr in THRESHOLDS:
        data = load(root, thr)
        for target in lm.TARGETS:
            try:
                rel = lm.rating_reliability(data, value_col=target)
            except ValueError:
                continue
            rows.append({
                "POSS >=": thr, "n lineups": len(data), "target": target,
                "median POSS": round(rel["median_poss"]),
                "SD(true)": rel["sd_true"], "SD(observed)": data[target].std(),
                "reliability": rel["reliability"],
            })
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nReading: reliability is the realistic R^2 ceiling. Raising the threshold")
    print("buys signal per row but costs rows; the table shows the trade directly.")


def experiment_split_strategy(root: Path, thr: int) -> None:
    banner("2. WHY THE SPLIT MATTERS MORE THAN THE MODEL")
    data = load(root, thr)
    fs = lm.feature_sets(data.columns)
    cols = fs["plus_clusters"]
    print(f"n={len(data)} lineups at POSS>={thr}, {len(cols)} features "
          f"(avg box + avg play types + STD + clusters)\n")
    rows = []
    for name, model in models(scaled=True).items():
        if name == "Baseline (mean)":
            continue
        for target in lm.TARGETS:
            rows.append({
                "model": name, "target": target,
                "random KFold R2": cv_r2(model, data[cols], data[target],
                                         data.GROUP, grouped=False),
                "GroupKFold R2": cv_r2(model, data[cols], data[target],
                                       data.GROUP, grouped=True),
            })
    out = pd.DataFrame(rows)
    out["inflation"] = out["random KFold R2"] - out["GroupKFold R2"]
    print(out.round(4).to_string(index=False))
    print("\nLineups from one roster share up to four of five players. A random split")
    print("puts near-duplicates in train and test. Every number below uses GroupKFold.")


def experiment_features(root: Path, thr: int) -> None:
    banner("3. WHICH FEATURES EARN THEIR PLACE?")
    data = load(root, thr)
    fs = lm.feature_sets(data.columns)
    rows = []
    for target in lm.TARGETS:
        for name, cols in fs.items():
            for model_name in ("Ridge (alpha=10)", "RandomForest"):
                model = models(scaled=True)[model_name]
                rows.append({
                    "target": target, "feature set": name, "n features": len(cols),
                    "model": model_name,
                    "GroupKFold R2": cv_r2(model, data[cols], data[target], data.GROUP),
                })
    out = pd.DataFrame(rows)
    for target in lm.TARGETS:
        print(f"\n--- {target} ---")
        sub = out[out.target == target].pivot(
            index=["feature set", "n features"], columns="model",
            values="GroupKFold R2")
        print(sub.round(4).to_string())
    print("\nNote on `plus_usage` (MIN / POSS / GP): a lineup plays more *because* it")
    print("performed well, so these encode the coach's hindsight. Any gain there is")
    print("not something you could use to evaluate a lineup that has not played yet.")


def experiment_scaling(root: Path, thr: int) -> None:
    banner("4. DOES NORMALISING HELP?")
    data = load(root, thr)
    cols = lm.feature_sets(data.columns)["plus_clusters"]
    rows = []
    for target in lm.TARGETS:
        for name in ("Ridge (alpha=10)", "kNN (k=25)", "DecisionTree (depth 4)",
                     "RandomForest"):
            rows.append({
                "target": target, "model": name,
                "raw": cv_r2(models(scaled=False)[name], data[cols],
                             data[target], data.GROUP),
                "StandardScaler": cv_r2(models(scaled=True)[name], data[cols],
                                        data[target], data.GROUP),
            })
    out = pd.DataFrame(rows)
    out["difference"] = out["StandardScaler"] - out["raw"]
    print(out.round(4).to_string(index=False))
    print("\nScaling matters for distance- and penalty-based models (kNN, Ridge) and")
    print("is irrelevant to trees, which split on thresholds within one column.")


def experiment_threshold(root: Path) -> None:
    banner("5. IS 1,476 LINEUPS ENOUGH? LOWERING THE POSSESSION THRESHOLD")
    rows = []
    for thr in THRESHOLDS:
        data = load(root, thr)
        cols = lm.feature_sets(data.columns)["plus_clusters"]
        for target in lm.TARGETS:
            rf = models(scaled=False)["RandomForest"]
            rows.append({
                "POSS >=": thr, "n lineups": len(data), "target": target,
                "SD(target)": data[target].std(),
                "GroupKFold R2": cv_r2(rf, data[cols], data[target], data.GROUP),
            })
    out = pd.DataFrame(rows)
    for target in lm.TARGETS:
        print(f"\n--- {target} (RandomForest) ---")
        print(out[out.target == target].drop(columns="target")
              .round(4).to_string(index=False))
    print("\nLowering the threshold adds rows but each row is a noisier measurement of")
    print("the same thing. R^2 falls because the target variance it is scored against")
    print("is increasingly unexplainable - not because the model got worse.")


#: Feature sets searched during tuning, alongside the hyper-parameters. The
#: section-3 ablation shows the choice of feature set matters at least as much
#: as the model, so it is treated as a tuned dimension rather than fixed.
TUNED_FEATURE_SETS = ["avg_box", "avg_box_playtype", "plus_clusters"]


def experiment_tuning(root: Path, thr: int, quick: bool) -> dict:
    banner("6. TUNING OVER FEATURE SET *AND* HYPER-PARAMETERS (GroupKFold)")
    data = load(root, thr)
    all_sets = lm.feature_sets(data.columns)
    cv = GroupKFold(n_splits=N_SPLITS)
    best: dict = {}
    best_cols: dict = {}

    tree_grid = {
        "max_depth": [2, 3, 4, 5, 6, 8, None],
        "min_samples_leaf": [5, 10, 20, 40, 80],
        "min_samples_split": [2, 10, 20],
        "ccp_alpha": [0.0, 0.01, 0.1],
    }
    rf_grid = {
        "n_estimators": [400],
        "max_depth": [4, 6, 8, None],
        "min_samples_leaf": [5, 10, 20, 40],
        "max_features": ["sqrt", 0.3, 1.0],
    }
    hgb_grid = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_iter": [200, 500],
        "min_samples_leaf": [10, 30],
        "l2_regularization": [0.0, 1.0],
    }
    ridge_grid = {"ridge__alpha": [0.1, 1, 10, 50, 100, 500, 1000]}
    if quick:
        tree_grid = {"max_depth": [3, 4, 6], "min_samples_leaf": [10, 40],
                     "min_samples_split": [2], "ccp_alpha": [0.0]}
        rf_grid = {"n_estimators": [200], "max_depth": [6, None],
                   "min_samples_leaf": [10], "max_features": ["sqrt", 0.3]}
        hgb_grid = {"max_depth": [3], "learning_rate": [0.05], "max_iter": [200],
                    "min_samples_leaf": [20], "l2_regularization": [0.0]}

    searches = {
        "DecisionTree": (DecisionTreeRegressor(random_state=RANDOM_STATE), tree_grid),
        "RandomForest": (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
                         rf_grid),
        "HistGradientBoosting": (
            HistGradientBoostingRegressor(random_state=RANDOM_STATE), hgb_grid),
        "Ridge": (make_pipeline(StandardScaler(), Ridge()), ridge_grid),
    }

    for target in lm.TARGETS:
        print(f"\n--- {target} ---")
        rows = []
        for name, (est, grid) in searches.items():
            top = None
            for set_name in TUNED_FEATURE_SETS:
                cols = all_sets[set_name]
                gs = GridSearchCV(est, grid, cv=cv, scoring="r2", n_jobs=-1)
                gs.fit(data[cols], data[target], groups=data.GROUP)
                if top is None or gs.best_score_ > top[0]:
                    top = (gs.best_score_, set_name, gs.best_params_,
                           gs.best_estimator_, cols)
            score, set_name, params, estimator, cols = top
            best[(target, name)] = estimator
            best_cols[(target, name)] = cols
            rows.append({"model": name, "best feature set": set_name,
                         "n feat": len(cols), "best GroupKFold R2": score,
                         "best params": str(params)})
        dummy = cv_r2(DummyRegressor(strategy="mean"), data[all_sets["avg_box"]],
                      data[target], data.GROUP)
        rows.append({"model": "Baseline (mean)", "best feature set": "-",
                     "n feat": 0, "best GroupKFold R2": dummy, "best params": "-"})
        print(pd.DataFrame(rows).sort_values("best GroupKFold R2", ascending=False)
              .round(4).to_string(index=False))
    best["_cols"] = best_cols
    return best


def experiment_holdout(root: Path, thr: int, best: dict) -> None:
    banner(f"7. HELD-OUT {TEST_SEASON} (trained on 2021-22 .. 2024-25)")
    data = load(root, thr)
    best_cols = best.get("_cols", {})
    default_cols = lm.feature_sets(data.columns)["plus_clusters"]
    train = data[data.SEASON != TEST_SEASON]
    test = data[data.SEASON == TEST_SEASON]
    print(f"train={len(train)}  test={len(test)}")
    print("Each model is refitted on the training seasons using the feature set "
          "and hyper-parameters section 6 selected.\n")
    rows = []
    for target in lm.TARGETS:
        base = train[target].mean()
        ss_tot = float(np.sum((test[target] - base) ** 2))
        rows.append({
            "target": target, "model": "Baseline (training mean)", "n feat": 0,
            "holdout R2": 0.0,
            "RMSE": lm.weighted_rmse(test[target], np.full(len(test), base)),
        })
        for name in ("DecisionTree", "RandomForest", "HistGradientBoosting", "Ridge"):
            est = best.get((target, name))
            if est is None:
                continue
            cols = best_cols.get((target, name), default_cols)
            est.fit(train[cols], train[target])
            pred = est.predict(test[cols])
            ss_res = float(np.sum((test[target] - pred) ** 2))
            rows.append({
                "target": target, "model": name, "n feat": len(cols),
                "holdout R2": 1 - ss_res / ss_tot if ss_tot else np.nan,
                "RMSE": lm.weighted_rmse(test[target], pred),
            })
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("\nBaseline R2 is 0 by construction (it predicts the training mean).")
    print("A negative value means the model is worse than that constant.")


def experiment_prior_season(root: Path, thr: int) -> None:
    banner("9. ARE THESE SCORES REAL? SAME-SEASON vs PRIOR-SEASON FEATURES")
    print("Every feature above is measured in the same season as the target, over")
    print("possessions that include the lineup's own. To see how much that flatters")
    print("the result, the same models are refitted using each member's profile from")
    print("the PREVIOUS season - information that genuinely predates the outcome.\n")

    lineup_path, player_path = lm.data_paths(root)
    lineups = pd.read_csv(lineup_path)
    players = pd.read_csv(player_path)
    seasons = sorted(players.SEASON.unique())
    nxt = {seasons[i]: seasons[i + 1] for i in range(len(seasons) - 1)}

    prior = players.copy()
    prior["SEASON"] = prior["SEASON"].map(nxt)
    prior = prior.dropna(subset=["SEASON"])

    same = lm.build_lineup_dataset(lineups, players, min_poss=thr)
    lag = lm.build_lineup_dataset(lineups, prior, min_poss=thr)
    # Compare on identical lineups so only the feature season differs.
    shared = set(zip(lag.GROUP_ID, lag.SEASON)) & set(zip(same.GROUP_ID, same.SEASON))
    same = same[[k in shared for k in zip(same.GROUP_ID, same.SEASON)]]
    lag = lag[[k in shared for k in zip(lag.GROUP_ID, lag.SEASON)]]
    for frame in (same, lag):
        frame["GROUP"] = frame.TEAM.astype(str) + "_" + frame.SEASON.astype(str)
    print(f"lineups with a complete prior-season profile for all five members: "
          f"{len(shared):,} (of {len(lm.build_lineup_dataset(lineups, players, min_poss=thr)):,})\n")

    rows = []
    for target in lm.TARGETS:
        for name in ("Ridge (alpha=10)", "RandomForest"):
            for label, frame in (("same-season", same), ("prior-season", lag)):
                cols = lm.feature_sets(frame.columns)["avg_box"]
                rows.append({
                    "target": target, "model": name, "features": label,
                    "GroupKFold R2": cv_r2(models(scaled=True)[name], frame[cols],
                                           frame[target], frame.GROUP),
                })
    out = pd.DataFrame(rows).pivot(
        index=["target", "model"], columns="features", values="GroupKFold R2")
    out["drop"] = out["same-season"] - out["prior-season"]
    print(out.round(4).to_string())
    print("\nThe same-season column answers 'how well do these five players' profiles")
    print("describe what this lineup did'. The prior-season column answers 'could we")
    print("have called it in advance'. Only the second is a forecast.")


def experiment_importances(root: Path, thr: int, best: dict) -> None:
    banner("8. WHAT THE MODELS ACTUALLY USE")
    data = load(root, thr)
    best_cols = best.get("_cols", {})
    for target in lm.TARGETS:
        est = best.get((target, "RandomForest"))
        if est is None:
            continue
        cols = best_cols.get((target, "RandomForest"),
                             lm.feature_sets(data.columns)["plus_clusters"])
        est.fit(data[cols], data[target])
        imp = (pd.Series(est.feature_importances_, index=cols)
               .sort_values(ascending=False).head(12))
        print(f"\n--- {target}: top RandomForest importances ---")
        print(imp.round(4).to_string())
    print("\nImpurity importances are biased toward high-cardinality continuous")
    print("features, so read these as a rough ranking, not an effect size.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small grids, for a fast run")
    ap.add_argument("--threshold", type=int, default=100,
                    help="possession threshold for the main experiments")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]

    print(f"Possession threshold for sections 2-4 and 6-8: POSS >= {args.threshold}")
    experiment_reliability(root)
    experiment_split_strategy(root, args.threshold)
    experiment_features(root, args.threshold)
    experiment_scaling(root, args.threshold)
    experiment_threshold(root)
    best = experiment_tuning(root, args.threshold, args.quick)
    experiment_holdout(root, args.threshold, best)
    experiment_importances(root, args.threshold, best)
    experiment_prior_season(root, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
