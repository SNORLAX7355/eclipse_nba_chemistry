# Modelling lineup OFF_RATING and DEF_RATING

My follow-up to `modeling.ipynb`. I used the cleaned dataset from that notebook and
tried to work out which features to use, whether to normalise, which model works best,
and whether 1,476 lineups is enough.

Targets: `OFF_RATING` and `DEF_RATING`, modelled separately.

---

## Data

| File | Shape | What it is |
|---|---|---|
| `using_dataset/lineups_five_seasons.csv` | 10,000 × 51 | five-man lineups, 2021-22 to 2025-26 |
| `using_dataset/full_data.csv` | 2,204 × 145 | player-seasons: box score + advanced + 18 synergy play-type shares + `CLUSTER` |

At `POSS >= 100` and requiring all five players to be found, this gives **1,476 lineups**,
which matches what `modeling.ipynb` got.

The K=6 play-style clusters are used as they are (labels from
`cluster_descriptions_and_players.txt`):

| Cluster | Label |
|---|---|
| 0 | Interior / two-way big |
| 1 | Primary ball handler |
| 2 | Interior big, less defensive |
| 3 | Off-ball role player / defensive wing |
| 4 | Spot-up specialist |
| 5 | Off-ball shooter |

---

## Files I added

| File | What it does |
|---|---|
| `lineup_modeling.py` | builds the lineup table, feature sets, reliability estimate |
| `modeling_ortg_drtg.ipynb` | the analysis with charts (already run, outputs saved) |
| `scripts/model_ortg_drtg.py` | same experiments from the command line |
| `tests/test_lineup_modeling.py` | 48 tests |

I also updated `lineup_chemistry.lineups_path()` so it looks in both `using_dataset/` and
`basketball_data/`. `lineups_five_seasons.csv` moved between those folders, which had
broken four of my older tests.

Two small things my builder does differently from the notebook version:

- It uses `TM_TOV_PCT` instead of `TOV_PCT`. `TOV_PCT` isn't a column in `full_data.csv`,
  so that feature wasn't actually making it into the model.
- Some players have an all-zero synergy row, which means no play-type data rather than
  zero usage of every play. I set those to `NaN` so they don't pull the lineup average
  down. Only 1 lineup is affected at `POSS >= 100`, so it doesn't change the results.

It's also vectorised rather than looping row by row, which matters when the possession
threshold is lowered and the table gets closer to 10,000 rows.

---

## Method

- 5-fold **`GroupKFold`, grouped by team-season**. Lineups from the same team overlap a
  lot (two lineups can share four of five players), so a random split ends up testing on
  rosters the model has partly already seen. Random `KFold` gave R² about 0.01–0.07
  higher across models, nearly always in the same direction.
- Final check on a **held-out season**: train on 2021-22 to 2024-25 (1,215 lineups), test
  on 2025-26 (261).
- Baseline is predicting the training mean, so R² is measured against that and a negative
  score means worse than a constant.
- `GridSearchCV` over hyper-parameters and over the feature set.

---

## How much is actually learnable

A lineup's rating is an average over `POSS` possessions, so
`Var(observed | POSS) = Var(true) + k/POSS`. Fitting that across possession bins splits
real variation in lineup quality from measurement noise. `reliability` is the share that
is real, which is roughly the highest R² anything could reach.

| POSS ≥ | n lineups | ORTG reliability | DRTG reliability |
|---|---|---|---|
| 200 | 535 | 0.433 | 0.214 |
| 150 | 828 | 0.322 | 0.082 |
| **100** | **1,476** | **0.190** | **0.125** |
| 75 | 2,213 | 0.072 | 0.000 |
| 50 | 4,102 | 0.119 | 0.044 |
| 25 | 9,180 | 0.081 | 0.125 |

Offence has more signal than defence at nearly every threshold. At `POSS >= 100` the
ceiling is about **0.19 for ORTG and 0.13 for DRTG**, so an R² of 0.14 should be compared
against 0.19, not against 1.0.

This estimate is itself noisy — it's a line fitted to a dozen binned variances, and the
`POSS >= 75` defensive value dropping to zero is an artefact of that fit. The trend is
the reliable part, not any single number.

---

## Results

### Which features

GroupKFold R² at `POSS >= 100`:

| Feature set | n | ORTG Ridge | ORTG RF | DRTG Ridge | DRTG RF |
|---|---|---|---|---|---|
| **`avg_box`** | **12** | **0.158** | 0.128 | **0.069** | 0.025 |
| `avg_box_playtype` | 30 | 0.146 | 0.129 | 0.047 | 0.013 |
| `plus_std` | 60 | 0.110 | 0.119 | 0.033 | 0.027 |
| `plus_clusters` | 66 | 0.100 | 0.119 | 0.032 | 0.027 |
| `plus_usage` | 69 | 0.114 | 0.123 | 0.062 | 0.037 |

The 12 average box-score columns are the best feature set for both targets. Everything
added after that is flat or worse.

- **Average play types didn't help.** I think this is because the shares are
  compositional (each player's offensive shares sum to 1, so one column is redundant) and
  because averaging across five players loses the information — one player who isolates
  constantly plus four who never do averages out to something unremarkable.
- **The `STD_` columns didn't help** and they double the number of features.
- **Cluster counts didn't help**, probably because the archetypes are mostly recoverable
  from the box-score averages already in the model.
- **`GP` / `POSS` / `MIN` help defence a little, but I left them out.** A lineup gets
  played more *because* it did well, so these describe the coach's decisions after the
  fact. A model using them couldn't score a lineup that hasn't played yet.

With ~1,476 rows and a noisy target, extra features seem to cost more in variance than
they add in signal.

### Normalising

It matters for Ridge and kNN and makes no difference to the tree models, which split on
thresholds inside one column at a time. The effect is biggest on kNN, where unscaled
distances are dominated by `FG3A` just because it has the widest numeric range. Ridge
ended up being the best model, so scaling is part of the final pipeline.

### Models (tuned, GroupKFold)

**OFF_RATING**

| Model | Feature set | n | CV R² | Params |
|---|---|---|---|---|
| **Ridge** | `avg_box` | 12 | **0.158** | `alpha=1` |
| RandomForest | `avg_box` | 12 | 0.135 | `max_depth=None, max_features='sqrt', min_samples_leaf=5, n=400` |
| HistGradientBoosting | `avg_box_playtype` | 30 | 0.134 | `lr=0.02, max_depth=2, max_iter=500` |
| DecisionTree | `avg_box_playtype` | 30 | 0.092 | `max_depth=3, min_samples_leaf=80` |
| Baseline (mean) | — | 0 | −0.009 | — |

**DEF_RATING**

| Model | Feature set | n | CV R² | Params |
|---|---|---|---|---|
| **Ridge** | `avg_box` | 12 | **0.073** | `alpha=1` |
| RandomForest | `avg_box` | 12 | 0.036 | `max_depth=8, max_features='sqrt', min_samples_leaf=10, n=400` |
| HistGradientBoosting | `avg_box` | 12 | 0.027 | `lr=0.02, max_depth=2, max_iter=200, l2=1.0` |
| Baseline (mean) | — | 0 | −0.001 | — |
| DecisionTree | `plus_clusters` | 66 | **−0.014** | `max_depth=2, min_samples_leaf=40` |

The decision tree is the weakest model I tried. Left untuned it scores about **−0.9**,
which is much worse than just predicting the mean — that's the overfitting showing up.
Tuning it to depth 3 fixes most of that but it still ends up behind everything else.

### Held-out 2025-26

| Target | Model | R² | RMSE |
|---|---|---|---|
| ORTG | Baseline (training mean) | 0.000 | 10.33 |
| ORTG | **RandomForest** | **0.141** | **9.57** |
| ORTG | Ridge | 0.138 | 9.59 |
| ORTG | HistGradientBoosting | 0.131 | 9.63 |
| ORTG | DecisionTree | 0.056 | 10.03 |
| DRTG | Baseline (training mean) | 0.000 | 10.52 |
| DRTG | **Ridge** | **0.104** | **9.96** |
| DRTG | RandomForest | 0.072 | 10.13 |
| DRTG | HistGradientBoosting | 0.062 | 10.19 |
| DRTG | DecisionTree | −0.008 | 10.56 |

I went with **Ridge on `avg_box`**. It's best in cross-validation for both targets and
clearly best for DRTG on the holdout. RandomForest is slightly ahead on the ORTG holdout
(0.141 vs 0.138) but that gap is smaller than the variation between folds.

One thing that looks odd in the scatter plots: the predictions cover a much narrower
range than the actual ratings. That's expected rather than a bug. When most of the target
is noise, a regularised model should stay near the mean unless the evidence is strong.
The untuned decision tree spreads its predictions across the whole range and is wrong,
which is what the −0.9 means.

### Possession threshold

| POSS ≥ | n lineups | SD(target) | ORTG RF R² |
|---|---|---|---|
| 200 | 535 | ~8.1 | higher |
| **100** | **1,476** | **~10.5** | **0.119** |
| 50 | 4,102 | ~14.1 | lower |
| 25 | 9,180 | ~18.1 | lowest |

Lowering the threshold made the results worse, not better. Going from `POSS >= 100` to
`POSS >= 25` grows the sample from 1,476 to 9,180 and R² goes down.

The model isn't getting worse — the target is. `SD(target)` rises from about 10.5 to
about 18 as the threshold drops, and most of that extra spread is sampling noise that
can't be explained by anything. So the number of rows doesn't seem to be the limiting
factor here; the noise in the target is. If anything, a higher threshold (150 or 200)
looks better for ORTG, at the cost of fewer rows.

### What the models use

For ORTG, shooting efficiency dominates: `AVG_TS_PCT` (0.161) and `AVG_EFG_PCT` (0.139).
That's mostly just talent — five efficient scorers outscore five inefficient ones.

For DRTG the importances are flat, nothing above 0.115, led by `AVG_USG_PCT`,
`AVG_TM_TOV_PCT` and `AVG_BLK`. That fits with the low reliability and weak scores.

These are impurity importances, which are biased towards continuous features with many
distinct values, so they're a rough ranking rather than an effect size.

### Same-season vs prior-season features

All the features above come from the same season as the target, measured over possessions
that include the lineup's own. So a player's `TS_PCT` partly reflects the minutes being
predicted. I refitted using each player's previous-season profile instead (795 lineups
have a complete one):

| Target | Model | Prior-season R² | Same-season R² |
|---|---|---|---|
| OFF_RATING | Ridge | **0.047** | 0.099 |
| OFF_RATING | RandomForest | 0.005 | 0.081 |
| DEF_RATING | Ridge | **−0.011** | 0.048 |
| DEF_RATING | RandomForest | −0.016 | 0.023 |

Offence still works but roughly half as well. Defence goes negative — using only
information from before the season, these features predict lineup defensive rating worse
than the mean does.

So the main results describe how well five players' profiles match what their lineup did.
They don't show that we could have predicted it beforehand, and for defence we currently
can't.

---

## Limitations

1. The headline results use same-season features, so they're descriptive rather than
   predictive.
2. The lineup data is capped at 2,000 rows per season by the API, so it's the most-used
   lineups rather than all of them. Lineups that were never tried aren't in the data at
   all, so none of this says anything about untested combinations.
3. No opponent adjustment. A lineup's DRTG partly reflects who it played against, which
   is probably part of why defence is so much harder to model.
4. The reliability numbers are noisy estimates and should be read as a trend.
5. Only 795 lineups have a full prior-season profile, and they skew towards established
   players since rookies are excluded by definition.
6. The holdout is one season and 261 lineups, so the RandomForest/Ridge gap on ORTG isn't
   meaningful.
7. The play-type shares are compositional, so individual linear coefficients on them
   aren't separately interpretable.
8. I only tried four model families. Something else might do better, though given the
   reliability ceiling there isn't much room.

---

## Running it

```bash
pip install -r requirements.txt
```

```bash
python scripts/model_ortg_drtg.py --quick
```

```bash
python scripts/model_ortg_drtg.py
```

```bash
python -m pytest tests -q
```

`--quick` uses smaller grids and takes about a minute; the full run takes a few minutes.
`modeling_ortg_drtg.ipynb` has everything with the charts.

---

## Tests

`python -m pytest tests -q` → **124 passed** (48 new, 76 from my earlier work).

The new tests use small made-up datasets where I worked out the expected answers by hand.
They cover:

- `parse_group_id` — normal groups, and that groups with 4 IDs, 6 IDs, duplicate IDs,
  or non-numeric values return `None` instead of a broken lineup.
- `explode_lineup_members` — five rows per lineup, right IDs and order.
- `build_lineup_dataset` — `AVG`/`STD` against hand-calculated values, the possession
  filter, targets carried through, cluster counts summing to five, a bad column name
  raising an error, all-zero synergy rows treated as missing, incomplete lineups dropped,
  and that a player's profile is matched within the correct season (a 2023-24 row with
  `USG_PCT = 0.99` must not end up on a 2024-25 lineup).
- `feature_sets` — the sets are properly nested, and no target column leaks into any of
  them.
- `rating_reliability` — recovers known variance components from 60,000 simulated
  lineups, and reports low reliability for pure noise.
- `weighted_rmse` — hand-calculated values and that weights change the result.

There are also checks on the real data: one row per player-season, all six cluster labels
present, synergy blocks summing to 1, `NET_RATING == OFF − DEF` within rounding, and the
builder producing exactly 1,476 lineups.
