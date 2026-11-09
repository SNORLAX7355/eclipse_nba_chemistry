# Lineup chemistry and player compatibility

Atharva's contribution to the Eclipse Analytics NBA research project (Topic B).
Builds on Eliot's data collection and player clustering; none of his work is modified.

---

## 1. Research objective

Do player **roles** and **pair-level co-play history** help explain five-player lineup
net rating beyond the average talent of the five players?

My assignment was to investigate two NBA Stats endpoints — `PlayerVsPlayer` and
`LeagueDashLineups` — determine what they actually measure, and connect them to the
team's lineup-chemistry and game-prediction work.

---

## 2. Data sources

### Already in the repository (Eliot's, unchanged)

| File | Rows × cols | Contents |
|---|---|---|
| `basketball_data/player_stats_<season>.csv` | ~540–606 × 126 | `LeagueDashPlayerStats`, Base + Advanced merged, PerGame |
| `basketball_data/player_stats_five_seasons.csv` | 2,867 × 126 | five seasons stacked |
| `basketball_data/lineups_<season>.csv` | 2,000 × 51 | `LeagueDashLineups`, `group_quantity=5`, Advanced, Per100Possessions |
| `basketball_data/lineups_five_seasons.csv` | 10,000 × 51 | five seasons stacked |
| `basketball_data/player_clusters_five_seasons.csv` | 2,025 × 131 | K-means archetypes (K=3), players with GP ≥ 20 and MIN ≥ 10 |
| `basketball_data/cluster_profiles_five_seasons.csv` | 3 × 13 | cluster centroids |

**Verified, not assumed.** Counts above were re-measured from the current `main`. Two
earlier reported figures were correct (582 players × 126 cols and 2,000 lineups × 51 cols
for 2025-26; 262 lineups with ≥ 100 possessions). Re-requesting the 2025-26 five-man data
live reproduced Eliot's committed CSV exactly — same 2,000 `GROUP_ID`s, zero difference in
`MIN`, `POSS` and `NET_RATING`.

### New in this contribution

| File | Rows × cols | Contents |
|---|---|---|
| `basketball_data/lineups_pairs_<season>.csv` | 2,000 × 51 | `LeagueDashLineups` with **`group_quantity=2`** — *measured* two-man on-court ratings |
| `basketball_data/lineups_pairs_five_seasons.csv` | 10,000 × 51 | five seasons stacked |
| `basketball_data/player_vs_player_onoff_2025-26.csv` | 30 × 19 | `PlayerVsPlayer` Advanced on/off splits, 15 high-minute teammate pairs |
| `basketball_data/pair_co_play_proxies_five_seasons.csv` | 15,333 × 15 | derived pair co-play proxies + measured ratings joined |
| `basketball_data/lineup_player_features_five_seasons.csv` | 10,000 × 49 | derived lineup composition features + archetype counts |

Seasons: 2021-22 … 2025-26.

---

## 3. API endpoints — what they actually measure

Both were tested **live** against `stats.nba.com` with `nba_api` 1.12.0 on 2026-11-09.
Nothing below is inferred from documentation.

### `PlayerVsPlayer`

Returns **10 result sets**: `Overall`, `OnOffCourt`, three shot-distance tables, three
shot-area tables, and two player-info tables.

`OnOffCourt` is the useful one. It takes **`player_id`'s own box score** and splits it by
whether **`vs_player_id` was on or off the court**:

```
PLAYER_NAME   VS_PLAYER_NAME  COURT_STATUS  GP  MIN      PTS  PLUS_MINUS
LeBron James  Hayes, Jaxson   On            49  545.22   352  37
LeBron James  Hayes, Jaxson   Off           60  1443.83  904  82
```

Findings:

* **It is not a chemistry score**, and it is **not symmetric** — it describes
  `player_id`'s production, not a joint pair measurement. Swapping the IDs gives a
  different table.
* **Teammate vs opponent is determined by who you pass**, not by a parameter. Same
  mechanic, different meaning:
  * teammate → "On" is their shared floor time → **teammate on/off analysis**
  * opponent → "On" is head-to-head floor time → **matchup split** (verified with
    LeBron vs Jokić: 54.5 shared minutes across 2 games)
* **It accepts `measure_type_detailed_defense="Advanced"`**, which returns
  `OFF_RATING` / `DEF_RATING` / `NET_RATING` / `POSS` per split instead of only Base
  counts. This gives a real **on/off rating differential** per pair and is not obvious
  from the endpoint's name.
* The **"Off" baseline is not a fixed context** — it pools every possession the player
  played without that teammate, including games the teammate missed entirely.
* **Cost:** one HTTP request per *ordered* pair. Covering the ~2,000 most-played pairs of
  one season needs thousands of requests.

**Verdict:** a tool for *targeted follow-up* on specific pairs, not bulk collection.

### `LeagueDashLineups`

* Eliot's call (`group_quantity=5`, Advanced, Per100Possessions) reproduces exactly.
* **`group_quantity=2` is the valuable unused parameter.** It returns *measured* two-man
  combination stats — 2,000 pairs in a single request, already possession-denominated and
  symmetric, with roughly **10× the per-observation reliability** of five-man rows
  (median 806 possessions vs 44).
* **Hard 2,000-row-per-request cap.** Server-side, not a choice in Eliot's code. Rows come
  back sorted by `MIN` descending, so every lineup dataset here is the *most-used* 2,000
  lineups per season — a biased sample, with untried lineups absent entirely.

**Verdict:** the right primary source for bulk pair data.

---

## 4. Methodology

### 4.1 Lineup → player join

`GROUP_ID` encodes members as `"-201145-203115-203932-203999-1628420-"`. IDs are parsed
and sorted so a group has one canonical key; this is safer than matching abbreviated
`GROUP_NAME` strings ("J. Green" is ambiguous). All 10,000 five-man groups parse to
exactly 5 distinct IDs, and **100%** of members join to `player_stats_five_seasons.csv`
on `(PLAYER_ID, SEASON)`. **97.5%** of members carry one of Eliot's cluster labels, and
**90.5%** of lineups have all five labelled.

Joins are always on `(PLAYER_ID, SEASON)`, never `PLAYER_ID` alone, so a player's
2021-22 attributes can never be attached to a 2024-25 lineup.

### 4.2 Signal versus noise (do this before anything else)

A lineup's net rating is a mean over `POSS` possessions, so sampling variance falls as
`1/POSS`. Across possession bins:

```
Var(observed | POSS) = Var(true) + k / POSS
```

Fitting that line against `1/POSS` separates real spread (intercept) from noise (slope).
`k / Var(true)` is the possession count at which an observation deserves equal weight
with the league prior.

| | five-man lineups | two-man lineups |
|---|---|---|
| SD of true quality | 6.36 net-rating pts | 5.78 |
| noise coefficient `k` | 30,015 | 27,902 |
| possessions for equal weight with prior | **742** | 836 |
| median possessions | **44** | **806** |
| **reliability at median** | **0.056** | **0.491** |

**At the median five-man lineup, ~94% of net-rating variance is sampling noise**, and
fewer than 2% of lineups reach 742 possessions. This governs every other result in the
contribution.

### 4.3 Pair co-play proxy

For every five-man lineup, all `C(5,2) = 10` unordered pairs are generated, then
aggregated per `(SEASON, TEAM, PLAYER_A_ID, PLAYER_B_ID)`:

```
PAIR_NET_PROXY = Σ(NET_RATING_i × POSS_i) / Σ(POSS_i)
```

Possessions, not minutes, because net rating is per 100 possessions.
`PAIR_NET_PROXY_SHRUNK` pulls this toward the league mean with prior weight **742**,
taken from §4.2 rather than guessed.

`SUM_LINEUP_POSS` **deliberately double-counts**: the ten pairs in one lineup all inherit
its possessions, and a pair's contributing lineups overlap heavily. It measures how much
evidence feeds the proxy, *not* possessions the pair actually shared — use
`MEASURED_PAIR_POSS` for that.

Grouping includes `TEAM` and `SEASON` so a traded pair is never pooled across rosters.

### 4.4 Lineup composition features

Per metric: `MEAN_` (level), `SD_` (role spread, population SD with `ddof=0` — a fixed
five-player group, not a sample), `MAX_` and `MIN_` (extremes). The spread terms are the
point: a mean cannot distinguish five average-usage players from one star plus four
low-usage teammates.

Metrics are **rates**, not counts (`USG_PCT`, `TS_PCT`, `AST_PCT`, `OREB_PCT`,
`DREB_PCT`, `TM_TOV_PCT`, `THREE_POINT_ATTEMPT_RATE`, `AST_TO_TOV`). A lineup's mean
`AST` mostly reflects how many minutes its members play — the earlier draft of this work
used raw `AST`/`REB`, which is why they were replaced.

Archetype counts reuse Eliot's saved labels rather than refitting, so the analyses cannot
drift apart. Lineups missing any label get `NaN` counts instead of a wrong count.

### 4.5 Feature definitions

| Column | Definition |
|---|---|
| `PAIR_NET_PROXY` | possession-weighted mean net rating of the five-man lineups containing the pair |
| `PAIR_NET_PROXY_SHRUNK` | above, shrunk toward the league mean with prior weight 742 possessions |
| `SAMPLED_LINEUPS` | distinct contributing five-man lineups |
| `SUM_LINEUP_POSS` | summed possessions of contributing lineups (**double-counts overlaps**) |
| `MEASURED_PAIR_NET_RATING` / `_POSS` / `_MIN` | *measured* two-man values from `group_quantity=2` |
| `MEAN_/SD_/MAX_/MIN_<metric>` | member-attribute level, spread and extremes |
| `N_CLUSTER_k` | members in Eliot's cluster `k` (`NaN` if any member unlabelled) |
| `ARCHETYPE_MIX` | counts as `interior-perimeter-creator`, e.g. `"1-3-1"` |
| `N_ARCHETYPES_PRESENT` | distinct archetypes in the lineup (1–3) |
| `NET_VS_TEAM` | lineup net rating minus its own team-season possession-weighted mean |

Eliot's K=3 clusters, read off `cluster_profiles_five_seasons.csv`:

| Cluster | Label | Profile |
|---|---|---|
| 0 | Interior finisher | highest TS% and rebounding, lowest 3PA rate (0.16), low FT% |
| 1 | Perimeter role player | highest 3PA rate (0.51), low usage, low rebounding |
| 2 | High-usage creator | highest USG% (0.25), AST% (0.25) and PIE |

---

## 5. Results

### 5.1 The pair proxy is a valid descriptive measure

Validated against measured two-man ratings — an independent check, since the proxy uses
only five-man data. 9,941 of 15,333 pairs (64.8%) have a measured counterpart.

| min contributing POSS | n pairs | r (raw) | r (shrunk) |
|---|---|---|---|
| 0 | 9,941 | 0.704 | **0.763** |
| 200 | 7,597 | 0.842 | 0.840 |
| 500 | 4,462 | **0.907** | 0.909 |
| 1,000 | 2,301 | **0.945** | 0.945 |
| 2,000 | 757 | 0.966 | 0.967 |

Mean absolute error vs measured: 6.04 net-rating points. Shrinkage helps most where the
sample is thin, exactly as intended.

**Caveats.** The agreement is **partly mechanical** — both numbers come from overlapping
on-court possessions of the same two players. It shows the proxy faithfully summarises
*observed co-play performance*; it does **not** show that either quantity isolates a pair
effect. The raw proxy is also **over-dispersed** (SD 12.6 vs 8.6 measured), which is what
noise inflation predicts.

**Recommendation:** use measured two-man data where it exists; use the proxy as the
fallback for the ~35% of pairs the row cap omits.

### 5.2 Usage spread, not average talent, is the chemistry-like signal

Possession-weighted correlations with lineup net rating, lineups with ≥ 100 possessions
(n = 1,492), also shown net of team strength:

| Feature | r vs NET | r net of team |
|---|---|---|
| `MEAN_TS_PCT` | 0.339 | 0.259 |
| `MIN_TS_PCT` | 0.281 | 0.226 |
| `MEAN_AST_TO_TOV` | 0.256 | 0.192 |
| **`SD_USG_PCT`** | **0.227** | **0.155** |
| `MEAN_TM_TOV_PCT` | −0.195 | −0.139 |
| `SD_THREE_POINT_ATTEMPT_RATE` | 0.132 | 0.117 |

`MEAN_TS_PCT` leading is **talent, not chemistry** — five efficient scorers outscore five
inefficient ones. The interesting result is **`SD_USG_PCT` being positive**: lineups with
a *wider spread* of usage rates outperform lineups where everyone wants the same number
of shots, and it survives team adjustment. Consistent with needing a clear offensive
hierarchy — but a correlation on 1,492 noisy observations, not a mechanism.

### 5.3 Archetype mixes — mostly talent, with one real finding

Possession-weighted net rating by archetype mix (five seasons, mixes with ≥ 3,000
possessions), `interior-perimeter-creator`:

| Mix | Composition | n lineups | wtd NET | net of team |
|---|---|---|---|---|
| 2-0-3 | 2 interior + 3 creators | 70 | 5.01 | **+2.19** |
| 1-0-4 | 1 interior + 4 creators | 66 | 5.89 | +2.02 |
| 2-1-2 | 2 interior + 1 perimeter + 2 creators | 469 | 5.09 | +1.99 |
| 1-2-2 | 1 + 2 + 2 | 2,306 | 2.96 | −0.08 |
| 1-3-1 | 1 + 3 + 1 | 2,188 | 2.50 | −0.13 |
| 1-4-0 | 1 interior + 4 perimeter | 470 | −3.06 | −3.02 |
| 0-5-0 | 5 perimeter role players | 96 | −6.23 | **−6.79** |

**Most of this is talent and minute allocation, not chemistry.** Cluster 1 is defined
partly by *low usage*, and low-usage players are disproportionately bench players.
"Lineups of five bench players perform worse" is close to a tautology, and the
team-strength adjustment does not remove it, because it compares a team's bench units to
that same team's average.

The result **not** explained by talent alone is role diversity at the extreme:

| Distinct archetypes | n lineups | wtd NET | net of team |
|---|---|---|---|
| 1 | 98 | −5.13 | −5.86 |
| 2 | 2,625 | 2.82 | −0.13 |
| 3 | 6,323 | 3.03 | +0.09 |

Going from **one** archetype to **two** is a large jump; **two to three** is negligible.
Role diversity matters at the degenerate extreme — don't field five of the same kind of
player — and flattens out immediately after. That is much weaker than "more diverse
lineups are better", and it is what the data supports.

### 5.4 Prediction: real but small, and leakage is the bigger lesson

Prior-season features only, temporal holdout (train 2022-23…2024-25, test 2025-26),
possession-weighted, baseline = training possession-weighted mean.

| Model | all lineups (n=891) | POSS ≥ 100 (n=158) |
|---|---|---|
| Baseline (training mean) | 0.0000 | 0.0000 |
| Ridge α=100 | 0.0235 | 0.0441 |
| Ridge α=1000 | **0.0238** | **0.0461** |
| RandomForest (300, depth 6) | 0.0069 | 0.0033 |

`GroupKFold` by team-season inside the training years: folds
`[0.0005, 0.0488, 0.0300, −0.0217, 0.0129]`, mean **0.0141**.

So the research question answers **yes, but barely**. Against the **~0.06 ceiling** from
§4.2, explaining 0.02–0.04 of total variance captures a non-trivial share of what is
explainable at all — a rough ratio of two estimates, not a precise claim.
`RandomForest` not beating Ridge is what a noise-dominated target looks like.

**The leakage demonstration is the most transferable result.** On the *same* lineups, the
same model with same-season features (including each member's same-season `NET_RATING`,
`PIE`, `PLUS_MINUS`) scores:

| Features | all lineups | POSS ≥ 100 |
|---|---|---|
| prior-season (honest) | 0.0235 | 0.0441 |
| same-season (leaky) | **0.0660** | **0.1361** |

**~3× better, purely from information it should not have** — a player's same-season net
rating is computed from the very possessions being predicted. Any lineup or game-win
model we build must use only pre-outcome information.

Ridge coefficients are **not** interpreted: the composition features are strongly
collinear (`MAX_TS_PCT` flips sign against `MEAN_TS_PCT`). §5.2's weighted correlations
are the honest descriptive summary.

**We cannot claim to predict genuinely new five-player combinations.** Every lineup in the
test set is one a coach actually chose to play; untried lineups are absent from the data.

---

## 6. Limitations

1. **The 2,000-row cap** makes every lineup dataset a biased sample of the most-used
   lineups. Untried combinations are absent, so nothing here supports claims about them.
2. **Small samples dominate** — ~94% of a median five-man lineup's net-rating variance is
   noise.
3. **No causal claims.** Net rating is confounded by the other three teammates, opponents,
   coaching and usage. The proxies are co-play summaries, not pair effects.
4. **Overlapping samples.** Pair proxies come from overlapping lineups and are not
   independent observations; `SUM_LINEUP_POSS` double-counts by construction.
5. **Archetype results are substantially talent**, since the clusters encode usage, which
   correlates with being a starter.
6. **Season-level attributes only** — no in-season form, injuries, rest or opponent
   adjustment. Traded players carry one blended season row (72 players had
   `TEAM_COUNT > 1` in 2025-26).
7. **Weak prediction** — held-out R² of 0.02–0.05, with one negative CV fold.
8. `nba_api` is an unofficial client for `stats.nba.com`; endpoints and the row cap can
   change without notice, and the underlying data may carry reuse restrictions.

---

## 7. Reproduction

```bash
pip install -r requirements.txt
```

From the repository root, in order:

```bash
# 1. Collect the new data (network; ~20 requests, bounded retries, 2s pauses)
python scripts/collect_pair_data.py

# 2. Build the derived datasets (offline, reads only committed CSVs)
python scripts/build_chemistry_datasets.py

# 3. Fit and evaluate the baseline models, including the leakage demonstration
python scripts/model_lineup_performance.py --demo-leakage

# 4. Run the tests
python -m pytest tests -q

# 5. The notebook reproduces the whole analysis with figures
jupyter notebook data_exploration_lineup_chemistry.ipynb
```

Step 2 alone regenerates both derived CSVs without network access, so the analysis is
reproducible offline from what is committed. `RUN_LIVE_API` in the notebook is `False` by
default; set it to `True` to re-issue the `PlayerVsPlayer` request.

---

## 8. Tests performed

`python -m pytest tests -q` — **76 passed**.

Synthetic fixtures with hand-computed expected values cover:

* **`parse_group_ids`** — five-man and two-man groups, sort-invariance, whitespace,
  and that `None`/`NaN`/`""`/`"-"`/`"-abc-"` give `()` rather than a wrong ID.
* **`weighted_mean`** — known value (7.5 from `[10,0]` weighted `[3,1]`), the documented
  pair formula, equal-weight reduction to a simple mean, zero/negative/`NaN` handling,
  empty input, and length-mismatch raising.
* **`explode_lineup_pairs`** — exactly 10 pairs per lineup, membership matching
  `itertools.combinations`, canonical `A < B` ordering, attribute pass-through, and that
  malformed groups (4 IDs, duplicate IDs) are dropped.
* **`aggregate_pair_proxies`** — the headline arithmetic
  `(10×100 + −5×50)/150 = +5.0` exactly; that the result is **possession**-weighted and
  not unweighted or minute-weighted (verified by perturbing `MIN` and confirming the
  proxy is unchanged); correct pair counts; that the same pair on two teams or in two
  seasons is **not** pooled; exclusion of zero-possession and missing-rating lineups; and
  no duplicate pair keys.
* **`shrink_to_prior`** — known value (20 at weight 100 with prior 0 and prior weight 100
  → 10.0), large samples barely moving while small samples collapse, never overshooting
  the prior, and the default prior being the weighted mean.
* **`attach_measured_pair_ratings`** — measured values landing on the right pair, row
  count preserved, unmatched pairs `NaN`, and duplicate measured keys not inflating the
  join.
* **`build_lineup_player_features`** — `MEAN`/`SD`/`MAX`/`MIN` against hand-computed
  values, `ddof=0` confirmed (and explicitly *not* `ddof=1`), zero spread for identical
  members, derived ratio computation, incomplete lineups dropped when required and
  flagged when not, **and that attributes are matched within season** (a 2023-24 row with
  `USG_PCT = 0.99` must not reach a 2024-25 lineup).
* **`add_archetype_composition`** — counts summing to five and matching labels,
  partially-labelled lineups giving `NaN` rather than a wrong count, row count preserved,
  within-season label matching.
* **`estimate_reliability`** — recovers known variance components from 60,000 simulated
  lineups (`Var(true) = 16`, `k = 24,000`, both within 25%), reports near-zero true
  variance for pure-noise data, never returns negative variance.
* **`possession_weighted_r2`** — 1.0 for perfect prediction, 0.0 for predicting the
  weighted mean, negative when worse than baseline, external baselines honoured, and
  weights actually changing the score.

Integrity checks against the committed CSVs: all 10,000 five-man groups parse to five
distinct players, lineup keys unique, 100% of members present in player stats,
`NET_RATING == OFF_RATING − DEF_RATING` within rounding, one row per player-season, all
two-man groups parse to two distinct players, and both derived outputs have the expected
structure (canonical pair ordering, no duplicate keys, positive possessions, archetype
counts summing to five, `MIN_ ≤ MEAN_ ≤ MAX_`).

---

## 9. Remaining work

1. **Make two-man data the backbone of pair analysis** — already collected for five
   seasons, and where the signal is (reliability 0.49 vs 0.06).
2. **Add `group_quantity=3`** for trio data, enabling the much better question: does a
   pair's performance survive changing the third player?
3. **Move to possession- or game-level data** (`PlayByPlayV3`, `LeagueDashPtStats`) to
   escape both the 2,000-row cap and the season-aggregate ceiling.
4. **Model pair effects jointly** — a ridge or mixed model with one coefficient per player
   plus pair interaction terms, fitted across all lineups at once. This is the step that
   separates "these two are good" from "these two fit together", which no per-pair average
   can do.
5. **Re-fit archetypes on rate-only features** so clusters stop encoding playing time,
   then re-run §5.3. Worth raising with Eliot rather than changing his notebook.
6. **For Topic A (game-win prediction):** carry the §5.4 leakage discipline across.
   Same-season aggregates inflated R² ~3× here, and a game-level model is easier to fool.

### Notes on Eliot's notebooks (not changed — flagged for him)

`data_gathering_five_seasons.ipynb` does not execute top-to-bottom in a fresh kernel:

* `fetch_lineup_stats` is **called in cell 6 but defined in cell 11** → `NameError`.
* Cell 4 calls `fetch_player_stats(SEASONS, "Base")`, passing the whole season **list**
  where a single season string is expected.

Neither affects the committed CSVs (the collection loop itself is correct, and the data
reproduces exactly from the live API), so I left his notebooks untouched. Worth a two-line
fix by him — moving the function definition above cell 6 — so the notebook reruns cleanly.
