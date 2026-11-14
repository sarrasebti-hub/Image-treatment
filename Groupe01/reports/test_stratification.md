# Test-set stratification procedure

The test set is drawn from the eligible `val2017` pool in `01_make_splits.ipynb`
cell 15. The validation set is drawn separately from the eligible `train2017`
pool. With the saved seed of 42, `sample_eval` is called for test with seed 43
(`42 + 1`) and for validation with seed 44 (`42 + 2`). The KB is sampled from
`train2017` with seed 45 (`42 + 3`), excluding validation IDs. The notebook
asserts that the selected IDs are unique and that test does not overlap KB or
validation.

The procedure below follows `sample_eval` in cell 9 and its helper functions in
cells 5, 7, and 13:

1. `load_pool` keeps images with at least the configured number of captions and
   objects, and whose shorter image side meets the configured minimum. It
   labels complexity from the number of non-crowd object annotations:
   simple = 1–2, medium = 3–5, complex = 6–9, crowded = 10 or more.
2. Before filling complexity quotas, `greedy_cover` selects up to
   `int(0.5 * n)` images (50 for each 100-image split) to cover classes, with
   `min_per_cat=1`. At each step it targets the rarest still-needed class and
   chooses an available image that covers the most other still-needed classes;
   the seeded RNG breaks ties.
3. Complexity targets are rounded from `eval_complexity_target`: simple 25%,
   medium 30%, complex 30%, crowded 15%. If rounding does not sum to `n`, the
   code adjusts the largest target when over, or adds to medium when under.
   The initial class-cover selection is accounted for; any overfilled
   complexity quota is rebalanced, and any remaining shortfall is assigned to
   medium.
4. The minimum interaction target is
   `ceil(eval_min_interaction_frac * n)` = 30 for a 100-image sample. An image
   counts as an interaction when it has a person and at least one other class,
   at least two person annotations, or object supercategories that include
   animal plus at least one other supercategory. While filling each complexity
   quota, candidates are considered in a context rotation of indoor, outdoor,
   mixed. Interactions are preferred while the global interaction target has
   not yet been reached.
5. Candidate IDs are sorted before being shuffled within complexity using the
   split's seeded RNG. The final result is truncated to `n`.

The implementation rotates contexts; it does not enforce a separate numeric
context quota in `sample_eval`, even though the configuration includes an
`eval_min_context_frac` value. The observed context and interaction counts
below are from the frozen `artifacts/splits/splits_meta.json` summaries; the
interaction count is the saved fraction multiplied by 100. The complexity
thresholds and interaction rule match the notebook helpers.

## Observed strata

| Split | Images | Simple | Medium | Complex | Crowded | Indoor | Outdoor | Mixed | Interactions | Classes covered | Minimum images/class |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 100 | 25 | 30 | 30 | 15 | 35 | 35 | 30 | 69 (69%) | 80 | 1 |
| Test | 100 | 25 | 30 | 30 | 15 | 37 | 33 | 30 | 61 (61%) | 80 | 1 |

The corresponding machine-readable table is [test_strata.csv](./tables/test_strata.csv).
