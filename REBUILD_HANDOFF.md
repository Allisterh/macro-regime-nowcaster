# Rebuild handoff — 2026-09-20

The feature panel is being rebuilt because the DFM's EM was diverging on
long panels. This file is the state of that job, so it can be picked up
without reconstructing the reasoning.

Branch: `fix/null-factor-detection` (pushed; merged into nothing yet).
Last commit: `d0a1514 dfm: stop the EM diverging, and name the fifth factor`.

---

## Why the rebuild is needed

`data/features.csv` is stale in two independent ways:

1. **Every factor value moves.** The EM diverged on panels longer than
   ~600 months, and the divergence guard kept an under-converged fit in
   which two of five factors carried no loadings at all. It now
   converges (parameters flat from iteration 80 to 99).
2. **A column is renamed.** `factor_inflation` → `factor_credit_premium`.
   That factor was never inflation: it correlates +0.371 with CPI
   year-on-year against +0.605 for `long_rates`, and its top loadings
   are `AAA10Y`, `BAA10Y`, `DFF`, `T10Y2Y`.

**No measured figure currently in the README can be trusted.** AUC 0.951,
Brier 0.0685, the volatility IC of +0.157 and every downstream table were
produced by the diverging estimator.

---

## The job

```bash
python scripts/build_features.py --start 1967-01-31 --step 1 \
    --output data/features.new.csv --log-level INFO \
    > data/features_rebuild.log 2>&1
```

- ~716 windows, roughly 4 hours.
- Writes `data/features.new.partial.csv` after every row and **resumes**
  from it. Re-running the same command after an interruption continues
  where it stopped — that is the intended path.
- Writes to `features.new.csv`, not over `features.csv`, so the old panel
  survives until the new one is complete.

### Resuming is safe *only while the model code is unchanged*

`generate_feature_panel` rejects a cache whose schema does not match
(no `knowable_at` column). It **cannot** detect a cache written by
different model code under the same schema. So:

- Interrupted and resumed on this commit → fine, resume.
- If `src/models/dynamic_factor_model.py` or `walk_forward.py` changes →
  delete `data/features.new.partial.csv` and start over, or the panel
  will silently mix two estimators.

### Checking progress

```bash
grep -c 'Walk-forward \[' data/features_rebuild.log   # windows done of 716
grep -c 'failed at'      data/features_rebuild.log    # should stay 0
grep -c 'EM diverged'    data/features_rebuild.log    # should stay 0
```

`EM diverged` staying at 0 is the point of the exercise. If it climbs,
the fix did not hold on some window and the result should not be shipped.

The run raises rather than writing if more than 5% of windows fail
(`max_failure_fraction`), so a silently short panel is not possible.

---

## After it finishes

### 1. Sanity-check the new panel

```python
import pandas as pd
d = pd.read_csv("data/features.new.csv", index_col=0, parse_dates=True)
fac = [c for c in d.columns
       if c.startswith("factor_") and not c.endswith(("_d1m","_d3m","_d6m"))]
print(len(d), d.index[0].date(), d.index[-1].date())
print(sorted(fac))                                    # expect factor_credit_premium, no factor_inflation
print(int((d[fac].abs() < 1e-12).all(axis=1).sum()))  # expect 0 all-zero rows
print(d["expected_recession_duration"].max())         # expect <= panel length, not 1e11
print(int((d["signal_rsm"] == 0.5).sum()))            # expect 0
```

Expect ~716 rows, 1967-01 to 2026-08.

### 2. Install it

```bash
mv data/features.new.csv data/features.csv
rm -f data/features.new.partial.csv
```

### 3. Re-measure everything

Recession metrics — a script for this was written but not committed; it
reads the panel, aligns NBER labels, and block-bootstraps ensemble minus
CFNAI over contiguous label runs. Rebuild it from
`src/models/regime_backtest.py` (`roc_auc`, `brier_score`,
`get_nber_recession_indicator`) and
`src/evaluation/horizon_curve.py` (`_label_blocks`, `horizon_auc_curve`).

Downstream:

```bash
python scripts/benchmark_features.py --horizon 3
python scripts/benchmark_features.py --horizon 6
python scripts/benchmark_features.py --horizon 12
```

### 4. Update the README

Every number in **What it does, measured**, **Measured Real-Time
Performance**, and **Is It Useful Downstream?**. Also the
`factor_inflation` → `factor_credit_premium` rename wherever the feature
schema is described.

`tests/test_ensemble_weights.py` will fail if the README's weights table
or architecture diagram drifts from `Nowcaster.DEFAULT_WEIGHTS`; it does
not check the performance numbers, which have to be done by hand.

### 5. Expect these to move differently

- **Recession detection** leans on probit and CFNAI, which carry weight
  0.50 each; the RSM is weighted 0. It survived the last rebuild almost
  unchanged (AUC 0.950 → 0.951) and will probably survive again.
- **The volatility result runs directly on the factor columns**, so it
  genuinely can move in either direction. Do not assume IC +0.157 holds.
  `src/models/volatility_outlook.py` refits from the panel, and its
  `predict` refuses a factor set that does not match rather than
  substituting a value — so the dashboard panel will show a message,
  not a wrong number, until the rebuild lands.

---

## Known gaps not addressed

- `data/oos_validation.csv` and `data/oos_validation_interim.csv` are
  March-2026 artefacts in a superseded schema. The stale-cache guard now
  ignores the interim one with a warning; both could simply be deleted.
- pytest's basetemp (`%LOCALAPPDATA%/Temp/pytest-of-Andrew`) is not
  writable, so `tmp_path` raises `PermissionError`. Tests use
  `tempfile.mkdtemp()` instead. Predates this work; could not be removed.
- `scripts/run_validation.py` regenerates its own panel rather than
  reading `features.csv`, so it duplicates several hours of work.
