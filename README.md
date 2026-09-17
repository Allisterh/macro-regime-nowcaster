# 📊 Macro Regime Nowcaster

> Real-time economic regime detection combining Dynamic Factor Models, Kalman filtering, Markov-switching models, and ensemble recession probability — with an LLM-powered narrative agent and interactive Streamlit dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Regime Probabilities](docs/images/regime_probabilities.png)

---

## Overview

This project implements a **real-time macroeconomic regime nowcaster** that ingests 75 economic indicators from the Federal Reserve (FRED), extracts latent factors using a mixed-frequency state-space model, and classifies the current economic regime as **expansion** or **recession** using a weighted ensemble of four signals.

### Key Features

- **Mixed-Frequency Dynamic Factor Model** — EM algorithm + Kalman filter extracts 4 interpretable latent factors from 75 noisy macro series (daily, weekly, monthly, quarterly) via Mariano–Murasawa cumulator, with varimax rotation for factor identification
- **Ensemble Recession Detection** — Weighted combination of a CFNAI threshold, a supervised probit, the Sahm rule and a Markov-switching model. The weights are set from measured out-of-sample performance, not judgement, and the components are correlated rather than independent
- **Estrella–Mishkin Yield-Curve Probit** — Canonical NY Fed specification (12-month-ahead recession on T10Y3M) available as a standalone baseline
- **Look-Ahead-Free Pipeline** — Expanding-window standardization, Kalman-filtered (not smoothed) factors *and* Hamilton-filtered (not Kim-smoothed) regime probabilities, ragged-edge masking with per-series publication lags, and NBER labels restricted to turning points that had actually been announced
- **Point-in-Time Feature Generation** — `walk_forward` refits the model once per as-of date and keeps only the final row, so a feature table for a downstream model contains only values that were computable at the time; `assert_point_in_time()` enforces it
- **ALFRED Vintage Support** — `DataPipeline(use_vintages=True)` fetches each series as it stood on the run date, so later revisions never reach the model
- **OLS-Calibrated GDP Nowcast** — Factor-augmented GDP prediction against actual GDPC1 growth, with the target quarter held out and a predictive (not residual) confidence interval
- **LLM Narrative Agent** — GPT-powered macro analyst that synthesizes quantitative signals with scraped Federal Reserve communications (FOMC minutes, Beige Book, speeches)
- **Interactive Streamlit Dashboard** — Full visualization suite with regime probabilities, factor dynamics, allocation weights, and one-click narrative generation
- **Regression-tested against its own failure modes** — prefix-invariance at both the pipeline and model layers, unit contracts on threshold signals, regime-orientation checks, and statistical recovery tests against simulated ground truth

---

### What it does, measured

Everything below is from a **670-point monthly walk-forward** (1967–2025, eight
recessions), refitting at each date and reading only that fit's final row.
Details in [Measured Real-Time Performance](#measured-real-time-performance)
and [Is It Useful Downstream?](#is-it-useful-downstream)

| | |
|---|---|
| Recession detection, real-time | AUC **0.917**, Brier **0.0837** |
| …against CFNAI alone | AUC 0.861, Brier 0.1039 |
| Calibration vs CFNAI | **19% better, P = 0.98** (CI excludes zero) |
| Discrimination vs CFNAI | +0.057 AUC — inside the noise band |
| Forecasting 12 months out | AUC 0.611 — modest |
| Predicting forward equity returns | **no skill** (negative R² everywhere) |
| Predicting forward vol / drawdown | **no skill either** on the full sample — an earlier positive result did not survive extending it |
| COVID, in real time | **missed** — two months is below the data's resolution |

**Read this before building on it.** The ensemble produces a
*better-calibrated* probability than CFNAI — significantly so — while its
ability to *rank* periods is statistically indistinguishable from it. It is a
coincident detector, not a forecaster: at twelve months it is barely better than
a coin flip, and it missed COVID entirely. Its defensible value is a calibrated
probability, an interpretable decomposition of 75 series, and a point-in-time
feature generator tested not to leak. The evaluation harness that establishes
all of this is part of the repository, and reproducing any number is one
command.

---

## Dashboard

The Streamlit dashboard provides real-time visualization of all model outputs:

![Dashboard Banner](docs/images/dashboard_banner.png)

### Ensemble Signal Decomposition

Four independent recession signals are combined with calibrated weights:

![Ensemble Breakdown](docs/images/ensemble_breakdown.png)

| Signal | Weight | Method |
|--------|--------|--------|
| **Probit** | 0.40 | Supervised model trained on NBER recession dates with lagged yield curve & credit spreads (class-balanced) |
| **RSM** | 0.20 | Hamilton (1989) Markov-switching on the *cyclical* DFM factors (real activity + labour) |
| **CFNAI** | 0.20 | Chicago Fed MA3 < −0.7 convention (logistic-smoothed) |
| **Sahm** | 0.20 | Sahm rule — 3-month MA unemployment minus 12-month min (logistic-smoothed) |

### Latent Factor Dynamics

Four interpretable factors extracted via PCA + EM + varimax rotation. Names are assigned by *loading evidence* rather than column position — varimax does not preserve ordering — and signs are fixed by a loading-sum convention so the factor orientation is reproducible across runs:

![Latent Factors](docs/images/latent_factors.png)

### Historical Regime Classification

Continuous regime probability mapped against NBER recession dates:

![Regime Timeline](docs/images/regime_timeline.png)

---

## Architecture

![Architecture](docs/images/architecture.png)

```
FRED API (75 series, daily/weekly/monthly/quarterly)
    │
    ▼
DataPipeline ── mixed-freq align, transform, expanding standardize, ragged-edge mask
    │                                          │
    │                                          └──► raw_levels_ (UNRATE, CFNAI)
    ▼                                               kept in published units for
DynamicFactorModel (EM + Kalman)                    the threshold signals below
    │  4 latent factors, Mariano–Murasawa cumulator, varimax rotation,
    │  names by loading evidence, filtered output (no look-ahead)
    │
    ├──► Markov-Switching RSM     (wt: 0.20)  ◄── cyclical factors only
    ├──► Recession Probit         (wt: 0.40)  ◄── NBER dates (announcement-lagged),
    │                                             yield curve, credit spreads
    ├──► CFNAI Signal             (wt: 0.20)  ◄── raw CFNAI level
    ├──► Sahm Rule                (wt: 0.20)  ◄── raw unemployment level
    │
    ▼
Ensemble P(Recession) + GDP Nowcast
    │
    ├──► RegimeAllocator ── conditional asset weights (net of transaction costs)
    ├──► Streamlit Dashboard ── interactive visualization
    ├──► NarrativeAgent (GPT) + FedScraper ── macro narrative
    └──► walk_forward ── point-in-time feature panel for downstream models
             (one refit per as-of date; only the final row is kept)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/AndrewFSee/macro-regime-nowcaster.git
cd macro-regime-nowcaster
pip install -e ".[dev]"
```

### 2. Set API keys

```bash
cp .env.example .env
# Edit .env and add:
#   FRED_API_KEY=your_key_here       (required — free at https://fred.stlouisfed.org/docs/api/api_key.html)
#   OPENAI_API_KEY=your_key_here     (optional — enables LLM narrative generation)
```

### 3. Run a nowcast

```bash
python scripts/run_nowcast.py
```

### 4. Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

Click **🔄 Run Nowcast** in the sidebar to fetch live data and generate results.

### 5. Build point-in-time features (for a downstream model)

```bash
python scripts/build_features.py --verify   # check the generator is leak-free first
python scripts/build_features.py            # then generate (slow; resumes from cache)
```

See [Point-in-Time Features](#point-in-time-features-for-downstream-models) for why this
is not the same as slicing the history out of a single `Nowcaster.run()`.

---

## Model Methodology

### Kalman Filter / State-Space Model

The core estimation engine is a linear Gaussian state-space model:

$$F_t = A \cdot F_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, Q)$$

$$Y_t = C \cdot F_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, R)$$

- $F_t \in \mathbb{R}^K$: latent factors ($K = 4$)
- $Y_t \in \mathbb{R}^N$: observed economic indicators ($N \approx 70$)
- Missing observations (ragged edge) are handled by skipping the Kalman update step
- Per-series publication lags (from `fred_series.yaml`) mask stale data at each date

### Dynamic Factor Model (EM Algorithm)

Parameters $(A, C, Q, R)$ are estimated via the EM algorithm:
- **E-step**: Kalman smoother computes sufficient statistics $\mathbb{E}[F_t | Y], \mathbb{E}[F_t F_t' | Y]$
- **M-step**: Closed-form OLS updates for all parameters
- **Initialization**: PCA on filled data for fast convergence
- **Post-processing**: Varimax rotation for interpretable factors, sign-alignment against anchor series
- **Filtered output**: Downstream models default to `filtered_factors_` (real-time safe) rather than `smoothed_factors_` to prevent look-ahead bias

### Mixed-Frequency Support (Mariano–Murasawa)

Quarterly series (GDP, credit conditions) are handled via state augmentation:

$$\tilde{F}_t = [f_t,\ f_{t-1},\ f_{t-2}]' \in \mathbb{R}^{3K}$$

The augmented observation equation for quarterly series becomes:

$$y_t^{(q)} = C^{(q)} (f_t + f_{t-1} + f_{t-2}) + \varepsilon_t$$

This allows monthly and quarterly data to be estimated jointly in a single Kalman pass without temporal aggregation.

### Hamilton (1989) Markov-Switching Model

Economic regimes follow a first-order Markov chain:

$$S_t | S_{t-1} \sim \text{Categorical}(P[S_{t-1}, :])$$

$$F_t | S_t = j \sim \mathcal{N}(\mu_j, \Sigma_j)$$

- **Filtered** probabilities via the Hamilton (1989) forward recursion — the default, and the only estimate safe to use as a historical series
- **Smoothed** probabilities via the Kim (1994) backward smoother, available as `smoothed_probs_` for retrospective turning-point dating (these condition on *t+1..T* and must not feed a supervised model)
- Fitted on the **cyclical** factor block (`real_activity`, `labor_market`) rather than all four.
  Hamilton's regime variable is the business-cycle state; including the inflation and
  financial-stress factors lets the mixture split on an inflation regime instead. Measured
  over the full sample: all four factors give AUC 0.705 with 49% average recession occupancy
  (it fires half the time), the cyclical block gives **AUC 0.820** at 33%. Configurable via
  `Nowcaster(rsm_factors=...)`; pass `None` to use every factor.
- Regime labels are assigned by ascending mean, so index 0 is always the recession state.
  `get_recession_probability()` resolves that column **positionally**, and a `regime_labels`
  ordering that contradicts the sort is rejected at construction rather than silently
  inverting the signal.
- Supports multi-regime (K > 2) with labels: recession / slowdown / recovery / expansion

### Sahm Rule

The Sahm rule indicator is:

$$\text{Sahm}_t = \overline{U}_{t}^{(3)} - \min_{s \in [t-12,\,t-1]} U_s$$

where $\overline{U}_{t}^{(3)}$ is the 3-month moving average of the unemployment rate. A value ≥ 0.50 has historically signalled every US recession since 1970.

The ensemble uses a logistic-smoothed version centred at the 0.50 threshold.

The threshold is in **percentage points of the unemployment level**, so this signal reads `DataPipeline.raw_levels_` rather than the modelling panel, whose columns are differenced and expanding-standardised. The same applies to the CFNAI −0.7 convention.

### Estrella–Mishkin Yield-Curve Probit

The canonical NY Fed recession indicator (Estrella & Mishkin, 1998):

$$P(\text{Recession}_{t+12}) = \Phi(\alpha + \beta \cdot \text{Spread}_t)$$

where $\text{Spread}_t$ = T10Y3M. Available as a standalone baseline via `RecessionProbit.estrella_mishkin()`.

### GDP Nowcast

OLS regression of actual quarterly GDPC1 growth against DFM factors:

$$\text{GDP}_q = \alpha + \boldsymbol{\beta}' \mathbf{f}_q + \varepsilon_q$$

The quarter being nowcast is held out of the fit, and the 90% interval uses the *predictive* standard error (residual variance inflated by leverage) rather than the in-sample residual standard error.

### Point-in-Time Features for Downstream Models

`Nowcaster.run()` fits every component on the sample it is given, so the
*history* it exposes is not what the model would have produced in real
time — the EM parameters, varimax rotation, factor normalisation and
probit coefficients all depend on the full sample. Slicing that history
into a feature table backtests on values nobody could have observed.

`src/models/walk_forward.py` builds the history the other way round: one
fit per as-of date, keeping only the final row. That row is safe because
at *t = T* the Kalman smoother coincides with the filter and the Kim
smoother coincides with the Hamilton filter.

```python
from src.models.walk_forward import generate_feature_panel, assert_point_in_time

features = generate_feature_panel(
    pipeline, start="2000-01-31", end="2024-12-31",
    step_months=1, cache_path="data/features.csv",   # slow; resumes from cache
)

# The gate to pass before a downstream model consumes the table:
assert_point_in_time(pipeline, early_cutoff="2016-12-31", late_cutoff="2019-12-31")
```

Each row carries more than a single probability, so a downstream model
can learn its own weighting rather than inheriting the fixed
0.20/0.40/0.20/0.20 blend:

| Feature group | Columns |
|---------------|---------|
| Latent factors | `factor_<name>`, plus 1/3/6-month changes |
| Ensemble components | `signal_rsm`, `signal_probit`, `signal_cfnai`, `signal_sahm` |
| Ensemble probability | `p_recession`, plus 1/3/6-month changes |
| Model uncertainty | `signal_dispersion` (spread across the four signals) |
| Regime dynamics | `p_stay_recession`, `expected_recession_duration`, `p_enter_recession`, `regime_age_months` |
| GDP | `gdp_nowcast`, `gdp_ci_width` |
| Timing | `knowable_at` |

**Join on `knowable_at`, not on the index.** The index is the reference
month; `knowable_at` is the date the row could first have been computed,
once its slowest input had printed. A January reading is not usable on
31 January.

Because the target is autocorrelated and the windows overlap, evaluate
with purged and embargoed cross-validation rather than standard k-fold.
And benchmark against the trivial alternative — the T10Y3M level, a
high-yield spread, and the Sahm rule as three raw features — before
concluding the factor machinery is earning its keep.

### How Performance Is Reported

Two conventions here are deliberate, and both make the headline numbers *lower*
than they would otherwise be:

**The decision threshold is fixed at 0.5.** Sweeping thresholds and reporting
metrics at the F1-maximising one selects a hyper-parameter on the evaluation
labels, which inflates accuracy, precision, recall and F1 together. Every
`RegimeBacktestResult` therefore also carries **ROC AUC** and the **Brier
score**, neither of which can be tuned on labels — those are the numbers to
compare across model versions.

**Signal orientation is fixed upstream, never inferred from labels.** The
recession column is column 0 by the ascending-mean sort, the DFM's factor signs
come from a loading-sum convention, and a contradictory `regime_labels`
ordering raises. Flipping a signal because it correlates negatively with the
test labels is a label leak, not a bug fix.

`RegimeBacktestResult.in_sample` marks results from the full-sample path, and
`summary()` prints a warning banner for them. `run()` dispatches to
`run_expanding()` by default, which refits at each step and reads only the final
row.

---

### Look-Ahead Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Expanding-window standardization | z-scores at time $t$ use only data through $t$ |
| Filtered factors | DFM defaults to `filtered_factors_`; RSM defaults to `filtered_probs_` |
| Ragged-edge masking | Per-series `publication_lag_days` masks observations not yet released at the run's `end_date` |
| ALFRED vintages | `DataPipeline(use_vintages=True)` fetches each series *as it stood* on the run date, so revisions published later never reach the model |
| NBER announcement lag | `nber_labels_available_at()` restricts labels to turning points the NBER had actually announced — they are published 5–21 months after the month they date |
| No forward fill or back fill | Stale data stays NaN; signals fall back to 0.5 rather than consuming a stale value |
| Fixed decision threshold | Metrics are reported at 0.5 alongside threshold-free AUC and Brier; tuning a cut-off on the evaluation labels inflates every number derived from it |
| Point-in-time features | `walk_forward.generate_feature_panel()` refits per as-of date and keeps only the final row |
| Prefix-invariance tests | `tests/test_no_leakage.py` (pipeline) and `tests/test_walk_forward.py` (model outputs) |

**What these do *not* cover.** A single `Nowcaster.run()` still fits every
component on the whole sample it is handed, so `get_ensemble_probabilities()`
returns an in-sample history. That is the right object for inspecting the
current state and for the dashboard, and the wrong one to use as a
model feature — use the walk-forward generator above for that.

---

## Project Structure

```
macro-regime-nowcaster/
├── config/
│   ├── settings.yaml              # Model hyperparameters, allocation weights
│   └── fred_series.yaml           # 75 FRED series with transforms, lags & frequencies
├── src/
│   ├── data/
│   │   ├── fred_client.py         # FREDClient with ALFRED vintage support (as_of)
│   │   ├── data_pipeline.py       # DataPipeline — transform, standardize, ragged-edge,
│   │   │                          #   raw_levels_ for threshold signals
│   │   ├── mixed_frequency.py     # Mariano–Murasawa cumulator & frequency alignment
│   │   └── ...                    # transforms, storage
│   ├── models/
│   │   ├── kalman_filter.py       # Kalman filter & smoother
│   │   ├── dynamic_factor_model.py # PCA + EM + varimax DFM (mixed-freq cumulator)
│   │   ├── regime_switching.py    # Hamilton Markov-switching (filtered + smoothed)
│   │   ├── recession_probit.py    # Supervised probit + Estrella-Mishkin classmethod
│   │   ├── sahm_rule.py           # Sahm recession indicator & logistic-smoothed signal
│   │   ├── regime_backtest.py     # NBER backtesting + announcement-lagged labels
│   │   ├── walk_forward.py        # Point-in-time feature generation for downstream ML
│   │   └── nowcaster.py           # End-to-end 4-signal ensemble orchestrator
│   ├── evaluation/                # Purged + embargoed CV splitter
│   ├── allocation/                # RegimeAllocator, Backtester
│   ├── agent/                     # NarrativeAgent (GPT), FedScraper, prompts
│   ├── dashboard/                 # Streamlit app (8 panels)
│   └── utils/                     # Logging, date utilities
├── tests/                         # pytest suite (fast subset: `-m 'not slow'`)
├── notebooks/                     # 5 Jupyter notebooks (EDA → full pipeline)
├── .github/workflows/tests.yml    # CI: fast suite on push, slow suite on PR
├── scripts/                       # CLI: fetch, train, nowcast, backtest, validate,
│                                  #   build_features, benchmark_features
├── docs/images/                   # README visualizations
├── pyproject.toml
├── requirements.txt
└── Makefile
```

---

## Data Sources

| Category | Example Series | Count |
|----------|---------------|-------|
| Labor Market | PAYEMS, UNRATE, ICSA, JTSJOL, JTSQUR, TEMPHELPS | 13 |
| Production & Activity | INDPRO, CFNAI, CFNAIDIFF, WEI, Empire/Philly/Dallas Fed | 15 |
| Financial Conditions | T10Y2Y, T10Y3M, BAMLH0A0HYM2, NFCI, ANFCI | 13 |
| Consumer & Housing | UMCSENT, PERMIT, HOUST, MORTGAGE30US, CSUSHPISA | 8 |
| Prices & Inflation | CPIAUCSL, PCEPI, T5YIE, MICH, DCOILWTICO | 9 |
| Money & Credit | M2SL, TOTBKCR, BUSLOANS, DRTSCILM, WRESBAL | 11 |
| International | DTWEXBGS, DCOILBRENTEU, VIXCLS | 3 |
| GDP | GDPC1, GDPDEF, A261RX1Q020SBEA | 3 |
| **Total** | | **75** |

Every id in `config/fred_series.yaml` is checked against the FRED API by
`tests/test_series_config.py` (opt-in, needs a key). Three shipped ids did not
exist — the Empire State and Philadelphia Fed codes had their `DI`/`DF`
infixes transposed, and the London bullion gold series had been delisted — and
because `DataPipeline` logs a warning and continues when a fetch fails, the
survey block was silently absent from the panel.

All data sourced from the [FRED database](https://fred.stlouisfed.org/) (Federal Reserve Bank of St. Louis). Quarterly series (GDP, credit conditions) are handled via Mariano–Murasawa state augmentation. ALFRED point-in-time vintages are supported for real-time backtesting.

---

## Configuration

Key settings in `config/settings.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.n_factors` | 4 | Number of latent factors |
| `model.n_regimes` | 2 | Recession / Expansion (supports 3 or 4) |
| `model.regime_labels` | `[recession, expansion]` | **Order matters.** Regimes sort by ascending mean, so the recession state must be listed first; the reverse order is rejected at construction |
| `model.ensemble.weights.probit` | 0.40 | Probit weight (supervised, class-balanced) |
| `model.ensemble.weights.rsm` | 0.20 | Markov-switching weight |
| `model.ensemble.weights.cfnai` | 0.20 | CFNAI signal weight |
| `model.ensemble.weights.sahm` | 0.20 | Sahm rule weight |
| `model.ensemble.probit.add_lags` | 3 | Lagged copies of each probit feature |
| `model.ensemble.probit.regularization` | 1.0 | L2 penalty; `fit_regularization_cv()` selects it by forward-chaining CV |
| `model.ensemble.probit.extra_features` | CFNAI, T10Y2Y, BAA10Y | Series added to the DFM factors |
| `data.start_date` | 1980-01-01 | Historical sample start |
| `data.vintage_dir` | data/vintages | ALFRED vintage storage |
| `agent.llm_model` | gpt-4o-mini | LLM for narrative generation |

The probit block under `model.ensemble` used to be read by nothing — `Nowcaster`
carried its own hard-coded copy with different values. It is now the single
source, passed through as `Nowcaster(probit_config=...)`.

---

## Running Tests

```bash
pytest tests/ -m "not slow"      # fast subset — use this while developing
pytest tests/ --cov=src          # everything, including the model refits
# or
make test
```

Tests that repeatedly refit the DFM/RSM are marked `slow`; they take
minutes rather than seconds because each one estimates the model many
times over.

Beyond the shape and smoke coverage, the suite pins the properties that
actually went wrong at some point:

| Test module | Property |
|-------------|----------|
| `test_no_leakage.py` | Pipeline output at *t* is unchanged by appending data after *t* |
| `test_walk_forward.py` | The same invariance for **model** outputs, plus a check that the guard fails on an injected leak |
| `test_signal_units.py` | Sahm/CFNAI receive published units, anchored to real 2008–09 history; a z-score is rejected |
| `test_regime_orientation.py` | P(recession) tracks the low-mean state under any labelling; reversed labels raise; output is not saturated |
| `test_dfm_recovery.py` | The DFM recovers a simulated factor space (canonical correlations), beats plain PCA, and survives 15% missing data |

---

---

## Measured Real-Time Performance

Every figure below comes from a **walk-forward**: 670 monthly refits from 1967
to 2025, each fitted only on data available at that date and contributing only
its own final row. No number here is in-sample.

The sample covers **eight** recessions — 1969-70, 1973-75, 1980, 1981-82,
1990-91, 2001, 2007-09 and 2020 — and 85 recession months out of 670. Reaching
back past 1980 is what the reconstructed `TERM_SPREAD` and `CREDIT_SPREAD`
series are for: FRED's ready-made `T10Y3M` and `BAA10Y` start only in 1982 and
1986, which would cap the evidence at four recessions.

| Signal | AUC | Brier |
|--------|-----|-------|
| **Ensemble** | **0.917** | **0.0837** |
| Probit | 0.901 | 0.0879 |
| CFNAI signal alone | 0.861 | 0.1039 |
| Sahm rule | 0.781 | 0.1790 |
| *Constant forecast at the 12.7% base rate* | 0.500 | 0.1108 |

### Does it beat CFNAI?

This is the question the project has to answer: its output should be worth more
than the free index it partly consumes. Block-bootstrapped over the eight
episodes, with weights fixed in advance:

| Metric | Ensemble − CFNAI | 95% CI | Verdict |
|--------|------------------|--------|---------|
| AUC | +0.057 | [−0.011, +0.178] | indistinguishable |
| **Brier** | **−0.0202** | **[−0.0512, −0.0004]** | **ensemble wins (P = 0.98)** |

**On calibration, yes — significantly.** The interval excludes zero, and the
ensemble's Brier is 19% better than CFNAI's. On *discrimination* the two remain
statistically indistinguishable, though the ensemble leads at every horizon out
to nine months.

That split matters more than it might look. Ranking periods correctly and
producing a number whose 0.30 means 30% are different properties, and the second
is the one that counts when the output is consumed as a feature by something
downstream. A well-calibrated probability is the defensible product here.

### Discrimination decays with horizon

| Horizon | Ensemble | CFNAI | Probit | Sahm |
|---------|----------|-------|--------|------|
| 0m | **0.917** | 0.861 | 0.901 | 0.781 |
| 3m | **0.843** | 0.802 | 0.824 | 0.700 |
| 6m | **0.758** | 0.728 | 0.738 | 0.661 |
| 9m | **0.676** | 0.660 | 0.642 | 0.641 |
| 12m | 0.611 | **0.630** | 0.558 | 0.619 |
| 18m | 0.539 | **0.586** | 0.487 | 0.555 |

The ensemble leads to nine months; CFNAI takes over beyond that. None of these
gaps is individually significant — the dashboard plots them with block-bootstrap
bands, which overlap heavily, and that is the honest way to read the table.

### Caveats that bound every number above

**Eight episodes is still a small sample.** Doubling it from four (1990-2026) to
eight reversed several conclusions during development. Treat differences of a
few AUC points as noise unless an interval says otherwise.

**It detects rather than forecasts.** At twelve months the ensemble is at 0.611
and CFNAI at 0.630, both modest. Anything needing warning rather than
confirmation should read the caveats in
[Is It Useful Downstream?](#is-it-useful-downstream) first.

**COVID was missed in real time.** The 2020 recession lasted two months; monthly
macro data published with a lag physically cannot resolve it. That is a
structural limit of the approach, not a tuning problem.

Reproduce with:

```bash
python scripts/build_features.py --step 1 --start 1967-01-31
```

**This takes several hours** — 670 separate model fits, which is the point: a
cheaper history would not be point-in-time. Results cache incrementally, so an
interrupted run resumes. `data/` is gitignored, so a fresh clone pays this cost
once before `scripts/benchmark_features.py` will run. Use `--step 3` while
iterating.

## Is It Useful Downstream?

The reason to build this rather than read CFNAI off the Chicago Fed website is
that its output should be worth more than the free index it partly consumes.
`scripts/benchmark_features.py` tests that on three-month forward NASDAQ
targets under purged and embargoed walk-forward CV.

Two rules make it honest: features join on **`knowable_at`**, not the reference
month, and folds are **purged and embargoed** — on a target built from
overlapping forward windows, shuffled k-fold reports ~+0.69 correlation where
the truth is zero (`tests/test_purged_cv.py` asserts exactly this).

Ridge, 4–5 folds, 670 monthly observations (1967–2025). IC is the mean fold
correlation; R² is measured against the *training* mean — the naive forecast
actually available at prediction time.

| Feature set | return IC | return R² | vol IC | vol R² | drawdown IC | drawdown R² |
|---|---|---|---|---|---|---|
| CFNAI only | +0.01 | −0.01 | −0.21 | −0.05 | −0.15 | −0.02 |
| p_recession only | +0.05 | −0.01 | −0.12 | −0.05 | −0.11 | −0.01 |
| regime signals (4) | −0.00 | −0.04 | −0.06 | −0.30 | −0.02 | −0.21 |
| factors only | −0.05 | −0.22 | +0.07 | −0.16 | +0.02 | −0.31 |
| full regime panel | −0.06 | −0.33 | −0.05 | −0.55 | −0.06 | −0.69 |

**Nothing here predicts equity outcomes.** Every R² is negative: no feature set
beats predicting the historical average, for returns, volatility or drawdown.

This *reverses* an earlier finding. On the 1990–2026 sample the same benchmark
showed genuine risk predictability — volatility IC +0.40 with positive R², and
drawdown +0.26. Extending to 1967 removed it, and several ICs changed sign.
Either the relationship is unstable across eras, or the shorter sample was
showing noise; on this evidence there is no way to tell, and the honest reading
is that the effect was never established.

**Fewer features still win.** The single-column sets are least bad on every
target, and the full 32-column panel is worst everywhere (vol R² −0.55,
drawdown −0.69). With a handful of recessions there is nothing to support a
wide fit; extra columns only add variance.

### What this means if you were planning to use it

- **Directional prediction: no.** Nothing in this repository supports it.
- **Risk conditioning: unproven.** It looked promising on 1990–2026 and did not
  survive 1967–2025. Re-run the benchmark on whatever sample you intend to
  trade before relying on it.
- **Regime state as a conditioner or interaction term** remains the most
  defensible use, and the calibration result above is the reason to prefer
  `p_recession` over a raw indicator for it — but that is an argument about the
  quality of the probability, not evidence of trading edge.

```bash
python scripts/benchmark_features.py --horizon 3
```

---

## Correctness Audit

An audit of this codebase found several defects that silently corrupted the
model's output while the whole test suite passed. They are fixed, and each now
has a regression test that fails without the fix. Recording them here because
the numbers this project produced before the fixes are not comparable with the
ones it produces now.

| Defect | Effect | Now |
|--------|--------|-----|
| Sahm rule read the panel's **z-scored first difference** of `UNRATE` instead of the unemployment level | The 0.50pp threshold was compared against a unitless z-score. The signal fired in 87% of months against 27% for the real rule, and correlated **−0.067** with it, while carrying 20% of the ensemble weight | Threshold signals read `DataPipeline.raw_levels_`, captured before the transform step. `Nowcaster._raw_level()` raises if given values outside published-unit ranges |
| CFNAI's −0.7 convention applied to an expanding z-score | Threshold drifted with the sample standard deviation | Same fix |
| `regime_labels: [expansion, recession]` in `settings.yaml` | Regimes sort by *ascending mean*, so the name "recession" landed on the expansion state. On a synthetic chain with known states, correlation with truth was **−1.000** instead of +1.000. Affected `run_nowcast.py` and `run_validation.py` | Column resolved positionally; contradictory orderings raise |
| Backtest evaluation used the test labels three ways: probit trained on its own evaluation window, threshold tuned for F1, RSM sign chosen by correlation | `corr(probit, NBER) = +0.906` was a leakage artefact | All three removed (a fourth instance was found in `_evaluate`). Fixed threshold plus AUC/Brier |
| RSM stored **Kim-smoothed** probabilities, which condition on *t+1..T* | Historical values shifted by up to 0.12 as future data arrived, contradicting the README's own look-ahead claims | Filtered and smoothed both retained; filtered is the default |
| Probabilities clipped to `[0.05, 0.95]` / `[0.02, 0.98]` | Output latched onto the bounds — 421 of 434 months sat at a bound, leaving no usable gradation | Numerical guard only, plus CV-selected regularisation and `calibration_report()` |
| DFM factor sign was unidentified when no anchor series matched | The SVD sign is arbitrary, so the ascending-mean regime sort was effectively a coin flip. The old label-correlation auto-flip had been masking this | Loading-sum sign convention, with anchors overriding |
| Factor names assigned by column position after varimax | Varimax does not preserve ordering, so the factor labelled "inflation" could be the financial-stress factor — and was then sign-aligned against the wrong anchors | Names matched to factors by loading evidence (`scipy.optimize.linear_sum_assignment`) |
| ALFRED vintage support was never called, and requested a *range* of vintages | Every backtest ran on fully-revised data | Wired through `DataPipeline(use_vintages=True)`; both realtime bounds pinned to the as-of date |
| NBER labels used before the NBER announced them | Turning points are published 5–21 months after the month they date | `nber_labels_available_at()` filters by announcement date |
| Three FRED ids did not exist: the Empire State and Philadelphia Fed codes had their `DI`/`DF` infixes **transposed**, and the gold series was delisted | `DataPipeline` warns and continues on a failed fetch, so the survey block was silently absent from the panel | Corrected; `tests/test_series_config.py` validates every id (opt-in live check) |
| `.gitignore` contained bare `data/` and `models/` | These match a directory of that name at **any depth**, so `src/data/` and `src/models/` — the DFM, Kalman filter, regime model, probit, nowcaster and backtester — were never under version control | Anchored to `/data/` and `/models/` |
| Allocation backtest charged no transaction costs | Turnover was computed and reported but never deducted, flattering a strategy that rotates most of the book at a regime switch | `Backtester(transaction_cost_bps=10.0)`, charged on the L1 weight change |
| **Three of four signals reported the 0.5 fallback as if it were a reading.** Publication lags mask the last month or two of every series — by design — but the ensemble read `.iloc[-1]`, saw NaN and fell back to 0.5. Separately, `max_column_missing` of 0.9 kept the *quarterly* `DRTSCILM` in a monthly panel, where it is 67% missing, and requiring it non-NaN then dropped two rows in three and emptied the probit's training set | The dashboard showed "Probit 50.0%", indistinguishable from a genuine coin-flip reading, and the headline probability averaged fill values with real ones | Signals use the latest *published* observation and report its date; the headline is computed from those point estimates so it cannot disagree with the breakdown beside it; the column threshold is 0.5 and the quarterly feature is gone |
| **The probit never trained in 64% of months.** `fit()` dropped every row containing a NaN, so one feature with no data in the training window emptied the whole training set; the exception was caught and the signal left at its 0.5 default | Measured as AUC 0.718 and "below chance beyond nine months", prompting a confident structural story about 1970s oil shocks. It was not underperforming — it was not running. Alive, it scores **0.901** | Columns are screened before rows: a feature absent from the window is excluded from the fit and from prediction. Partial missingness is imputed with training-fold means |
| **An extended walk-forward silently lost six of ten recessions.** `_quarterly_to_monthly` called `.index.min()` on an empty series, giving `NaT`; series starting after the as-of date correctly return nothing and took down the entire window | A run launched to cover 1967-2025 produced only 1996-2025, and reported "DONE" with a row count that looked plausible | Empty input is treated as the normal condition it is. The runner keeps WARNING-level logging so failed windows are visible |
| `ragged_edge_mask` did ~38k scalar `.loc` writes per call | Dominated pipeline time in expanding backtests | Vectorised: **37× faster**, bit-identical output |


**A note on how these were found.** All of the failures above were silent: an
exception swallowed by a `try/except`, a fallback of 0.5 that looks like a
legitimate probability, a "DONE" line with a plausible row count, and a NaN
comparing false against a threshold. Each produced *numbers*, and those numbers
supported confident, wrong conclusions until something was reconciled against a
count that did not match. The tests added alongside each fix assert the
behaviour rather than the absence of an exception, because an exception was
never raised.

The 0.5 fallback is the worst offender, and it is worth stating as a design
rule: **a sentinel that is indistinguishable from a legitimate output will be
mistaken for one.** Every signal now carries the reference date of the
observation behind it, so a stale or unavailable reading is visible rather than
inferred.

The one change here that is a **modelling opinion rather than a bug fix** is the
RSM factor restriction described under [Hamilton (1989) Markov-Switching
Model](#hamilton-1989-markov-switching-model): fitting on the cyclical block
instead of all four factors lifts AUC from 0.705 to 0.820. Set
`Nowcaster(rsm_factors=None)` to restore the previous behaviour.

---

## Academic References

1. Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2), 357–384.
2. Kim, C.J. (1994). *Dynamic Linear Models with Markov-Switching*. Journal of Econometrics, 60(1-2), 1–22.
3. Doz, C., Giannone, D., & Reichlin, L. (2012). *A Quasi Maximum Likelihood Approach for Large Approximate Dynamic Factor Models*. Review of Economics and Statistics, 94(4), 1014–1024.
4. Stock, J.H., & Watson, M.W. (2002). *Forecasting Using Principal Components from a Large Number of Predictors*. JASA, 97(460), 1167–1179.
5. Bańbura, M., & Rünstler, G. (2011). *A Look into the Factor Model Black Box*. IJF, 27(2), 333–346.
6. Mariano, R.S., & Murasawa, Y. (2003). *A New Coincident Index of Business Cycles Based on Monthly and Quarterly Series*. Journal of Applied Econometrics, 18(4), 427–443.
7. Estrella, A., & Mishkin, F.S. (1998). *Predicting U.S. Recessions: Financial Variables as Leading Indicators*. Review of Economics and Statistics, 80(1), 45–61.
8. Sahm, C.R. (2019). *Direct Stimulus Payments to Individuals*. In *Recession Ready: Fiscal Policies to Stabilize the American Economy*, Hamilton Project, Brookings Institution.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
