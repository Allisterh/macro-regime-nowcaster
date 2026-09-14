# 📊 Macro Regime Nowcaster

> Real-time economic regime detection combining Dynamic Factor Models, Kalman filtering, Markov-switching models, and ensemble recession probability — with an LLM-powered narrative agent and interactive Streamlit dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Regime Probabilities](docs/images/regime_probabilities.png)

---

## Overview

This project implements a **real-time macroeconomic regime nowcaster** that ingests 75 economic indicators from the Federal Reserve (FRED), extracts latent factors using a mixed-frequency state-space model, and classifies the current economic regime as **expansion** or **recession** using an ensemble of four independent signals.

### Key Features

- **Mixed-Frequency Dynamic Factor Model** — EM algorithm + Kalman filter extracts 4 interpretable latent factors from 75 noisy macro series (daily, weekly, monthly, quarterly) via Mariano–Murasawa cumulator, with varimax rotation for factor identification
- **Ensemble Recession Detection** — Weighted combination of four signals: Markov-switching model, supervised probit, CFNAI threshold, and Sahm rule
- **Estrella–Mishkin Yield-Curve Probit** — Canonical NY Fed specification (12-month-ahead recession on T10Y3M) available as a standalone baseline
- **Look-Ahead-Free Pipeline** — Expanding-window standardization, Kalman-filtered (not smoothed) factors *and* Hamilton-filtered (not Kim-smoothed) regime probabilities, ragged-edge masking with per-series publication lags, and NBER labels restricted to turning points that had actually been announced
- **Point-in-Time Feature Generation** — `walk_forward` refits the model once per as-of date and keeps only the final row, so a feature table for a downstream model contains only values that were computable at the time; `assert_point_in_time()` enforces it
- **ALFRED Vintage Support** — `DataPipeline(use_vintages=True)` fetches each series as it stood on the run date, so later revisions never reach the model
- **OLS-Calibrated GDP Nowcast** — Factor-augmented GDP prediction against actual GDPC1 growth, with the target quarter held out and a predictive (not residual) confidence interval
- **LLM Narrative Agent** — GPT-powered macro analyst that synthesizes quantitative signals with scraped Federal Reserve communications (FOMC minutes, Beige Book, speeches)
- **Interactive Streamlit Dashboard** — Full visualization suite with regime probabilities, factor dynamics, allocation weights, and one-click narrative generation
- **Regression-tested against its own failure modes** — prefix-invariance at both the pipeline and model layers, unit contracts on threshold signals, regime-orientation checks, and statistical recovery tests against simulated ground truth

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

Every figure below comes from a **walk-forward**: 434 monthly refits from 1990
to 2026, each fitted only on data available at that date and contributing only
its own final row. No number here is in-sample.

The sample covers **four** recessions (1990-91, 2001, 2007-09, 2020) and 36
recession months out of 434.

| Signal | AUC | Brier |
|--------|-----|-------|
| CFNAI signal alone | **0.947** | **0.0521** |
| **Ensemble** (current weights) | 0.944 | 0.0544 |
| Probit | 0.919 | 0.0855 |
| Sahm rule | 0.841 | 0.1432 |
| Markov-switching (RSM) | 0.714 | 0.3937 |
| *Constant forecast at the 8.3% base rate* | 0.500 | 0.0761 |

### What this does and does not support

**The ensemble still does not beat CFNAI alone.** 0.944 / 0.0544 against
0.947 / 0.0521. The Chicago Fed publishes that index for free. On this
evidence the DFM, Kalman filter, EM and Markov-switching machinery is not
earning its keep as a *recession detector* — its value is as a factor
extractor and as a second, partly-independent read (see below).

**It detects; it does not forecast.** AUC against a recession *h* months ahead:

| Horizon | 0m | 3m | 6m | 9m | 12m |
|---------|----|----|----|----|-----|
| AUC | 0.944 | 0.916 | 0.820 | 0.742 | 0.597 |

At a one-year horizon it is close to a coin flip. For any use that needs
warning rather than confirmation, this is the number that matters.

**Per-episode detection is uneven, and COVID was missed outright:**

| Recession | Months | Detected | First signal | Peak |
|-----------|--------|----------|--------------|------|
| 1990-07 – 1991-02 | 8 | 3/8 | +5 months | 0.72 |
| 2001-03 – 2001-10 | 8 | 8/8 | +0 months | 0.73 |
| 2007-12 – 2009-05 | 18 | 12/18 | +4 months | 1.00 |
| 2020-02 – 2020-03 | 2 | **0/2** | fired 2020-04, *after it ended* | 0.10 |

False alarms are rare — 11 of 398 expansion months (2.8%) — so the model is
conservative rather than trigger-happy. But a typical detection lag of 4-5
months, and a complete miss on a two-month shock, bound what this can be used
for. The COVID miss is not a bug: publication lags mean March 2020 data did not
exist in March 2020, and a monthly macro panel physically cannot resolve a
two-month exogenous shock. It is a structural limit of the approach.

### CFNAI is an input, not an independent check

CFNAI reaches the ensemble through four paths: the direct threshold signal, the
probit's feature set, the DFM panel (so it shapes the latent factors), and as a
sign anchor for `real_activity`. Removing it from **all four** and refitting
costs little, so the signal does not depend on it — but the residual
correlation is still +0.688. CFNAI is itself a factor model over 85 overlapping
indicators, so the two read the same economy. They corroborate; they cannot
independently confirm. Combining them is the best configuration measured, which
is what the shipped weights approximate.

### The RSM: fixed, improved, still not weighted

The Markov-switching signal used to report >0.99 in half of all real-time
fits, including deep expansions. The cause was the regime sort: it ranked
regimes by the *first* factor only, so when `real_activity` and `labor_market`
disagreed about which regime was weaker, the labels inverted. Fitting as of
July 2019 gave

    regime A: real_activity -0.23, labor_market +1.05   (composite +0.41)
    regime B: real_activity +0.14, labor_market -0.66   (composite -0.26)

and dim-0 ordering called A "recession" although it was the *stronger* state —
so with labour running hot the model reported P(recession) = 0.999 through a
late-cycle expansion. Ranking on the composite mean across all fitted
dimensions fixes it (that 2019 fit goes from 0.999 to 0.001), and a warning now
fires when the dimensions disagree.

The fix helped materially — AUC 0.600 → 0.714, latching 51% → 35% of months —
but the signal still does not earn a weight:

| RSM weight | Ensemble AUC | Ensemble Brier |
|------------|--------------|----------------|
| **0.00 (shipped)** | **0.944** | **0.0544** |
| 0.10 | 0.931 | 0.0609 |
| 0.20 | 0.924 | 0.0736 |

It still reports 42.5% recession probability during expansions. It stays at
0.0, now on stronger evidence; `tests/test_ensemble_weights.py` fails if the
weight is restored without re-running this evaluation.

Reproduce all of it with:

```bash
python scripts/build_features.py --step 1 --start 1990-01-31
```

---

## Is It Useful Downstream?

The reason to build this rather than read CFNAI off the Chicago Fed website
is that its output should be worth more than the free index it is partly built
from. `scripts/benchmark_features.py` tests that directly, on three-month
forward NASDAQ targets, under purged and embargoed walk-forward CV.

Two rules make it honest: features are joined on **`knowable_at`**, not the
reference month, and folds are **purged and embargoed** — on a target built
from overlapping forward windows, shuffled k-fold reports ~+0.69 correlation
where the truth is zero (`tests/test_purged_cv.py` asserts this).

Ridge, 4 folds, 434 monthly observations. IC is the mean fold correlation;
R² is measured against the *training* mean, the naive forecast actually
available at prediction time.

| Feature set | return IC | return R² | vol IC | vol R² | drawdown IC | drawdown R² |
|---|---|---|---|---|---|---|
| CFNAI only | −0.02 | −0.02 | **+0.40** | **+0.04** | **+0.26** | **+0.02** |
| p_recession only | −0.04 | −0.04 | +0.37 | +0.01 | +0.24 | +0.01 |
| regime signals (4) | +0.10 | −0.12 | +0.07 | −0.25 | +0.07 | −0.17 |
| factors only | +0.01 | −0.28 | +0.08 | −0.51 | −0.00 | −0.25 |
| full regime panel | +0.07 | −0.52 | −0.07 | −1.09 | −0.10 | −0.62 |

**Returns are not predictable here.** Every R² is negative — no configuration
beats predicting the historical average. This is consistent with the 0.597
twelve-month AUC above, and it is the expected result: a coincident recession
signal tells you about a drawdown that has largely already happened.

**Risk is modestly predictable.** Volatility and drawdown both show positive
out-of-sample R² and IC around +0.25 to +0.40. This is the use the signal
actually supports — vol targeting, drawdown control, regime-conditional
sizing — not directional forecasting.

**Fewer features win, decisively.** A single column beats the full 32-column
panel on every target, and the full panel is catastrophic for volatility
(R² −1.09). With four folds and a handful of recessions there is not enough
data to fit anything wide; the extra columns only add variance.

**CFNAI alone still edges out `p_recession`** on all three targets, though
the gap is small (vol IC +0.40 vs +0.37). Combined with the recession-detection
results above, the consistent finding across every test in this repository is
that the factor machinery does not beat the single published index it uses as
an input. Its defensible value is as an interpretable decomposition and a
second, partly-independent read — not as a better recession detector.

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
| `ragged_edge_mask` did ~38k scalar `.loc` writes per call | Dominated pipeline time in expanding backtests | Vectorised: **37× faster**, bit-identical output |

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
