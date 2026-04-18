# 📊 Macro Regime Nowcaster

> Real-time economic regime detection combining Dynamic Factor Models, Kalman filtering, Markov-switching models, and ensemble recession probability — with an LLM-powered narrative agent and interactive Streamlit dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-140%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Regime Probabilities](docs/images/regime_probabilities.png)

---

## Overview

This project implements a **real-time macroeconomic regime nowcaster** that ingests 70+ economic indicators from the Federal Reserve (FRED), extracts latent factors using a mixed-frequency state-space model, and classifies the current economic regime as **expansion** or **recession** using an ensemble of four independent signals.

### Key Features

- **Mixed-Frequency Dynamic Factor Model** — EM algorithm + Kalman filter extracts 4 interpretable latent factors from 70+ noisy macro series (daily, weekly, monthly, quarterly) via Mariano–Murasawa cumulator, with varimax rotation for factor identification
- **Ensemble Recession Detection** — Weighted combination of four signals: Markov-switching model, supervised probit, CFNAI threshold, and Sahm rule
- **Estrella–Mishkin Yield-Curve Probit** — Canonical NY Fed specification (12-month-ahead recession on T10Y3M) available as a standalone baseline
- **Look-Ahead-Free Pipeline** — Expanding-window standardization, filtered (not smoothed) factors, ragged-edge masking with per-series publication lags, and dynamic NBER cutoffs eliminate data leakage
- **ALFRED Vintage Support** — Fetch point-in-time vintages from FRED for true real-time backtesting
- **OLS-Calibrated GDP Nowcast** — Factor-augmented GDP prediction regressed against actual GDPC1 quarterly growth
- **LLM Narrative Agent** — GPT-powered macro analyst that synthesizes quantitative signals with scraped Federal Reserve communications (FOMC minutes, Beige Book, speeches)
- **Interactive Streamlit Dashboard** — Full visualization suite with regime probabilities, factor dynamics, allocation weights, and one-click narrative generation
- **140 passing tests** — Comprehensive test coverage including no-leakage, ragged-edge, and expanding-standardization regression tests

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
| **RSM** | 0.20 | Hamilton (1989) Markov-switching on multivariate DFM factors |
| **CFNAI** | 0.20 | Chicago Fed MA3 < −0.7 convention (logistic-smoothed) |
| **Sahm** | 0.20 | Sahm rule — 3-month MA unemployment minus 12-month min (logistic-smoothed) |

### Latent Factor Dynamics

Four interpretable factors extracted via PCA + EM + varimax rotation, with sign alignment against known anchor series:

![Latent Factors](docs/images/latent_factors.png)

### Historical Regime Classification

Continuous regime probability mapped against NBER recession dates:

![Regime Timeline](docs/images/regime_timeline.png)

---

## Architecture

![Architecture](docs/images/architecture.png)

```
FRED API (70+ series, daily/weekly/monthly/quarterly)
    │
    ▼
DataPipeline ── transform, mixed-freq align, expanding standardize, ragged-edge mask
    │
    ▼
DynamicFactorModel (EM + Kalman) ── 4 latent factors, Mariano–Murasawa cumulator,
    │                                 varimax rotation, filtered output (no look-ahead)
    │
    ├──► Markov-Switching RSM     (wt: 0.20)
    ├──► Recession Probit         (wt: 0.40)  ◄── NBER dates, yield curve, credit spreads
    ├──► CFNAI Signal             (wt: 0.20)
    ├──► Sahm Rule                (wt: 0.20)  ◄── unemployment rate
    │
    ▼
Ensemble P(Recession) + OLS-calibrated GDP Nowcast
    │
    ├──► RegimeAllocator ── conditional asset weights
    ├──► Streamlit Dashboard ── interactive visualization
    └──► NarrativeAgent (GPT) + FedScraper ── macro narrative
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

- Filtered probabilities via Hamilton (1989) forward recursion
- Smoothed probabilities via Kim (1994) backward smoother
- Multivariate specification uses all DFM factors jointly
- Regime labels assigned by ascending mean: index 0 = recession, index 1 = expansion
- Supports multi-regime (K > 2) with labels: recession / slowdown / recovery / expansion

### Sahm Rule

The Sahm rule indicator is:

$$\text{Sahm}_t = \overline{U}_{t}^{(3)} - \min_{s \in [t-12,\,t-1]} U_s$$

where $\overline{U}_{t}^{(3)}$ is the 3-month moving average of the unemployment rate. A value ≥ 0.50 has historically signalled every US recession since 1970.

The ensemble uses a logistic-smoothed version centred at the 0.50 threshold.

### Estrella–Mishkin Yield-Curve Probit

The canonical NY Fed recession indicator (Estrella & Mishkin, 1998):

$$P(\text{Recession}_{t+12}) = \Phi(\alpha + \beta \cdot \text{Spread}_t)$$

where $\text{Spread}_t$ = T10Y3M. Available as a standalone baseline via `RecessionProbit.estrella_mishkin()`.

### GDP Nowcast

OLS regression of actual quarterly GDPC1 growth against DFM factors:

$$\text{GDP}_q = \alpha + \boldsymbol{\beta}' \mathbf{f}_q + \varepsilon_q$$

Calibrated in-sample, with 90% confidence intervals from the residual standard error.

### Look-Ahead Safeguards

| Safeguard | Implementation |
|-----------|---------------|
| Expanding-window standardization | z-scores at time $t$ use only data through $t$ |
| Filtered factors | DFM defaults to `filtered_factors_` (no future data in Kalman pass) |
| Ragged-edge masking | Per-series `publication_lag_days` from YAML config masks stale observations |
| Dynamic NBER cutoff | Probit training end-date derived from NBER recession table, not hard-coded |
| No forward fill + back fill | Stale data is NaN, never silently carried forward |
| Expanding backtest | `run_expanding()` is the default backtest path |

---

## Project Structure

```
macro-regime-nowcaster/
├── config/
│   ├── settings.yaml              # Model hyperparameters, allocation weights
│   └── fred_series.yaml           # 70+ FRED series with transforms, lags & frequencies
├── src/
│   ├── data/
│   │   ├── fred_client.py         # FREDClient with ALFRED vintage support (as_of)
│   │   ├── data_pipeline.py       # DataPipeline — transform, standardize, ragged-edge
│   │   ├── mixed_frequency.py     # Mariano–Murasawa cumulator & frequency alignment
│   │   └── ...                    # transforms, storage
│   ├── models/
│   │   ├── kalman_filter.py       # Kalman filter & smoother
│   │   ├── dynamic_factor_model.py # PCA + EM + varimax DFM (mixed-freq cumulator)
│   │   ├── regime_switching.py    # Hamilton Markov-switching (K≥2 regimes)
│   │   ├── recession_probit.py    # Supervised probit + Estrella-Mishkin classmethod
│   │   ├── sahm_rule.py           # Sahm recession indicator & logistic-smoothed signal
│   │   ├── regime_backtest.py     # NBER backtesting (expanding-window default)
│   │   └── nowcaster.py           # End-to-end 4-signal ensemble orchestrator
│   ├── allocation/                # RegimeAllocator, Backtester
│   ├── agent/                     # NarrativeAgent (GPT), FedScraper, prompts
│   ├── dashboard/                 # Streamlit app (8 panels)
│   └── utils/                     # Logging, date utilities
├── tests/                         # 140 pytest tests (19 test files)
├── notebooks/                     # 5 Jupyter notebooks (EDA → full pipeline)
├── scripts/                       # CLI: fetch, train, nowcast, backtest, validation
├── docs/images/                   # README visualizations
├── pyproject.toml
├── requirements.txt
└── Makefile
```

---

## Data Sources

| Category | Example Series | Count |
|----------|---------------|-------|
| Labor Market | PAYEMS, UNRATE, ICSA, JTSJOL, U6RATE, IC4WSA, TEMPHELPS, AWHAETP | ~12 |
| Production & Activity | INDPRO, TCU, CFNAI, RSXFS, WEI, USSLIND, Empire/Philly Fed | ~12 |
| Financial Conditions | T10Y2Y, T10Y3M, BAA10Y, SP500, NFCI, VIXCLS, STLFSI2 | ~11 |
| Consumer & Housing | UMCSENT, PERMIT, HOUST, MORTGAGE30US, MSACSR, CSUSHPISA | ~8 |
| Prices & Inflation | CPIAUCSL, PCEPI, T5YIE, DCOILWTICO | ~8 |
| Money & Credit | M2SL, TOTBKCR, BUSLOANS, DRTSCILM, DRALACBN, CORBLACBS | ~10 |
| International | DTWEXBGS, DCOILBRENTEU, GOLDAMGBD228NLBM, VIXCLS | ~4 |
| GDP | GDPC1, GDPDEF, A261RX1Q020SBEA | ~3 |

All data sourced from the [FRED database](https://fred.stlouisfed.org/) (Federal Reserve Bank of St. Louis). Quarterly series (GDP, credit conditions) are handled via Mariano–Murasawa state augmentation. ALFRED point-in-time vintages are supported for real-time backtesting.

---

## Configuration

Key settings in `config/settings.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.n_factors` | 4 | Number of latent factors |
| `model.n_regimes` | 2 | Expansion / Recession (supports 3 or 4) |
| `ensemble.weights.probit` | 0.40 | Probit model weight (supervised, class-balanced) |
| `ensemble.weights.rsm` | 0.20 | Markov-switching weight |
| `ensemble.weights.cfnai` | 0.20 | CFNAI signal weight |
| `ensemble.weights.sahm` | 0.20 | Sahm rule weight |
| `data.start_date` | 1980-01-01 | Historical sample start |
| `data.vintage_dir` | data/vintages | ALFRED vintage storage |
| `agent.llm_model` | gpt-4o-mini | LLM for narrative generation |

---

## Running Tests

```bash
pytest tests/ -v --cov=src
# or
make test
```

140 tests across 19 test files covering: Kalman filter, DFM (including mixed-frequency cumulator and varimax rotation), regime switching, recession probit (including Estrella–Mishkin), Sahm rule, nowcaster integration, ALFRED vintage isolation, expanding standardization, no-leakage regression, ragged-edge masking, narrative agent, regime allocator, data pipeline, and utility modules.

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
