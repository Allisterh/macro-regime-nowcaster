"""Streamlit dashboard for the Macro Regime Nowcaster.

Panels:
    1. Colour-coded regime banner with key metrics
    2. Ensemble recession probability gauge + breakdown
    3. Regime probability time-series chart (stacked area, NBER shaded)
    4. Latent factor time series (4 panels)
    5. Regime timeline with NBER recession shading
    6. Asset allocation pie chart (from RegimeAllocator)
    7. LLM narrative summary card (NarrativeAgent + FedScraper)
    8. Recent regime probability data table

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path when run directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from src.utils.logging_config import setup_logging

setup_logging(level="WARNING")

# ---------------------------------------------------------------------------
# NBER recession dates for shading
# ---------------------------------------------------------------------------
NBER_RECESSIONS = [
    ("1980-01-01", "1980-07-31"),
    ("1981-07-01", "1982-11-30"),
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
    ("2020-02-01", "2020-04-30"),
]

# ---------------------------------------------------------------------------
# Real-time (walk-forward) history
# ---------------------------------------------------------------------------
# A single Nowcaster.run() fits the DFM, RSM and probit on the whole sample it
# is given, so the *history* it returns is an in-sample fit: at any past month
# it embeds parameters estimated from data that had not happened yet.  The
# latest point is real-time-safe (there, the Kalman smoother coincides with the
# filter), which is what the headline metrics use — but the historical curve is
# not what the model would have printed at the time.
#
# scripts/build_features.py refits once per as-of date and keeps only each
# fit's final row.  When that panel exists we overlay it, so the gap between
# the two lines is visible rather than implied.
#
# Only the current-format panel is read.  An earlier version of this loader
# also fell back to data/oos_validation.csv, which was produced by a
# superseded run_validation.py: annually spaced, with a probit pinned at its
# old 0.05 clip floor in 25 of 27 rows and an inverted RSM sitting at 0.98 in
# 17 of 27.  Drawn next to NBER bands it read as a plausible track record and
# invited conclusions the underlying numbers could not support.  A stale panel
# is worse than no panel, so the fallback is gone.
WALK_FORWARD_PATH = Path("data/features.csv")

# Sampling coarser than this cannot resolve a short recession — the 2020
# downturn lasted two months — so the caption says so rather than letting a
# flat line read as "the model missed it".
_COARSE_SAMPLING_DAYS = 45


def _rgba(hex_colour: str, alpha: float) -> str:
    """``#rrggbb`` to an ``rgba()`` string, for translucent CI bands."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


@st.cache_data(show_spinner="Computing horizon curve…")
def compute_horizon_curve():
    """AUC vs forecast horizon for each signal, with bootstrap intervals.

    Returns ``None`` when no walk-forward panel has been generated, in
    which case the chart is replaced by instructions rather than by a
    plot of the in-sample history — which would be the same mistake this
    dashboard already made once.
    """
    panel = WALK_FORWARD_PATH
    if not panel.exists():
        return None
    try:
        from src.evaluation import horizon_auc_curve
        from src.models.regime_backtest import get_nber_recession_indicator

        frame = pd.read_csv(panel, index_col=0, parse_dates=True).sort_index()
        labels = get_nber_recession_indicator(
            start=str(frame.index[0].date()),
            end=str(frame.index[-1].date()),
        )
        signals = {
            "Ensemble": "p_recession",
            "CFNAI": "signal_cfnai",
            "Probit": "signal_probit",
            "Sahm": "signal_sahm",
        }
        signals = {k: v for k, v in signals.items() if v in frame.columns}
        if not signals:
            return None
        return horizon_auc_curve(frame, labels, signals, n_boot=1000)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not compute the horizon curve: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_walk_forward() -> dict | None:
    """Load the precomputed point-in-time panel, if one has been generated.

    Returns ``None`` when no panel is on disk — the dashboard then shows
    only the in-sample curve, clearly labelled as such.

    The returned dict carries sampling metadata alongside the series so the
    chart can state its resolution and date range instead of leaving the
    reader to infer them from the line.
    """
    path = WALK_FORWARD_PATH
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not read walk-forward panel {path}: {exc}")
        return None

    if "p_recession" not in frame.columns:
        logger.warning(f"{path} has no p_recession column; ignoring")
        return None

    series = pd.to_numeric(frame["p_recession"], errors="coerce").dropna()
    if len(series) < 2:
        return None

    spacing_days = float(series.index.to_series().diff().dt.days.median())
    return {
        "series": series,
        "source": str(path),
        "n": len(series),
        "start": series.index[0],
        "end": series.index[-1],
        "spacing_days": spacing_days,
        "coarse": spacing_days > _COARSE_SAMPLING_DAYS,
    }

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Macro Regime Nowcaster",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
REGIME_COLORS = {
    "expansion": "#2ecc71",
    "recession": "#e74c3c",
}

ASSET_COLORS = {
    "equities": "#2ecc71",
    "bonds": "#3498db",
    "commodities": "#f39c12",
    "cash": "#bdc3c7",
}

SIGNAL_COLORS = {
    "rsm": "#9b59b6",
    "probit": "#3498db",
    "cfnai": "#e67e22",
    "ensemble": "#e74c3c",
}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    start_date = st.date_input("Start Date", value=pd.Timestamp("2000-01-01"))
    end_date = st.date_input("End Date", value=pd.Timestamp.today())
    # Default from the model, not a literal. This slider read value=4
    # after the validated default moved to 5, so the dashboard was
    # silently running a configuration nobody had measured — and at 4 it
    # also truncates DEFAULT_FACTOR_NAMES, dropping long_rates.
    from src.models.nowcaster import Nowcaster as _NowcasterDefaults

    _default_k = len(_NowcasterDefaults.DEFAULT_FACTOR_NAMES)
    n_factors = st.slider(
        "Latent Factors", min_value=1, max_value=8, value=_default_k,
        help=(
            f"{_default_k} is the validated default: the smallest number at "
            f"which every factor has a name the loadings support."
        ),
    )
    if n_factors != _default_k:
        st.warning(
            f"Running with {n_factors} factors instead of the validated "
            f"{_default_k}. Below {_default_k} the factor names are "
            f"truncated and one of the rates factors is dropped; above it, "
            f"the extra factors are unnamed. Measured performance in the "
            f"README does not apply.",
            icon="⚠️",
        )
    run_button = st.button("🔄 Run Nowcast", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown(
        "DFM → RSM + Probit + CFNAI + Sahm → Ensemble → Allocation"
    )
    # Read the weights from the model rather than restating them here.
    # Hard-coding is how this panel came to advertise a three-signal
    # 0.25 / 0.50 / 0.25 split long after the shipped ensemble moved to
    # four signals at 0.20 / 0.40 / 0.20 / 0.20.
    try:
        from src.models.nowcaster import Nowcaster as _NowcasterWeights

        st.markdown(
            "\n".join(
                f"- **{name.upper()} weight:** {weight:.2f}"
                for name, weight in _NowcasterWeights.DEFAULT_WEIGHTS.items()
            )
        )
    except Exception:  # noqa: BLE001
        logger.warning("Could not read ensemble weights for the sidebar")

# ---------------------------------------------------------------------------
# Main title
# ---------------------------------------------------------------------------
st.title("📊 Macro Regime Nowcaster")
st.caption(
    "Real-time economic regime detection · Dynamic Factor Model · "
    "Markov-Switching · Ensemble Recession Probability"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_nber_shading(fig: go.Figure, x_min=None, x_max=None) -> None:
    """Add NBER recession shading rectangles to a Plotly figure."""
    for start, end in NBER_RECESSIONS:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if x_max and s > pd.Timestamp(x_max):
            continue
        if x_min and e < pd.Timestamp(x_min):
            continue
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor="rgba(200,200,200,0.25)",
            layer="below",
            line_width=0,
            annotation_text="",
        )


def _show_instructions() -> None:
    st.info(
        """
        **No nowcast data available.**

        To get started:
        1. Copy `.env.example` → `.env` and add your `FRED_API_KEY`
        2. Run `make fetch-data` to download FRED data
        3. Click **🔄 Run Nowcast** in the sidebar

        The pipeline will fetch data from FRED, fit the Dynamic Factor Model,
        run ensemble recession detection, and display results here.
        """,
        icon="ℹ️",
    )


@st.cache_data(ttl=3600, show_spinner="Running nowcast pipeline…")
def _run_nowcast(start: str, end: str, n_fac: int):
    """Cached nowcast execution.  Returns all artefacts needed for display."""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY", "")
    if not api_key:
        return None, "FRED_API_KEY not set — add it to your .env file."

    try:
        from src.data.data_pipeline import DataPipeline
        from src.data.fred_client import FREDClient
        from src.models.nowcaster import Nowcaster

        client = FREDClient(api_key=api_key)
        pipeline = DataPipeline(fred_client=client, start_date=start)
        nowcaster = Nowcaster(
            pipeline=pipeline,
            n_factors=n_fac,
            n_regimes=2,
            use_ensemble=True,
        )
        result = nowcaster.run(end_date=end)
        factors = nowcaster._last_factors
        regime_probs = nowcaster.get_ensemble_probabilities()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return {
            "result": result,
            "factors": factors,
            "regime_probs": regime_probs,
            "timestamp": timestamp,
            # Reference date behind each signal. The panel edge is ragged
            # by design, so a reading is typically one or two months old.
            "signal_as_of": dict(getattr(nowcaster, "_signal_as_of", {}) or {}),
            "weights": dict(nowcaster.ensemble_weights),
            # How well the loadings support each factor name.
            "factor_quality": dict(
                getattr(nowcaster._dfm, "_factor_match_quality", {}) or {}
            ),
        }, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Nowcast run failed")
        return None, str(exc)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
data = None
error_msg = None

if run_button:
    data, error_msg = _run_nowcast(str(start_date), str(end_date), n_factors)
    if data is not None:
        st.session_state["nowcast_data"] = data
    if error_msg:
        st.session_state.pop("nowcast_data", None)

# Restore from session state so data survives button reruns
if data is None and "nowcast_data" in st.session_state:
    data = st.session_state["nowcast_data"]

if error_msg:
    st.error(f"❌ {error_msg}")
elif data is None:
    _show_instructions()
else:
    result = data["result"]
    factors: pd.DataFrame = data["factors"]
    regime_probs: pd.DataFrame = data["regime_probs"]
    timestamp: str = data["timestamp"]

    # ==================================================================
    # 1. Regime banner
    # ==================================================================
    regime = result.current_regime
    p_recession = result.recession_probability
    color = REGIME_COLORS.get(regime, "#95a5a6")

    st.markdown(
        f"""
        <div style="background-color:{color}; padding:24px; border-radius:12px;
                    text-align:center; margin-bottom:16px;">
            <h1 style="color:white; margin:0; font-size:2.2em;">
                {regime.upper()}
            </h1>
            <p style="color:rgba(255,255,255,0.9); margin:6px 0 0 0; font-size:1.15em;">
                GDP Nowcast: {result.gdp_nowcast:.2f}% ann.
                &nbsp;|&nbsp;
                Recession Probability: {p_recession:.1%}
                &nbsp;|&nbsp;
                {timestamp}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==================================================================
    # 2. Key metrics row
    # ==================================================================
    m1, m2, m3, m4 = st.columns(4)

    # Streamlit renders a metric's delta with an arrow and a colour, which
    # reads as a *change* and as good news. A confidence interval is
    # neither, so it goes in the label and the arrow is turned off.
    ci_width = result.gdp_ci_upper - result.gdp_ci_lower
    m1.metric(
        "GDP Nowcast",
        f"{result.gdp_nowcast:.2f}%",
        f"90% CI [{result.gdp_ci_lower:.1f}%, {result.gdp_ci_upper:.1f}%]",
        delta_color="off",
    )
    m2.metric("Recession Prob", f"{p_recession:.1%}")
    m3.metric("Current Regime", regime.title())

    # "N active" counted len(ensemble_detail), which includes the
    # "ensemble" entry itself — so four signals were reported as five —
    # and counted components carrying zero weight as though they were
    # contributing.
    weights = data.get("weights", {})
    signal_keys = [k for k in result.ensemble_detail if k != "ensemble"]
    weighted = [k for k in signal_keys if weights.get(k, 0.0) > 0]
    m4.metric(
        "Signals Weighted",
        f"{len(weighted)} of {len(signal_keys)}",
        ", ".join(sorted(weighted)) or "none",
        delta_color="off",
    )

    if ci_width > 8.0:
        st.caption(
            f":warning: The GDP interval spans {ci_width:.0f} percentage "
            f"points. The factors explain little quarter-to-quarter GDP "
            f"variation — the residual standard error is inflated by the "
            f"2020 quarters — so treat the point estimate as weakly "
            f"identified rather than as a forecast."
        )

    st.markdown("---")

    # ==================================================================
    # 3. Ensemble breakdown + Asset allocation (side by side)
    # ==================================================================
    col_ens, col_alloc = st.columns([3, 2])

    with col_ens:
        st.subheader("Ensemble Signal Breakdown")
        detail = result.ensemble_detail
        if detail:
            weights = data.get("weights", {})
            as_of = data.get("signal_as_of", {})
            signal_names = []
            signal_vals = []
            signal_colors = []
            # Sahm was missing from this chart entirely even though it is
            # part of the ensemble; a component with weight 0 still belongs
            # here, labelled, rather than silently absent.
            labels = {
                "rsm": "RSM (Markov)", "probit": "Probit",
                "cfnai": "CFNAI", "sahm": "Sahm",
            }
            for key in ["rsm", "probit", "cfnai", "sahm"]:
                if key in detail:
                    weight = weights.get(key)
                    suffix = f"  (w={weight:.2f})" if weight is not None else ""
                    signal_names.append(labels[key] + suffix)
                    signal_vals.append(detail[key])
                    signal_colors.append(SIGNAL_COLORS.get(key, "#95a5a6"))

            # Add ensemble bar
            if "ensemble" in detail:
                signal_names.append("Ensemble")
                signal_vals.append(detail["ensemble"])
                signal_colors.append(SIGNAL_COLORS["ensemble"])

            fig_ens = go.Figure()
            fig_ens.add_trace(go.Bar(
                x=signal_vals,
                y=signal_names,
                orientation="h",
                marker_color=signal_colors,
                text=[f"{v:.1%}" for v in signal_vals],
                textposition="auto",
            ))
            fig_ens.add_vline(x=0.5, line_dash="dash", line_color="grey",
                              annotation_text="50% threshold")
            fig_ens.update_layout(
                xaxis=dict(range=[0, 1], title="P(Recession)", tickformat=".0%"),
                yaxis=dict(autorange="reversed"),
                height=250,
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig_ens, use_container_width=True)

            # Say how old each reading is. Publication lags mask the last
            # month or two of every series, so a signal is normally one or
            # two months stale — and a bar labelled 50.0% is usually a
            # signal that could not be computed, not a genuine coin flip.
            stamps = [
                f"{labels[k]} {as_of[k]:%b %Y}"
                for k in ["rsm", "probit", "cfnai", "sahm"]
                if as_of.get(k) is not None
            ]
            if stamps:
                st.caption(
                    "Latest published reading behind each signal — the panel "
                    "edge is ragged by design, so these lag the current "
                    "month: " + " · ".join(stamps) + ". "
                    "Signals weighted 0.00 are computed and shown but do not "
                    "enter the ensemble; see the Architecture panel for why."
                )
        else:
            st.info("Ensemble detail not available")

    with col_alloc:
        st.subheader("Portfolio Allocation")
        from src.allocation.regime_allocator import RegimeAllocator
        allocator = RegimeAllocator()
        allocation = allocator.get_allocation_from_nowcast(result)

        fig_pie = px.pie(
            names=list(allocation.keys()),
            values=list(allocation.values()),
            color=list(allocation.keys()),
            color_discrete_map=ASSET_COLORS,
        )
        fig_pie.update_traces(
            textinfo="label+percent",
            textposition="inside",
        )
        fig_pie.update_layout(
            height=250,
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        # Show allocation table
        alloc_df = pd.DataFrame(
            {"Weight": [f"{v:.1%}" for v in allocation.values()]},
            index=[k.title() for k in allocation.keys()],
        )
        st.dataframe(alloc_df, use_container_width=True)

    st.markdown("---")

    # ==================================================================
    # 4. Regime probability time series with NBER shading
    # ==================================================================
    st.subheader("Regime Probabilities Over Time")

    if isinstance(regime_probs, pd.DataFrame) and len(regime_probs) > 0:
        fig_probs = go.Figure()

        for col_name in regime_probs.columns:
            fig_probs.add_trace(
                go.Scatter(
                    x=regime_probs.index,
                    y=regime_probs[col_name],
                    name=col_name.title(),
                    mode="lines",
                    stackgroup="one",
                    fillcolor=REGIME_COLORS.get(col_name, "#95a5a6"),
                    line=dict(width=0.5, color=REGIME_COLORS.get(col_name, "#95a5a6")),
                )
            )

        _add_nber_shading(fig_probs)

        fig_probs.update_layout(
            yaxis=dict(title="Probability", range=[0, 1], tickformat=".0%"),
            xaxis=dict(
                title="Date",
                range=[regime_probs.index.min(), regime_probs.index.max()],
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=350,
            margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig_probs, use_container_width=True)
        st.caption(
            "In-sample: the model is fitted on the full selected window, so past "
            "months are scored with parameters estimated partly from later data. "
            "The most recent point is real-time-safe; earlier points are not a "
            "track record."
        )
    else:
        st.warning("No regime probabilities to display")

    # ==================================================================
    # 5. Latent factor time series
    # ==================================================================
    st.subheader("Latent Factors (DFM)")
    factor_cols = factors.columns.tolist()
    quality = data.get("factor_quality", {})

    # Every factor gets a plot. This used to cap at four
    # (min(len(factor_cols), 4)) and lay them out in a single row, so the
    # fifth factor silently vanished when the default moved to K=5 —
    # another literal restating something the model owns. Wrap instead.
    per_row = 4

    # Determine the actual data range from factor values.
    # The Kalman smoother fills early rows with near-zero values when
    # few series are available; trim those by finding the first row
    # where any factor deviates meaningfully from zero (|z| > 0.05).
    factors_plot = factors[factor_cols].copy()
    meaningful = factors_plot.abs().max(axis=1) > 0.05
    if meaningful.any():
        factors_plot = factors_plot.loc[meaningful.idxmax():]

    row_slots: list = []
    for i, col_name in enumerate(factor_cols):
        if i % per_row == 0:
            remaining = len(factor_cols) - i
            row_slots = st.columns(min(per_row, remaining))
        with row_slots[i % per_row]:
            series = factors_plot[col_name].dropna()
            # Clip to ±3σ for display only (model uses unclipped values)
            series_display = series.clip(-3.0, 3.0)
            fig_f = go.Figure()
            fig_f.add_trace(
                go.Scatter(
                    x=series_display.index,
                    y=series_display,
                    mode="lines",
                    line=dict(color="#3498db", width=1.5),
                )
            )
            # Zero line
            fig_f.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5)
            _add_nber_shading(fig_f)

            clean_name = col_name.replace("_", " ").title()
            # Show how well the loadings support the name. A factor whose
            # anchors are weak is a label of convenience, and the reader
            # should be able to see that on the plot rather than trusting
            # the title — this is exactly how a yield-curve factor came to
            # be read as "labor market".
            score = quality.get(col_name)
            if score is not None and score == score:
                mark = "✓" if score >= 1.0 else "?"
                title_text = f"{clean_name}  <sub>{mark} {score:.2f}</sub>"
            else:
                title_text = clean_name
            fig_f.update_layout(
                title=dict(text=title_text, font=dict(size=13)),
                height=200,
                margin=dict(t=35, b=30, l=10, r=10),
                xaxis=dict(
                    showticklabels=True,
                    tickformat="%Y",
                    dtick="M60",
                    tickangle=-45,
                    tickfont=dict(size=9),
                    range=[series_display.index.min(), series_display.index.max()],
                ),
                yaxis=dict(title="z-score", range=[-3.5, 3.5]),
            )
            st.plotly_chart(fig_f, use_container_width=True)

    if quality:
        weak = [n for n, v in quality.items() if v == v and v < 1.0]
        st.caption(
            "The number beside each name is how strongly that factor's "
            "anchor series load on it, relative to the factor's own "
            "90th-percentile loading — ✓ means the name is supported by "
            "the loadings, ? means it is a label of convenience."
            + (
                f" Currently weak: {', '.join(sorted(weak))}."
                if weak else " All factor names are currently evidenced."
            )
        )

    st.markdown("---")

    # ==================================================================
    # 6. Regime timeline with NBER shading
    # ==================================================================
    st.subheader("Historical Regime Classification")
    walk_forward = load_walk_forward()

    if isinstance(regime_probs, pd.DataFrame) and "recession" in regime_probs.columns:
        fig_timeline = go.Figure()

        # In-sample recession probability as a filled area
        fig_timeline.add_trace(
            go.Scatter(
                x=regime_probs.index,
                y=regime_probs["recession"],
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.3)",
                line=dict(color="#e74c3c", width=1.5),
                name="P(Recession) — in-sample",
            )
        )

        # Real-time line, when a walk-forward panel has been generated.
        # The gap between the two is the look-ahead advantage the
        # in-sample curve enjoys, and it is the most honest thing this
        # chart can show.
        if walk_forward is not None:
            wf_series = walk_forward["series"]
            fig_timeline.add_trace(
                go.Scatter(
                    x=wf_series.index,
                    y=wf_series.values,
                    mode="lines+markers",
                    line=dict(color="#2c3e50", width=1.8, dash="dot"),
                    marker=dict(size=5),
                    name="P(Recession) — real-time (walk-forward)",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
                )
            )

        # 50% threshold
        fig_timeline.add_hline(
            y=0.5, line_dash="dash", line_color="grey",
            annotation_text="50% threshold",
        )
        _add_nber_shading(fig_timeline)

        fig_timeline.update_layout(
            yaxis=dict(title="P(Recession)", range=[0, 1], tickformat=".0%"),
            xaxis=dict(
                title="Date",
                range=[regime_probs.index.min(), regime_probs.index.max()],
            ),
            height=280,
            margin=dict(l=10, r=10, t=10, b=30),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

        if walk_forward is not None:
            spacing = walk_forward["spacing_days"]
            cadence = (
                f"~{spacing / 30.4:.0f}-month" if spacing > 45 else "monthly"
            )
            st.caption(
                "**Red (in-sample)**: every past month is scored by a model fitted "
                "on the whole sample, so it embeds parameters estimated from data "
                "that had not happened yet — it is not what the model would have "
                "printed at the time. "
                f"**Dark dotted (real-time)**: one refit per date, final row only — "
                f"{walk_forward['n']} points at {cadence} spacing, "
                f"{walk_forward['start']:%b %Y} to {walk_forward['end']:%b %Y} "
                f"(`{walk_forward['source']}`). Judge historical skill from the "
                "dotted line, and only within that range."
            )
            if walk_forward["coarse"]:
                st.warning(
                    f"The real-time line is sampled every ~{spacing:.0f} days. "
                    "Recessions shorter than that can fall entirely between two "
                    "points — the 2020 downturn lasted two months — so a flat "
                    "line across one is a limit of the sampling, not evidence "
                    "the model missed it. Regenerate with "
                    "`python scripts/build_features.py --step 1` for monthly "
                    "resolution.",
                    icon="⚠️",
                )
        else:
            st.caption(
                ":warning: **This is an in-sample fit, not a track record.** Each "
                "past month is scored by a model fitted on the whole sample, "
                "including data that post-dates it, so the curve hugs the NBER "
                "bands more closely than real-time performance would. The *latest* "
                "point is real-time-safe; the history is not. "
                "Run `python scripts/build_features.py` to generate the "
                "walk-forward panel and a real-time line will be overlaid here."
            )

    st.markdown("---")

    # ==================================================================
    # 6b. Discrimination vs forecast horizon, with confidence intervals
    # ==================================================================
    st.subheader("Discrimination by Forecast Horizon")

    horizon_curve = compute_horizon_curve()
    if horizon_curve is None:
        st.caption(
            "Needs the walk-forward panel. Run "
            "`python scripts/build_features.py --step 1 --start 1990-01-31`, "
            "then this chart shows how each signal's discrimination decays "
            "as the forecast horizon lengthens."
        )
    else:
        frame = horizon_curve.to_frame()
        fig_h = go.Figure()
        palette = {
            "Ensemble": "#e74c3c", "CFNAI": "#e67e22",
            "Probit": "#3498db", "Sahm": "#9b59b6", "RSM": "#7f8c8d",
        }
        for signal in frame["signal"].unique():
            sub = frame[frame["signal"] == signal].sort_values("horizon")
            colour = palette.get(signal, "#95a5a6")
            # Interval first, so the point estimates draw on top of it.
            fig_h.add_trace(
                go.Scatter(
                    x=list(sub["horizon"]) + list(sub["horizon"])[::-1],
                    y=list(sub["hi"]) + list(sub["lo"])[::-1],
                    fill="toself",
                    fillcolor=_rgba(colour, 0.13),
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{signal} CI",
                )
            )
            fig_h.add_trace(
                go.Scatter(
                    x=sub["horizon"], y=sub["auc"],
                    mode="lines+markers",
                    line=dict(color=colour, width=2),
                    marker=dict(size=7),
                    name=signal,
                    hovertemplate=(
                        f"<b>{signal}</b><br>%{{x}} months ahead<br>"
                        "AUC %{y:.3f}<extra></extra>"
                    ),
                )
            )
        # 0.5 is the no-skill line: below it, the signal is worse than a
        # coin flip at that horizon.
        fig_h.add_hline(
            y=0.5, line_dash="dash", line_color="grey",
            annotation_text="no skill",
        )
        fig_h.update_layout(
            xaxis=dict(title="Forecast horizon (months ahead)"),
            yaxis=dict(title="AUC", range=[0.1, 1.0]),
            height=380,
            margin=dict(l=10, r=10, t=10, b=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
            hovermode="x unified",
        )
        st.plotly_chart(fig_h, use_container_width=True)

        n_pos = int(frame["n_positive"].max())
        st.caption(
            "**The orderings reverse across the horizon.** CFNAI is a "
            "*coincident* index: close to unbeatable at 0 months and below "
            "the no-skill line by 18. The probit carries the yield-curve "
            "and credit-spread features, so it is weaker at 0 and stronger "
            "further out. "
            "**Read the bands, not the lines.** They are 95% block-bootstrap "
            f"intervals resampled over contiguous label runs, on a sample "
            f"with {n_pos} recession months in a handful of episodes — they "
            "overlap almost everywhere, so the orderings are suggestive "
            "rather than established."
        )

    st.markdown("---")

    # ==================================================================
    # 7. LLM narrative card (with FedScraper)
    # ==================================================================
    st.subheader("📝 Narrative Analysis")
    with st.expander("Generate LLM Narrative", expanded=False):
        st.markdown(
            "Generates a macro narrative by combining the quantitative nowcast with "
            "recent Federal Reserve communications (FOMC minutes, statements, Beige Book)."
        )
        fetch_fed = st.checkbox("Fetch latest Fed documents", value=True)
        gen_button = st.button("Generate Narrative", type="secondary")

        if gen_button:
            fed_docs = []
            if fetch_fed:
                with st.spinner("Scraping Federal Reserve communications…"):
                    try:
                        from src.agent.fed_scraper import FedScraper
                        scraper = FedScraper(request_delay=1.0)
                        fed_docs = scraper.fetch_all_recent(n_each=2)
                        st.success(f"Fetched {len(fed_docs)} Fed documents")
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Fed scraping failed: {exc}")

            with st.spinner("Generating narrative…"):
                try:
                    from src.agent.narrative_agent import NarrativeAgent
                    agent = NarrativeAgent()
                    report = agent.generate(
                        nowcast_result=result.to_dict(),
                        fed_documents=fed_docs if fed_docs else None,
                    )

                    # If parsing extracted structured fields, show them;
                    # otherwise fall back to rendering the raw LLM markdown.
                    if report.key_drivers or report.risk_flags:
                        st.markdown(f"### Summary\n{report.summary}")
                        st.markdown("### Key Drivers")
                        for driver in report.key_drivers:
                            st.markdown(f"- {driver}")
                        st.markdown("### Risk Flags")
                        for flag in report.risk_flags:
                            st.markdown(f"- {flag}")
                        st.markdown(
                            f"**Fed Alignment:** {report.corroboration_score.title()}"
                        )
                    else:
                        # Parsing didn't extract bullets — show full response
                        st.markdown(report.raw_response)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Narrative generation failed: {exc}")

    # ==================================================================
    # 8. Data tables
    # ==================================================================
    st.subheader("📋 Recent Data")
    tab1, tab2, tab3 = st.tabs(["Regime Probabilities", "Factor Values", "Allocation Weights"])

    with tab1:
        if isinstance(regime_probs, pd.DataFrame) and len(regime_probs) > 0:
            st.dataframe(
                regime_probs.tail(24).style.format("{:.1%}"),
                use_container_width=True,
            )
        else:
            st.info("No regime probabilities to display")

    with tab2:
        if isinstance(factors, pd.DataFrame) and len(factors) > 0:
            st.dataframe(
                factors.tail(24).style.format("{:.3f}"),
                use_container_width=True,
            )
        else:
            st.info("No factor data to display")

    with tab3:
        # Build allocation time series
        if isinstance(regime_probs, pd.DataFrame) and len(regime_probs) > 0:
            alloc_ts = allocator.get_allocation_dataframe(regime_probs.tail(24))
            st.dataframe(
                alloc_ts.style.format("{:.1%}"),
                use_container_width=True,
            )

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#7f8c8d; font-size:0.85em;'>"
        f"Macro Regime Nowcaster · Last updated: {timestamp} · "
        f"2-regime ensemble (RSM + Probit + CFNAI)"
        f"</div>",
        unsafe_allow_html=True,
    )
