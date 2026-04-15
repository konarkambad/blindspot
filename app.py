"""
app.py — Blind Spot: Data Integrity Auditor

Main Streamlit entry point. Handles layout, upload, preview, audit execution,
and results rendering. All analysis modules are wired here.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import duckdb

from utils.loader       import load_data
from utils.profiler     import profile_data
from utils.paradox      import scan_simpsons_paradox
from utils.survivorship import detect_survivorship_bias
from utils.confounders  import detect_confounders
from utils.metric_risk  import score_metric_gaming_risk
from utils.confidence   import compute_confidence_scores
from utils.blind_spots  import generate_blind_spots
from utils.report       import generate_html_report
from utils.sample_data  import generate_demo_dataset


# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Blind Spot — Data Integrity Auditor",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  /* ── Base ── */
  [data-testid="stAppViewContainer"] { background:#0e1117; color:#e0e0e0; }
  [data-testid="stSidebar"]          { background:#161b27; border-right:1px solid #2a2f3e; }
  [data-testid="stSidebar"] *        { color:#c9d1e0 !important; }

  /* ── Shrink Streamlit's default block padding ── */
  .block-container { padding-top:1.2rem !important; padding-bottom:1rem !important; max-width:100% !important; }

  /* ── Hide default Streamlit chrome ── */
  #MainMenu, footer { visibility:hidden; }
  header { background:transparent !important; }

  /* ── Tighten default Streamlit element spacing ── */
  div[data-testid="stVerticalBlock"] > div { gap:.4rem !important; }
  div[data-testid="stHorizontalBlock"] { gap:.6rem !important; }
  [data-testid="stMarkdownContainer"] p { margin-bottom:.25rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] .block-container { padding-top:.6rem !important; padding-bottom:.5rem !important; }
  .sidebar-logo {
      display:flex; align-items:center; gap:.6rem;
      padding:.55rem .2rem .55rem .2rem;
      border-bottom:1px solid #2a2f3e; margin-bottom:.7rem;
  }
  .sidebar-logo .app-name    { font-size:1.05rem; font-weight:800; letter-spacing:.04em; color:#fff !important; }
  .sidebar-logo .app-tagline { font-size:.65rem; color:#7a8599 !important; text-transform:uppercase; letter-spacing:.08em; line-height:1.3; }
  .sidebar-section-label { font-size:.65rem; text-transform:uppercase; letter-spacing:.12em; color:#4a5568 !important; margin:.7rem 0 .25rem 0; }

  /* ── Buttons ── */
  div[data-testid="stButton"] > button {
      width:100%; background:linear-gradient(135deg,#6c63ff 0%,#48b0f7 100%);
      color:#fff; font-weight:700; font-size:.88rem; letter-spacing:.04em;
      border:none; border-radius:8px; padding:.5rem 0; margin-top:.3rem; transition:opacity .2s;
  }
  div[data-testid="stButton"] > button:hover    { opacity:.88; }
  div[data-testid="stButton"] > button:disabled { background:#2a2f3e; color:#4a5568; cursor:not-allowed; }

  /* ── Streamlit tab strip ── */
  [data-testid="stTabs"] [role="tablist"] { gap:.2rem; border-bottom:1px solid #2a2f3e; }
  [data-testid="stTabs"] [role="tab"]     { font-size:.82rem; padding:.35rem .75rem; color:#7a8599; border-radius:6px 6px 0 0; }
  [data-testid="stTabs"] [aria-selected="true"] { color:#e0e0e0 !important; border-bottom:2px solid #6c63ff !important; }

  /* ── Hero (compact) ── */
  .hero { display:flex; align-items:center; gap:1.2rem; padding:.8rem 0 .6rem 0; }
  .hero-eye   { font-size:2rem; line-height:1; flex-shrink:0; }
  .hero-title {
      font-size:1.6rem; font-weight:900;
      background:linear-gradient(135deg,#6c63ff,#48b0f7);
      -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      background-clip:text; margin:0;
  }
  .hero-subtitle { font-size:.85rem; color:#7a8599; margin:.1rem 0 0 0; line-height:1.4; }

  /* ── Feature cards ── */
  .cards-grid {
      display:grid; grid-template-columns:repeat(3,1fr);
      gap:.7rem; margin:.8rem 0 .6rem 0;
  }
  .feature-card { background:#161b27; border:1px solid #2a2f3e; border-radius:10px; padding:.75rem .9rem; transition:border-color .2s; }
  .feature-card:hover { border-color:#6c63ff; }
  .card-icon  { font-size:1.1rem; margin-bottom:.2rem; }
  .card-title { font-size:.85rem; font-weight:700; color:#e0e0e0; margin-bottom:.2rem; }
  .card-desc  { font-size:.78rem; color:#7a8599; line-height:1.45; }

  /* ── Upload prompt ── */
  .upload-prompt { text-align:center; padding:.9rem 1rem .6rem 1rem; color:#4a5568; font-size:.88rem; }
  .upload-prompt .arrow { font-size:1.3rem; margin-bottom:.2rem; }

  /* ── Preview ── */
  .preview-header  { display:flex; align-items:center; gap:.6rem; margin-bottom:.7rem; }
  .preview-title   { font-size:1.1rem; font-weight:800; color:#e0e0e0; }
  .preview-badge   { background:#1e2535; border:1px solid #2a2f3e; border-radius:20px; padding:.15rem .6rem; font-size:.75rem; color:#7a8599; }

  /* ── Stat chips ── */
  .stat-row  { display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.8rem; }
  .stat-chip { background:#161b27; border:1px solid #2a2f3e; border-radius:8px; padding:.4rem .75rem; text-align:center; min-width:100px; flex:1; }
  .stat-value { font-size:1.2rem; font-weight:800; color:#6c63ff; }
  .stat-label { font-size:.68rem; color:#7a8599; text-transform:uppercase; letter-spacing:.07em; }

  /* ── Column table ── */
  .col-table { width:100%; border-collapse:collapse; font-size:.82rem; }
  .col-table th { text-align:left; padding:.4rem .65rem; border-bottom:1px solid #2a2f3e; color:#4a5568; text-transform:uppercase; font-size:.68rem; letter-spacing:.09em; font-weight:600; }
  .col-table td { padding:.35rem .65rem; border-bottom:1px solid #1e2535; color:#c9d1e0; vertical-align:middle; }
  .col-table tr:hover td { background:#1a1f2e; }
  .dtype-badge    { display:inline-block; padding:.08rem .45rem; border-radius:4px; font-size:.7rem; font-weight:600; }
  .dtype-numeric  { background:#1a2744; color:#48b0f7; }
  .dtype-category { background:#1f1a44; color:#a78bfa; }
  .dtype-datetime { background:#1a3030; color:#34d399; }
  .dtype-boolean  { background:#2d1f1f; color:#f87171; }
  .dtype-other    { background:#1e2535; color:#7a8599; }

  /* ── Score card ── */
  .score-big  { font-size:5rem; font-weight:900; line-height:1; }
  .grade-pill { display:inline-block; font-size:1.2rem; font-weight:800; padding:.15rem 1rem; border-radius:30px; margin-top:.4rem; }

  /* ── Risk card ── */
  .risk-card { background:#161b27; border-left:4px solid #f87171; border-radius:0 8px 8px 0; padding:.6rem .9rem; margin-bottom:.4rem; font-size:.85rem; color:#c9d1e0; line-height:1.45; }

  /* ── Risk level badges ── */
  .badge       { display:inline-block; padding:.12rem .5rem; border-radius:4px; font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
  .badge-high  { background:#3d1515; color:#f87171; }
  .badge-medium{ background:#3d2d0a; color:#fbbf24; }
  .badge-low   { background:#0a2d1a; color:#34d399; }

  /* ── Finding card ── */
  .finding-card { background:#161b27; border:1px solid #2a2f3e; border-radius:10px; padding:.75rem .9rem; margin-bottom:.5rem; }
  .finding-title { font-size:.88rem; font-weight:700; color:#e0e0e0; margin-bottom:.2rem; }
  .finding-desc  { font-size:.81rem; color:#7a8599; line-height:1.5; }

  /* ── Recommendation ── */
  .recommendation-box { background:#161b27; border:1px solid #2a2f3e; border-radius:10px; padding:.85rem 1.1rem; font-size:.86rem; color:#c9d1e0; line-height:1.6; }

  /* ── Blind spot cards ── */
  .bs-card { border-radius:10px; padding:.7rem .9rem .65rem .9rem; margin-bottom:.5rem; border-left:4px solid; position:relative; }
  .bs-card-high   { background:#1e1010; border-color:#f87171; }
  .bs-card-medium { background:#1c1a0e; border-color:#fbbf24; }
  .bs-card-low    { background:#0e1c14; border-color:#34d399; }
  .bs-impact { display:inline-block; font-size:.65rem; font-weight:800; text-transform:uppercase; letter-spacing:.1em; padding:.08rem .45rem; border-radius:4px; margin-bottom:.35rem; }
  .bs-impact-high   { background:#3d1515; color:#f87171; }
  .bs-impact-medium { background:#3d2d0a; color:#fbbf24; }
  .bs-impact-low    { background:#0a2d1a; color:#34d399; }
  .bs-title   { font-size:.88rem; font-weight:700; color:#e0e0e0; margin-bottom:.25rem; line-height:1.4; }
  .bs-missing { font-size:.72rem; color:#7a8599; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.35rem; }
  .bs-source  { font-size:.8rem; color:#94a3b8; line-height:1.5; border-top:1px solid #2a2f3e; padding-top:.35rem; margin-top:.3rem; }
  .bs-source::before { content:"💡 "; }

  /* ── Misc ── */
  [data-testid="stDataFrame"] { border:1px solid #2a2f3e; border-radius:8px; overflow:hidden; }
  [data-testid="stExpander"]  { border:1px solid #2a2f3e !important; border-radius:8px; background:#161b27; }
  hr { border-color:#2a2f3e !important; margin:.4rem 0 !important; }
  section[data-testid="stSidebar"] > div { padding-top:.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

defaults = {
    "df": None, "metadata": None, "audit_results": None, "file_name": None,
    "max_combinations": 500, "confidence_threshold": 70,
    "enable_paradox": True, "enable_survivorship": True,
    "enable_confounders": True, "enable_gaming": True,
    "_auto_run_audit": False,
    "_demo_active": False,   # True while demo dataset is the active source
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
      <span style="font-size:1.4rem;line-height:1;">👁</span>
      <div>
        <div class="app-name">BLIND SPOT</div>
        <div class="app-tagline">Data Integrity Auditor</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        help="Accepts .csv, .xlsx, .xls up to 200 MB",
    )

    if uploaded_file is not None:
        # clear demo mode
        if st.session_state._demo_active:
            st.session_state._demo_active = False

        if uploaded_file.name != st.session_state.file_name:
            with st.spinner("Loading file…"):
                result = load_data(uploaded_file)
            if isinstance(result[0], pd.DataFrame):
                st.session_state.df        = result[0]
                st.session_state.metadata  = result[1]
                st.session_state.file_name = uploaded_file.name
                st.session_state.audit_results = None
                for warn_msg in (result[1] or {}).get("warnings", []):
                    st.warning(warn_msg)
            else:
                st.error(f"Could not load file: {result[1]}")
                st.session_state.df = None
    else:
        # preserve demo state
        if not st.session_state._demo_active and st.session_state.file_name is not None:
            st.session_state.df            = None
            st.session_state.metadata      = None
            st.session_state.file_name     = None
            st.session_state.audit_results = None

    # demo button
    st.markdown(
        '<div style="text-align:center;margin:.4rem 0 .2rem 0;">'
        '<span style="font-size:.72rem;color:#4a5568;letter-spacing:.08em;">OR</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Try Demo Dataset",
        use_container_width=True,
        help="Loads a synthetic 1 000-row dataset engineered to trigger every analysis module.",
    ):
        with st.spinner("Generating demo dataset…"):
            _demo_df, _demo_meta = generate_demo_dataset()
        st.session_state.df              = _demo_df
        st.session_state.metadata        = _demo_meta
        st.session_state.file_name       = "demo_dataset.csv"
        st.session_state.audit_results   = None
        st.session_state._demo_active    = True
        st.session_state._auto_run_audit = True
        st.rerun()

    st.markdown('<div class="sidebar-section-label">Analysis</div>', unsafe_allow_html=True)
    run_audit = st.button(
        "Run Full Audit",
        disabled=(st.session_state.df is None),
        use_container_width=True,
    )

    # download report button
    if st.session_state.get("audit_results") and st.session_state.get("df") is not None:
        st.markdown('<div class="sidebar-section-label">Export</div>', unsafe_allow_html=True)
        try:
            html_report = generate_html_report(
                df=st.session_state.df,
                metadata=st.session_state.metadata or {},
                results=st.session_state.audit_results,
                file_name=st.session_state.file_name or "dataset",
            )
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            stem = (st.session_state.file_name or "dataset").rsplit(".", 1)[0]
            st.download_button(
                label="Download Audit Report",
                data=html_report.encode("utf-8"),
                file_name=f"blindspot_audit_{stem}_{ts}.html",
                mime="text/html",
                use_container_width=True,
                help="Opens in any browser. Print → Save as PDF for a portable report.",
            )
        except Exception as e:
            st.warning(f"Report generation failed: {e}")

    with st.expander("About", expanded=False):
        st.markdown("""
        **Blind Spot** audits data for hidden biases and structural problems
        before any model or conclusion is drawn.

        **Modules**
        - Simpson's Paradox Scanner
        - Survivorship Bias Detector
        - Confounder Detection
        - Metric Gaming Risk Scorer
        - Data Confidence Score (A–F)
        - Blind Spot Report
        - Exportable HTML Audit

        Accepts CSV or Excel files up to 200 MB.
        All computation runs locally — no data leaves your machine.

        Built with [Claude](https://claude.ai) by Anthropic.

        [View on GitHub](https://github.com/konarkambad/blindspot)
        """)

    with st.expander("Advanced Settings", expanded=False):
        st.session_state.max_combinations = st.slider(
            "Max paradox combinations", 100, 1000,
            value=st.session_state.max_combinations, step=50,
        )
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold", 0, 100,
            value=st.session_state.confidence_threshold,
        )
        st.markdown("**Enable / disable modules**")
        st.session_state.enable_paradox      = st.checkbox("Simpson's Paradox",   value=st.session_state.enable_paradox)
        st.session_state.enable_survivorship = st.checkbox("Survivorship Bias",   value=st.session_state.enable_survivorship)
        st.session_state.enable_confounders  = st.checkbox("Confounder Detection",value=st.session_state.enable_confounders)
        st.session_state.enable_gaming       = st.checkbox("Metric Gaming Risk",  value=st.session_state.enable_gaming)


# ---------------------------------------------------------------------------
# Helper: dtype badge HTML
# ---------------------------------------------------------------------------

def _dtype_badge(dtype_str: str) -> str:  # noqa: D401
    s = str(dtype_str).lower()
    if any(x in s for x in ("int", "float", "complex")):
        return f'<span class="dtype-badge dtype-numeric">{dtype_str}</span>'
    if "datetime" in s:
        return f'<span class="dtype-badge dtype-datetime">{dtype_str}</span>'
    if "bool" in s:
        return f'<span class="dtype-badge dtype-boolean">{dtype_str}</span>'
    if "categ" in s:
        return f'<span class="dtype-badge dtype-category">{dtype_str}</span>'
    return f'<span class="dtype-badge dtype-other">{dtype_str}</span>'


# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
    font=dict(color="#c9d1e0", family="sans-serif"),
)


# ---------------------------------------------------------------------------
# Audit runner
# ---------------------------------------------------------------------------

def run_full_audit(df: pd.DataFrame, profile_dict: dict, settings: dict) -> dict:
    """
    Execute all audit modules in sequence with a live progress bar.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    profile_dict : dict
        Pre-computed loader metadata (not the profiler output — that runs here).
    settings : dict
        User-configured flags from session state (enable_paradox, etc.).

    Returns
    -------
    dict mapping module keys to their results, plus ``_warnings`` for any
    module-level failures caught during execution.
    """
    results:  dict = {}
    warnings: list = []

    # (pct, label, key)
    steps = [
        (0,  "Profiling data…",                "profile"),
        (15, "Scanning for Simpson's Paradox…", "paradox"),
        (30, "Checking survivorship bias…",     "survivorship"),
        (48, "Detecting confounders…",          "confounders"),
        (64, "Scoring metric gaming risk…",     "gaming"),
        (80, "Computing confidence scores…",    "confidence"),
        (92, "Mapping blind spots…",            "blind_spots"),
    ]

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for pct_start, label, key in steps:
        status_text.markdown(f"**{label}**")
        progress_bar.progress(pct_start)

        try:
            if key == "profile":
                results["profile"] = profile_data(df)

            elif key == "paradox":
                results["paradox"] = (
                    scan_simpsons_paradox(df) if settings["enable_paradox"] else []
                )

            elif key == "survivorship":
                results["survivorship"] = (
                    detect_survivorship_bias(df, results.get("profile", {}))
                    if settings["enable_survivorship"] else []
                )

            elif key == "confounders":
                results["confounders"] = (
                    detect_confounders(df) if settings["enable_confounders"] else []
                )

            elif key == "gaming":
                results["gaming"] = (
                    score_metric_gaming_risk(df, results.get("profile", {}))
                    if settings["enable_gaming"] else []
                )

            elif key == "confidence":
                results["confidence"] = compute_confidence_scores(
                    df,
                    results.get("profile", {}),
                    results.get("paradox", []),
                    results.get("survivorship", []),
                    results.get("confounders", []),
                    results.get("gaming", []),
                )

            elif key == "blind_spots":
                results["blind_spots"] = generate_blind_spots(
                    df, results.get("profile", {})
                )

        except Exception as exc:
            warnings.append(f"**{label.rstrip('…')}** failed: {exc}")
            results[key] = {} if key in ("profile", "confidence") else []

    progress_bar.progress(100)
    status_text.empty()
    progress_bar.empty()

    results["_warnings"] = warnings
    return results


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

def _score_color(score):
    if score >= 75: return "#34d399"
    if score >= 55: return "#fbbf24"
    return "#f87171"

def _grade_style(grade):
    return {"A": ("#34d399", "#0a2d1a"), "B": ("#48b0f7", "#0a1f3d"),
            "C": ("#fbbf24", "#3d2d0a"), "D": ("#fb923c", "#3d1a0a"),
            "F": ("#f87171", "#3d1515")}.get(grade, ("#7a8599", "#1e2535"))


# ── Tab 1: Overview ────────────────────────────────────────────────────────

def render_overview(conf: dict, threshold: int, blind_spots: list | None = None) -> None:
    score    = conf.get("overall_score", 0)
    grade    = conf.get("overall_grade", "F")
    fg, bg   = _grade_style(grade)

    if score < threshold:
        st.warning(f"Score {score:.1f} is below your confidence threshold of {threshold}. Treat conclusions from this data with caution.")

    # ── Gauge + component bar side by side ──────────────────────────────
    col_gauge, col_bar = st.columns([1, 1.4])

    with col_gauge:
        gauge_color = _score_color(score)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(
                font=dict(size=52, color=gauge_color, family="sans-serif"),
                suffix="",
                valueformat=".1f",
            ),
            gauge=dict(
                axis=dict(
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0", "25", "50", "75", "100"],
                    tickfont=dict(color="#4a5568", size=11),
                ),
                bar=dict(color=gauge_color, thickness=0.28),
                bgcolor="#161b27",
                borderwidth=0,
                steps=[
                    dict(range=[0,  45], color="#3d1515"),
                    dict(range=[45, 70], color="#3d2d0a"),
                    dict(range=[70, 100], color="#0a2d1a"),
                ],
                threshold=dict(
                    line=dict(color="#ffffff", width=3),
                    thickness=0.85,
                    value=score,
                ),
            ),
            title=dict(
                text=f"<b>Grade {grade}</b><br><span style='font-size:13px;color:#7a8599'>Data Confidence Score</span>",
                font=dict(size=18, color=fg),
            ),
            domain=dict(x=[0, 1], y=[0, 1]),
        ))
        fig_gauge.update_layout(
            **PLOTLY_LAYOUT,
            height=260,
            margin=dict(l=20, r=20, t=50, b=5),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_bar:
        comp = conf.get("component_scores", {})
        if comp:
            labels     = [l.replace("_", " ").title() for l in comp.keys()]
            values     = list(comp.values())
            bar_colors = [_score_color(v) for v in values]

            fig_bar = go.Figure()

            # bg track at 100
            fig_bar.add_trace(go.Bar(
                x=[100] * len(labels),
                y=labels,
                orientation="h",
                marker=dict(color="#1e2535", line=dict(width=0)),
                showlegend=False,
                hoverinfo="skip",
            ))

            # score bars
            fig_bar.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(color=bar_colors, line=dict(width=0)),
                text=[f"{v:.1f}" for v in values],
                textposition="outside",
                textfont=dict(color="#e0e0e0", size=13, family="sans-serif"),
                cliponaxis=False,   # keep labels visible
                hovertemplate="%{y}: %{x:.1f}/100<extra></extra>",
                showlegend=False,
            ))

            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                title="Component Scores",
                barmode="overlay",
                xaxis=dict(range=[0, 130], showgrid=False, zeroline=False, visible=False),
                yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
                height=260,
                margin=dict(l=10, r=55, t=35, b=5),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Risks + Recommendation side by side ───────────────────────────────
    risk_col, rec_col = st.columns([1, 1])

    with risk_col:
        top_risks = conf.get("top_risks", [])
        st.markdown("**Top Risk Factors**")
        if top_risks:
            for risk in top_risks:
                st.markdown(f'<div class="risk-card">⚠ {risk}</div>', unsafe_allow_html=True)
        else:
            st.success("No major risk factors detected.")

    with rec_col:
        rec = conf.get("recommendation", "")
        if rec:
            st.markdown("**Recommendation**")
            st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

    # ── Blind Spots ────────────────────────────────────────────────────────
    if blind_spots is not None:
        st.markdown("""
        <div style="margin:.6rem 0 .4rem 0;">
          <span style="font-size:1rem;font-weight:800;color:#e0e0e0;">🕳 Blind Spots</span>
          <span style="font-size:.8rem;color:#7a8599;margin-left:.5rem;">— what this data structurally cannot answer</span>
        </div>
        """, unsafe_allow_html=True)
        render_blind_spots(blind_spots)


# ── Blind Spots renderer (called from Overview) ────────────────────────────

def render_blind_spots(blind_spots: list) -> None:
    """
    Render the 'What This Data Cannot Tell You' section.
    Each finding is an impact-coded card with a description, missing data type,
    and suggested source.
    """
    if not blind_spots:
        st.success("No structural blind spots detected — this dataset covers the expected analytical dimensions.")
        return

    high   = [b for b in blind_spots if b.get("impact_level") == "high"]
    medium = [b for b in blind_spots if b.get("impact_level") == "medium"]
    low    = [b for b in blind_spots if b.get("impact_level") == "low"]
    st.markdown(f"""
    <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1.1rem;">
      <div class="stat-chip" style="min-width:unset;padding:.4rem .9rem;">
        <div class="stat-value" style="font-size:1.3rem;color:#f87171;">{len(high)}</div>
        <div class="stat-label">High impact</div>
      </div>
      <div class="stat-chip" style="min-width:unset;padding:.4rem .9rem;">
        <div class="stat-value" style="font-size:1.3rem;color:#fbbf24;">{len(medium)}</div>
        <div class="stat-label">Medium impact</div>
      </div>
      <div class="stat-chip" style="min-width:unset;padding:.4rem .9rem;">
        <div class="stat-value" style="font-size:1.3rem;color:#34d399;">{len(low)}</div>
        <div class="stat-label">Low impact</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    for spot in blind_spots:
        impact  = spot.get("impact_level", "low")
        desc    = spot.get("blind_spot_description", "")
        missing = spot.get("missing_data_type", "")
        source  = spot.get("suggested_source", "")

        st.markdown(f"""
        <div class="bs-card bs-card-{impact}">
          <span class="bs-impact bs-impact-{impact}">{impact} impact</span>
          <div class="bs-title">{desc}</div>
          <div class="bs-missing">Missing: {missing}</div>
          <div class="bs-source">{source}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Tab 2: Data Profile ────────────────────────────────────────────────────

def render_profile(df: pd.DataFrame, profile: dict) -> None:
    for warn_msg in profile.get("warnings", []):
        st.warning(warn_msg)

    # ── Row 1: missing heatmap + correlation heatmap side by side ─────────
    prof_left, prof_right = st.columns(2)

    with prof_left:
        st.markdown("**Missing Value Density**")
        N_BANDS   = 40
        n         = len(df)
        band_size = max(1, n // N_BANDS)
        all_cols  = list(df.columns)
        cols_with_nulls = [c for c in all_cols if df[c].isnull().any()]

        bands, band_labels = [], []
        for b in range(N_BANDS):
            start = b * band_size
            end   = min(start + band_size, n)
            bands.append(df.iloc[start:end][all_cols].isnull().mean().values)
            band_labels.append(f"{start}–{end}")

        z  = np.array(bands).T
        cs = [[0.0, "#161b27"], [0.01, "#2d1515"], [0.5, "#c0392b"], [1.0, "#f87171"]]

        null_info = (
            f"{len(cols_with_nulls)}/{len(all_cols)} cols have nulls"
            if cols_with_nulls else "No missing values"
        )
        st.caption(null_info)

        tick_labels = band_labels[::5]
        fig_miss = go.Figure(go.Heatmap(
            z=z, x=band_labels, y=all_cols,
            colorscale=cs, zmin=0, zmax=1,
            colorbar=dict(
                title=dict(text="Missing %", font=dict(color="#7a8599", size=10)),
                tickformat=".0%", tickfont=dict(color="#7a8599"), thickness=10, len=0.8,
            ),
            hovertemplate="Rows %{x}<br>Column: %{y}<br>Missing: %{z:.1%}<extra></extra>",
        ))
        fig_miss.update_layout(
            **PLOTLY_LAYOUT,
            height=max(260, len(all_cols) * 18 + 80),
            xaxis=dict(tickvals=tick_labels, ticktext=tick_labels,
                       tickangle=40, tickfont=dict(size=8, color="#7a8599"), showgrid=False),
            yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
            margin=dict(l=10, r=65, t=20, b=60),
        )
        st.plotly_chart(fig_miss, use_container_width=True)

    with prof_right:
        st.markdown("**Correlation Matrix**")
        numeric_df = df.select_dtypes(include=[np.number])
        numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]
        if numeric_df.shape[1] < 2:
            st.info("Need ≥ 2 non-constant numeric columns.")
        else:
            if numeric_df.shape[1] > 15:
                numeric_df = numeric_df.iloc[:, :15]
                st.caption("Showing first 15 numeric columns.")
            corr = numeric_df.corr(method="pearson").round(2)
            text_vals = np.where(
                np.eye(len(corr), dtype=bool), "—",
                np.array([[f"{v:.2f}" for v in row] for row in corr.values])
            )
            corr_cs = [
                [0.0, "#f87171"], [0.35, "#2d1f2f"],
                [0.5, "#161b27"], [0.65, "#1a2744"], [1.0, "#48b0f7"],
            ]
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                text=text_vals, texttemplate="%{text}",
                textfont=dict(size=9, color="#c9d1e0"),
                colorscale=corr_cs, zmin=-1, zmax=1,
                colorbar=dict(
                    title=dict(text="r", font=dict(color="#7a8599", size=10)),
                    tickfont=dict(color="#7a8599"), thickness=10, len=0.8,
                    tickvals=[-1, -0.5, 0, 0.5, 1],
                ),
                hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.2f}<extra></extra>",
            ))
            high_pairs = profile.get("high_correlation_pairs", [])
            for pair in high_pairs:
                ca, cb = pair["col_a"], pair["col_b"]
                if ca in corr.columns and cb in corr.index:
                    fig_corr.add_shape(
                        type="rect",
                        x0=corr.columns.tolist().index(ca) - 0.5,
                        x1=corr.columns.tolist().index(ca) + 0.5,
                        y0=corr.index.tolist().index(cb) - 0.5,
                        y1=corr.index.tolist().index(cb) + 0.5,
                        line=dict(color="#fbbf24", width=2),
                        fillcolor="rgba(0,0,0,0)",
                    )
            fig_corr.update_layout(
                **PLOTLY_LAYOUT,
                height=max(260, numeric_df.shape[1] * 30 + 60),
                xaxis=dict(tickangle=40, tickfont=dict(size=9), side="bottom"),
                yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
                margin=dict(l=10, r=65, t=20, b=70),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    # ── Row 2: type issues + outliers side by side ────────────────────────
    ti_col, out_col = st.columns(2)

    with ti_col:
        st.markdown("**Type Issues**")
        type_issues = profile.get("type_issues", {})
        if type_issues:
            st.dataframe(pd.DataFrame([{"Column": c, "Issue": i} for c, i in type_issues.items()]),
                         use_container_width=True, hide_index=True, height=200)
        else:
            st.success("No dtype mismatches detected.")

    with out_col:
        st.markdown("**Outlier Summary**")
        outliers = profile.get("outlier_flags", {})
        if outliers:
            rows = [
                (col, ev.get("zscore_outliers", "—"), ev.get("iqr_outliers", "—"))
                for col, ev in outliers.items()
                if ev.get("iqr_outliers") or ev.get("zscore_outliers")
            ]
            if rows:
                tbody = "".join(
                    f"<tr style='background:{'#1a1f2e' if i % 2 == 0 else '#161b27'};'>"
                    f"<td style='padding:.4rem .65rem;font-size:.82rem;'>"
                    f"<code style='color:#a78bfa;background:none;'>{col}</code></td>"
                    f"<td style='text-align:center;padding:.4rem .65rem;font-size:.82rem;color:#c9d1e0;'>{z}</td>"
                    f"<td style='text-align:center;padding:.4rem .65rem;font-size:.82rem;color:#c9d1e0;'>{iqr}</td>"
                    f"</tr>"
                    for i, (col, z, iqr) in enumerate(rows)
                )
                st.markdown(f"""
                <div style="max-height:200px;overflow-y:auto;border:1px solid #2a2f3e;border-radius:8px;">
                <table style="width:100%;border-collapse:collapse;">
                  <thead><tr style="background:#1e2535;">
                    <th style="text-align:left;padding:.4rem .65rem;font-size:.72rem;text-transform:uppercase;
                               letter-spacing:.08em;color:#c9d1e0;font-weight:700;border-bottom:1px solid #2a2f3e;">Column</th>
                    <th style="text-align:center;padding:.4rem .65rem;font-size:.72rem;text-transform:uppercase;
                               letter-spacing:.08em;color:#c9d1e0;font-weight:700;border-bottom:1px solid #2a2f3e;">Z-score</th>
                    <th style="text-align:center;padding:.4rem .65rem;font-size:.72rem;text-transform:uppercase;
                               letter-spacing:.08em;color:#c9d1e0;font-weight:700;border-bottom:1px solid #2a2f3e;">IQR</th>
                  </tr></thead>
                  <tbody>{tbody}</tbody>
                </table>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("No significant outliers detected.")
        else:
            st.info("No numeric columns for outlier analysis.")


# ── Tab 3: Simpson's Paradox ───────────────────────────────────────────────

# subgroup palette
_SUBGROUP_PALETTE = [
    "#a78bfa", "#34d399", "#fbbf24", "#f87171",
    "#60a5fa", "#f472b6", "#fb923c", "#86efac",
]

def render_paradox(paradox_results: list, df: pd.DataFrame | None = None) -> None:
    if not paradox_results:
        st.success("No Simpson's Paradox cases detected in this dataset.")
        return

    st.markdown(f"**{len(paradox_results)} case(s) detected** — sorted by severity.")
    st.markdown("")

    for i, case in enumerate(paradox_results):
        severity    = case.get("severity_score", 0)
        x_col       = case["x_column"]
        y_col       = case["y_column"]
        group_col   = case["grouping_column"]
        overall_r   = case["overall_correlation"]

        st.markdown(
            f'<div style="border-left:4px solid #6c63ff;padding:.55rem .9rem;'
            f'margin:1.4rem 0 .8rem 0;background:#161b27;border-radius:0 6px 6px 0;">'
            f'<span style="font-weight:800;color:#e0e0e0;font-size:.97rem;">'
            f'#{i+1} — {x_col} vs {y_col} grouped by {group_col}</span>'
            f'<span style="color:#7a8599;font-size:.82rem;margin-left:.6rem;">'
            f'| Severity {severity:.2f}</span></div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Overall correlation:** `{overall_r:.4f}`")
            st.markdown(f"**Reversal type:** `{case['reversal_type']}`")
            st.markdown(f"**Severity score:** `{severity:.4f}`")
        with col2:
            st.markdown("**Subgroup correlations:**")
            for grp, corr in case.get("subgroup_correlations", {}).items():
                if corr is None:
                    st.markdown(f"- `{grp}`: *insufficient data*")
                else:
                    flipped = (corr < 0 < overall_r) or (corr > 0 > overall_r)
                    emoji = "🔴" if flipped else "🟢"
                    st.markdown(f"- `{grp}`: **{corr:.4f}** {emoji}")

        # ── Grouped scatter with trend lines ──────────────────────────────
        if df is not None and x_col in df.columns and y_col in df.columns and group_col in df.columns:
            fig_sc = go.Figure()

            # overall trend line
            x_all = df[x_col].dropna()
            y_all = df[y_col].reindex(x_all.index).dropna()
            x_all = x_all.reindex(y_all.index)
            if len(x_all) >= 2:
                m, b = np.polyfit(x_all, y_all, 1)
                x_range = np.linspace(float(x_all.min()), float(x_all.max()), 80)
                fig_sc.add_trace(go.Scatter(
                    x=x_range, y=m * x_range + b,
                    mode="lines",
                    line=dict(color="#ffffff", width=2.5, dash="dash"),
                    name=f"Overall (r={overall_r:.2f})",
                    hoverinfo="skip",
                ))

            # Per-subgroup scatter + trend line
            groups = df[group_col].dropna().unique()
            for gi, grp_val in enumerate(sorted(groups)):
                color = _SUBGROUP_PALETTE[gi % len(_SUBGROUP_PALETTE)]
                mask  = df[group_col] == grp_val
                gx    = df.loc[mask, x_col].dropna()
                gy    = df.loc[mask, y_col].reindex(gx.index).dropna()
                gx    = gx.reindex(gy.index)

                grp_r = case["subgroup_correlations"].get(str(grp_val))
                r_label = f"  r={grp_r:.2f}" if grp_r is not None else ""

                # sample for perf
                sample_gx = gx.sample(min(200, len(gx)), random_state=0) if len(gx) > 200 else gx
                sample_gy = gy.reindex(sample_gx.index)
                fig_sc.add_trace(go.Scatter(
                    x=sample_gx, y=sample_gy,
                    mode="markers",
                    marker=dict(color=color, size=5, opacity=0.55),
                    name=f"{grp_val}{r_label}",
                    hovertemplate=f"{group_col}={grp_val}<br>{x_col}=%{{x:.2f}}<br>{y_col}=%{{y:.2f}}<extra></extra>",
                ))

                # Subgroup trend line
                if len(gx) >= 3 and gx.std() > 0 and gy.std() > 0:
                    m_g, b_g = np.polyfit(gx, gy, 1)
                    x_rng = np.linspace(float(gx.min()), float(gx.max()), 60)
                    fig_sc.add_trace(go.Scatter(
                        x=x_rng, y=m_g * x_rng + b_g,
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo="skip",
                    ))

            fig_sc.update_layout(
                **PLOTLY_LAYOUT,
                title=f"Scatter: {x_col} vs {y_col}  |  dashed = overall trend",
                xaxis=dict(title=x_col, showgrid=False, zeroline=False,
                           gridcolor="#2a2f3e"),
                yaxis=dict(title=y_col, showgrid=True, gridcolor="#2a2f3e",
                           zeroline=False),
                legend=dict(
                    bgcolor="#161b27", bordercolor="#2a2f3e", borderwidth=1,
                    font=dict(size=11),
                ),
                height=380,
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        # ── Correlation bar chart ──────────────────────────────────────────
        subgroups = case.get("subgroup_correlations", {})
        valid = {k: v for k, v in subgroups.items() if v is not None}
        if valid:
            all_labels = ["OVERALL"] + list(valid.keys())
            all_values = [overall_r] + list(valid.values())
            bar_colors = [
                "#48b0f7" if j == 0
                else ("#f87171" if ((v < 0 < overall_r) or (v > 0 > overall_r))
                      else "#34d399")
                for j, v in enumerate(all_values)
            ]
            fig_bar = go.Figure(go.Bar(
                x=all_labels, y=all_values,
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in all_values],
                textposition="outside",
                textfont=dict(color="#c9d1e0"),
            ))
            fig_bar.add_hline(y=0, line_color="#4a5568", line_width=1)
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                height=280,
                yaxis=dict(range=[-1.15, 1.15], zeroline=False,
                           showgrid=True, gridcolor="#2a2f3e"),
                xaxis=dict(showgrid=False),
                title=f"Correlation by group  ({group_col})",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        if i < len(paradox_results) - 1:
            st.markdown('<hr style="border-color:#2a2f3e;margin:1.5rem 0;">', unsafe_allow_html=True)


# ── Tab 4: Survivorship Bias ───────────────────────────────────────────────

def _render_timeline(df):
    """
    Find a datetime column and draw a monthly record-density bar chart
    with interior gaps highlighted in red.
    """
    date_col = None
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        date_col = col
        break
    if date_col is None:
        for col in df.select_dtypes(include="object").columns:
            try:
                coerced = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                if coerced.notna().mean() > 0.80:
                    df = df.copy()
                    df[col] = coerced
                    date_col = col
                    break
            except Exception:
                continue
    if date_col is None:
        return None

    periods = df[date_col].dropna().dt.to_period("M")
    if len(periods) < 4:
        return None

    counts = periods.value_counts().sort_index()
    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    all_counts = counts.reindex(full_range, fill_value=0)

    x_strs = [str(p) for p in all_counts.index]
    y_vals  = all_counts.values.tolist()

    # detect interior gaps
    is_gap = []
    for j, v in enumerate(y_vals):
        left  = any(y_vals[:j])
        right = any(y_vals[j+1:])
        is_gap.append(v == 0 and left and right)

    bar_colors = ["#f87171" if g else "#6c63ff" for g in is_gap]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_strs, y=y_vals,
        marker_color=bar_colors,
        hovertemplate="%{x}<br>Records: %{y}<extra></extra>",
        name="Records per month",
    ))

    # shade gaps
    gap_starts = [j for j in range(len(is_gap)) if is_gap[j] and (j == 0 or not is_gap[j-1])]
    gap_ends   = [j for j in range(len(is_gap)) if is_gap[j] and (j == len(is_gap)-1 or not is_gap[j+1])]
    for gs, ge in zip(gap_starts, gap_ends):
        fig.add_vrect(
            x0=x_strs[gs], x1=x_strs[ge],
            fillcolor="rgba(248,113,113,0.12)",
            line_width=0,
            annotation_text="GAP",
            annotation_position="top left",
            annotation_font=dict(color="#f87171", size=10),
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Monthly Record Density — '{date_col}'  (red = interior gaps)",
        xaxis=dict(tickangle=45, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(title="Records", showgrid=True, gridcolor="#2a2f3e", zeroline=False),
        height=320,
        margin=dict(l=20, r=20, t=50, b=60),
        showlegend=False,
    )
    return fig


def render_survivorship(survivorship_results: list, df: pd.DataFrame | None = None) -> None:
    if not survivorship_results:
        st.success("No survivorship bias signals detected.")
        # Still try to show the timeline even when no findings
        if df is not None:
            fig = _render_timeline(df)
            if fig:
                st.markdown("#### Record Density Over Time")
                st.plotly_chart(fig, use_container_width=True)
        return

    # ── Timeline (always attempt, even without temporal_gap finding) ───────
    if df is not None:
        fig_tl = _render_timeline(df)
        if fig_tl:
            st.markdown("#### Record Density Over Time")
            st.plotly_chart(fig_tl, use_container_width=True)

    # ── Findings list ──────────────────────────────────────────────────────
    st.markdown(f"#### {len(survivorship_results)} Finding(s)")
    for finding in survivorship_results:
        risk      = finding.get("risk_level", "low")
        badge_cls = f"badge-{risk}"
        ftype     = finding.get("finding_type", "").replace("_", " ").title()
        cols_str  = ", ".join(finding.get("affected_columns", []))

        st.markdown(f"""
        <div class="finding-card">
          <div class="finding-title">
            <span class="badge {badge_cls}">{risk.upper()}</span>&nbsp;&nbsp;
            {ftype} — <code style="color:#a78bfa;">{cols_str}</code>
          </div>
          <div class="finding-desc">{finding.get('description','')}</div>
        </div>
        """, unsafe_allow_html=True)

        evidence = finding.get("evidence", {})
        if evidence:
            ev_rows = [{"Key": k, "Value": str(v)} for k, v in evidence.items()]
            st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)


# ── Tab 5: Confounders ─────────────────────────────────────────────────────

def render_confounders(confounder_results: list) -> None:
    if not confounder_results:
        st.success("No confounding variables detected.")
        return

    numeric_conf = [r for r in confounder_results if r.get("type") == "numeric_confounder"]
    cat_conf     = [r for r in confounder_results if r.get("type") == "categorical_association"]

    if numeric_conf:
        st.markdown(f"#### Numeric Confounders ({len(numeric_conf)})")
        fig = go.Figure()
        for r in numeric_conf:
            pair_label = f"{r['variable_x']} × {r['variable_y']}"
            fig.add_trace(go.Scatter(
                x=[r["raw_correlation"], r["partial_correlation"]],
                y=[pair_label, pair_label],
                mode="lines+markers",
                line=dict(color="#2a2f3e", width=2),
                marker=dict(
                    color=["#48b0f7", "#f87171"],
                    size=10,
                    symbol=["circle", "diamond"],
                ),
                name=pair_label,
                showlegend=False,
                hovertemplate=(
                    f"<b>{pair_label}</b><br>"
                    f"Confounder: {r['likely_confounder']}<br>"
                    f"Raw: {r['raw_correlation']:.3f}<br>"
                    f"Partial: {r['partial_correlation']:.3f}<br>"
                    f"Drop: {r['correlation_drop_pct']:.1f}%"
                    "<extra></extra>"
                ),
            ))

        fig.add_vline(x=0, line_color="#2a2f3e", line_width=1)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=max(250, len(numeric_conf) * 38 + 60),
            xaxis=dict(range=[-1.1, 1.1], title="Correlation", showgrid=False),
            yaxis=dict(autorange="reversed"),
            title="Raw (●) vs Partial (◆) Correlation after controlling for confounder",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        rows = [{
            "X": r["variable_x"],
            "Y": r["variable_y"],
            "Confounder (Z)": r["likely_confounder"],
            "Raw r": f"{r['raw_correlation']:.3f}",
            "Partial r": f"{r['partial_correlation']:.3f}",
            "Drop %": f"{r['correlation_drop_pct']:.1f}%",
        } for r in numeric_conf]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("")
        for r in numeric_conf:
            st.markdown(
                f'<div style="border-left:3px solid #2a2f3e;padding:.5rem .85rem;'
                f'margin:.5rem 0;border-radius:0 4px 4px 0;background:#161b27;">'
                f'<div style="font-size:.8rem;font-weight:700;color:#7a8599;'
                f'text-transform:uppercase;letter-spacing:.07em;margin-bottom:.3rem;">'
                f'{r["variable_x"]} × {r["variable_y"]}</div>'
                f'<div style="font-size:.86rem;color:#94a3b8;line-height:1.6;">'
                f'{r.get("interpretation", "")}</div></div>',
                unsafe_allow_html=True,
            )

    if cat_conf:
        st.markdown(f"#### Categorical Associations ({len(cat_conf)})")
        rows = [{
            "Variable A": r["variable_x"],
            "Variable B": r["variable_y"],
            "Chi² stat": f"{r.get('chi2_statistic', '—'):.2f}",
            "p-value": f"{r.get('p_value', '—'):.4f}",
            "Cramér's V": f"{r.get('cramers_v', '—'):.3f}",
        } for r in cat_conf]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab 6: Metric Gaming ───────────────────────────────────────────────────

# Benford expected
_BENFORD_EXPECTED = np.array([np.log10(1 + 1/d) for d in range(1, 10)])
_BENFORD_DIGITS   = list(range(1, 10))

def _benford_chart(col_name: str, breakdown: dict) -> go.Figure | None:
    """
    Build a Benford's Law observed vs expected first-digit bar chart
    for one column, using the precomputed distribution stored in breakdown.
    """
    benford_ev = breakdown.get("benford", {})
    observed   = benford_ev.get("observed_first_digit_dist")
    if not observed or len(observed) != 9:
        return None

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_BENFORD_DIGITS,
        y=[v * 100 for v in _BENFORD_EXPECTED],
        name="Benford expected",
        marker_color="#2a2f3e",
        marker_line=dict(color="#48b0f7", width=1.5),
        hovertemplate="Digit %{x}<br>Expected: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=_BENFORD_DIGITS,
        y=[v * 100 for v in observed],
        name="Observed",
        marker_color="#a78bfa",
        opacity=0.85,
        hovertemplate="Digit %{x}<br>Observed: %{y:.1f}%<extra></extra>",
    ))
    p = benford_ev.get("p_value")
    p_label = f"  (chi² p={p:.4f})" if p is not None else ""
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"First-Digit Distribution vs Benford's Law — '{col_name}'{p_label}",
        barmode="overlay",
        xaxis=dict(
            tickvals=_BENFORD_DIGITS, ticktext=[str(d) for d in _BENFORD_DIGITS],
            title="Leading digit", showgrid=False,
        ),
        yaxis=dict(title="Frequency (%)", showgrid=True, gridcolor="#2a2f3e", zeroline=False),
        legend=dict(bgcolor="#161b27", bordercolor="#2a2f3e", borderwidth=1),
        height=300,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def render_gaming(gaming_results: list) -> None:
    if not gaming_results:
        st.info("No numeric columns available for gaming risk analysis.")
        return

    dim_keys   = ["concentration", "round_numbers", "benford", "spike", "weekend_pattern"]
    dim_labels = ["Concentration", "Round Numbers", "Benford", "Spike", "Weekend Pattern"]
    dim_colors = ["#6c63ff", "#48b0f7", "#a78bfa", "#fbbf24", "#f87171"]
    col_names  = [r["column_name"] for r in gaming_results]
    fig_stack  = go.Figure()
    for dk, dl, dc in zip(dim_keys, dim_labels, dim_colors):
        sub_scores = [r["breakdown"].get(dk, {}).get("sub_score", 0) for r in gaming_results]
        fig_stack.add_trace(go.Bar(
            name=dl,
            y=col_names,
            x=sub_scores,
            orientation="h",
            marker_color=dc,
            hovertemplate=f"<b>%{{y}}</b><br>{dl}: %{{x:.1f}} pts<extra></extra>",
        ))

    # Total score label at the right end of each bar
    totals = [r["gaming_risk_score"] for r in gaming_results]
    fig_stack.add_trace(go.Scatter(
        x=totals,
        y=col_names,
        mode="text",
        text=[f"  {t:.1f}" for t in totals],
        textposition="middle right",
        textfont=dict(color="#e0e0e0", size=12, family="sans-serif"),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_stack.update_layout(
        **PLOTLY_LAYOUT,
        barmode="stack",
        title="Gaming Risk Sub-Score Breakdown per Column",
        xaxis=dict(
            title="Score contribution (pts)",
            range=[0, 120],   # extra room for the total labels
            showgrid=True, gridcolor="#2a2f3e", zeroline=False,
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="#161b27", bordercolor="#2a2f3e", borderwidth=1,
            font=dict(size=11),
        ),
        height=max(240, len(gaming_results) * 32 + 80),
        margin=dict(l=20, r=20, t=70, b=30),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    top_col     = gaming_results[0]
    fig_benford = _benford_chart(top_col["column_name"], top_col.get("breakdown", {}))
    if fig_benford:
        st.markdown(f"#### Benford's Law — highest-risk column: `{top_col['column_name']}`")
        st.plotly_chart(fig_benford, use_container_width=True)

    st.markdown("#### Per-Column Detail")
    for r in gaming_results:
        risk      = r.get("risk_level", "low")
        badge     = f'<span class="badge badge-{risk}">{risk.upper()}</span>'
        total     = r["gaming_risk_score"]
        breakdown = r.get("breakdown", {})
        is_notable = risk in ("medium", "high")

        if is_notable:
            # medium/high inline
            st.markdown(
                f'<div style="border-left:4px solid '
                f'{"#f87171" if risk == "high" else "#fbbf24"};'
                f'padding:.55rem .9rem;margin:1.2rem 0 .6rem 0;'
                f'background:#161b27;border-radius:0 6px 6px 0;">'
                f'<span style="font-weight:800;color:#e0e0e0;font-size:.95rem;">'
                f'{r["column_name"]}</span>'
                f'<span style="margin-left:.7rem;">{badge}</span>'
                f'<span style="color:#7a8599;font-size:.82rem;margin-left:.6rem;">'
                f'{total:.1f} / 100</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"*{r.get('interpretation','')}*")
            st.markdown("")

            d_scores = [breakdown.get(dk, {}).get("sub_score", 0) for dk in dim_keys]
            fig_mini = go.Figure(go.Bar(
                x=dim_labels, y=d_scores,
                marker_color=dim_colors,
                text=[f"{s:.1f}" for s in d_scores],
                textposition="outside",
                textfont=dict(color="#c9d1e0", size=12),
            ))
            fig_mini.update_layout(
                **PLOTLY_LAYOUT, height=230,
                xaxis=dict(showgrid=False),
                yaxis=dict(range=[0, 30], showgrid=False, zeroline=False, visible=False),
                margin=dict(l=10, r=10, t=10, b=30),
            )
            st.plotly_chart(fig_mini, use_container_width=True)

            fig_b = _benford_chart(r["column_name"], breakdown)
            if fig_b:
                st.plotly_chart(fig_b, use_container_width=True)

            ev_rows = []
            for dim, ev in breakdown.items():
                for k, v in ev.items():
                    if k != "sub_score":
                        ev_rows.append({"Dimension": dim, "Metric": k, "Value": str(v)})
            if ev_rows:
                st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)

        else:
            # low risk expander
            with st.expander(f"{r['column_name']}  —  {total:.1f} / 100  (low risk)", expanded=False):
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(f"*{r.get('interpretation','')}*")
                st.markdown("")

                d_scores = [breakdown.get(dk, {}).get("sub_score", 0) for dk in dim_keys]
                fig_mini = go.Figure(go.Bar(
                    x=dim_labels, y=d_scores,
                    marker_color=dim_colors,
                    text=[f"{s:.1f}" for s in d_scores],
                    textposition="outside",
                    textfont=dict(color="#c9d1e0", size=12),
                ))
                fig_mini.update_layout(
                    **PLOTLY_LAYOUT, height=230,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(range=[0, 30], showgrid=False, zeroline=False, visible=False),
                    margin=dict(l=10, r=10, t=10, b=30),
                )
                st.plotly_chart(fig_mini, use_container_width=True)

                fig_b = _benford_chart(r["column_name"], breakdown)
                if fig_b:
                    st.plotly_chart(fig_b, use_container_width=True)

                ev_rows = []
                for dim, ev in breakdown.items():
                    for k, v in ev.items():
                        if k != "sub_score":
                            ev_rows.append({"Dimension": dim, "Metric": k, "Value": str(v)})
                if ev_rows:
                    st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)


# ── Tab 7: Confidence Detail ───────────────────────────────────────────────

def render_confidence_detail(conf: dict) -> None:
    at_scores = conf.get("analysis_type_scores", {})
    comp      = conf.get("component_scores", {})
    if at_scores:
        labels   = [k.replace("_confidence", "").replace("_", " ").title() for k in at_scores]
        values   = list(at_scores.values())
        values_c = values + [values[0]]
        labels_c = labels + [labels[0]]

        fig_radar = go.Figure()

        # bg ring at 100
        fig_radar.add_trace(go.Scatterpolar(
            r=[100] * (len(labels) + 1),
            theta=labels_c,
            fill="toself",
            fillcolor="rgba(42,47,62,0.35)",
            line=dict(color="#2a2f3e", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Threshold ring (e.g., 70)
        fig_radar.add_trace(go.Scatterpolar(
            r=[70] * (len(labels) + 1),
            theta=labels_c,
            fill="none",
            line=dict(color="#fbbf24", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ))
        # actual scores
        fig_radar.add_trace(go.Scatterpolar(
            r=values_c,
            theta=labels_c,
            fill="toself",
            fillcolor="rgba(108,99,255,0.20)",
            line=dict(color="#6c63ff", width=2.5),
            marker=dict(color=[_score_color(v) for v in values_c], size=9),
            name="Confidence",
            hovertemplate="%{theta}<br>Score: %{r:.0f}/100<extra></extra>",
        ))
        fig_radar.update_layout(
            **PLOTLY_LAYOUT,
            title="Analysis-Type Confidence Scores  (dotted = 70 threshold)",
            polar=dict(
                bgcolor="#161b27",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    tickvals=[25, 50, 75, 100],
                    tickfont=dict(color="#4a5568", size=10),
                    gridcolor="#2a2f3e", linecolor="#2a2f3e",
                    angle=90,
                ),
                angularaxis=dict(
                    tickfont=dict(color="#c9d1e0", size=13),
                    gridcolor="#2a2f3e", linecolor="#2a2f3e",
                    direction="clockwise",
                ),
            ),
            showlegend=False,
            height=360,
            margin=dict(l=30, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        cols_st = st.columns(len(at_scores))
        for col_w, (k, v) in zip(cols_st, at_scores.items()):
            label = k.replace("_confidence", "").replace("_", " ").title()
            color = _score_color(v)
            with col_w:
                st.markdown(f"""
                <div class="stat-chip" style="min-width:unset;">
                  <div class="stat-value" style="color:{color};font-size:1.3rem;">{v:.1f}</div>
                  <div class="stat-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Component evidence ─────────────────────────────────────────────────
    st.markdown("**Component Evidence**")
    comp_ev = conf.get("component_evidence", {})
    for component, evidence in comp_ev.items():
        score = comp.get(component, 0)
        color = _score_color(score)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.75rem;'
            f'padding:.5rem .85rem;margin:.8rem 0 .3rem 0;'
            f'background:#161b27;border-radius:6px;border:1px solid #2a2f3e;">'
            f'<span style="font-weight:700;color:#e0e0e0;font-size:.92rem;">'
            f'{component.replace("_"," ").title()}</span>'
            f'<span style="font-size:1.1rem;font-weight:800;color:{color};">'
            f'{score:.1f}</span>'
            f'<span style="font-size:.78rem;color:#4a5568;">/100</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        ev_rows = [{"Metric": k, "Value": str(v)} for k, v in evidence.items()]
        st.dataframe(pd.DataFrame(ev_rows), use_container_width=True, hide_index=True)


# ── Tab 8: Dashboard ──────────────────────────────────────────────────────

def render_dashboard(df: pd.DataFrame, profile: dict) -> None:
    """Auto-generated visual dashboard: KPI cards, heatmap, histograms, time series, bar chart."""

    # ── KPI cards ─────────────────────────────────────────────────────────
    total_cells  = df.shape[0] * df.shape[1]
    missing_pct  = (df.isnull().sum().sum() / total_cells * 100) if total_cells else 0
    n_numeric    = df.select_dtypes(include="number").shape[1]
    n_cat        = df.select_dtypes(include="object").shape[1]

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_defs = [
        (kpi_col1, f"{df.shape[0]:,}", "Rows",           "#6c63ff"),
        (kpi_col2, str(df.shape[1]),   "Columns",         "#48b0f7"),
        (kpi_col3, f"{missing_pct:.1f}%", "Missing",     "#fbbf24" if missing_pct > 5 else "#34d399"),
        (kpi_col4, str(n_numeric),     "Numeric cols",    "#a78bfa"),
    ]
    for col_w, val, label, color in kpi_defs:
        with col_w:
            st.markdown(f"""
            <div class="stat-chip" style="width:100%;box-sizing:border-box;min-width:unset;">
              <div class="stat-value" style="color:{color};font-size:1.8rem;">{val}</div>
              <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: correlation heatmap + top-category bar chart ───────────────
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]
    cat_cols   = df.select_dtypes(include="object").columns.tolist()

    has_corr = numeric_df.shape[1] >= 2
    has_cat  = len(cat_cols) > 0

    if has_corr or has_cat:
        left_w   = 3 if has_corr and has_cat else 1
        right_w  = 2 if has_corr and has_cat else 1
        col_left, col_right = st.columns([left_w, right_w]) if has_corr and has_cat else (st.columns(1)[0], None)

        if has_corr:
            with col_left:
                _num = numeric_df.iloc[:, :15] if numeric_df.shape[1] > 15 else numeric_df
                corr = _num.corr(method="pearson").round(2)
                corr_cs = [
                    [0.0, "#f87171"], [0.35, "#2d1f2f"],
                    [0.5, "#161b27"], [0.65, "#1a2744"],
                    [1.0, "#48b0f7"],
                ]
                fig_ch = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    text=[[f"{v:.2f}" for v in row] for row in corr.values],
                    texttemplate="%{text}",
                    textfont=dict(size=9, color="#c9d1e0"),
                    colorscale=corr_cs, zmin=-1, zmax=1,
                    colorbar=dict(
                        title=dict(text="r", font=dict(color="#7a8599", size=10)),
                        tickfont=dict(color="#7a8599"), thickness=10, len=0.8,
                    ),
                    hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.2f}<extra></extra>",
                ))
                fig_ch.update_layout(
                    **PLOTLY_LAYOUT,
                    title="Correlation Heatmap",
                    height=max(280, corr.shape[1] * 32 + 80),
                    xaxis=dict(tickangle=40, tickfont=dict(size=9), side="bottom"),
                    yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
                    margin=dict(l=10, r=60, t=50, b=80),
                )
                st.plotly_chart(fig_ch, use_container_width=True)

        if has_cat and col_right is not None:
            with col_right:
                top_cat = cat_cols[0]
                vc = df[top_cat].value_counts().head(12)
                fig_bar = go.Figure(go.Bar(
                    x=vc.index.tolist(),
                    y=vc.values.tolist(),
                    marker=dict(
                        color=vc.values.tolist(),
                        colorscale=[[0, "#2a1f5e"], [1, "#6c63ff"]],
                        line=dict(width=0),
                    ),
                    hovertemplate="%{x}: %{y:,}<extra></extra>",
                ))
                fig_bar.update_layout(
                    **PLOTLY_LAYOUT,
                    title=f"Top categories — {top_cat}",
                    height=280,
                    xaxis=dict(tickangle=35, tickfont=dict(size=10)),
                    yaxis=dict(showgrid=True, gridcolor="#2a2f3e"),
                    margin=dict(l=10, r=10, t=50, b=80),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        elif has_cat and not has_corr:
            # full-width bar chart
            top_cat = cat_cols[0]
            vc = df[top_cat].value_counts().head(12)
            fig_bar = go.Figure(go.Bar(
                x=vc.index.tolist(), y=vc.values.tolist(),
                marker=dict(color=vc.values.tolist(),
                            colorscale=[[0, "#2a1f5e"], [1, "#6c63ff"]],
                            line=dict(width=0)),
                hovertemplate="%{x}: %{y:,}<extra></extra>",
            ))
            fig_bar.update_layout(
                **PLOTLY_LAYOUT, title=f"Top categories — {top_cat}",
                height=280,
                xaxis=dict(tickangle=35, tickfont=dict(size=10)),
                yaxis=dict(showgrid=True, gridcolor="#2a2f3e"),
                margin=dict(l=10, r=10, t=50, b=80),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    # ── Row 3: distribution histograms ────────────────────────────────────
    num_cols_plot = numeric_df.columns.tolist()[:4]
    if num_cols_plot:
        n_plots = len(num_cols_plot)
        fig_hist = make_subplots(
            rows=1, cols=n_plots,
            subplot_titles=num_cols_plot,
        )
        hist_colors = ["#6c63ff", "#48b0f7", "#a78bfa", "#34d399"]
        for i, col_name in enumerate(num_cols_plot, start=1):
            vals = df[col_name].dropna()
            fig_hist.add_trace(
                go.Histogram(
                    x=vals,
                    marker=dict(color=hist_colors[(i - 1) % 4], line=dict(width=0)),
                    opacity=0.85,
                    nbinsx=30,
                    hovertemplate=f"{col_name}<br>%{{x}}: %{{y:,}}<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=i,
            )
        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            title="Distributions",
            height=260,
            margin=dict(l=20, r=20, t=55, b=30),
            bargap=0.06,
        )
        fig_hist.update_annotations(font=dict(color="#c9d1e0", size=11))
        for ax in fig_hist.layout:
            if ax.startswith("xaxis") or ax.startswith("yaxis"):
                fig_hist.layout[ax].update(
                    gridcolor="#2a2f3e", zeroline=False,
                    tickfont=dict(color="#7a8599", size=9),
                )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Row 4: time-series line chart (conditional) ───────────────────────
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    # also try object cols that parse as dates
    if not date_cols:
        for c in df.select_dtypes(include="object").columns:
            try:
                parsed = pd.to_datetime(df[c], infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df = df.copy()
                    df[c] = parsed
                    date_cols.append(c)
                    break
            except Exception:
                pass

    if date_cols:
        date_col = date_cols[0]
        ts = df[date_col].dt.to_period("M").astype(str)
        counts = ts.value_counts().sort_index()

        fig_ts = go.Figure(go.Scatter(
            x=counts.index.tolist(),
            y=counts.values.tolist(),
            mode="lines+markers",
            line=dict(color="#48b0f7", width=2),
            marker=dict(color="#6c63ff", size=5),
            fill="tozeroy",
            fillcolor="rgba(108,99,255,0.08)",
            hovertemplate="%{x}: %{y:,} records<extra></extra>",
        ))
        fig_ts.update_layout(
            **PLOTLY_LAYOUT,
            title=f"Record count over time — {date_col}",
            height=240,
            xaxis=dict(tickangle=35, tickfont=dict(size=9), showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#2a2f3e", zeroline=False),
            margin=dict(l=20, r=20, t=50, b=60),
        )
        st.plotly_chart(fig_ts, use_container_width=True)


# ---------------------------------------------------------------------------
# Main area — no file
# ---------------------------------------------------------------------------

if st.session_state.df is None:
    st.markdown("""
    <div class="hero">
      <div class="hero-eye">👁</div>
      <div>
        <div class="hero-title">Blind Spot</div>
        <div class="hero-subtitle">Upload any CSV or Excel file → get a full bias &amp; integrity audit before drawing conclusions.</div>
      </div>
    </div>
    <div class="cards-grid">
      <div class="feature-card"><div class="card-icon">🔀</div><div class="card-title">Simpson's Paradox</div><div class="card-desc">Correlations that flip when data is split into subgroups.</div></div>
      <div class="feature-card"><div class="card-icon">👻</div><div class="card-title">Survivorship Bias</div><div class="card-desc">Temporal gaps and status skew that reveal what was silently filtered out.</div></div>
      <div class="feature-card"><div class="card-icon">🧵</div><div class="card-title">Confounder Detection</div><div class="card-desc">Third variables that inflate or fabricate correlations between two others.</div></div>
      <div class="feature-card"><div class="card-icon">🎯</div><div class="card-title">Metric Gaming Risk</div><div class="card-desc">Benford deviation, round-number clusters, and spike patterns per column.</div></div>
      <div class="feature-card"><div class="card-icon">🛡️</div><div class="card-title">Confidence Score</div><div class="card-desc">One A–F grade with a plain-English recommendation for stakeholders.</div></div>
      <div class="feature-card"><div class="card-icon">🔬</div><div class="card-title">Health Profiler</div><div class="card-desc">Nulls, type mismatches, outliers, and high-correlation pairs.</div></div>
    </div>
    <div class="upload-prompt">
      <div class="arrow">←</div>
      Upload a dataset in the sidebar to begin.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main area — file loaded
# ---------------------------------------------------------------------------

else:
    df   = st.session_state.df
    meta = st.session_state.metadata

    # run or auto-run
    auto_run = st.session_state.get("_auto_run_audit", False)
    if auto_run:
        st.session_state._auto_run_audit = False   # consume the flag

    if run_audit or auto_run:
        settings = {
            "max_combinations":    st.session_state.max_combinations,
            "enable_paradox":      st.session_state.enable_paradox,
            "enable_survivorship": st.session_state.enable_survivorship,
            "enable_confounders":  st.session_state.enable_confounders,
            "enable_gaming":       st.session_state.enable_gaming,
        }
        try:
            with st.spinner("Running audit — this may take a moment on large datasets…"):
                st.session_state.audit_results = run_full_audit(df, meta, settings)
        except Exception as _outer_exc:
            st.error(f"Audit failed unexpectedly: {_outer_exc}")
            st.session_state.audit_results = None


    # ── Preview (always visible) ───────────────────────────────────────────
    if st.session_state.audit_results is None:
        fname = st.session_state.file_name or ""
        dtype_s  = meta.get("dtypes_summary", {})
        dup_rows = meta.get("duplicate_row_count", 0)
        mem_mb   = meta.get("memory_usage_mb", 0)

        # ── Header + stat chips ────────────────────────────────────────────
        st.markdown(f"""
        <div class="preview-header">
          <span class="preview-title">Dataset Preview</span>
          <span class="preview-badge">{fname}</span>
        </div>
        <div class="stat-row">
          <div class="stat-chip"><div class="stat-value">{meta["row_count"]:,}</div><div class="stat-label">Rows</div></div>
          <div class="stat-chip"><div class="stat-value">{meta["column_count"]}</div><div class="stat-label">Cols</div></div>
          <div class="stat-chip"><div class="stat-value">{dtype_s.get("numeric",0)}</div><div class="stat-label">Numeric</div></div>
          <div class="stat-chip"><div class="stat-value">{dtype_s.get("categorical",0)}</div><div class="stat-label">Categorical</div></div>
          <div class="stat-chip"><div class="stat-value">{dup_rows:,}</div><div class="stat-label">Duplicates</div></div>
          <div class="stat-chip"><div class="stat-value">{mem_mb:.1f}</div><div class="stat-label">MB</div></div>
        </div>
        """, unsafe_allow_html=True)

        if not meta.get("has_header", True):
            st.warning("No header row detected — column names have been auto-assigned.")
        for warn_msg in (meta or {}).get("warnings", []):
            st.warning(warn_msg)

        # ── Two-column layout: sample rows | column table ──────────────────
        prev_left, prev_right = st.columns([1, 1])

        with prev_left:
            st.caption("Sample rows (first 5)")
            st.dataframe(df.head(5), use_container_width=True, height=195)

        with prev_right:
            rows_html = ""
            for col in df.columns:
                dtype_str  = str(df[col].dtype)
                badge      = _dtype_badge(dtype_str)
                null_count = int(df[col].isnull().sum())
                null_pct   = f"{null_count / len(df) * 100:.1f}%"
                n_unique   = df[col].nunique(dropna=True)
                sample_vals = df[col].dropna().unique()[:2]
                sample_str  = ", ".join(str(v) for v in sample_vals)
                if len(sample_str) > 36:
                    sample_str = sample_str[:33] + "…"
                rows_html += (
                    f"<tr><td><code style='color:#a78bfa;background:none;'>{col}</code></td>"
                    f"<td>{badge}</td>"
                    f"<td>{null_count:,} <span style='color:#4a5568;'>({null_pct})</span></td>"
                    f"<td>{n_unique:,}</td>"
                    f"<td style='color:#7a8599;'>{sample_str}</td></tr>"
                )
            st.markdown(f"""
            <div style="max-height:195px;overflow-y:auto;border:1px solid #2a2f3e;border-radius:8px;">
            <table class="col-table">
              <thead><tr><th>Column</th><th>Type</th><th>Nulls</th><th>Uniq</th><th>Sample</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin:.5rem 0;border-top:1px solid #2a2f3e;'></div>", unsafe_allow_html=True)

        # ── SQL query box ──────────────────────────────────────────────────
        st.markdown("""
        <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.4rem;">
          <span style="font-size:.95rem;font-weight:800;color:#e0e0e0;">SQL Query</span>
          <span style="font-size:.72rem;color:#4a5568;background:#1e2535;border:1px solid #2a2f3e;
                       border-radius:4px;padding:.08rem .45rem;">table: <code style="color:#a78bfa;">data</code></span>
          <span style="font-size:.75rem;color:#4a5568;">
            e.g. <code style="color:#7a8599;">SELECT * FROM data LIMIT 10</code> ·
            <code style="color:#7a8599;">SELECT region, AVG(salary) FROM data GROUP BY region</code>
          </span>
        </div>
        """, unsafe_allow_html=True)

        sql_col, btn_col = st.columns([5, 1])
        with sql_col:
            sql_query = st.text_area(
                label="sql_input",
                value="SELECT * FROM data LIMIT 10",
                height=80,
                label_visibility="collapsed",
                key="sql_query_input",
            )
        with btn_col:
            st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
            run_sql = st.button("Run Query", use_container_width=True, key="run_sql_btn")

        if run_sql and sql_query.strip():
            try:
                con = duckdb.connect()
                con.register("data", df)
                result_df = con.execute(sql_query.strip()).df()
                con.close()
                st.caption(f"{len(result_df):,} row(s) returned")
                st.dataframe(result_df, use_container_width=True, height=min(320, 36 * len(result_df) + 38))
            except Exception as e:
                st.error(f"Query error: {e}")

        st.markdown("""<div class="upload-prompt">
          <div class="arrow">←</div>
          Click <strong>Run Full Audit</strong> in the sidebar to analyse this dataset.
        </div>""", unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────
    else:
        results = st.session_state.audit_results

        # module warnings
        for w in results.get("_warnings", []):
            st.warning(w)

        conf = results.get("confidence", {})
        score = conf.get("overall_score", 0)
        grade = conf.get("overall_grade", "?")
        fg, bg = _grade_style(grade)
        color  = _score_color(score)

        # score banner
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;padding:.6rem 1rem;
                    background:#161b27;border:1px solid #2a2f3e;border-radius:10px;margin-bottom:.8rem;">
          <div style="font-size:2rem;font-weight:900;color:{color};line-height:1;">{score:.1f}</div>
          <div>
            <div style="font-size:.67rem;color:#7a8599;text-transform:uppercase;letter-spacing:.1em;">Data Confidence Score</div>
            <span style="background:{bg};color:{fg};padding:.08rem .6rem;border-radius:20px;font-weight:800;font-size:.88rem;">
              Grade {grade}
            </span>
          </div>
          <div style="flex:1;"></div>
          <div style="font-size:.75rem;color:#7a8599;">{st.session_state.file_name}</div>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        tabs = st.tabs([
            "Overview", "Data Profile", "Simpson's Paradox",
            "Survivorship Bias", "Confounders",
            "Metric Gaming", "Confidence Detail", "Dashboard",
        ])

        with tabs[0]:
            render_overview(conf, st.session_state.confidence_threshold,
                            blind_spots=results.get("blind_spots", []))
        with tabs[1]:
            render_profile(df, results.get("profile", {}))
        with tabs[2]:
            render_paradox(results.get("paradox", []), df=df)
        with tabs[3]:
            render_survivorship(results.get("survivorship", []), df=df)
        with tabs[4]:
            render_confounders(results.get("confounders", []))
        with tabs[5]:
            render_gaming(results.get("gaming", []))
        with tabs[6]:
            render_confidence_detail(conf)
        with tabs[7]:
            render_dashboard(df, results.get("profile", {}))

# ---------------------------------------------------------------------------
# Footer — shown on every page state
# ---------------------------------------------------------------------------

st.markdown("""
<div style="
    margin-top: 1rem;
    padding: .5rem 0 .4rem 0;
    border-top: 1px solid #2a2f3e;
    text-align: center;
    font-size: .75rem;
    color: #4a5568;
    line-height: 1.6;
">
  Made by <strong style="color:#c9d1e0;">Konark</strong>
  &nbsp;·&nbsp; Built with <strong style="color:#c9d1e0;">Claude</strong> (Anthropic)
  &nbsp;
  <a href="https://github.com/konarkambad/blindspot"
     target="_blank"
     style="color:#c9d1e0; text-decoration:none; vertical-align:middle;"
     title="View on GitHub">
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
         fill="currentColor" style="vertical-align:middle;">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577
               0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61-.546-1.385-1.333-1.754
               -1.333-1.754-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237
               1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305
               -5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524
               .117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005
               2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118
               3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823
               1.102.823 2.222 0 1.606-.015 2.896-.015 3.286 0 .322.216.694.825.576C20.565
               21.795 24 17.295 24 12c0-6.63-5.37-12-12-12z"/>
    </svg>
  </a>
</div>
""", unsafe_allow_html=True)
