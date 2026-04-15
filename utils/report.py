"""
report.py — HTML audit report generator.

Produces a self-contained, single-file HTML document that can be opened in
any browser and printed to PDF via Ctrl+P → Save as PDF. All CSS is inline
or in a <style> block so the file has zero external dependencies.

This report IS the portfolio piece — it must look polished without the app.
"""

from datetime import datetime


# ---------------------------------------------------------------------------
# Colour palette (professional dark-header / light-body)
# ---------------------------------------------------------------------------

C = {
    "bg":           "#ffffff",
    "surface":      "#f8f9fc",
    "border":       "#e2e8f0",
    "header_bg":    "#0f172a",
    "header_text":  "#f1f5f9",
    "accent":       "#6c63ff",
    "accent_light": "#ede9fe",
    "text":         "#1e293b",
    "muted":        "#64748b",
    "green":        "#15803d",
    "green_bg":     "#dcfce7",
    "amber":        "#b45309",
    "amber_bg":     "#fef3c7",
    "red":          "#b91c1c",
    "red_bg":       "#fee2e2",
    "blue":         "#1d4ed8",
    "blue_bg":      "#dbeafe",
}


# ---------------------------------------------------------------------------
# HTML building blocks
# ---------------------------------------------------------------------------

def _style() -> str:
    return f"""
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: {C['text']};
    background: {C['bg']};
    padding: 0;
  }}

  /* ── Cover header ── */
  .cover {{
    background: {C['header_bg']};
    color: {C['header_text']};
    padding: 3rem 3.5rem 2.5rem 3.5rem;
    page-break-after: avoid;
  }}
  .cover .eyebrow {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #94a3b8;
    margin-bottom: 0.6rem;
  }}
  .cover h1 {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.35rem;
    color: #f1f5f9;
  }}
  .cover .subtitle {{
    font-size: 0.9rem;
    color: #94a3b8;
    margin-bottom: 1.8rem;
  }}
  .cover .meta-row {{
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
    border-top: 1px solid #1e293b;
    padding-top: 1.2rem;
    margin-top: 0.5rem;
  }}
  .cover .meta-item {{ }}
  .cover .meta-label {{
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #64748b;
    margin-bottom: 0.15rem;
  }}
  .cover .meta-value {{
    font-size: 0.92rem;
    font-weight: 600;
    color: #e2e8f0;
  }}

  /* ── Body content wrapper ── */
  .body {{ padding: 2.5rem 3.5rem; max-width: 960px; margin: 0 auto; }}

  /* ── Score hero ── */
  .score-hero {{
    display: flex;
    align-items: center;
    gap: 2rem;
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1.8rem;
  }}
  .score-number {{
    font-size: 4rem;
    font-weight: 900;
    line-height: 1;
  }}
  .score-grade {{
    font-size: 1rem;
    font-weight: 700;
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    display: inline-block;
    margin-top: 0.3rem;
  }}
  .score-label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {C['muted']};
    margin-top: 0.15rem;
  }}
  .score-divider {{
    width: 1px;
    height: 60px;
    background: {C['border']};
    flex-shrink: 0;
  }}
  .score-right {{ flex: 1; }}
  .score-right p {{
    font-size: 0.88rem;
    color: {C['muted']};
    line-height: 1.55;
  }}

  /* ── Section ── */
  .section {{
    margin-bottom: 2.8rem;
    page-break-inside: avoid;
  }}
  .section-title {{
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: {C['muted']};
    border-bottom: 2px solid {C['border']};
    padding-bottom: 0.45rem;
    margin-bottom: 1.1rem;
  }}
  h2.section-title {{ font-size: 0.68rem; }}

  /* ── Component score bars ── */
  .comp-row {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.55rem;
  }}
  .comp-label {{
    width: 160px;
    font-size: 0.82rem;
    font-weight: 500;
    color: {C['text']};
    flex-shrink: 0;
  }}
  .comp-bar-track {{
    flex: 1;
    background: {C['border']};
    border-radius: 4px;
    height: 8px;
    overflow: hidden;
  }}
  .comp-bar-fill {{
    height: 100%;
    border-radius: 4px;
  }}
  .comp-score {{
    width: 36px;
    font-size: 0.82rem;
    font-weight: 700;
    text-align: right;
    flex-shrink: 0;
  }}

  /* ── Risk cards ── */
  .risk-card {{
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.65rem;
    font-size: 0.87rem;
    line-height: 1.5;
  }}
  .risk-high   {{ background:{C['red_bg']};   border-color:{C['red']};   color:{C['red']};   }}
  .risk-medium {{ background:{C['amber_bg']}; border-color:{C['amber']}; color:{C['amber']}; }}
  .risk-low    {{ background:{C['green_bg']}; border-color:{C['green']}; color:{C['green']}; }}
  .risk-neutral{{ background:{C['surface']};  border-color:{C['border']}; color:{C['text']}; }}

  /* ── Blind spot cards ── */
  .bs-card {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    margin-bottom: 0.9rem;
    overflow: hidden;
  }}
  .bs-card-header {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.7rem 1rem;
    background: {C['surface']};
    border-bottom: 1px solid {C['border']};
  }}
  .bs-badge {{
    font-size: 0.65rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.12rem 0.55rem;
    border-radius: 4px;
    flex-shrink: 0;
  }}
  .bs-badge-high   {{ background:{C['red_bg']};   color:{C['red']};   }}
  .bs-badge-medium {{ background:{C['amber_bg']}; color:{C['amber']}; }}
  .bs-badge-low    {{ background:{C['green_bg']}; color:{C['green']}; }}
  .bs-desc  {{ font-size: 0.87rem; font-weight: 600; color: {C['text']}; }}
  .bs-body  {{ padding: 0.7rem 1rem; }}
  .bs-missing {{ font-size: 0.75rem; color: {C['muted']}; text-transform: uppercase;
                 letter-spacing: 0.07em; margin-bottom: 0.35rem; }}
  .bs-source {{ font-size: 0.83rem; color: {C['text']}; line-height: 1.5; }}

  /* ── Tables ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
    margin-bottom: 1rem;
  }}
  thead tr {{ background: {C['header_bg']}; color: {C['header_text']}; }}
  thead th {{
    padding: 0.55rem 0.85rem;
    text-align: left;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
  }}
  tbody tr:nth-child(even) {{ background: {C['surface']}; }}
  tbody td {{
    padding: 0.5rem 0.85rem;
    border-bottom: 1px solid {C['border']};
    vertical-align: top;
  }}
  .tag {{
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 4px;
    font-size: 0.73rem;
    font-weight: 600;
  }}
  .tag-high   {{ background:{C['red_bg']};   color:{C['red']};   }}
  .tag-medium {{ background:{C['amber_bg']}; color:{C['amber']}; }}
  .tag-low    {{ background:{C['green_bg']}; color:{C['green']}; }}

  /* ── Info boxes ── */
  .info-box {{
    background: {C['blue_bg']};
    border: 1px solid #93c5fd;
    border-radius: 7px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: {C['blue']};
    margin-bottom: 1rem;
  }}
  .success-box {{
    background: {C['green_bg']};
    border: 1px solid #86efac;
    border-radius: 7px;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: {C['green']};
    margin-bottom: 1rem;
  }}

  /* ── Recommendation box ── */
  .rec-box {{
    background: {C['accent_light']};
    border: 1px solid #c4b5fd;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    font-size: 0.88rem;
    color: #3730a3;
    line-height: 1.65;
    margin-top: 1rem;
  }}

  /* ── Footer ── */
  .footer {{
    background: {C['header_bg']};
    color: #475569;
    text-align: center;
    padding: 1.2rem;
    font-size: 0.75rem;
    margin-top: 3rem;
  }}
  .footer a {{ color: #6c63ff; text-decoration: none; }}

  /* ── Print ── */
  @media print {{
    .cover {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    thead tr {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .section {{ page-break-inside: avoid; }}
  }}
</style>
"""


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    if score >= 75: return C["green"]
    if score >= 55: return C["amber"]
    return C["red"]

def _grade_colors(grade: str) -> tuple:
    """Returns (text_color, bg_color)."""
    return {
        "A": (C["green"],  C["green_bg"]),
        "B": (C["blue"],   C["blue_bg"]),
        "C": (C["amber"],  C["amber_bg"]),
        "D": ("#9a3412",   "#ffedd5"),
        "F": (C["red"],    C["red_bg"]),
    }.get(grade, (C["muted"], C["surface"]))

def _comp_bar_color(score: float) -> str:
    if score >= 75: return C["green"]
    if score >= 55: return C["amber"]
    return C["red"]

def _risk_tag(level: str) -> str:
    return f'<span class="tag tag-{level}">{level.upper()}</span>'


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _cover(file_name: str, row_count: int, col_count: int) -> str:
    now = datetime.now().strftime("%B %d, %Y  %H:%M")
    return f"""
<div class="cover">
  <div class="eyebrow">Data Integrity Audit Report</div>
  <h1>👁 Blind Spot</h1>
  <div class="subtitle">Analytical Bias &amp; Data Integrity Auditor</div>
  <div class="meta-row">
    <div class="meta-item">
      <div class="meta-label">Dataset</div>
      <div class="meta-value">{file_name}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Rows × Columns</div>
      <div class="meta-value">{row_count:,} × {col_count}</div>
    </div>
    <div class="meta-item">
      <div class="meta-label">Generated</div>
      <div class="meta-value">{now}</div>
    </div>
  </div>
</div>
"""


def _executive_summary(confidence: dict) -> str:
    score  = confidence.get("overall_score", 0)
    grade  = confidence.get("overall_grade", "?")
    risks  = confidence.get("top_risks", [])
    rec    = confidence.get("recommendation", "")
    fg, bg = _grade_colors(grade)
    color  = _score_color(score)

    risks_html = "".join(
        f'<div class="risk-card risk-neutral">⚠&nbsp; {r}</div>'
        for r in risks
    ) if risks else '<div class="success-box">No major risk factors identified.</div>'

    rec_html = f'<div class="rec-box"><strong>Recommendation:</strong> {rec}</div>' if rec else ""

    return f"""
<div class="section">
  <h2 class="section-title">Executive Summary</h2>
  <div class="score-hero">
    <div>
      <div class="score-number" style="color:{color};">{score:.0f}</div>
      <div class="score-grade" style="color:{fg};background:{bg};">Grade {grade}</div>
      <div class="score-label">Data Confidence Score</div>
    </div>
    <div class="score-divider"></div>
    <div class="score-right">
      <p>This score synthesises completeness, consistency, bias risk, confounding,
      and metric gaming into a single 0–100 rating. Scores above 75 are suitable
      for decision-making. Scores below 55 require remediation before analysis.</p>
    </div>
  </div>
  <strong style="font-size:.82rem;text-transform:uppercase;letter-spacing:.1em;color:{C['muted']};">Top Risk Factors</strong>
  <div style="margin-top:.6rem;">{risks_html}</div>
  {rec_html}
</div>
"""


def _blind_spots_section(blind_spots: list) -> str:
    if not blind_spots:
        return f"""
<div class="section">
  <h2 class="section-title">Blind Spots — What This Data Cannot Tell You</h2>
  <div class="success-box">No structural blind spots detected.</div>
</div>"""

    cards = []
    for spot in blind_spots:
        impact  = spot.get("impact_level", "low")
        desc    = spot.get("blind_spot_description", "")
        missing = spot.get("missing_data_type", "")
        source  = spot.get("suggested_source", "")
        cards.append(f"""
<div class="bs-card">
  <div class="bs-card-header">
    <span class="bs-badge bs-badge-{impact}">{impact} impact</span>
    <span class="bs-desc">{desc}</span>
  </div>
  <div class="bs-body">
    <div class="bs-missing">Missing: {missing}</div>
    <div class="bs-source">💡 {source}</div>
  </div>
</div>""")

    return f"""
<div class="section">
  <h2 class="section-title">Blind Spots — What This Data Cannot Tell You</h2>
  {''.join(cards)}
</div>"""


def _profile_section(profile: dict, metadata: dict) -> str:
    dtype_s  = metadata.get("dtypes_summary", {})
    rows_html = f"""
<table>
  <thead><tr>
    <th>Metric</th><th>Value</th>
  </tr></thead>
  <tbody>
    <tr><td>Rows</td><td>{metadata.get('row_count', '—'):,}</td></tr>
    <tr><td>Columns</td><td>{metadata.get('column_count', '—')}</td></tr>
    <tr><td>Numeric columns</td><td>{dtype_s.get('numeric', 0)}</td></tr>
    <tr><td>Categorical columns</td><td>{dtype_s.get('categorical', 0)}</td></tr>
    <tr><td>Datetime columns</td><td>{dtype_s.get('datetime', 0)}</td></tr>
    <tr><td>Boolean columns</td><td>{dtype_s.get('boolean', 0)}</td></tr>
    <tr><td>Memory usage</td><td>{metadata.get('memory_usage_mb', 0):.2f} MB</td></tr>
    <tr><td>Duplicate rows</td><td>{metadata.get('duplicate_row_count', 0):,}</td></tr>
    <tr><td>Header detected</td><td>{'Yes' if metadata.get('has_header', True) else 'No'}</td></tr>
  </tbody>
</table>"""

    # Missing summary
    missing = profile.get("missing_summary", {})
    cols_with_nulls = {c: v for c, v in missing.items() if v.get("null_count", 0) > 0}
    if cols_with_nulls:
        miss_rows = "".join(
            f"<tr><td><code>{c}</code></td>"
            f"<td>{v['null_count']:,}</td>"
            f"<td>{v['pct_missing']:.1f}%</td>"
            f"<td>{v.get('missing_pattern','—')}</td></tr>"
            for c, v in sorted(cols_with_nulls.items(),
                                key=lambda x: -x[1]['pct_missing'])
        )
        miss_table = f"""
<table>
  <thead><tr><th>Column</th><th>Null count</th><th>% Missing</th><th>Pattern</th></tr></thead>
  <tbody>{miss_rows}</tbody>
</table>"""
    else:
        miss_table = '<div class="success-box">No missing values detected.</div>'

    # Type issues
    type_issues = profile.get("type_issues", {})
    if type_issues:
        ti_rows = "".join(
            f"<tr><td><code>{c}</code></td><td>{issue}</td></tr>"
            for c, issue in type_issues.items()
        )
        ti_table = f"""
<table>
  <thead><tr><th>Column</th><th>Issue</th></tr></thead>
  <tbody>{ti_rows}</tbody>
</table>"""
    else:
        ti_table = '<div class="success-box">No dtype mismatches detected.</div>'

    # High correlations
    pairs = profile.get("high_correlation_pairs", [])
    if pairs:
        p_rows = "".join(
            "<tr><td><code>{}</code></td><td><code>{}</code></td>"
            "<td style='font-weight:700;color:{};'>{:.3f}</td></tr>".format(
                p["col_a"], p["col_b"],
                _score_color(abs(p["correlation"]) * 100),
                p["correlation"],
            )
            for p in pairs
        )
        pairs_table = f"""
<table>
  <thead><tr><th>Column A</th><th>Column B</th><th>Pearson r</th></tr></thead>
  <tbody>{p_rows}</tbody>
</table>"""
    else:
        pairs_table = '<div class="success-box">No high-correlation pairs (|r| &gt; 0.85) detected.</div>'

    # Constant & zero-variance
    const = profile.get("constant_columns", [])
    zv    = profile.get("zero_variance_columns", [])
    extras = ""
    if const:
        extras += f"<p style='font-size:.84rem;margin-bottom:.4rem;'><strong>Constant columns:</strong> {', '.join(f'<code>{c}</code>' for c in const)}</p>"
    if zv:
        extras += f"<p style='font-size:.84rem;'><strong>Near-zero variance columns:</strong> {', '.join(f'<code>{c}</code>' for c in zv)}</p>"

    return f"""
<div class="section">
  <h2 class="section-title">Data Profile</h2>
  <p style="font-size:.84rem;font-weight:600;margin-bottom:.5rem;">Dataset Summary</p>
  {rows_html}
  <p style="font-size:.84rem;font-weight:600;margin:.9rem 0 .5rem 0;">Missing Data</p>
  {miss_table}
  <p style="font-size:.84rem;font-weight:600;margin:.9rem 0 .5rem 0;">Type Issues</p>
  {ti_table}
  <p style="font-size:.84rem;font-weight:600;margin:.9rem 0 .5rem 0;">High Correlation Pairs</p>
  {pairs_table}
  {extras}
</div>"""


def _paradox_section(paradox: list) -> str:
    if not paradox:
        return f"""
<div class="section">
  <h2 class="section-title">Simpson's Paradox</h2>
  <div class="success-box">No Simpson's Paradox cases detected.</div>
</div>"""

    rows = "".join(f"""
<tr>
  <td><code>{c['x_column']}</code></td>
  <td><code>{c['y_column']}</code></td>
  <td><code>{c['grouping_column']}</code></td>
  <td style="font-weight:700;">{c['overall_correlation']:.3f}</td>
  <td>{c.get('reversal_type','—').replace('_',' ')}</td>
  <td style="font-weight:700;color:{_score_color(100 - c.get('severity_score',0)*100)};">
    {c.get('severity_score',0):.3f}
  </td>
</tr>""" for c in paradox)

    desc_blocks = "".join(f"""
<div class="risk-card risk-neutral" style="margin-bottom:.5rem;">
  <strong>{c['x_column']} × {c['y_column']}</strong>
  grouped by <strong>{c['grouping_column']}</strong> —
  overall r = {c['overall_correlation']:.3f},
  reversal: {c.get('reversal_type','').replace('_',' ')},
  severity {c.get('severity_score',0):.3f}.
  Subgroups: {
    ', '.join(
      f"{k}: {v:.3f}" for k, v in c.get('subgroup_correlations', {}).items()
      if v is not None
    )
  }.
</div>""" for c in paradox)

    return f"""
<div class="section">
  <h2 class="section-title">Simpson's Paradox ({len(paradox)} case(s))</h2>
  <table>
    <thead><tr>
      <th>X</th><th>Y</th><th>Group by</th>
      <th>Overall r</th><th>Reversal</th><th>Severity</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <div style="margin-top:.8rem;">{desc_blocks}</div>
</div>"""


def _survivorship_section(survivorship: list) -> str:
    if not survivorship:
        return f"""
<div class="section">
  <h2 class="section-title">Survivorship Bias</h2>
  <div class="success-box">No survivorship bias signals detected.</div>
</div>"""

    cards = "".join(f"""
<div class="risk-card risk-{f.get('risk_level','low')}" style="color:{C['text']};background:{'#fee2e2' if f.get('risk_level')=='high' else '#fef3c7' if f.get('risk_level')=='medium' else '#dcfce7'};">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
    <div>
      {_risk_tag(f.get('risk_level','low'))}
      <strong style="margin-left:.5rem;">{f.get('finding_type','').replace('_',' ').title()}</strong>
      — <em>{', '.join(f.get('affected_columns', []))}</em>
    </div>
  </div>
  <p style="margin-top:.4rem;font-size:.84rem;color:{C['text']};">{f.get('description','')}</p>
  {'<p style="margin-top:.3rem;font-size:.78rem;color:' + C['muted'] + ';"><strong>Evidence:</strong> ' + str(f.get('evidence',{})) + '</p>' if f.get('evidence') else ''}
</div>""" for f in survivorship)

    return f"""
<div class="section">
  <h2 class="section-title">Survivorship Bias ({len(survivorship)} finding(s))</h2>
  {cards}
</div>"""


def _confounders_section(confounders: list) -> str:
    if not confounders:
        return f"""
<div class="section">
  <h2 class="section-title">Confounding Variables</h2>
  <div class="success-box">No confounding variables detected.</div>
</div>"""

    numeric = [r for r in confounders if r.get("type") == "numeric_confounder"]
    cat     = [r for r in confounders if r.get("type") == "categorical_association"]

    num_table = ""
    if numeric:
        rows = "".join(f"""
<tr>
  <td><code>{r['variable_x']}</code></td>
  <td><code>{r['variable_y']}</code></td>
  <td><code>{r.get('likely_confounder','—')}</code></td>
  <td>{r.get('raw_correlation',0):.3f}</td>
  <td>{r.get('partial_correlation',0):.3f}</td>
  <td style="font-weight:700;color:{C['red']};">{r.get('correlation_drop_pct',0):.1f}%</td>
</tr>""" for r in numeric)
        num_table = f"""
<p style="font-size:.84rem;font-weight:600;margin-bottom:.5rem;">Numeric Confounders</p>
<table>
  <thead><tr><th>X</th><th>Y</th><th>Confounder Z</th><th>Raw r</th><th>Partial r</th><th>Drop %</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""

    cat_table = ""
    if cat:
        rows = "".join(f"""
<tr>
  <td><code>{r['variable_x']}</code></td>
  <td><code>{r['variable_y']}</code></td>
  <td>{r.get('chi2_statistic',0):.2f}</td>
  <td>{r.get('p_value',0):.4f}</td>
  <td>{r.get('cramers_v',0):.3f}</td>
</tr>""" for r in cat)
        cat_table = f"""
<p style="font-size:.84rem;font-weight:600;margin:.9rem 0 .5rem 0;">Categorical Associations</p>
<table>
  <thead><tr><th>Variable A</th><th>Variable B</th><th>Chi² stat</th><th>p-value</th><th>Cramér's V</th></tr></thead>
  <tbody>{rows}</tbody>
</table>"""

    return f"""
<div class="section">
  <h2 class="section-title">Confounding Variables ({len(confounders)} finding(s))</h2>
  {num_table}{cat_table}
</div>"""


def _gaming_section(gaming: list) -> str:
    if not gaming:
        return f"""
<div class="section">
  <h2 class="section-title">Metric Gaming Risk</h2>
  <div class="info-box">No numeric columns available for gaming risk analysis.</div>
</div>"""

    rows = "".join(f"""
<tr>
  <td><code>{r['column_name']}</code></td>
  <td style="font-weight:700;color:{_score_color(100 - r['gaming_risk_score'])};">{r['gaming_risk_score']:.0f}</td>
  <td>{_risk_tag(r.get('risk_level','low'))}</td>
  <td>{r['breakdown'].get('concentration',{}).get('sub_score',0):.1f}</td>
  <td>{r['breakdown'].get('round_numbers',{}).get('sub_score',0):.1f}</td>
  <td>{r['breakdown'].get('benford',{}).get('sub_score',0):.1f}</td>
  <td>{r['breakdown'].get('spike',{}).get('sub_score',0):.1f}</td>
  <td>{r['breakdown'].get('weekend_pattern',{}).get('sub_score',0):.1f}</td>
</tr>""" for r in gaming)

    interps = "".join(
        f'<p style="font-size:.83rem;margin-bottom:.3rem;"><code>{r["column_name"]}</code> — {r.get("interpretation","")}</p>'
        for r in gaming if r.get("risk_level") in ("high", "medium")
    )

    return f"""
<div class="section">
  <h2 class="section-title">Metric Gaming Risk</h2>
  <table>
    <thead><tr>
      <th>Column</th><th>Score /100</th><th>Risk</th>
      <th>Conc.</th><th>Round#</th><th>Benford</th><th>Spike</th><th>Weekend</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
  {('<div style="margin-top:.7rem;">' + interps + '</div>') if interps else ''}
</div>"""


def _confidence_section(confidence: dict) -> str:
    comp   = confidence.get("component_scores", {})
    at     = confidence.get("analysis_type_scores", {})

    # Component bars
    bars = "".join(f"""
<div class="comp-row">
  <div class="comp-label">{k.replace('_',' ').title()}</div>
  <div class="comp-bar-track">
    <div class="comp-bar-fill" style="width:{v}%;background:{_comp_bar_color(v)};"></div>
  </div>
  <div class="comp-score" style="color:{_comp_bar_color(v)};">{v:.0f}</div>
</div>""" for k, v in comp.items())

    # Analysis-type scores table
    at_rows = "".join(f"""
<tr>
  <td>{k.replace('_confidence','').replace('_',' ').title()}</td>
  <td style="font-weight:700;color:{_score_color(v)};">{v:.0f} / 100</td>
  <td>{_risk_tag('high' if v < 55 else 'medium' if v < 75 else 'low')}</td>
</tr>""" for k, v in at.items())

    return f"""
<div class="section">
  <h2 class="section-title">Confidence Breakdown</h2>
  <p style="font-size:.84rem;font-weight:600;margin-bottom:.6rem;">Component Scores</p>
  {bars}
  <p style="font-size:.84rem;font-weight:600;margin:1rem 0 .5rem 0;">Analysis-Type Confidence</p>
  <table>
    <thead><tr><th>Analysis Type</th><th>Score</th><th>Risk</th></tr></thead>
    <tbody>{at_rows}</tbody>
  </table>
</div>"""


def _footer() -> str:
    return f"""
<div class="footer">
  Generated by <strong>Blind Spot — Data Integrity Auditor</strong> &nbsp;|&nbsp;
  <a href="https://github.com/YOUR_USERNAME/blindspot">github.com/YOUR_USERNAME/blindspot</a>
  &nbsp;|&nbsp; {datetime.now().strftime("%Y")}
</div>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_html_report(
    df,
    metadata: dict,
    results: dict,
    file_name: str = "dataset",
) -> str:
    """
    Generate a self-contained HTML audit report from all analysis results.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset (used for shape metadata only).
    metadata : dict
        Output of loader.load_data() — the metadata dict.
    results : dict
        The full results dict from run_full_audit(), containing keys:
        profile, paradox, survivorship, confounders, gaming, confidence, blind_spots.
    file_name : str
        Original filename for display in the report header.

    Returns
    -------
    str — complete HTML document as a string.
    """
    row_count = len(df) if df is not None else metadata.get("row_count", 0)
    col_count = len(df.columns) if df is not None else metadata.get("column_count", 0)

    confidence   = results.get("confidence",   {})
    profile      = results.get("profile",      {})
    paradox      = results.get("paradox",      [])
    survivorship = results.get("survivorship", [])
    confounders  = results.get("confounders",  [])
    gaming       = results.get("gaming",       [])
    blind_spots  = results.get("blind_spots",  [])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Blind Spot Audit Report — {file_name}</title>
  {_style()}
</head>
<body>
  {_cover(file_name, row_count, col_count)}
  <div class="body">
    {_executive_summary(confidence)}
    {_blind_spots_section(blind_spots)}
    {_profile_section(profile, metadata)}
    {_paradox_section(paradox)}
    {_survivorship_section(survivorship)}
    {_confounders_section(confounders)}
    {_gaming_section(gaming)}
    {_confidence_section(confidence)}
  </div>
  {_footer()}
</body>
</html>"""

    return html
