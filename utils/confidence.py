"""
confidence.py — Data Confidence Score aggregator.

Takes the outputs of every prior analysis module and synthesises them into
a single interpretable score that a non-technical stakeholder can read in
two seconds, alongside the granular breakdown an analyst needs to act on it.

Score architecture:
  Overall (0–100)
    ├── Completeness   25 %  ← missing data
    ├── Consistency    20 %  ← type issues, outliers, constants
    ├── Bias Risk      25 %  ← survivorship + Simpson's Paradox
    ├── Confounding    15 %  ← confounder findings
    └── Gaming Risk    15 %  ← metric gaming scores

Per-analysis-type scores flag which *kinds* of conclusions are most at risk.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------

GRADE_THRESHOLDS = [
    (90, "A"),
    (75, "B"),
    (60, "C"),
    (45, "D"),
    (  0, "F"),
]

# ---------------------------------------------------------------------------
# Weight map (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS = {
    "completeness":  0.25,
    "consistency":   0.20,
    "bias_risk":     0.25,
    "confounding":   0.15,
    "gaming":        0.15,
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    for threshold, letter in GRADE_THRESHOLDS:
        if score >= threshold:
            return letter
    return "F"


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(np.clip(value, lo, hi))


def _penalty(*penalties: float) -> float:
    """
    Combine independent penalty fractions multiplicatively so that multiple
    moderate penalties stack without instantly collapsing the score to zero.
    Each penalty is in [0, 1] where 0 = no penalty, 1 = full deduction.
    Returns the total fractional reduction to apply to a 100-point component.
    """
    survival = 1.0
    for p in penalties:
        survival *= (1.0 - float(np.clip(p, 0, 1)))
    return 1.0 - survival          # total fractional loss


# ---------------------------------------------------------------------------
# Component scorers  (each returns a float in [0, 100])
# ---------------------------------------------------------------------------

def _score_completeness(profile_dict: dict) -> tuple[float, dict]:
    """
    Penalise missing data.  A dataset with zero nulls scores 100.
    Each column's missing fraction contributes proportionally.
    Columns with >50 % missing are weighted double (they are functionally absent).
    """
    missing = profile_dict.get("missing_summary", {})
    if not missing:
        return 100.0, {"avg_missing_pct": 0.0, "columns_over_50pct": 0}

    fractions    = [v["pct_missing"] / 100.0 for v in missing.values()]
    w            = [2 if f > 0.50 else 1 for f in fractions]
    avg_missing  = sum(f * wi for f, wi in zip(fractions, w)) / sum(w) if w else 0.0
    score        = _clamp(100 * (1 - avg_missing) ** 1.5)

    return score, {
        "avg_missing_pct":       round(avg_missing * 100, 2),
        "columns_over_50pct":    sum(1 for f in fractions if f > 0.50),
        "total_columns_checked": len(fractions),
    }


def _score_consistency(profile_dict: dict, df: pd.DataFrame) -> tuple[float, dict]:
    """
    Penalise structural inconsistencies:
      - Type mismatches (column dtype doesn't match content)
      - Outlier prevalence (average % of outlier rows per numeric column)
      - Constant / zero-variance columns (carry no information)
    """
    type_issues    = profile_dict.get("type_issues", {})
    outlier_flags  = profile_dict.get("outlier_flags", {})
    constant_cols  = profile_dict.get("constant_columns", [])
    total_cols     = len(df.columns) if df is not None else 1

    # type penalty
    type_issue_rate = len(type_issues) / max(total_cols, 1)
    type_penalty    = min(type_issue_rate * 2, 1.0)

    # outlier penalty
    total_rows = len(df) if df is not None else 1
    outlier_rates = []
    for col_ev in outlier_flags.values():
        iqr_count = col_ev.get("iqr_outliers")
        if iqr_count is not None:
            outlier_rates.append(iqr_count / max(total_rows, 1))
    avg_outlier_rate = float(np.mean(outlier_rates)) if outlier_rates else 0.0
    outlier_penalty  = min(avg_outlier_rate * 5, 1.0)

    # constant penalty
    constant_rate    = len(constant_cols) / max(total_cols, 1)
    constant_penalty = min(constant_rate * 3, 1.0)

    total_penalty = _penalty(type_penalty, outlier_penalty, constant_penalty)
    score = _clamp(100 * (1 - total_penalty))

    return score, {
        "type_issue_count": len(type_issues),
        "avg_outlier_rate_pct": round(avg_outlier_rate * 100, 2),
        "constant_column_count": len(constant_cols),
    }


def _score_bias_risk(
    survivorship_results: list,
    paradox_results: list,
) -> tuple[float, dict]:
    """
    Penalise evidence of survivorship bias and Simpson's Paradox.

    Survivorship:  high-risk findings carry more weight than low-risk ones.
    Simpson's:     each paradox case is weighted by its severity_score (0–1).
    """
    # survivorship penalty
    risk_weights = {"high": 1.0, "medium": 0.5, "low": 0.2}
    surv_weighted = sum(
        risk_weights.get(f.get("risk_level", "low"), 0.2)
        for f in survivorship_results
    )
    surv_penalty = min(surv_weighted / 6.0, 0.75)

    # paradox penalty
    if paradox_results:
        avg_severity  = float(np.mean([r.get("severity_score", 0) for r in paradox_results]))
        n_paradoxes   = len(paradox_results)
        # scale by count up to 5
        paradox_penalty = min(avg_severity * (n_paradoxes / 5.0), 1.0)
    else:
        avg_severity    = 0.0
        n_paradoxes     = 0
        paradox_penalty = 0.0

    total_penalty = _penalty(surv_penalty, paradox_penalty)
    score = _clamp(100 * (1 - total_penalty))

    return score, {
        "survivorship_findings": len(survivorship_results),
        "high_risk_survivorship": sum(
            1 for f in survivorship_results if f.get("risk_level") == "high"
        ),
        "simpsons_paradox_cases": n_paradoxes,
        "avg_paradox_severity": round(avg_severity, 4),
    }


def _score_confounding(confounder_results: list) -> tuple[float, dict]:
    """
    Penalise the presence of confounding variables.

    Numeric confounders are weighted by their correlation_drop_pct (a 90 % drop
    is far more serious than a 41 % drop). Categorical associations are weighted
    uniformly at 0.5 (structural concern but less directly quantified).
    """
    numeric_conf = [
        r for r in confounder_results if r.get("type") == "numeric_confounder"
    ]
    cat_conf = [
        r for r in confounder_results if r.get("type") == "categorical_association"
    ]

    # normalise to [0, 1]
    numeric_weights = [
        r.get("correlation_drop_pct", 0) / 100.0 for r in numeric_conf
    ]
    numeric_severity = float(np.mean(numeric_weights)) if numeric_weights else 0.0

    # scale by count
    numeric_penalty = min(
        numeric_severity * len(numeric_conf) / 5.0, 1.0
    ) if numeric_conf else 0.0

    # flat cat penalty
    cat_penalty = min(len(cat_conf) * 0.10, 1.0)

    total_penalty = _penalty(numeric_penalty, cat_penalty)
    score = _clamp(100 * (1 - total_penalty))

    return score, {
        "numeric_confounders": len(numeric_conf),
        "categorical_associations": len(cat_conf),
        "avg_numeric_drop_pct": round(numeric_severity * 100, 2),
    }


def _score_gaming(gaming_results: list) -> tuple[float, dict]:
    """
    Penalise columns with high metric gaming risk scores.

    Uses the distribution of gaming_risk_score values rather than just the
    mean — a dataset with one extremely gamed column is different from one
    where all columns are moderately suspicious.
    """
    if not gaming_results:
        return 100.0, {"columns_scored": 0, "avg_gaming_risk": 0.0, "high_risk_columns": 0}

    raw_scores    = [r.get("gaming_risk_score", 0) for r in gaming_results]
    avg_risk      = float(np.mean(raw_scores))
    max_risk      = float(np.max(raw_scores))
    high_risk_n   = sum(1 for s in raw_scores if s >= 60)

    # avg + max signal
    combined_penalty = (avg_risk / 100.0) * 0.6 + (max_risk / 100.0) * 0.4
    score = _clamp(100 * (1 - combined_penalty))

    return score, {
        "columns_scored": len(gaming_results),
        "avg_gaming_risk": round(avg_risk, 2),
        "max_gaming_risk": round(max_risk, 2),
        "high_risk_columns": high_risk_n,
    }


# ---------------------------------------------------------------------------
# Per-analysis-type scores
# ---------------------------------------------------------------------------

def _analysis_type_scores(
    component_scores: dict,
    paradox_results: list,
    survivorship_results: list,
    confounder_results: list,
    profile_dict: dict,
) -> dict:
    """
    Derive four targeted confidence scores for common analytical tasks.
    Each starts at 100 and is penalised by the findings most relevant to it.
    """

    # correlation confidence
    n_numeric_conf = sum(
        1 for r in confounder_results if r.get("type") == "numeric_confounder"
    )
    n_paradox = len(paradox_results)
    avg_paradox_sev = float(np.mean(
        [r.get("severity_score", 0) for r in paradox_results]
    )) if paradox_results else 0.0

    corr_penalty = _penalty(
        min(n_numeric_conf * 0.12, 1.0),
        min(n_paradox * avg_paradox_sev * 0.25, 1.0),
    )
    correlation_confidence = _clamp(100 * (1 - corr_penalty))

    # trend confidence
    temporal_findings = [
        f for f in survivorship_results if f.get("finding_type") == "temporal_gap"
    ]
    n_temporal = len(temporal_findings)
    n_surv_high = sum(
        1 for f in survivorship_results if f.get("risk_level") == "high"
    )

    trend_penalty = _penalty(
        min(n_temporal * 0.30, 1.0),
        min(n_surv_high * 0.20, 1.0),
    )
    trend_confidence = _clamp(100 * (1 - trend_penalty))

    # segmentation confidence
    cardinality = profile_dict.get("cardinality", {})
    low_cardinality_cols = sum(
        1 for v in cardinality.values()
        if v.get("unique_count", 99) <= 2
    )
    imbalance_findings = [
        f for f in survivorship_results if f.get("finding_type") == "outcome_imbalance"
    ]
    total_cols = max(len(cardinality), 1)

    seg_penalty = _penalty(
        min(low_cardinality_cols / total_cols * 2, 1.0),
        min(len(imbalance_findings) * 0.25, 1.0),
    )
    segmentation_confidence = _clamp(100 * (1 - seg_penalty))

    # comparison confidence
    n_cat_conf = sum(
        1 for r in confounder_results if r.get("type") == "categorical_association"
    )
    comp_penalty = _penalty(
        min(n_paradox * 0.20, 1.0),
        min(n_cat_conf * 0.10, 1.0),
    )
    comparison_confidence = _clamp(100 * (1 - comp_penalty))

    return {
        "correlation_confidence": round(correlation_confidence, 1),
        "trend_confidence":       round(trend_confidence, 1),
        "segmentation_confidence": round(segmentation_confidence, 1),
        "comparison_confidence":  round(comparison_confidence, 1),
    }


# ---------------------------------------------------------------------------
# Top risks extractor
# ---------------------------------------------------------------------------

def _extract_top_risks(
    component_scores: dict,
    component_evidence: dict,
    survivorship_results: list,
    paradox_results: list,
    confounder_results: list,
    gaming_results: list,
) -> list:
    """
    Rank all risk signals by severity and return the top 3 as plain-English strings.
    Each risk is (penalty_magnitude, message) so we can sort and pick the worst.
    """
    risks = []

    # Missing data
    avg_missing = component_evidence["completeness"].get("avg_missing_pct", 0)
    if avg_missing > 5:
        risks.append((avg_missing, f"Average {avg_missing:.1f}% of values are missing across columns — null patterns may bias every downstream statistic."))

    # Type issues
    n_type = component_evidence["consistency"].get("type_issue_count", 0)
    if n_type > 0:
        risks.append((n_type * 10, f"{n_type} column(s) have dtype mismatches (e.g. numbers stored as text) — aggregations on these columns will silently produce wrong results."))

    # Survivorship — high-risk findings
    high_surv = component_evidence["bias_risk"].get("high_risk_survivorship", 0)
    if high_surv > 0:
        risks.append((high_surv * 30, f"{high_surv} high-risk survivorship signal(s) detected — the dataset likely represents only 'winners' and excludes failed or churned cases."))

    # Simpson's Paradox
    n_paradox = component_evidence["bias_risk"].get("simpsons_paradox_cases", 0)
    avg_sev   = component_evidence["bias_risk"].get("avg_paradox_severity", 0)
    if n_paradox > 0:
        risks.append((n_paradox * avg_sev * 50, f"{n_paradox} Simpson's Paradox case(s) found — overall correlations reverse direction inside subgroups, making aggregate trends actively misleading."))

    # Confounders
    n_conf = component_evidence["confounding"].get("numeric_confounders", 0)
    avg_drop = component_evidence["confounding"].get("avg_numeric_drop_pct", 0)
    if n_conf > 0:
        risks.append((n_conf * avg_drop, f"{n_conf} likely confounding variable(s) found — correlations drop by an average of {avg_drop:.0f}% when controlled for, suggesting spurious relationships."))

    # Gaming risk
    high_game = component_evidence["gaming"].get("high_risk_columns", 0)
    avg_game  = component_evidence["gaming"].get("avg_gaming_risk", 0)
    if high_game > 0:
        risks.append((avg_game, f"{high_game} column(s) show high metric gaming risk — round-number clustering, Benford deviation, or spike patterns suggest data may not reflect natural behaviour."))

    # Constant columns
    n_const = component_evidence["consistency"].get("constant_column_count", 0)
    if n_const > 0:
        risks.append((n_const * 15, f"{n_const} constant column(s) carry zero information — they should be dropped before any modelling or correlation analysis."))

    # top 3 by severity
    risks.sort(key=lambda x: x[0], reverse=True)
    return [msg for _, msg in risks[:3]]


# ---------------------------------------------------------------------------
# Recommendation builder
# ---------------------------------------------------------------------------

def _build_recommendation(
    overall_score: float,
    top_risks: list,
    component_scores: dict,
    analysis_type_scores: dict,
) -> str:
    """
    Produce a short, actionable paragraph tailored to what the scores reveal.
    """
    grade = _grade(overall_score)

    # opening by grade
    if grade == "A":
        opening = "This dataset has strong integrity and is suitable for analysis with minimal pre-processing."
    elif grade == "B":
        opening = "This dataset is generally reliable but has some issues worth addressing before drawing firm conclusions."
    elif grade == "C":
        opening = "This dataset has moderate integrity concerns. Conclusions drawn from it should be treated as directional, not definitive."
    elif grade == "D":
        opening = "This dataset has significant integrity issues. Any conclusions require validation against an independent source."
    else:
        opening = "This dataset has critical integrity problems. It should not be used for decision-making without substantial remediation."

    # action items
    actions = []

    if component_scores["completeness"] < 75:
        actions.append("Address missing data through imputation or explicit exclusion, and document which rows were removed and why.")
    if component_scores["consistency"] < 75:
        actions.append("Fix dtype mismatches and investigate outlier-heavy columns before running any statistical model.")
    if component_scores["bias_risk"] < 70:
        actions.append("Investigate survivorship and paradox findings — consider whether excluded records (churned users, failed projects, closed accounts) need to be re-sourced.")
    if component_scores["confounding"] < 75:
        actions.append("Control for the identified confounding variables in any regression or causal analysis — raw correlations here are unreliable.")
    if component_scores["gaming"] < 75:
        actions.append("Treat high-gaming-risk columns with scepticism in performance reporting — validate them against an independent audit trail if possible.")

    # weakest analysis type
    weakest_type = min(analysis_type_scores, key=analysis_type_scores.get)
    weakest_score = analysis_type_scores[weakest_type]
    if weakest_score < 70:
        type_labels = {
            "correlation_confidence":    "correlation-based analysis",
            "trend_confidence":          "time-series trend analysis",
            "segmentation_confidence":   "segmentation and group comparisons",
            "comparison_confidence":     "cross-group comparisons",
        }
        actions.append(f"Be especially cautious with {type_labels.get(weakest_type, weakest_type)} — it has the lowest confidence score ({weakest_score:.0f}/100).")

    if not actions:
        actions.append("No immediate remediation required — standard analytical hygiene applies.")

    body = " ".join(actions)
    return f"{opening} {body}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_confidence_scores(
    df: pd.DataFrame,
    profile_dict: dict,
    paradox_results: list,
    survivorship_results: list,
    confounder_results: list,
    gaming_results: list,
) -> dict:
    """
    Synthesise all audit findings into a single Data Confidence Score.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    profile_dict : dict
        Output of profiler.profile_data(df).
    paradox_results : list
        Output of paradox.scan_simpsons_paradox(df).
    survivorship_results : list
        Output of survivorship.detect_survivorship_bias(df, profile_dict).
    confounder_results : list
        Output of confounders.detect_confounders(df).
    gaming_results : list
        Output of metric_risk.score_metric_gaming_risk(df, profile_dict).

    Returns
    -------
    dict with keys:
        overall_score         — float in [0, 100]
        overall_grade         — "A", "B", "C", "D", or "F"
        component_scores      — dict of five weighted sub-scores (each 0–100)
        component_evidence    — raw numbers behind each sub-score
        analysis_type_scores  — dict of four task-specific confidence scores
        top_risks             — list of up to 3 plain-English risk descriptions
        recommendation        — short actionable paragraph
    """
    if df is None or df.empty or not profile_dict:
        return {
            "overall_score": 0.0,
            "overall_grade": "F",
            "component_scores": {},
            "component_evidence": {},
            "analysis_type_scores": {},
            "top_risks": ["Dataset is empty or could not be profiled."],
            "recommendation": "No data available to score.",
        }

    # compute components
    completeness_score, completeness_ev = _score_completeness(profile_dict)
    consistency_score,  consistency_ev  = _score_consistency(profile_dict, df)
    bias_score,         bias_ev         = _score_bias_risk(survivorship_results, paradox_results)
    confounding_score,  confounding_ev  = _score_confounding(confounder_results)
    gaming_score,       gaming_ev       = _score_gaming(gaming_results)

    raw = [completeness_score, consistency_score, bias_score, confounding_score, gaming_score]
    keys = ["completeness", "consistency", "bias_risk", "confounding", "gaming"]
    component_scores   = {k: round(v, 1) for k, v in zip(keys, raw)}
    component_evidence = dict(zip(keys, [completeness_ev, consistency_ev, bias_ev,
                                         confounding_ev, gaming_ev]))

    # weighted overall
    overall = round(_clamp(sum(component_scores[k] * WEIGHTS[k] for k in WEIGHTS)), 1)

    # analysis-type scores
    a_type_scores = _analysis_type_scores(
        component_scores, paradox_results, survivorship_results,
        confounder_results, profile_dict,
    )

    # top risks
    top_risks = _extract_top_risks(
        component_scores, component_evidence,
        survivorship_results, paradox_results,
        confounder_results, gaming_results,
    )

    # recommendation
    recommendation = _build_recommendation(
        overall, top_risks, component_scores, a_type_scores
    )

    return {
        "overall_score":       overall,
        "overall_grade":       _grade(overall),
        "component_scores":    component_scores,
        "component_evidence":  component_evidence,
        "analysis_type_scores": a_type_scores,
        "top_risks":           top_risks,
        "recommendation":      recommendation,
    }
