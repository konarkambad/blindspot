"""
survivorship.py — Survivorship bias detector.

Survivorship bias occurs when analysis is performed only on the subset of
entities that 'survived' some selection process, while the failures, dropouts,
or filtered-out cases are invisible in the data. Classic examples:

  - Studying only successful companies ignores the many that failed similarly.
  - A customer dataset with only 'active' users hides why churned users left.
  - A loan portfolio showing only performing loans omits the defaulted ones.

This module looks for four structural signals that suggest survivorship bias
may be distorting the dataset before any model or conclusion is drawn.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POSITIVE_STATUS_KEYWORDS = {"active", "open", "live", "current", "enrolled", "running"}
NEGATIVE_STATUS_KEYWORDS = {"inactive", "closed", "cancelled", "canceled", "churned",
                            "failed", "completed", "dropped", "expired", "terminated",
                            "rejected", "lost", "dead", "deleted", "suspended"}
ALL_STATUS_KEYWORDS = POSITIVE_STATUS_KEYWORDS | NEGATIVE_STATUS_KEYWORDS

STATUS_SKEW_THRESHOLD       = 0.80
BINARY_IMBALANCE_THRESHOLD  = 0.75
SUSPICIOUS_ROUND_NUMBERS    = {0, 1, 10, 100, 1_000, 10_000, 100_000, 1_000_000}
BOUNDARY_DENSITY_THRESHOLD  = 0.02
MIN_TIME_PERIODS            = 4


# ---------------------------------------------------------------------------
# Risk classification helper
# ---------------------------------------------------------------------------

def _risk(fraction: float, low: float = 0.60, high: float = 0.85) -> str:
    """Map a dominance fraction to a risk level string."""
    if fraction >= high:
        return "high"
    if fraction >= low:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# 1. Temporal gap detection
# ---------------------------------------------------------------------------

def _detect_temporal_gaps(df: pd.DataFrame) -> list:
    """
    Look for date/datetime columns and check whether any calendar periods
    (month-level granularity) are completely missing from the record sequence.

    A gap is only meaningful when it is an interior gap — a missing month
    surrounded by months that do have records — because that suggests records
    from that period were removed rather than not yet collected.
    """
    findings  = []
    date_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()

    # try object columns
    for col in df.select_dtypes(include="object").columns:
        try:
            coerced = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if coerced.notna().mean() > 0.80:
                df = df.copy()
                df[col] = coerced
                date_cols.append(col)
        except Exception:
            continue

    # no date columns
    if not date_cols:
        return findings

    for col in date_cols:
        series = df[col].dropna()
        if len(series) < MIN_TIME_PERIODS:
            continue

        # monthly periods
        periods       = series.dt.to_period("M")
        period_counts = periods.value_counts().sort_index()

        if len(period_counts) < MIN_TIME_PERIODS:
            continue

        all_months = pd.period_range(
            start=period_counts.index.min(),
            end=period_counts.index.max(),
            freq="M",
        )

        present = set(period_counts.index)
        gaps    = [str(m) for m in all_months if m not in present]

        if gaps:
            # interior gaps only
            interior_gaps = [
                g for g in gaps
                if pd.Period(g, "M") > period_counts.index.min()
                and pd.Period(g, "M") < period_counts.index.max()
            ]
            if interior_gaps:
                risk = "high" if len(interior_gaps) > 3 else "medium"
                findings.append({
                    "finding_type": "temporal_gap",
                    "affected_columns": [col],
                    "description": (
                        f"Column '{col}' has {len(interior_gaps)} interior month(s) "
                        "with zero records surrounded by months with data. "
                        "Records from these periods may have been filtered or lost."
                    ),
                    "evidence": {
                        "missing_periods": interior_gaps,
                        "total_periods_present": len(period_counts),
                        "total_periods_expected": len(all_months),
                    },
                    "risk_level": risk,
                })

    return findings


# ---------------------------------------------------------------------------
# 2. Status column skew
# ---------------------------------------------------------------------------

def _detect_status_skew(df: pd.DataFrame) -> list:
    """
    Find columns that look like status fields (contain known status keywords)
    and flag those where one status value dominates beyond the skew threshold.

    A dataset that is 95 % 'active' almost certainly had 'churned' or 'closed'
    records removed before analysis — the survivors are all that remain.
    """
    findings    = []
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        try:
            raw = df[col].dropna()
            if len(raw) == 0:
                continue

            # safe string cast
            values_lower = raw.apply(lambda v: str(v).strip().lower()[:200])
            n_total      = len(values_lower)

            # skip high-cardinality
            if values_lower.nunique() > 50:
                continue

            # check status keywords
            has_status_keyword = values_lower.apply(
                lambda v: any(kw in v for kw in ALL_STATUS_KEYWORDS)
            ).any()

            if not has_status_keyword:
                continue

            value_counts = values_lower.value_counts(normalize=True)
            if len(value_counts) == 0:
                continue

            dominant_value    = value_counts.index[0]
            dominant_fraction = float(value_counts.iloc[0])
        except Exception:
            continue

        if dominant_fraction >= STATUS_SKEW_THRESHOLD:
            # survivor vs terminal
            is_survivor = any(kw in dominant_value for kw in POSITIVE_STATUS_KEYWORDS)
            note = (
                "Dominant status is a 'survivor' type — records in other states "
                "may have been excluded from this dataset."
                if is_survivor else
                "Dominant status is a terminal/negative type — dataset may only "
                "capture a specific stage of the lifecycle."
            )

            findings.append({
                "finding_type": "status_skew",
                "affected_columns": [col],
                "description": (
                    f"Column '{col}' appears to be a status field and {dominant_fraction:.1%} "
                    f"of records have the value '{dominant_value}'. {note}"
                ),
                "evidence": {
                    "dominant_value": dominant_value,
                    "dominant_fraction": round(dominant_fraction, 4),
                    "all_value_fractions": value_counts.round(4).to_dict(),
                },
                "risk_level": _risk(dominant_fraction, low=0.80, high=0.92),
            })

    return findings


# ---------------------------------------------------------------------------
# 3. Truncation detection
# ---------------------------------------------------------------------------

def _detect_truncation(df: pd.DataFrame, profile_dict: dict) -> list:
    """
    Check numeric columns for suspiciously clean hard boundaries that suggest
    the data was filtered before reaching this analysis.

    Two signals:
      a) All values are strictly positive in a column where negatives are
         conceptually plausible (e.g., revenue change, score delta, temperature).
      b) A round-number boundary concentrates an unusually high density of
         exact-value records (pile-up at a reporting threshold).
    """
    findings     = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue

        col_min = float(series.min())
        col_max = float(series.max())

        # signal A: no negatives
        if col_min == 0 or col_min > 0:
            # name suggests delta
            delta_keywords = {"change", "diff", "delta", "growth", "return",
                              "score", "gain", "loss", "rate", "margin"}
            col_lower = col.lower()
            if any(kw in col_lower for kw in delta_keywords):
                frac_at_zero = float((series == 0).mean())
                findings.append({
                    "finding_type": "truncation_boundary",
                    "affected_columns": [col],
                    "description": (
                        f"Column '{col}' has a minimum of {col_min} with no negative "
                        "values despite its name suggesting it could be negative. "
                        "Negative outcomes may have been filtered out."
                    ),
                    "evidence": {
                        "min_value": col_min,
                        "max_value": col_max,
                        "fraction_at_zero": round(frac_at_zero, 4),
                    },
                    "risk_level": "medium",
                })

        # signal B: round boundary
        for boundary in SUSPICIOUS_ROUND_NUMBERS:
            if col_min <= boundary <= col_max:
                frac_at_boundary = float((series == boundary).mean())
                if frac_at_boundary >= BOUNDARY_DENSITY_THRESHOLD:
                    findings.append({
                        "finding_type": "truncation_boundary",
                        "affected_columns": [col],
                        "description": (
                            f"Column '{col}' has {frac_at_boundary:.1%} of values "
                            f"sitting exactly at the round boundary {boundary}. "
                            "This may indicate a reporting threshold or hard filter cutoff."
                        ),
                        "evidence": {
                            "boundary_value": boundary,
                            "fraction_at_boundary": round(frac_at_boundary, 4),
                            "total_non_null_rows": int(len(series)),
                        },
                        "risk_level": _risk(frac_at_boundary, low=0.05, high=0.15),
                    })

    return findings


# ---------------------------------------------------------------------------
# 4. Missing failure cases (binary outcome imbalance)
# ---------------------------------------------------------------------------

def _detect_outcome_imbalance(df: pd.DataFrame) -> list:
    """
    Identify binary outcome columns (0/1 or yes/no or true/false) and flag
    those where the positive class dominates beyond the imbalance threshold.

    A dataset where 90 % of rows are 'success' / '1' / 'yes' likely had
    failures removed, or was collected in a way that only captures survivors.
    """
    findings = []
    outcome_keywords = {"outcome", "result", "target", "label", "success",
                        "converted", "churned", "defaulted", "survived",
                        "response", "flag", "status", "class"}

    for col in df.columns:
        series = df[col].dropna()
        n      = len(series)
        if n < 10:
            continue

        # binary numeric
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = set(series.unique())
            if unique_vals <= {0, 1}:
                positive_frac = float((series == 1).mean())
                negative_frac = 1.0 - positive_frac
                dominant_frac = max(positive_frac, negative_frac)

                if dominant_frac >= BINARY_IMBALANCE_THRESHOLD:
                    dominant_class = 1 if positive_frac >= negative_frac else 0
                    minority_class = 1 - dominant_class
                    findings.append({
                        "finding_type": "outcome_imbalance",
                        "affected_columns": [col],
                        "description": (
                            f"Binary column '{col}' is {dominant_frac:.1%} class {dominant_class} "
                            f"and only {1 - dominant_frac:.1%} class {minority_class}. "
                            "The minority class may represent filtered-out failure cases."
                        ),
                        "evidence": {
                            "positive_class_fraction": round(positive_frac, 4),
                            "negative_class_fraction": round(negative_frac, 4),
                            "dominant_class": int(dominant_class),
                            "total_rows": n,
                        },
                        "risk_level": _risk(dominant_frac, low=0.75, high=0.90),
                    })

        # binary categorical
        elif df[col].dtype == object:
            col_lower      = col.lower()
            is_outcome_col = any(kw in col_lower for kw in outcome_keywords)

            str_series   = series.astype(str).str.strip().str.lower()
            binary_pairs = [{"yes", "no"}, {"true", "false"}, {"pass", "fail"},
                            {"win", "lose"}, {"success", "failure"},
                            {"approved", "rejected"}]
            unique_lower = set(str_series.unique())

            for pair in binary_pairs:
                if unique_lower <= pair or (is_outcome_col and len(unique_lower) == 2):
                    value_counts  = str_series.value_counts(normalize=True)
                    dominant_frac = float(value_counts.iloc[0])

                    if dominant_frac >= BINARY_IMBALANCE_THRESHOLD:
                        findings.append({
                            "finding_type": "outcome_imbalance",
                            "affected_columns": [col],
                            "description": (
                                f"Categorical binary column '{col}' is "
                                f"{dominant_frac:.1%} '{value_counts.index[0]}'. "
                                "The minority outcome may represent cases excluded from this dataset."
                            ),
                            "evidence": {
                                "value_fractions": value_counts.round(4).to_dict(),
                                "dominant_value": value_counts.index[0],
                                "total_rows": n,
                            },
                            "risk_level": _risk(dominant_frac, low=0.75, high=0.90),
                        })
                    break  # one per column

    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_survivorship_bias(df: pd.DataFrame, profile_dict: dict) -> list:
    """
    Scan a DataFrame for structural signals of survivorship bias.

    Survivorship bias arises when the data you can see is systematically
    different from the data that was filtered, lost, or never collected.
    This function checks four independent signals:

      1. Temporal gaps   — interior date periods with no records.
      2. Status skew     — one lifecycle status dominates heavily.
      3. Truncation      — suspiciously clean numeric boundaries.
      4. Outcome imbalance — binary target heavily skewed toward success.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset to inspect.
    profile_dict : dict
        Output of profiler.profile_data(df). Used to avoid recomputing
        basic statistics already available from the profiling step.

    Returns
    -------
    list of dict, each containing:
        finding_type      — one of: temporal_gap, status_skew,
                            truncation_boundary, outcome_imbalance
        affected_columns  — list of column names involved
        description       — human-readable explanation of the finding
        evidence          — dict of supporting numbers
        risk_level        — "low", "medium", or "high"

    Returns an empty list if no signals are detected.
    """
    if df is None or df.empty:
        return []

    findings = []
    findings.extend(_detect_temporal_gaps(df))
    findings.extend(_detect_status_skew(df))
    findings.extend(_detect_truncation(df, profile_dict))
    findings.extend(_detect_outcome_imbalance(df))

    # sort by risk
    risk_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: risk_order.get(f["risk_level"], 3))

    return findings
