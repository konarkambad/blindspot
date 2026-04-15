"""
paradox.py — Simpson's Paradox scanner.

Simpson's Paradox occurs when a trend present in aggregated data reverses
(or disappears) when the data is split into subgroups. This is one of the
most dangerous and least-intuitive traps in data analysis.

Example: a drug appears effective overall, but when patients are split by
age group, it is harmful in every single group — the aggregate result was
driven purely by confounded group sizes.

This module scans all viable (numeric_x, numeric_y, categorical_group)
combinations and flags reversals using Pearson correlation as the trend proxy.
"""

import numpy as np
import pandas as pd
from itertools import combinations, product


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_GROUP_CARDINALITY = 2
MAX_GROUP_CARDINALITY = 10
MIN_SUBGROUP_ROWS     = 5
MAX_COMBINATIONS      = 500
MIN_OVERALL_CORR      = 0.05


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_categorical_candidates(df: pd.DataFrame) -> list:
    """
    Return column names suitable as grouping variables, sorted by cardinality
    ascending so low-cardinality columns are prioritised in the scan budget.

    A column qualifies if:
      - dtype is object, category, or boolean, OR
      - dtype is integer with cardinality within the allowed range
    """
    candidates = []

    for col in df.columns:
        dtype    = df[col].dtype
        n_unique = df[col].nunique(dropna=True)

        if not (MIN_GROUP_CARDINALITY <= n_unique <= MAX_GROUP_CARDINALITY):
            continue

        is_categorical = (
            pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or hasattr(dtype, "categories")
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_integer_dtype(dtype)
        )

        if is_categorical:
            candidates.append((col, n_unique))

    # low cardinality first
    candidates.sort(key=lambda x: x[1])
    return [col for col, _ in candidates]


def _safe_correlation(series_x: pd.Series, series_y: pd.Series) -> float | None:
    """
    Compute Pearson correlation between two series, returning None if it
    cannot be computed (insufficient data, zero variance, NaN result, etc.).
    """
    try:
        combined = pd.concat([series_x, series_y], axis=1).dropna()
        if len(combined) < MIN_SUBGROUP_ROWS:
            return None

        x_vals = combined.iloc[:, 0]
        y_vals = combined.iloc[:, 1]

        # skip zero-variance
        if x_vals.std() == 0 or y_vals.std() == 0:
            return None

        r = float(x_vals.corr(y_vals))
        # guard NaN/Inf
        if not np.isfinite(r):
            return None
        return r
    except Exception:
        return None


def _compute_severity(overall_corr: float, subgroup_corrs: list) -> float:
    """
    Score how dramatic the reversal is on a 0-1 scale.

    The severity combines two signals:
      1. Magnitude of the overall correlation (stronger trend = more surprising reversal).
      2. Fraction of subgroups that actually reversed sign.
      3. Average magnitude of the reversed subgroup correlations.

    A score near 1.0 means: strong overall trend, all subgroups reversed strongly.
    A score near 0.0 means: weak overall trend, only one subgroup barely reversed.
    """
    overall_sign      = np.sign(overall_corr)
    reversed_corrs    = [c for c in subgroup_corrs if np.sign(c) != overall_sign]
    fraction_reversed = len(reversed_corrs) / len(subgroup_corrs) if subgroup_corrs else 0
    avg_rev_mag       = np.mean([abs(c) for c in reversed_corrs]) if reversed_corrs else 0.0
    severity = min(abs(overall_corr), 1.0) * fraction_reversed * (0.5 + 0.5 * avg_rev_mag)
    return round(float(np.clip(severity, 0.0, 1.0)), 4)


def _reversal_type(overall_corr: float) -> str:
    """
    Label the direction of the paradox from the overall correlation's perspective.
    """
    if overall_corr > 0:
        return "positive_to_negative"
    return "negative_to_positive"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_simpsons_paradox(df: pd.DataFrame) -> list:
    """
    Scan a DataFrame for instances of Simpson's Paradox.

    For every combination of (numeric_x, numeric_y, categorical_group) within
    the scan budget, this function:
      1. Computes the overall Pearson correlation between x and y.
      2. Computes the within-subgroup correlations for each level of the grouping column.
      3. Flags cases where the overall trend sign is reversed in the majority of subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to scan. Should already be loaded and profiled.

    Returns
    -------
    list of dict, each containing:
        x_column              — name of the first numeric column
        y_column              — name of the second numeric column
        grouping_column       — name of the categorical column used for splitting
        overall_correlation   — Pearson r across the full dataset
        subgroup_correlations — dict mapping each group label to its within-group r
                                (None if the subgroup had insufficient data)
        reversal_type         — "positive_to_negative" or "negative_to_positive"
        severity_score        — float in [0, 1]; higher = more dramatic paradox

    Returns an empty list if the DataFrame is too small, has no numeric pairs,
    or no qualifying categorical columns are found.
    """
    if df is None or df.empty:
        return []

    # identify candidates
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # skip zero-variance columns
    numeric_cols = [
        c for c in numeric_cols
        if df[c].dropna().nunique() > 1 and df[c].dropna().std(ddof=0) > 0
    ]

    categorical_cols = _get_categorical_candidates(df)

    if len(numeric_cols) < 2:
        return []  # need a pair

    if not categorical_cols:
        return []  # no grouping variable

    numeric_pairs = list(combinations(numeric_cols, 2))
    all_combos    = [(x, y, g) for (x, y), g in product(numeric_pairs, categorical_cols)]

    if len(all_combos) > MAX_COMBINATIONS:
        all_combos = all_combos[:MAX_COMBINATIONS]

    # scan combinations
    flagged = []

    for x_col, y_col, group_col in all_combos:
        # overall r
        overall_corr = _safe_correlation(df[x_col], df[y_col])
        if overall_corr is None or abs(overall_corr) < MIN_OVERALL_CORR:
            continue

        overall_sign       = np.sign(overall_corr)
        subgroup_corrs_raw = {}
        valid_corrs        = []

        for group_val, subgroup in df.groupby(group_col, observed=True):
            r = _safe_correlation(subgroup[x_col], subgroup[y_col])
            subgroup_corrs_raw[str(group_val)] = round(r, 4) if r is not None else None
            if r is not None:
                valid_corrs.append(r)

        if not valid_corrs:
            continue

        reversed_count = sum(1 for r in valid_corrs if np.sign(r) != overall_sign)
        if reversed_count > len(valid_corrs) / 2:
            severity = _compute_severity(overall_corr, valid_corrs)

            flagged.append({
                "x_column":             x_col,
                "y_column":             y_col,
                "grouping_column":      group_col,
                "overall_correlation":  round(overall_corr, 4),
                "subgroup_correlations": subgroup_corrs_raw,
                "reversal_type":        _reversal_type(overall_corr),
                "severity_score":       severity,
            })

    # sort by severity
    flagged.sort(key=lambda x: x["severity_score"], reverse=True)
    return flagged
