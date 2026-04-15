"""
confounders.py — Confounding variable detection.

A confounding variable (confounder) is a third variable that correlates with
both an exposure (x) and an outcome (y), creating a spurious or inflated
association between them. Failing to account for confounders is one of the
most common ways that correlational analysis is mistaken for causal evidence.

Classic example: ice cream sales and drowning rates are strongly correlated —
but the confounder is summer temperature, which independently drives both.
Control for temperature and the correlation disappears.

This module detects two classes of confounding:
  1. Numeric confounders — via partial correlation after controlling for z.
  2. Categorical association — via chi-squared test between categorical pairs.
"""

import numpy as np
import pandas as pd
from itertools import combinations, permutations
from scipy import stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum absolute correlation for an (x, y) pair to be worth investigating
XY_CORR_THRESHOLD = 0.50

# Minimum absolute correlation z must have with both x AND y to be a candidate
Z_CORR_THRESHOLD = 0.40

# A confounder is flagged if the partial correlation drops by this fraction
DROP_THRESHOLD = 0.40

# Max (x, y) pairs to evaluate — keeps runtime predictable on wide datasets
MAX_PAIRS = 200

# Chi-squared p-value below which a categorical association is flagged
CHI2_PVALUE_THRESHOLD = 0.05

# Minimum expected cell frequency for chi-squared to be reliable
MIN_EXPECTED_FREQ = 5


# ---------------------------------------------------------------------------
# Internal helpers — numeric partial correlation
# ---------------------------------------------------------------------------

def _pearson(a: pd.Series, b: pd.Series) -> float | None:
    """
    Pearson correlation between two series on their shared non-null rows.
    Returns None if computation is impossible (< 4 rows, zero variance).
    """
    combined = pd.concat([a, b], axis=1).dropna()
    if len(combined) < 4:
        return None
    x, y = combined.iloc[:, 0], combined.iloc[:, 1]
    if x.std() == 0 or y.std() == 0:
        return None
    return float(x.corr(y))


def _partial_correlation(x: pd.Series, y: pd.Series, z: pd.Series) -> float | None:
    """
    Compute the partial correlation of x and y controlling for z.

    Method: regress both x and z, y and z via OLS; correlate the residuals.
    This is equivalent to the textbook partial correlation formula and does
    not require statsmodels — scipy.stats.linregress is sufficient.

    Returns None if any intermediate step fails.
    """
    try:
        combined = pd.concat([x, y, z], axis=1).dropna()
        if len(combined) < 6:   # Need enough df for two regressions
            return None

        xv = combined.iloc[:, 0].values.astype(float)
        yv = combined.iloc[:, 1].values.astype(float)
        zv = combined.iloc[:, 2].values.astype(float)

        # constant z, skip
        if np.std(zv) == 0:
            return None

        # zero variance, skip
        if np.std(xv) == 0 or np.std(yv) == 0:
            return None

        # residuals of x ~ z
        slope_xz, intercept_xz, *_ = stats.linregress(zv, xv)
        res_x = xv - (slope_xz * zv + intercept_xz)

        # residuals of y ~ z
        slope_yz, intercept_yz, *_ = stats.linregress(zv, yv)
        res_y = yv - (slope_yz * zv + intercept_yz)

        if np.std(res_x) == 0 or np.std(res_y) == 0:
            return None

        r = float(np.corrcoef(res_x, res_y)[0, 1])
        return r if np.isfinite(r) else None
    except Exception:
        return None


def _drop_pct(raw: float, partial: float) -> float:
    """
    Percentage drop in absolute correlation magnitude after controlling for z.
    A positive value means the correlation weakened (confounder absorbed variance).
    A negative value means the correlation strengthened (suppressor effect).
    """
    if abs(raw) == 0:
        return 0.0
    return round((abs(raw) - abs(partial)) / abs(raw) * 100, 2)


def _interpret_numeric(
    x: str, y: str, z: str,
    raw: float, partial: float, drop: float,
) -> str:
    """Build a plain-English sentence describing the numeric confounder finding."""
    direction = "positive" if raw > 0 else "negative"
    if drop >= DROP_THRESHOLD * 100:
        strength = "substantially" if drop < 70 else "almost entirely"
        return (
            f"'{x}' and '{y}' share a {direction} correlation (r={raw:.2f}), "
            f"but when controlling for '{z}' the correlation {strength} weakens "
            f"to {partial:.2f} (a {drop:.0f}% drop). "
            f"'{z}' is likely driving much of the apparent relationship."
        )
    return (
        f"'{x}' and '{y}' correlate (r={raw:.2f}); controlling for '{z}' "
        f"reduces this to {partial:.2f} — a modest {drop:.0f}% drop. "
        f"'{z}' may be a partial confounder."
    )


# ---------------------------------------------------------------------------
# Internal helpers — categorical association (chi-squared)
# ---------------------------------------------------------------------------

def _chi2_association(df: pd.DataFrame, col_a: str, col_b: str) -> dict | None:
    """
    Run a chi-squared test of independence between two categorical columns.

    Returns a result dict if the test is reliable (expected frequencies ≥ 5
    in at least 80 % of cells), otherwise None.
    """
    contingency = pd.crosstab(df[col_a].astype(str), df[col_b].astype(str))

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None

    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    except Exception:
        return None

    # Reliability check: Cochran's rule — at most 20 % of cells below expected 5
    low_expected = (expected < MIN_EXPECTED_FREQ).mean()
    if low_expected > 0.20:
        return None

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": int(dof),
        "cramers_v": _cramers_v(chi2, contingency),
    }


def _cramers_v(chi2: float, contingency: pd.DataFrame) -> float:
    """
    Cramér's V — a normalised effect size for chi-squared tests in [0, 1].
    0 = no association, 1 = perfect association.
    """
    n = contingency.values.sum()
    k = min(contingency.shape) - 1
    if n == 0 or k == 0:
        return 0.0
    return round(float(np.sqrt(chi2 / (n * k))), 4)


def _interpret_categorical(col_a: str, col_b: str, p_value: float, v: float) -> str:
    strength = (
        "strong" if v >= 0.30 else
        "moderate" if v >= 0.15 else
        "weak"
    )
    return (
        f"'{col_a}' and '{col_b}' have a statistically significant {strength} "
        f"association (chi-squared p={p_value:.4f}, Cramér's V={v:.2f}). "
        "Their distributions shift substantially across each other's categories — "
        "either variable could confound analyses involving the other."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_confounders(df: pd.DataFrame) -> list:
    """
    Detect likely confounding variables in a DataFrame.

    Two detection strategies are combined:

    Strategy 1 — Numeric partial correlation
      For every (x, y) numeric pair with |r| > 0.5, find candidate third
      variables z that correlate with both (|r_xz| > 0.4 and |r_yz| > 0.4).
      Compute partial correlation of x and y controlling for z. Flag z as a
      confounder if the correlation drops by more than 40 %.

    Strategy 2 — Categorical chi-squared association
      For every pair of categorical columns, run a chi-squared test of
      independence. Flag pairs where p < 0.05 as potentially confounding —
      if either column is also used as a grouping or filter variable, the
      other may introduce bias through uncontrolled association.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse. Should be loaded via loader.load_data().

    Returns
    -------
    list of dict, each containing:

      For numeric findings:
        type                 — "numeric_confounder"
        variable_x           — first numeric column name
        variable_y           — second numeric column name
        likely_confounder    — third column name driving the relationship
        raw_correlation      — Pearson r between x and y
        partial_correlation  — Pearson r between x and y controlling for z
        correlation_drop_pct — percentage reduction in |r| after controlling
        interpretation       — plain-English explanation

      For categorical findings:
        type                 — "categorical_association"
        variable_x           — first categorical column
        variable_y           — second categorical column
        likely_confounder    — None (both columns flag each other)
        chi2_statistic       — chi-squared test statistic
        p_value              — significance of the association
        cramers_v            — effect size (0 = none, 1 = perfect)
        interpretation       — plain-English explanation

    Results are sorted by descending correlation_drop_pct (numeric) or
    ascending p_value (categorical), with numeric findings listed first.
    """
    if df is None or df.empty:
        return []

    # skip zero-variance cols
    all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [
        c for c in all_numeric
        if df[c].dropna().nunique() > 1 and df[c].dropna().std(ddof=0) > 0
    ]
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    findings = []

    # ------------------------------------------------------------------
    # Strategy 1: Numeric confounders
    # ------------------------------------------------------------------
    if len(numeric_cols) >= 3:

        # precompute correlations
        corr_matrix = df[numeric_cols].corr(method="pearson")

        # top pairs by |r|
        all_pairs = [
            (col_x, col_y, abs(corr_matrix.loc[col_x, col_y]))
            for col_x, col_y in combinations(numeric_cols, 2)
            if pd.notna(corr_matrix.loc[col_x, col_y])
            and abs(corr_matrix.loc[col_x, col_y]) >= XY_CORR_THRESHOLD
        ]
        all_pairs.sort(key=lambda t: t[2], reverse=True)
        top_pairs = all_pairs[:MAX_PAIRS]

        for col_x, col_y, raw_abs in top_pairs:
            raw_corr = corr_matrix.loc[col_x, col_y]

            # candidate confounders
            candidate_z = [
                z for z in numeric_cols
                if z not in (col_x, col_y)
                and pd.notna(corr_matrix.loc[col_x, z])
                and pd.notna(corr_matrix.loc[col_y, z])
                and abs(corr_matrix.loc[col_x, z]) >= Z_CORR_THRESHOLD
                and abs(corr_matrix.loc[col_y, z]) >= Z_CORR_THRESHOLD
            ]

            # sort by combined strength
            candidate_z.sort(
                key=lambda z: (
                    abs(corr_matrix.loc[col_x, z]) + abs(corr_matrix.loc[col_y, z])
                ),
                reverse=True,
            )

            for z_col in candidate_z:
                partial = _partial_correlation(df[col_x], df[col_y], df[z_col])
                if partial is None:
                    continue

                drop = _drop_pct(raw_corr, partial)

                if drop >= DROP_THRESHOLD * 100:
                    findings.append({
                        "type": "numeric_confounder",
                        "variable_x": col_x,
                        "variable_y": col_y,
                        "likely_confounder": z_col,
                        "raw_correlation": round(raw_corr, 4),
                        "partial_correlation": round(partial, 4),
                        "correlation_drop_pct": drop,
                        "interpretation": _interpret_numeric(
                            col_x, col_y, z_col, raw_corr, partial, drop
                        ),
                    })
                    # one per pair
                    break

    # ------------------------------------------------------------------
    # Strategy 2: Categorical associations
    # ------------------------------------------------------------------
    if len(categorical_cols) >= 2:
        for col_a, col_b in combinations(categorical_cols, 2):
            result = _chi2_association(df, col_a, col_b)
            if result is None:
                continue
            if result["p_value"] < CHI2_PVALUE_THRESHOLD:
                findings.append({
                    "type": "categorical_association",
                    "variable_x": col_a,
                    "variable_y": col_b,
                    "likely_confounder": None,
                    "chi2_statistic": result["chi2_statistic"],
                    "p_value": result["p_value"],
                    "cramers_v": result["cramers_v"],
                    "interpretation": _interpret_categorical(
                        col_a, col_b, result["p_value"], result["cramers_v"]
                    ),
                })

    # ------------------------------------------------------------------
    # Sort: numeric findings first (by drop %), then categorical (by p-value)
    numeric_findings     = sorted([f for f in findings if f["type"] == "numeric_confounder"],
                                   key=lambda f: f["correlation_drop_pct"], reverse=True)
    categorical_findings = sorted([f for f in findings if f["type"] == "categorical_association"],
                                   key=lambda f: f["p_value"])
    return numeric_findings + categorical_findings
