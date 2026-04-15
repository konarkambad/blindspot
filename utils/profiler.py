"""
profiler.py — Data health-check utility.

Produces a comprehensive profiling dictionary covering missingness, type issues,
cardinality, outliers, correlation, and variance. Runs before any bias analysis
so that downstream modules work on well-understood data.
"""

import numpy as np
import pandas as pd
from itertools import combinations


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID_UNIQUENESS_THRESHOLD   = 0.95
HIGH_CORR_THRESHOLD       = 0.85
ZSCORE_CUTOFF             = 3.0
IQR_MULTIPLIER            = 1.5
ZERO_VARIANCE_THRESHOLD   = 0.001
MIN_ROWS_FOR_PATTERN      = 10


# ---------------------------------------------------------------------------
# 1. Missing summary
# ---------------------------------------------------------------------------

def _classify_missing_pattern(series: pd.Series) -> str:
    """
    Classify the pattern of missing values in a column.

    - 'monotone' : all nulls appear before all non-nulls (or vice-versa).
                   Common in time-series where a sensor was added/removed.
    - 'block'    : nulls cluster in a contiguous interior block.
                   Suggests a data outage or import gap.
    - 'random'   : nulls are scattered with no obvious structure.
    """
    if series.isnull().sum() == 0:
        return "none"

    null_mask = series.isnull().values
    n = len(null_mask)

    if n < MIN_ROWS_FOR_PATTERN:
        return "random"

    null_indices = np.where(null_mask)[0]
    first_null, last_null = null_indices[0], null_indices[-1]

    if null_mask[:last_null + 1].all() or null_mask[first_null:].all():
        return "monotone"

    non_null_before = not null_mask[:first_null].any() if first_null > 0 else True
    non_null_after  = not null_mask[last_null + 1:].any() if last_null < n - 1 else True
    if non_null_before and non_null_after:
        return "block"

    return "random"


def _missing_summary(df: pd.DataFrame) -> dict:
    """
    For each column: null count, percentage missing, and missing pattern type.
    Columns that are entirely null are flagged with pattern 'all_null'.
    """
    summary = {}
    n_rows = len(df)

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        pct = round(null_count / n_rows * 100, 2) if n_rows > 0 else 0.0

        if null_count == n_rows:
            pattern = "all_null"
        elif null_count > 0:
            pattern = _classify_missing_pattern(df[col])
        else:
            pattern = "none"

        summary[col] = {
            "null_count": null_count,
            "pct_missing": pct,
            "missing_pattern": pattern,
        }

    return summary


# ---------------------------------------------------------------------------
# 2. Type issues
# ---------------------------------------------------------------------------

def _looks_numeric(series: pd.Series) -> bool:
    """Return True if >80% of non-null values can be coerced to a number."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    return coerced.notna().mean() > 0.80


def _looks_datetime(series: pd.Series) -> bool:
    """Return True if >80% of non-null values parse as dates."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    # sample for speed
    sample = non_null.sample(min(200, len(non_null)), random_state=0)
    try:
        coerced = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        return coerced.notna().mean() > 0.80
    except Exception:
        return False


def _type_issues(df: pd.DataFrame) -> dict:
    """
    Detect columns where the stored dtype does not match the actual content.

    Returns a dict mapping column name -> issue description (only flagged cols).
    """
    issues = {}

    for col in df.columns:
        dtype = df[col].dtype

        # mistyped columns
        if dtype == object:
            if _looks_numeric(df[col]):
                issues[col] = "stored as object but content appears numeric"
            elif _looks_datetime(df[col]):
                issues[col] = "stored as object but content appears to be datetime"

        # possible boolean
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            unique_vals = df[col].dropna().unique()
            if set(unique_vals).issubset({0, 1}):
                issues[col] = "stored as numeric but content appears boolean (only 0/1)"

        # low-cardinality int
        elif pd.api.types.is_integer_dtype(dtype):
            n_unique = df[col].nunique()
            if 2 < n_unique <= 20:
                issues[col] = (
                    f"stored as integer with only {n_unique} unique values "
                    "— may be an encoded categorical"
                )

    return issues


# ---------------------------------------------------------------------------
# 3. Cardinality
# ---------------------------------------------------------------------------

def _cardinality(df: pd.DataFrame) -> dict:
    """
    For each column: unique count, uniqueness ratio, and ID-column flag.
    """
    result = {}
    n_rows = len(df)

    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))
        ratio    = round(n_unique / n_rows, 4) if n_rows > 0 else 0.0
        is_id    = ratio >= ID_UNIQUENESS_THRESHOLD and n_unique > 1

        result[col] = {
            "unique_count": n_unique,
            "uniqueness_ratio": ratio,
            "likely_id_column": is_id,
        }

    return result


# ---------------------------------------------------------------------------
# 4. Outlier flags
# ---------------------------------------------------------------------------

def _outlier_flags(df: pd.DataFrame) -> dict:
    """
    For each numeric column: count of outliers by z-score and by IQR method.
    Skips all-null columns and single-row datasets gracefully.
    """
    result = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        return {}

    for col in numeric_cols:
        series = df[col].dropna()

        if len(series) == 0:
            result[col] = {"zscore_outliers": None, "iqr_outliers": None,
                           "note": "all values are null"}
            continue

        if len(series) < 4:
            result[col] = {"zscore_outliers": None, "iqr_outliers": None,
                           "note": f"only {len(series)} non-null value(s) — insufficient for outlier analysis"}
            continue

        # z-score outliers
        mean, std = series.mean(), series.std()
        if std == 0 or pd.isna(std):
            zscore_count = 0
        else:
            z_scores     = ((series - mean) / std).abs()
            zscore_count = int((z_scores > ZSCORE_CUTOFF).sum())

        # IQR outliers
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            iqr_count = 0
        else:
            lower     = q1 - IQR_MULTIPLIER * iqr
            upper     = q3 + IQR_MULTIPLIER * iqr
            iqr_count = int(((series < lower) | (series > upper)).sum())

        result[col] = {
            "zscore_outliers": zscore_count,
            "iqr_outliers": iqr_count,
        }

    return result


# ---------------------------------------------------------------------------
# 5. Constant columns
# ---------------------------------------------------------------------------

def _constant_columns(df: pd.DataFrame) -> list:
    """
    Return list of column names that contain only one distinct value.
    """
    return [col for col in df.columns if df[col].nunique(dropna=False) <= 1]


# ---------------------------------------------------------------------------
# 6. High-correlation pairs
# ---------------------------------------------------------------------------

def _high_correlation_pairs(df: pd.DataFrame) -> list:
    """
    Return list of (col_a, col_b, correlation) for pairs with |r| > threshold.
    Only considers numeric columns with at least 2 non-null values and
    non-zero variance (constant columns produce undefined correlations).
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2 or len(numeric_df) < 2:
        return []

    # skip constant columns
    valid_cols = [
        col for col in numeric_df.columns
        if numeric_df[col].dropna().nunique() > 1
        and numeric_df[col].dropna().std() > 0
    ]
    if len(valid_cols) < 2:
        return []

    try:
        corr_matrix = numeric_df[valid_cols].corr(method="pearson")
    except Exception:
        return []

    flagged = []
    for col_a, col_b in combinations(corr_matrix.columns, 2):
        r = corr_matrix.loc[col_a, col_b]
        if pd.notna(r) and abs(r) > HIGH_CORR_THRESHOLD:
            flagged.append({
                "col_a": col_a,
                "col_b": col_b,
                "correlation": round(float(r), 4),
            })

    flagged.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return flagged


# ---------------------------------------------------------------------------
# 7. Zero-variance columns
# ---------------------------------------------------------------------------

def _zero_variance_columns(df: pd.DataFrame) -> list:
    """
    Return list of numeric column names whose variance is below the threshold.
    These carry almost no information and can destabilise statistical models.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return [
        col for col in numeric_df.columns
        if numeric_df[col].var(ddof=0) < ZERO_VARIANCE_THRESHOLD
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_data(df: pd.DataFrame) -> dict:
    """
    Run a full health-check on a DataFrame and return a nested profiling dict.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to profile. Should already be loaded via loader.load_data().

    Returns
    -------
    dict with keys:
        missing_summary        — per-column null stats and pattern classification
        type_issues            — columns whose dtype mismatches their content
        cardinality            — per-column unique counts and ID-column flags
        outlier_flags          — per numeric column outlier counts (z-score & IQR)
        constant_columns       — list of columns with only one unique value
        high_correlation_pairs — list of strongly correlated numeric column pairs
        zero_variance_columns  — list of near-constant numeric columns
    """
    if df is None or df.empty:
        return {"error": "DataFrame is None or empty — nothing to profile."}

    warnings = []

    if df.select_dtypes(include=[np.number]).shape[1] == 0:
        warnings.append("No numeric columns found — outlier, correlation, and variance analyses are skipped.")
    if len(df) == 1:
        warnings.append("Dataset has only 1 row — most statistical analyses are not meaningful.")

    return {
        "missing_summary":        _missing_summary(df),
        "type_issues":            _type_issues(df),
        "cardinality":            _cardinality(df),
        "outlier_flags":          _outlier_flags(df),
        "constant_columns":       _constant_columns(df),
        "high_correlation_pairs": _high_correlation_pairs(df),
        "zero_variance_columns":  _zero_variance_columns(df),
        "warnings":               warnings,
    }
