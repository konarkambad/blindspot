"""
metric_risk.py — Metric gaming risk scorer.

Senior analysts know that data is not always passively collected — it is
sometimes actively shaped by the people being measured. When employees,
vendors, or systems know which metrics are tracked, those metrics get gamed.

This module treats each numeric column as potentially adversarial and scores
it across five dimensions of gaming susceptibility:

  1. Concentration risk   — a few inflated records dominate the aggregate.
  2. Round number bias    — suspiciously frequent multiples suggest manual entry.
  3. Benford's Law        — first-digit distribution diverges from natural data.
  4. Spike detection      — unusual pile-ups at psychologically salient values.
  5. Weekend/time pattern — metric behaves unnaturally uniformly across time.

Each dimension contributes a weighted sub-score. The total gaming_risk_score
is 0–100, where higher means more suspicious.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Weights (must sum to 100)
# ---------------------------------------------------------------------------

W_CONCENTRATION = 25
W_ROUND_NUMBER  = 20
W_BENFORD       = 25
W_SPIKE         = 15
W_WEEKEND       = 15

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Benford expected probs
BENFORD_EXPECTED = np.array([
    np.log10(1 + 1 / d) for d in range(1, 10)
])

# round moduli
ROUND_MODULI = [5, 10, 50, 100]

# spike anchors
SPIKE_ANCHORS_EXPONENTS = range(-1, 7)

# spike threshold (3 %)
SPIKE_DENSITY_THRESHOLD = 0.03

# min rows
MIN_ROWS = 20

# min Benford rows
MIN_ROWS_BENFORD = 50

# min weekend rows
MIN_WEEKEND_ROWS = 10


# ---------------------------------------------------------------------------
# Risk label helper
# ---------------------------------------------------------------------------

def _risk_label(score: float) -> str:
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Sub-scorer 1 — Concentration risk (Pareto / top-10 % share)
# ---------------------------------------------------------------------------

def _score_concentration(series: pd.Series) -> tuple[float, dict]:
    """
    Measure what share of the total absolute sum is held by the top 10 % of
    values.  A 100 % Pareto concentration (one record = everything) scores 25.
    A perfectly uniform distribution scores 0.

    Returns (sub_score, evidence_dict).
    """
    pos = series.dropna()
    pos = pos[pos > 0]   # positive values only

    if len(pos) < MIN_ROWS:
        return 0.0, {"note": "insufficient positive values"}

    cutoff      = pos.quantile(0.90)
    top10_share = float(pos[pos >= cutoff].sum() / pos.sum()) if pos.sum() > 0 else 0.0
    sub_score   = round(max(0.0, (top10_share - 0.10) / 0.90) * W_CONCENTRATION, 2)
    return sub_score, {"top10pct_share": round(top10_share, 4), "p90_threshold": round(float(cutoff), 4)}


# ---------------------------------------------------------------------------
# Sub-scorer 2 — Round number bias
# ---------------------------------------------------------------------------

def _score_round_numbers(series: pd.Series) -> tuple[float, dict]:
    """
    Count what fraction of values are exact multiples of 5, 10, 50, or 100.
    Naturally measured data rarely clusters at round numbers; manual or
    target-chasing data frequently does.

    Returns (sub_score, evidence_dict).
    """
    vals = series.dropna()
    if len(vals) < MIN_ROWS:
        return 0.0, {"note": "insufficient data"}

    # integer check (floats with .0 included)
    integer_mask = (vals % 1 == 0)
    int_vals = vals[integer_mask]

    if len(int_vals) == 0:
        return 0.0, {"round_fraction": 0.0, "integer_fraction": 0.0}

    round_mask = pd.Series(False, index=int_vals.index)
    for mod in ROUND_MODULI:
        round_mask |= (int_vals % mod == 0)

    round_fraction   = float(round_mask.sum() / len(vals))
    integer_fraction = float(len(int_vals) / len(vals))
    sub_score        = round(min(max(0.0, round_fraction - 0.20) / 0.80, 1.0) * W_ROUND_NUMBER, 2)
    return sub_score, {
        "round_fraction":   round(round_fraction, 4),
        "integer_fraction": round(integer_fraction, 4),
        "moduli_checked":   ROUND_MODULI,
    }


# ---------------------------------------------------------------------------
# Sub-scorer 3 — Benford's Law deviation
# ---------------------------------------------------------------------------

def _extract_first_digits(series: pd.Series) -> np.ndarray | None:
    """
    Extract the leading significant digit (1–9) from each non-zero value.
    Uses absolute values so negative columns are handled correctly.
    Returns None if there are not enough values.
    """
    non_null = series.dropna()
    # use absolute values
    pos = non_null.abs()
    pos = pos[pos > 0]   # exclude exact zeros (no leading digit)

    if len(pos) < MIN_ROWS_BENFORD:
        return None

    try:
        magnitudes = np.floor(np.log10(pos.values.astype(float)))
        leading = np.floor(pos.values / (10 ** magnitudes)).astype(int)
        leading = leading[(leading >= 1) & (leading <= 9)]
    except Exception:
        return None

    return leading if len(leading) >= MIN_ROWS_BENFORD else None


def _score_benford(series: pd.Series) -> tuple[float, dict]:
    """
    Compare the observed first-digit distribution against Benford's Law using
    a chi-squared goodness-of-fit test. Large deviation scores up to 25 points.

    Natural transaction-like data (counts, revenues, populations) follows
    Benford's Law. Fabricated or heavily manipulated data typically does not.

    Handles negative columns by applying Benford to absolute values.
    Note: Benford's Law does NOT apply to bounded data (e.g., percentages,
    ages, temperatures). We skip columns whose range spans less than one
    order of magnitude to avoid false positives.
    """
    vals = series.dropna()

    # All-zero column — no first digits to analyse
    if (vals == 0).all():
        return 0.0, {"note": "all values are zero — Benford not applicable"}

    # Use absolute values for magnitude analysis
    vals_abs = vals.abs()
    vals_pos = vals_abs[vals_abs > 0]

    if len(vals_pos) < MIN_ROWS_BENFORD:
        return 0.0, {"note": "insufficient non-zero values for Benford analysis"}

    # Benford's Law only applies to data spanning at least 2 orders of magnitude
    # (i.e. max/min ratio > 100). Bounded columns like ages, scores, or years
    # produce false positives because their first-digit distribution is
    # determined by the fixed range, not by any natural logarithmic process.
    try:
        data_range_orders = (
            np.log10(vals_pos.max()) - np.log10(vals_pos.min())
            if vals_pos.min() > 0 else 0
        )
    except Exception:
        data_range_orders = 0
    if data_range_orders < 2.0:
        return 0.0, {"note": "range too narrow for Benford analysis (requires max/min ratio > 100)"}

    first_digits = _extract_first_digits(series)
    if first_digits is None:
        return 0.0, {"note": "could not extract first digits"}

    n = len(first_digits)
    observed_counts = np.array([
        np.sum(first_digits == d) for d in range(1, 10)
    ])
    expected_counts = BENFORD_EXPECTED * n

    try:
        chi2, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
    except Exception:
        return 0.0, {"note": "chi-squared test failed"}

    observed_dist = (observed_counts / n).round(4).tolist()

    # p-value to score
    if p_value >= 0.10:
        sub_score = 0.0
    elif p_value <= 0.001:
        sub_score = float(W_BENFORD)
    else:
        # log-linear interpolation
        normalised = (np.log10(0.10) - np.log10(p_value)) / (np.log10(0.10) - np.log10(0.001))
        sub_score = round(float(np.clip(normalised, 0, 1) * W_BENFORD), 2)

    return sub_score, {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "observed_first_digit_dist": observed_dist,
        "expected_first_digit_dist": BENFORD_EXPECTED.round(4).tolist(),
        "n_values_used": n,
    }


# ---------------------------------------------------------------------------
# Sub-scorer 4 — Spike detection
# ---------------------------------------------------------------------------

def _score_spikes(series: pd.Series) -> tuple[float, dict]:
    """
    Detect unusual pile-ups at psychologically salient values: powers of 10,
    their halves (50, 500, …), and small multiples (200, 2000, …).

    Target-chasing behaviour (hitting 100 %, reaching $1 000 revenue, staying
    just above a 0 threshold) produces sharp spikes at specific values that
    would be extremely unlikely in natural data.

    Returns (sub_score, evidence_dict).
    """
    vals = series.dropna()
    if len(vals) < MIN_ROWS:
        return 0.0, {"note": "insufficient data"}

    n = len(vals)

    # anchor values
    anchors = set()
    for exp in SPIKE_ANCHORS_EXPONENTS:
        base = 10 ** exp
        for multiplier in [1, 2, 5, 0.5, 0.25]:
            anchors.add(round(base * multiplier, 10))

    spikes_found = []
    for anchor in sorted(anchors):
        density = float((vals == anchor).sum() / n)
        if density >= SPIKE_DENSITY_THRESHOLD:
            spikes_found.append({"value": anchor, "density": round(density, 4)})

    if not spikes_found:
        return 0.0, {"spikes_detected": []}

    max_density = max(s["density"] for s in spikes_found)
    sub_score   = round(min((max_density - SPIKE_DENSITY_THRESHOLD) / (0.30 - SPIKE_DENSITY_THRESHOLD), 1.0) * W_SPIKE, 2)
    return sub_score, {"spikes_detected": spikes_found}


# ---------------------------------------------------------------------------
# Sub-scorer 5 — Weekend / time pattern
# ---------------------------------------------------------------------------

def _find_datetime_column(df: pd.DataFrame) -> str | None:
    """Return the name of the first usable datetime column in df, or None."""
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        return col
    # Try coercing object columns
    for col in df.select_dtypes(include="object").columns:
        try:
            coerced = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if coerced.notna().mean() > 0.80:
                return col
        except Exception:
            continue
    return None


def _score_weekend_pattern(
    series: pd.Series, df: pd.DataFrame, date_col: str | None
) -> tuple[float, dict]:
    """
    Compare the metric's mean and variance on weekends vs. weekdays.

    Legitimate transactional data almost always differs between weekdays and
    weekends (lower sales on Sunday, fewer support tickets, etc.). Suspiciously
    uniform behaviour across all days — or unnaturally high weekend values —
    may indicate automated entry, copy-paste filling, or fabrication.

    Scores based on how similar (rather than different) the two distributions
    are — uniformity is the red flag here.
    """
    if date_col is None:
        return 0.0, {"note": "no datetime column found"}

    try:
        dt = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    except Exception:
        return 0.0, {"note": "could not parse datetime column"}

    combined = pd.concat([series, dt], axis=1).dropna()
    combined.columns = ["metric", "date"]

    if len(combined) < MIN_WEEKEND_ROWS * 2:
        return 0.0, {"note": "insufficient rows for weekend analysis"}

    combined["is_weekend"] = combined["date"].dt.dayofweek >= 5  # Sat=5, Sun=6
    weekday = combined.loc[~combined["is_weekend"], "metric"]
    weekend = combined.loc[combined["is_weekend"], "metric"]

    if len(weekday) < MIN_WEEKEND_ROWS or len(weekend) < MIN_WEEKEND_ROWS:
        return 0.0, {"note": "insufficient weekend or weekday rows"}

    weekday_mean = float(weekday.mean())
    weekend_mean = float(weekend.mean())
    overall_std = float(combined["metric"].std())

    # Run a two-sample t-test
    try:
        t_stat, p_value = stats.ttest_ind(weekday, weekend, equal_var=False)
    except Exception:
        return 0.0, {"note": "t-test failed"}

    # Cohen's d proxy
    mean_diff = abs(weekday_mean - weekend_mean)
    cohens_d = mean_diff / overall_std if overall_std > 0 else 0.0

    # uniformity is the red flag
    if p_value > 0.50 and len(combined) >= 100:
        # Uniformity is suspicious — score based on how high the p-value is
        normalised = min((p_value - 0.50) / 0.50, 1.0)
        sub_score = round(normalised * W_WEEKEND, 2)
    else:
        sub_score = 0.0

    return sub_score, {
        "weekday_mean": round(weekday_mean, 4),
        "weekend_mean": round(weekend_mean, 4),
        "t_statistic": round(float(t_stat), 4),
        "p_value_weekday_vs_weekend": round(float(p_value), 6),
        "cohens_d": round(cohens_d, 4),
        "n_weekday": int(len(weekday)),
        "n_weekend": int(len(weekend)),
    }


# ---------------------------------------------------------------------------
# Interpretation builder
# ---------------------------------------------------------------------------

def _build_interpretation(col: str, score: float, breakdown: dict) -> str:
    """
    Compose a plain-English interpretation prioritising the highest sub-scores.
    """
    risk = _risk_label(score)
    parts = []

    if breakdown["concentration"]["top10pct_share"] > 0.60:
        share = breakdown["concentration"]["top10pct_share"]
        parts.append(
            f"top 10 % of values account for {share:.0%} of the total sum "
            "(high Pareto concentration — aggregate easily moved by a few records)"
        )

    if breakdown["round_numbers"].get("round_fraction", 0) > 0.40:
        frac = breakdown["round_numbers"]["round_fraction"]
        parts.append(
            f"{frac:.0%} of values are round numbers "
            "(suggests manual entry, estimation, or target-matching)"
        )

    if breakdown["benford"].get("p_value", 1.0) < 0.05:
        p = breakdown["benford"]["p_value"]
        parts.append(
            f"first-digit distribution deviates from Benford's Law (p={p:.4f}) "
            "(natural transaction data typically follows Benford's — deviation is a red flag)"
        )

    spikes = breakdown["spike"].get("spikes_detected", [])
    if spikes:
        spike_vals = [str(s["value"]) for s in spikes[:3]]
        parts.append(
            f"unusual value pile-ups at {', '.join(spike_vals)} "
            "(possible target-chasing or threshold gaming)"
        )

    if breakdown["weekend_pattern"].get("p_value_weekday_vs_weekend", 0) > 0.50:
        parts.append(
            "weekday and weekend values are suspiciously uniform "
            "(legitimate data almost always varies by day of week)"
        )

    if not parts:
        return f"'{col}' shows {risk} gaming risk (score {score:.0f}/100). No dominant red flags detected."

    bullets = "; ".join(parts)
    return (
        f"'{col}' shows {risk} gaming risk (score {score:.0f}/100). "
        f"Key signals: {bullets}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_metric_gaming_risk(df: pd.DataFrame, profile_dict: dict) -> list:
    """
    Evaluate how easily each numeric column could be gamed or misinterpreted.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to analyse.
    profile_dict : dict
        Output of profiler.profile_data(df) — used to skip constant/zero-
        variance columns that cannot be meaningfully scored.

    Returns
    -------
    list of dict, sorted by gaming_risk_score descending, each containing:
        column_name        — name of the numeric column
        gaming_risk_score  — float in [0, 100]; higher = more suspicious
        breakdown          — dict with sub-scores and evidence for each dimension:
                               concentration, round_numbers, benford,
                               spike, weekend_pattern
        risk_level         — "low", "medium", or "high"
        interpretation     — plain-English summary of the main risk signals
    """
    if df is None or df.empty:
        return []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return []

    # find date column once
    date_col = _find_datetime_column(df)

    # columns to skip
    zero_var  = set(profile_dict.get("zero_variance_columns", []))
    constant  = set(profile_dict.get("constant_columns", []))
    all_null  = set(profile_dict.get("all_null_columns", []))
    skip      = zero_var | constant | all_null

    # min sample
    MIN_SAMPLE = 30

    results = []

    for col in numeric_cols:
        if col in skip:
            continue

        series = df[col]
        non_null = series.dropna()

        # row count check
        if len(non_null) < MIN_ROWS:
            continue

        insufficient_sample = len(non_null) < MIN_SAMPLE

        # all-zero check
        all_zeros = (non_null == 0).all()

        # run sub-scorers
        conc_score,    conc_ev    = _score_concentration(series)
        round_score,   round_ev   = _score_round_numbers(series)

        if all_zeros:
            benford_score = 0.0
            benford_ev    = {"note": "all values are zero — Benford not applicable"}
            spike_score   = 0.0
            spike_ev      = {"note": "all values are zero — spike detection skipped"}
        else:
            benford_score, benford_ev = _score_benford(series)
            spike_score,   spike_ev   = _score_spikes(series)

        weekend_score, weekend_ev = _score_weekend_pattern(series, df, date_col)

        total = conc_score + round_score + benford_score + spike_score + weekend_score
        total = round(min(total, 100.0), 2)

        if insufficient_sample:
            # dampen small samples
            total = round(total * 0.5, 2)
            benford_ev.setdefault("note", f"small sample ({len(non_null)} rows) — scores dampened")

        breakdown = {
            "concentration":   {"sub_score": conc_score,    **conc_ev},
            "round_numbers":   {"sub_score": round_score,   **round_ev},
            "benford":         {"sub_score": benford_score, **benford_ev},
            "spike":           {"sub_score": spike_score,   **spike_ev},
            "weekend_pattern": {"sub_score": weekend_score, **weekend_ev},
        }

        results.append({
            "column_name": col,
            "gaming_risk_score": total,
            "breakdown": breakdown,
            "risk_level": _risk_label(total),
            "interpretation": _build_interpretation(col, total, breakdown),
        })

    results.sort(key=lambda r: r["gaming_risk_score"], reverse=True)
    return results
