"""
sample_data.py — Synthetic demo dataset generator.

Produces a 1 000-row, 12-column DataFrame that is deliberately engineered to
trigger every Blind Spot analysis module:

  Module                   What is planted
  ─────────────────────── ──────────────────────────────────────────────────────
  Simpson's Paradox        ad_spend vs conversions: overall r > 0, but r < 0 in
                           2 of 3 regions (region is the hidden driver)
  Survivorship Bias        status: 85 % "active" — churned customers absent
  Confounding              experience_years drives both salary and performance;
                           salary-performance r drops ~50 % when controlled
  Metric Gaming            sales_target: 30 % pile-up on round numbers (100/200/500)
  Block Missing            engagement_score: 25 % NaN in an interior row band
  Temporal Gap             signup_date: no records in March or April 2023
  Blind Spots              no demographic columns, no cost/margin data
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.loader import _build_metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_demo_dataset(seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    Generate the synthetic demo dataset.

    Returns
    -------
    (df, metadata)
        df       — 1 000-row DataFrame with engineered bias signals
        metadata — same structure that load_data() would return
    """
    rng = np.random.default_rng(seed)
    n   = 1_000

    # 1. Region
    region = rng.choice(["North", "South", "West"], size=n, p=[0.60, 0.30, 0.10])

    # 2. Ad spend
    mean_spend = np.where(region == "North", 300,
                 np.where(region == "South", 150, 60)).astype(float)
    ad_spend = np.clip(mean_spend + rng.normal(0, 28, n), 10, None).astype(int)

    # 3. Conversions (Simpson's Paradox)
    base_conv = np.where(region == "North", 80,
                np.where(region == "South", 40, 15)).astype(float)
    region_mean_spend = np.where(region == "North", 300,
                        np.where(region == "South", 150, 60)).astype(float)
    conversions = np.clip(
        base_conv - 0.28 * (ad_spend - region_mean_spend) + rng.normal(0, 7, n),
        1, None
    ).astype(int)

    # 4. Experience years (confounder)
    experience_years = np.round(rng.uniform(1, 20, n), 1)

    # 5 & 6. Salary + Performance (partial correlation)
    talent = rng.normal(0, 1, n)   # latent ability

    performance_score = np.round(
        np.clip(38 + 2.6 * experience_years + 8.0 * talent + rng.normal(0, 8, n), 0, 100), 1
    )

    salary = np.clip(
        np.round(38_000 + 3_400 * experience_years + 4_000 * talent + rng.normal(0, 6_000, n), -2),
        25_000, 220_000,
    ).astype(int)

    # 7. Status (survivorship bias)
    status = rng.choice(
        ["active"] * 85 + ["churned"] * 8 + ["inactive"] * 5 + ["suspended"] * 2,
        size=n,
    )

    # 8. Sales target (metric gaming)
    sales_target = rng.integers(50, 1_001, size=n)
    mask = rng.random(n) < 0.30
    sales_target[mask] = rng.choice([100, 200, 500], size=int(mask.sum()))

    # 9. Engagement score (block missing)
    engagement_score = np.round(
        np.clip(50 + 1.2 * performance_score + rng.normal(0, 15, n), 0, 100), 1
    ).astype(float)
    engagement_score[375:625] = np.nan     # interior block → 25 % NaN

    # 10. Revenue
    revenue = np.round(
        np.clip(conversions * rng.uniform(40, 80, n) + rng.normal(0, 500, n), 0, None), 2
    )

    # 11. Customer segment
    segment = rng.choice(["Enterprise", "SMB", "Consumer"], size=n, p=[0.25, 0.45, 0.30])

    # 12. Signup date (temporal gap)
    all_days    = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    signup_date = pd.to_datetime(np.sort(
        rng.choice(all_days[~all_days.month.isin([3, 4])].values, size=n, replace=True)
    ))

    # assemble
    df = pd.DataFrame({
        "signup_date":       signup_date,
        "region":            region,
        "customer_segment":  segment,
        "status":            status,
        "experience_years":  experience_years,
        "salary":            salary,
        "performance_score": performance_score,
        "ad_spend":          ad_spend,
        "conversions":       conversions,
        "revenue":           revenue,
        "sales_target":      sales_target,
        "engagement_score":  engagement_score,
    })

    metadata = _build_metadata(df)
    return df, metadata
