"""
blind_spots.py — "What This Data Cannot Tell You" generator.

Most analytical tools describe what data shows. This module does the opposite:
it articulates what the data structurally *cannot* answer — the questions that
will never appear in any chart because the required columns simply don't exist.

This is the most important thing an analyst can communicate to a decision-maker,
and the thing that is almost never documented.
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Keyword vocabularies for domain inference and column classification
# ---------------------------------------------------------------------------

DOMAIN_SIGNALS = {
    "sales":      ["revenue", "order", "sale", "purchase", "transaction", "invoice",
                   "quantity", "price", "discount", "cart", "checkout", "sku"],
    "hr":         ["employee", "salary", "hire", "termination", "department", "tenure",
                   "headcount", "payroll", "performance", "attrition", "leave", "fte"],
    "marketing":  ["campaign", "impression", "click", "conversion", "ctr", "cpm", "cpc",
                   "channel", "lead", "funnel", "ad", "spend", "engagement", "reach"],
    "finance":    ["profit", "loss", "asset", "liability", "balance", "cash", "budget",
                   "forecast", "expense", "cost", "margin", "ebitda", "debt", "equity"],
    "healthcare": ["patient", "diagnosis", "treatment", "drug", "dosage", "hospital",
                   "admission", "discharge", "icd", "claim", "provider", "condition"],
    "product":    ["feature", "release", "bug", "ticket", "sprint", "user", "session",
                   "event", "funnel", "retention", "dau", "mau", "churn", "cohort"],
    "logistics":  ["shipment", "delivery", "carrier", "warehouse", "route", "tracking",
                   "freight", "inventory", "sku", "lead_time", "fulfillment", "return"],
}

# column patterns
TEMPORAL_PATTERNS   = [r"date", r"time", r"timestamp", r"_at$", r"_on$", r"year",
                       r"month", r"day", r"week", r"quarter", r"period"]
DEMOGRAPHIC_PATTERNS= [r"age", r"gender", r"sex", r"race", r"ethnicity", r"income",
                       r"education", r"marital", r"household", r"segment", r"cohort",
                       r"persona", r"generation"]
GEOGRAPHIC_PATTERNS = [r"country", r"region", r"city", r"state", r"zip", r"postal",
                       r"location", r"territory", r"market", r"lat", r"lon",
                       r"address", r"district", r"province"]
COST_PATTERNS       = [r"cost", r"expense", r"spend", r"cogs", r"overhead",
                       r"budget", r"opex", r"capex", r"loss"]
REVENUE_PATTERNS    = [r"revenue", r"sales", r"income", r"gmv", r"arr", r"mrr",
                       r"price", r"amount", r"value", r"total"]
ENTITY_ID_PATTERNS  = [r"customer_id", r"user_id", r"client_id", r"account_id",
                       r"patient_id", r"employee_id", r"member_id", r"person_id",
                       r"entity_id", r"subject_id"]
OUTCOME_PATTERNS    = [r"target", r"label", r"outcome", r"result", r"score",
                       r"rating", r"flag", r"churn", r"converted", r"success"]
COMPETITOR_PATTERNS = [r"competitor", r"benchmark", r"market_share", r"industry",
                       r"peer", r"rival", r"comparison"]
QUALITY_PATTERNS    = [r"quality", r"defect", r"error", r"complaint", r"return",
                       r"refund", r"reject", r"fail", r"issue", r"incident"]
SENTIMENT_PATTERNS  = [r"sentiment", r"review", r"feedback", r"rating", r"nps",
                       r"satisfaction", r"comment", r"survey", r"opinion"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matches_any(col_name: str, patterns: list) -> bool:
    """Return True if a column name matches any regex pattern in the list."""
    name = col_name.lower().replace(" ", "_")
    return any(re.search(p, name) for p in patterns)


def _cols_matching(df: pd.DataFrame, patterns: list) -> list:
    """Return list of column names that match any pattern."""
    return [c for c in df.columns if _matches_any(c, patterns)]


def _infer_domain(df: pd.DataFrame) -> list:
    """
    Score each domain by counting how many of its signal words appear in
    column names.  Returns the top 1-2 domains (those above a threshold).
    """
    col_text = " ".join(df.columns).lower().replace(" ", "_")
    scores   = {d: sum(1 for kw in kws if kw in col_text)
                for d, kws in DOMAIN_SIGNALS.items()}
    scores   = {d: s for d, s in scores.items() if s > 0}
    if not scores:
        return ["general"]
    threshold = max(1, max(scores.values()) * 0.5)
    return [d for d, s in sorted(scores.items(), key=lambda x: -x[1]) if s >= threshold][:2]


def _has_repeated_entity(df: pd.DataFrame) -> bool:
    """
    Return True if an entity-ID column exists AND that column has repeated
    values — meaning the same entity appears in multiple rows (panel structure).
    """
    id_cols = _cols_matching(df, ENTITY_ID_PATTERNS)
    for col in id_cols:
        if df[col].duplicated().any():
            return True
    return False


def _low_cardinality_categoricals(df: pd.DataFrame) -> list:
    """Return categorical columns with fewer than 3 unique non-null values."""
    return [
        col for col in df.select_dtypes(include="object").columns
        if 1 < df[col].nunique(dropna=True) < 3
    ]


def _all_numeric(df: pd.DataFrame) -> bool:
    """True if virtually every column is numeric (no categorical signals)."""
    n_numeric = df.select_dtypes(include="number").shape[1]
    return n_numeric >= len(df.columns) * 0.85


def _has_high_missing(profile_dict: dict, threshold: float = 0.30) -> list:
    """Return columns with > threshold fraction missing."""
    missing = profile_dict.get("missing_summary", {})
    return [
        col for col, ev in missing.items()
        if ev.get("pct_missing", 0) / 100 > threshold
    ]


# ---------------------------------------------------------------------------
# Blind spot rules
# ---------------------------------------------------------------------------

def _check_temporal(df, profile_dict, domains) -> dict | None:
    temporal_cols = _cols_matching(df, TEMPORAL_PATTERNS)
    # Also accept datetime-dtype columns
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    if not temporal_cols and not datetime_cols:
        return {
            "blind_spot_description": "This data cannot show trends, seasonality, or changes over time.",
            "missing_data_type": "Temporal / time-series dimension",
            "suggested_source": (
                "Add a transaction timestamp, event date, or reporting period column. "
                "Without it, every observation is treated as simultaneous and no "
                "before/after or trend analysis is possible."
            ),
            "impact_level": "high",
        }
    return None


def _check_demographic(df, profile_dict, domains) -> dict | None:
    demo_cols = _cols_matching(df, DEMOGRAPHIC_PATTERNS)
    if not demo_cols and "hr" not in domains:
        return {
            "blind_spot_description": (
                "This data cannot reveal demographic disparities, equity gaps, "
                "or segment-level differences across population groups."
            ),
            "missing_data_type": "Demographic attributes (age, gender, income band, etc.)",
            "suggested_source": (
                "Join with a CRM, HR system, or survey dataset that captures "
                "demographic dimensions. Census linkage is also possible for "
                "geographic data using postal codes."
            ),
            "impact_level": "high" if any(d in domains for d in ["hr", "healthcare", "marketing"]) else "medium",
        }
    return None


def _check_cost_vs_revenue(df, profile_dict, domains) -> dict | None:
    has_revenue = bool(_cols_matching(df, REVENUE_PATTERNS))
    has_cost    = bool(_cols_matching(df, COST_PATTERNS))
    if has_revenue and not has_cost:
        return {
            "blind_spot_description": (
                "Revenue is captured but costs are absent — profitability, margin, "
                "and ROI analysis are structurally impossible with this data alone."
            ),
            "missing_data_type": "Cost / expense data",
            "suggested_source": (
                "Join with a finance or ERP system (e.g., NetSuite, SAP) that holds "
                "COGS, operating expenses, and overhead allocations per "
                "transaction or time period."
            ),
            "impact_level": "high",
        }
    return None


def _check_geographic(df, profile_dict, domains) -> dict | None:
    geo_cols = _cols_matching(df, GEOGRAPHIC_PATTERNS)
    if not geo_cols:
        return {
            "blind_spot_description": (
                "No geographic dimension exists — regional performance differences, "
                "location-based clustering, and spatial risk cannot be assessed."
            ),
            "missing_data_type": "Geographic / location data (country, region, postal code, lat/lon)",
            "suggested_source": (
                "Add country/region/city columns from your CRM or order management "
                "system. IP-based geolocation can fill gaps for digital event data."
            ),
            "impact_level": "medium",
        }
    return None


def _check_competitor(df, profile_dict, domains) -> dict | None:
    comp_cols = _cols_matching(df, COMPETITOR_PATTERNS)
    if not comp_cols and any(d in domains for d in ["sales", "marketing", "finance"]):
        return {
            "blind_spot_description": (
                "There is no benchmark or competitor data — whether your metrics "
                "are good or bad cannot be determined in isolation. A 20% conversion "
                "rate is excellent or terrible depending on the industry baseline."
            ),
            "missing_data_type": "Competitive benchmarks / industry reference data",
            "suggested_source": (
                "Supplement with industry reports (Gartner, Forrester), "
                "public company filings, or platforms like Similarweb, Semrush, "
                "or Crunchbase for competitive context."
            ),
            "impact_level": "medium",
        }
    return None


def _check_panel_structure(df, profile_dict, domains) -> dict | None:
    id_cols = _cols_matching(df, ENTITY_ID_PATTERNS)
    if id_cols and not _has_repeated_entity(df):
        entity_word = {
            "hr": "employee", "healthcare": "patient",
            "sales": "customer", "product": "user",
        }.get(domains[0] if domains else "", "entity")
        return {
            "blind_spot_description": (
                f"Each {entity_word} appears only once — this is a cross-sectional "
                "snapshot, not a panel. Individual-level change over time, "
                "cohort trajectories, and before/after effects cannot be measured."
            ),
            "missing_data_type": "Repeated observations per entity (longitudinal / panel data)",
            "suggested_source": (
                f"Collect or join historical records for each {entity_word} across "
                "multiple time periods. Even two snapshots (before/after) unlock "
                "difference-in-differences and growth analysis."
            ),
            "impact_level": "high" if any(d in domains for d in ["hr", "healthcare", "product"]) else "medium",
        }
    return None


def _check_low_cardinality_groups(df, profile_dict, domains) -> dict | None:
    low_card = _low_cardinality_categoricals(df)
    if low_card:
        col_list = ", ".join(f"'{c}'" for c in low_card[:4])
        return {
            "blind_spot_description": (
                f"Categorical column(s) {col_list} have fewer than 3 distinct groups. "
                "Statistical comparisons between groups will have limited power — "
                "small subgroup sizes inflate uncertainty and make effect detection unreliable."
            ),
            "missing_data_type": "Finer-grained categorical segmentation",
            "suggested_source": (
                "Expand the taxonomy: if 'region' only has North/South, add city-level "
                "data. If 'tier' only has Free/Paid, add sub-tiers. More granular "
                "segmentation enables meaningful group-level analysis."
            ),
            "impact_level": "medium",
        }
    return None


def _check_outcome_variable(df, profile_dict, domains) -> dict | None:
    outcome_cols = _cols_matching(df, OUTCOME_PATTERNS)
    if not outcome_cols:
        return {
            "blind_spot_description": (
                "No clear outcome or target variable is present. Without knowing "
                "what 'success' looks like in this data, it is impossible to determine "
                "which features predict good outcomes vs. bad ones."
            ),
            "missing_data_type": "Outcome / target variable (conversion, churn, score, label)",
            "suggested_source": (
                "Define and join a ground-truth outcome: did the customer churn? "
                "Did the patient recover? Did the campaign convert? This is the "
                "minimum requirement for any causal or predictive analysis."
            ),
            "impact_level": "high" if any(d in domains for d in ["marketing", "product", "healthcare"]) else "medium",
        }
    return None


def _check_qualitative_context(df, profile_dict, domains) -> dict | None:
    sentiment_cols = _cols_matching(df, SENTIMENT_PATTERNS)
    quality_cols   = _cols_matching(df, QUALITY_PATTERNS)
    all_num = _all_numeric(df)
    if all_num and not sentiment_cols and not quality_cols:
        return {
            "blind_spot_description": (
                "All columns are numeric — the 'why' behind the numbers is absent. "
                "Quantitative data can show that a metric dropped; it cannot explain "
                "whether that is because of product changes, market shifts, or "
                "operational failures."
            ),
            "missing_data_type": "Qualitative / contextual signals (feedback, notes, event logs, sentiment)",
            "suggested_source": (
                "Integrate support tickets, NPS comments, Slack incident logs, "
                "or customer feedback surveys. Text data provides the causal "
                "narrative that numbers alone never can."
            ),
            "impact_level": "medium",
        }
    return None


def _check_external_factors(df, profile_dict, domains) -> dict | None:
    """No macro/external columns → can't separate internal performance from external forces."""
    macro_keywords = ["gdp", "inflation", "interest", "macro", "index",
                      "market", "economy", "pandemic", "season", "weather"]
    col_text = " ".join(df.columns).lower()
    if not any(kw in col_text for kw in macro_keywords) and any(d in domains for d in ["finance", "sales", "marketing"]):
        return {
            "blind_spot_description": (
                "External / macroeconomic context is absent. It is impossible to "
                "separate organic performance from market tailwinds or headwinds — "
                "a revenue increase during a boom may mean nothing, and a dip during "
                "a recession may hide strong underlying performance."
            ),
            "missing_data_type": "External economic / environmental signals",
            "suggested_source": (
                "Overlay public datasets: FRED for economic indicators, Google Trends "
                "for search demand, weather APIs for seasonal signals, or industry "
                "indices from Bloomberg / Reuters."
            ),
            "impact_level": "medium",
        }
    return None


def _check_missing_heavy(df, profile_dict, domains) -> dict | None:
    """Columns with >30% missing are blind spots themselves."""
    heavy = _has_high_missing(profile_dict, threshold=0.30)
    if heavy:
        col_list = ", ".join(f"'{c}'" for c in heavy[:4])
        more     = f" (+{len(heavy)-4} more)" if len(heavy) > 4 else ""
        return {
            "blind_spot_description": (
                f"Column(s) {col_list}{more} are more than 30% empty. "
                "Any analysis involving these columns applies only to the subset "
                "of records where data exists — and that subset may be "
                "systematically different from the rest."
            ),
            "missing_data_type": "Complete records for heavily incomplete columns",
            "suggested_source": (
                "Investigate why values are missing at the source system level. "
                "If missingness is non-random (e.g., only certain user types fill "
                "in a field), imputation will introduce bias — the better fix is "
                "upstream data collection."
            ),
            "impact_level": "high",
        }
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# rule functions
_RULES = [
    _check_temporal,
    _check_outcome_variable,
    _check_cost_vs_revenue,
    _check_panel_structure,
    _check_demographic,
    _check_geographic,
    _check_competitor,
    _check_low_cardinality_groups,
    _check_qualitative_context,
    _check_external_factors,
    _check_missing_heavy,
]

_IMPACT_ORDER = {"high": 0, "medium": 1, "low": 2}


def generate_blind_spots(df: pd.DataFrame, profile_dict: dict) -> list:
    """
    Analyse column names and data structure to identify what questions
    this dataset structurally cannot answer.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    profile_dict : dict
        Output of profiler.profile_data(df).

    Returns
    -------
    list of dict, sorted by impact_level (high → medium → low), each with:
        blind_spot_description — what analysis is impossible and why
        missing_data_type      — the category of data that is absent
        suggested_source       — where to find it
        impact_level           — "high", "medium", or "low"
    """
    if df is None or df.empty:
        return []

    domains = _infer_domain(df)
    findings = []

    for rule_fn in _RULES:
        try:
            result = rule_fn(df, profile_dict, domains)
            if result is not None:
                findings.append(result)
        except Exception:
            continue

    # sort by impact
    findings.sort(key=lambda f: _IMPACT_ORDER.get(f.get("impact_level", "low"), 2))
    return findings
