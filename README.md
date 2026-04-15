# Blind Spot — Know What Your Data Hides

**Upload any CSV or Excel file. Get a full audit of what it conceals before you draw a single conclusion.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-ff4b4b)

---

## Why This Exists

Most tools tell you what is *in* your data. Blind Spot tells you what is **missing, misleading, or dangerous** before you model, report, or act on it. The problems that destroy analyses — Simpson's Paradox, survivorship bias, hidden confounders, metric gaming — are invisible to standard profilers. Blind Spot makes them visible.

---

## Features

| Module | What it catches |
|---|---|
| **Simpson's Paradox Scanner** | Correlations that flip sign when data is split into subgroups |
| **Survivorship Bias Detector** | Temporal gaps, status skew, and outcome imbalance revealing filtered-out cases |
| **Confounder Warnings** | Third variables that inflate or fabricate correlations between two others |
| **Metric Gaming Risk Scorer** | Benford deviation, round-number pile-ups, and spike patterns in numeric columns |
| **Data Confidence Score** | A single A–F grade synthesising every finding, with a plain-English recommendation |
| **Blind Spot Report** | Structural gaps the dataset cannot answer regardless of analytical method |
| **Exportable Audit Report** | Self-contained HTML report, printable to PDF, with all findings and charts |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/konarkambad/blindspot.git
cd blindspot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

No API key, no database, no configuration required. Open your browser to `http://localhost:8501` and upload any CSV or Excel file — or click **Try Demo Dataset** to run the full audit instantly on an engineered synthetic dataset.

---

## How It Works

```
Upload            Profile              Audit                 Report
──────────────    ─────────────────    ──────────────────    ─────────────────────
CSV / Excel   →   Missingness,     →   7 bias detectors  →   Confidence score,
up to 200 MB      type issues,         run in sequence,      per-module findings,
                  cardinality,         each producing        downloadable HTML
                  outliers,            structured evidence   audit report
                  correlations         and risk ratings
```

---

## Sample Output

> Screenshot coming soon — run `streamlit run app.py` and click **Try Demo Dataset** to see a live example with engineered Simpson's Paradox, survivorship bias, and metric gaming signals.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data | Python 3.10+, Pandas, NumPy |
| Statistics | SciPy, Statsmodels, Scikit-learn |
| Visualisation | Plotly |
| Interface | Streamlit |

---

## Project Structure

```
blindspot/
├── app.py                  # Streamlit entry point — layout, sidebar, tab renderers
├── requirements.txt
└── utils/
    ├── loader.py           # CSV / Excel ingestion, size validation, metadata
    ├── profiler.py         # Missingness, type issues, outliers, correlation matrix
    ├── paradox.py          # Simpson's Paradox scanner
    ├── survivorship.py     # Survivorship bias detector
    ├── confounders.py      # Partial correlation confounder detection
    ├── metric_risk.py      # Benford's Law, round-number, spike scorer
    ├── confidence.py       # Overall A–F confidence score synthesis
    ├── blind_spots.py      # Structural gap inference from column vocabulary
    ├── report.py           # Self-contained HTML report generator
    └── sample_data.py      # Synthetic demo dataset (triggers every module)
```

---

## Contributing

Bug reports, feature requests, and pull requests are welcome — please open an issue first so we can discuss the change. Keep pull requests focused on a single concern and include a brief description of what the change does and why.

---

## License

MIT — see [LICENSE](LICENSE) for full text.
