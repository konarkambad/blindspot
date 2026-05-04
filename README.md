# Blind Spot

**Live demo:** _add your Streamlit Cloud URL here_

A small Streamlit app that audits any CSV or Excel file for the kinds of problems that quietly break analyses – things like Simpson's Paradox, survivorship bias, hidden confounders and metric gaming. The idea is simple: most tools tell you what's in your data; this one tries to tell you what's missing or misleading before you draw a conclusion.

## What it does

You upload a file. The app then runs through seven small modules in sequence:

- **Profile** – missingness, outliers, type issues, correlations
- **Simpson's Paradox Scanner** – flags pairs of variables whose correlation flips sign across subgroups
- **Survivorship Bias Detector** – looks for temporal gaps and outcome imbalance suggesting filtered-out cases
- **Confounder Detection** – uses partial correlation to surface third-variable effects
- **Metric Gaming Risk Scorer** – checks for Benford's Law violations, round-number pile-ups and suspicious spikes
- **Confidence Score** – pulls everything into a single A–F grade with a one-line recommendation
- **Blind Spot Report** – calls out structural gaps the dataset can't answer regardless of method

You can also export a self-contained HTML audit report that opens in any browser or prints to PDF.

## Tech

- Python, Streamlit
- Pandas, NumPy, SciPy, Statsmodels, Scikit-learn
- Plotly for visuals
- DuckDB for fast in-memory aggregation

## How to run it locally

```bash
git clone https://github.com/konarkambad/blindspot.git
cd blindspot
pip install -r requirements.txt
streamlit run app.py
```

That's it – no API key, no database, no setup. Click **Try Demo Dataset** in the sidebar to run the audit on a synthetic dataset that's been engineered to trigger every module.

## A note on big files

The app accepts files up to 200 MB, and locally that works fine on most laptops. The hosted version runs on a free tier with limited memory, so very large or very wide datasets can still crash the browser session even after the recent config bump – there's a hard ceiling on how much data Streamlit can ship to the browser in a single message.

If you want to push the limits or test the app on a heavy dataset, run it locally instead. The same modules work on any size your own machine can handle.

## Files

- `app.py` – Streamlit entry point: sidebar, layout, tab renderers, audit runner
- `requirements.txt` – Python dependencies
- `.streamlit/config.toml` – server settings (upload limit, message size)
- `utils/loader.py` – CSV / Excel ingestion
- `utils/profiler.py` – missingness, type issues, outliers, correlations
- `utils/paradox.py` – Simpson's Paradox scanner
- `utils/survivorship.py` – survivorship bias detector
- `utils/confounders.py` – partial correlation confounder detection
- `utils/metric_risk.py` – Benford / round-number / spike scorer
- `utils/confidence.py` – overall A–F score synthesis
- `utils/blind_spots.py` – structural gap inference
- `utils/report.py` – self-contained HTML report generator
- `utils/sample_data.py` – synthetic demo dataset
