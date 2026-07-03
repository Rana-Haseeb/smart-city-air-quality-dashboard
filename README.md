# 🌍 Smart City Air Quality Dashboard

### Urban Environmental Intelligence Engine

A full analytics pipeline + interactive Streamlit dashboard that simulates and analyzes **100 air-quality sensors across 6 environmental variables over a full year (8,760 hours)** to uncover pollution patterns, health-threshold violations, and industrial-vs-residential disparities in a synthetic smart city.

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://smart-city-air-quality-dashboard.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/status-active-success)]()

**🔗 Live App:** https://smart-city-air-quality-dashboard.streamlit.app/

---

## 📖 Overview

This project models a city-wide sensor network (50 Industrial + 50 Residential zones) tracking **PM2.5, PM10, NO₂, Ozone, Temperature, and Humidity**. It fetches real data from the [OpenAQ API](https://openaq.org/) (with a realistic synthetic-data fallback), then runs it through four analytical modules covering dimensionality reduction, temporal pattern detection, distribution modeling, and chart-integrity auditing — all wrapped in a polished dark-themed Streamlit dashboard.

## ✨ Key Features

| Module | What it does |
|---|---|
| 🧬 **Dimensionality Analysis** | PCA projects 6 correlated variables into 2D, revealing clear Industrial vs Residential clustering |
| 📈 **Temporal Analysis** | Heatmaps across all 100 sensors simultaneously to expose daily (24h) and monthly (30-day) pollution cycles without line-chart clutter |
| 📊 **Distribution Modeling** | Dual KDE + log-scale histograms to capture both the "typical day" peak and rare, extreme hazard events (99th percentile analysis) |
| 🔍 **Visual Integrity Audit** | Formally rejects a proposed 3D bar chart (Lie Factor & Data-Ink Ratio violations) in favor of small multiples + perceptually accurate sequential color scales |
| 🖥️ **Interactive Dashboard** | Streamlit + Plotly UI for exploring all of the above live, with health-threshold overlays |

## 🛠️ Tech Stack

- **Data & Compute:** Python, Pandas, NumPy, PyArrow (Parquet storage)
- **ML/Stats:** scikit-learn (PCA, StandardScaler), SciPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **App/UI:** Streamlit
- **Data Source:** OpenAQ API v3 (live) with synthetic generation fallback

## 📂 Project Structure

```
├── main.py                  # Orchestrates the full pipeline (data → 4 tasks → outputs)
├── config.py                 # Central config: paths, thresholds, styling, API settings
├── data_pipeline.py           # OpenAQ fetching + synthetic data generation/preprocessing
├── dashboard.py               # Interactive Streamlit dashboard (all tasks combined)
├── task1_dimensionality.py    # PCA-based dimensionality reduction
├── task2_temporal.py          # High-density heatmap temporal analysis
├── task3_distribution.py      # KDE + tail/extreme-event distribution modeling
├── task4_integrity.py         # Chart-choice justification & visual integrity audit
└── requirements.txt
```

## 🚀 Getting Started

1. **Clone & install dependencies**
   ```bash
   git clone https://github.com/Rana-Haseeb/smart-city-air-quality-dashboard.git
   cd smart-city-air-quality-dashboard
   pip install -r requirements.txt
   ```

2. **(Optional) Use live OpenAQ data** — otherwise the pipeline auto-generates realistic synthetic data.
   ```powershell
   setx OPENAQ_API_KEY "your_real_key_here"
   ```

3. **Run the pipeline**
   ```bash
   python main.py          # synthetic data (fast)
   python main.py --live   # live OpenAQ data
   ```

4. **Launch the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

## 📡 Deployment

Deployed on **Streamlit Community Cloud** — pushes to `main` auto-redeploy the live app. See [`deployment_links.txt`](deployment_links.txt) for live app, write-up, and announcement links.

## 📝 License

This project is provided for educational and portfolio purposes.
