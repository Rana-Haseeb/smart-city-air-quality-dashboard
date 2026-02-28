"""
config.py — Central configuration for Urban Environmental Intelligence Challenge.
All constants, paths, thresholds, and styling parameters.
"""

import os

# ─── Project Paths ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_FILE = os.path.join(DATA_DIR, "sensor_data.parquet")

# ─── Data Parameters ─────────────────────────────────────────────────────────
NUM_STATIONS = 100
YEAR = 2025
HOURS_IN_YEAR = 8760  # 365 * 24

ENVIRONMENTAL_VARS = ["PM2.5", "PM10", "NO2", "Ozone", "Temperature", "Humidity"]

# Zone split: 50 Industrial, 50 Residential
NUM_INDUSTRIAL = 50
NUM_RESIDENTIAL = 50

# ─── OpenAQ API ──────────────────────────────────────────────────────────────
OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_RATE_LIMIT = 55  # requests per minute (stay under 60)

# API key: try environment variable first, then fallback to api_keys.py
from api_keys import OPENAQ_API_KEY as FALLBACK_KEY

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY") or FALLBACK_KEY

# OpenAQ fetch robustness
OPENAQ_MAX_PAGES = int(os.environ.get("OPENAQ_MAX_PAGES", 20))
OPENAQ_RETRIES = int(os.environ.get("OPENAQ_RETRIES", 3))
OPENAQ_BACKOFF = float(os.environ.get("OPENAQ_BACKOFF", 2.0))

# Raw API response directory for audit/logging
RAW_OPENAQ_DIR = os.path.join(DATA_DIR, "raw_openaq")
os.makedirs(RAW_OPENAQ_DIR, exist_ok=True)

# ─── Health Thresholds ───────────────────────────────────────────────────────
PM25_HEALTH_THRESHOLD = 35.0      # μg/m³ — Health Threshold Violation
PM25_EXTREME_HAZARD = 200.0       # μg/m³ — Extreme Hazard level

# ─── Visualization Style ─────────────────────────────────────────────────────
PLOT_STYLE = {
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#aaaaaa",
    "ytick.color": "#aaaaaa",
    "grid.alpha": 0,                # No unnecessary grids
    "font.family": "sans-serif",
    "font.size": 11,
}

# Color palettes
ZONE_COLORS = {"Industrial": "#e74c3c", "Residential": "#3498db"}
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdYlGn_r"
HEATMAP_CMAP = "inferno"

# ─── Population Density (synthetic, per region) ─────────────────────────────
REGIONS = [
    "North America", "Europe", "East Asia",
    "South Asia", "Middle East", "Africa",
    "South America", "Southeast Asia", "Oceania", "Central Asia"
]

POPULATION_DENSITY = {
    "North America": 36, "Europe": 120, "East Asia": 200,
    "South Asia": 450, "Middle East": 90, "Africa": 45,
    "South America": 55, "Southeast Asia": 300, "Oceania": 8, "Central Asia": 18
}

# ─── Random Seed ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42
