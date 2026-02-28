"""
data_pipeline.py — Data acquisition, synthetic generation, and preprocessing.
Handles OpenAQ API fetching with fallback to realistic synthetic data.
Stores data in Parquet format for efficient big-data handling.
"""

import numpy as np
import pandas as pd
import requests
import time
import os
import logging
import json
from pathlib import Path

from config import (
    NUM_STATIONS, YEAR, HOURS_IN_YEAR, ENVIRONMENTAL_VARS,
    NUM_INDUSTRIAL, NUM_RESIDENTIAL, DATA_FILE, DATA_DIR,
    OPENAQ_BASE_URL, OPENAQ_RATE_LIMIT, OPENAQ_API_KEY,
    REGIONS, POPULATION_DENSITY, RANDOM_SEED,
    OPENAQ_MAX_PAGES, OPENAQ_RETRIES, OPENAQ_BACKOFF, RAW_OPENAQ_DIR
)

# Ensure raw dir exists
Path(RAW_OPENAQ_DIR).mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAQ API Functions
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_openaq_locations(n=100):
    """Fetch n air quality station locations from OpenAQ API v3."""
    headers = {"Accept": "application/json"}
    if OPENAQ_API_KEY:
        headers["X-API-Key"] = OPENAQ_API_KEY

    locations = []
    page = 1
    while len(locations) < n:
        try:
            resp = requests.get(
                f"{OPENAQ_BASE_URL}/locations",
                params={"limit": min(100, n - len(locations)), "page": page,
                        "parameter_id": 2, "order_by": "id"},  # parameter_id=2 is PM2.5
                headers=headers, timeout=30
            )
            if resp.status_code == 429:
                logger.warning("Rate limited. Sleeping 60s...")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                break
            locations.extend(results)
            page += 1
            time.sleep(60 / OPENAQ_RATE_LIMIT)  # respect rate limit
        except Exception as e:
            logger.error(f"OpenAQ API error: {e}")
            break
    return locations[:n]



def fetch_sensor_measurements(sensor_id, date_from, date_to):
    """Fetch measurements for a single sensor from OpenAQ API v3 with retries,
    paginated fetch and raw response saving for audit.
    """
    headers = {"Accept": "application/json"}
    if OPENAQ_API_KEY:
        headers["X-API-Key"] = OPENAQ_API_KEY

    measurements = []
    page = 1
    max_pages = OPENAQ_MAX_PAGES

    while page <= max_pages:
        attempt = 0
        while attempt < OPENAQ_RETRIES:
            try:
                resp = requests.get(
                    f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/measurements",
                    params={"date_from": date_from, "date_to": date_to,
                            "limit": 1000, "page": page},
                    headers=headers, timeout=60
                )

                # Save raw response for audit
                try:
                    raw_path = Path(RAW_OPENAQ_DIR) / f"sensor_{sensor_id}_page_{page}.json"
                    with open(raw_path, "w", encoding="utf-8") as fh:
                        fh.write(resp.text)
                except Exception:
                    # Non-fatal — continue processing
                    logger.debug("Failed to write raw OpenAQ response for sensor %s page %s", sensor_id, page)

                if resp.status_code == 429:
                    logger.warning("Rate limited when fetching sensor %s page %s — sleeping 60s", sensor_id, page)
                    time.sleep(60)
                    attempt += 1
                    continue

                if resp.status_code == 404:
                    logger.warning("Sensor %s not found (404).", sensor_id)
                    return measurements

                resp.raise_for_status()

                data = resp.json()
                results = data.get("results", [])

                if not results:
                    return measurements

                measurements.extend(results)

                # Respect rate limit between successful pages
                time.sleep(60 / OPENAQ_RATE_LIMIT)

                # Advance to next page
                page += 1
                break  # break out of retry loop on success

            except requests.RequestException as req_err:
                wait = OPENAQ_BACKOFF ** attempt
                logger.warning("Request error for sensor %s page %s (attempt %d): %s — retrying in %.1fs",
                               sensor_id, page, attempt + 1, req_err, wait)
                time.sleep(wait)
                attempt += 1
            except Exception as e:
                logger.error("Unexpected error fetching sensor %s page %s: %s", sensor_id, page, e)
                return measurements

        else:
            # Exhausted retries for this page
            logger.error("Exhausted retries fetching sensor %s page %s — skipping remaining pages.", sensor_id, page)
            break

    return measurements


def fetch_real_data():
    """
    Orchestrate fetching of real data from OpenAQ for 100 stations.
    Iterates through sensors of each location to fetch specific environmental variables.
    """
    logger.info("Fetching real locations from OpenAQ...")
    locations = fetch_openaq_locations(n=NUM_STATIONS)
    
    if not locations:
        logger.error("No locations found. Falling back to synthetic data.")
        return generate_synthetic_data()

    logger.info(f"Found {len(locations)} locations. Starting measurement fetch...")
    
    dfs = []
    rng = np.random.default_rng(RANDOM_SEED)
    meta_df = _assign_station_metadata(rng)
    
    if len(locations) < len(meta_df):
        meta_df = meta_df.iloc[:len(locations)]
    
    fetched_count = 0
    start_date = f"{YEAR}-01-01"
    end_date = f"{YEAR}-12-31"

    # Mapping OpenAQ parameter names/IDs to our internal names
    # OpenAQ v3 uses ID or name. Names are like "pm25", "pm10", "no2", "o3", "temperature", "relativehumidity"
    # We will look for these in the sensor list.
    target_params = {
        'pm25': 'PM2.5', 
        'pm10': 'PM10', 
        'no2': 'NO2', 
        'o3': 'Ozone', 
        'temperature': 'Temperature', 
        'relativehumidity': 'Humidity'
    }

    for idx, loc in enumerate(locations):
        loc_id = loc['id']
        sensors = loc.get('sensors', [])
        
        station_measurements = []
        
        # Identify relevant sensors
        for sensor in sensors:
            # Determine parameter name from sensor structure (API has variations)
            param = sensor.get('parameter', sensor.get('parameter_id', ''))
            param_name = ''
            if isinstance(param, dict):
                param_name = param.get('name') or param.get('param') or ''
            elif isinstance(param, (int, float)):
                param_name = str(param)
            else:
                param_name = str(param)
            param_name = param_name.lower().strip()

            # Match parameter robustly
            mapped_name = None
            if param_name in target_params:
                mapped_name = target_params[param_name]
            else:
                # Try normalization (e.g., 'pm25' vs 'pm2.5')
                norm = param_name.replace('.', '').replace('_', '')
                for k, v in target_params.items():
                    if k.replace('.', '').replace('_', '') == norm:
                        mapped_name = v
                        break

            if mapped_name:
                sensor_id = sensor.get('id')
                logger.info(f"  Fetching {mapped_name} (Sensor {sensor_id}) for Loc {loc_id}...")

                meas = fetch_sensor_measurements(sensor_id, start_date, end_date)
                if meas:
                    # Convert to minimal DF
                    df_s = pd.DataFrame(meas)
                    
                    # Normalize time and value
                    # v3 structure: value, period{datetimeFrom, ...}
                    if 'period' in df_s.columns:
                        # period is a dict with datetimeFrom as a dict {utc: ..., local: ...}
                        def extract_utc(period):
                            if isinstance(period, dict):
                                dt = period.get('datetimeFrom')
                                if isinstance(dt, dict):
                                    return dt.get('utc')
                                return dt
                            return None
                            
                        df_s['timestamp'] = pd.to_datetime(df_s['period'].apply(extract_utc))
                        
                    if 'value' in df_s.columns:
                         # Extract value if it is dict (unlikely for measurements endpoint, usually float)
                         pass
                    
                    df_s['parameter'] = mapped_name
                    # keep only timestamp, parameter, value
                    df_s = df_s[['timestamp', 'parameter', 'value']]
                    station_measurements.append(df_s)

        if not station_measurements:
            logger.warning(f"No relevant data found for Location {loc_id}. Skipping.")
            continue
            
        # Combine all sensors for this station
        station_df = pd.concat(station_measurements, ignore_index=True)
        
        # Pivot to get columns: timestamp, PM2.5, PM10, etc.
        pivot = station_df.pivot_table(index='timestamp', columns='parameter', values='value', aggfunc='mean')
        
        # Resample to hourly
        pivot = pivot.resample('H').mean()
        
        # Add metadata
        meta = meta_df.iloc[idx]
        pivot['station_id'] = meta['station_id']
        pivot['zone'] = meta['zone']
        pivot['region'] = meta['region']
        pivot['population_density'] = meta['population_density']
        
        pivot = pivot.reset_index()
        dfs.append(pivot)
        
        fetched_count += 1
        logger.info(f"Fetched station {fetched_count}/{len(locations)} (ID: {loc_id}) - {len(pivot)} rows")

    if not dfs:
        logger.warning("No measurements fetched. Generating synthetic data.")
        return generate_synthetic_data()

    full_df = pd.concat(dfs, ignore_index=True)
    
    # Fill remaining columns
    for col in ENVIRONMENTAL_VARS:
        if col not in full_df.columns:
            full_df[col] = np.nan
            
    # Derive time features
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    
    # FILTER FOR YEAR 2025 ONLY
    full_df = full_df[full_df["timestamp"].dt.year == YEAR]
    
    # CLIP NEGATIVE VALUES (Physical impossibility check)
    # OpenAQ sometimes returns -999 for errors or negative values for calibration drift
    for col in ENVIRONMENTAL_VARS:
        if col in full_df.columns:
            full_df[col] = full_df[col].clip(lower=0)
    
    full_df["hour"] = full_df["timestamp"].dt.hour
    full_df["month"] = full_df["timestamp"].dt.month
    full_df["day_of_week"] = full_df["timestamp"].dt.dayofweek
    full_df["date"] = full_df["timestamp"].dt.date
    
    return full_df, meta_df


# ═══════════════════════════════════════════════════════════════════════════════
# Synthetic Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

def _assign_station_metadata(rng):
    """Create metadata for 100 stations with zone and region assignments."""
    zones = (["Industrial"] * NUM_INDUSTRIAL) + (["Residential"] * NUM_RESIDENTIAL)
    rng.shuffle(zones)

    # Distribute stations across regions
    region_assignment = []
    per_region = NUM_STATIONS // len(REGIONS)
    for region in REGIONS:
        region_assignment.extend([region] * per_region)
    region_assignment = region_assignment[:NUM_STATIONS]
    rng.shuffle(region_assignment)

    stations = []
    for i in range(NUM_STATIONS):
        stations.append({
            "station_id": f"S{i+1:03d}",
            "zone": zones[i],
            "region": region_assignment[i],
            "population_density": POPULATION_DENSITY[region_assignment[i]],
            "latitude": rng.uniform(-60, 70),
            "longitude": rng.uniform(-180, 180),
        })
    return pd.DataFrame(stations)


def _generate_hourly_series(station_meta, rng):
    """Generate realistic hourly environmental data for one station."""
    zone = station_meta["zone"]
    is_industrial = zone == "Industrial"

    # Time index for 2025
    timestamps = pd.date_range(f"{YEAR}-01-01", periods=HOURS_IN_YEAR, freq="h")
    hours = timestamps.hour.values
    months = timestamps.month.values
    day_of_year = timestamps.dayofyear.values

    # ── Base profiles based on zone ──────────────────────────────────
    if is_industrial:
        pm25_base = rng.uniform(25, 55)
        pm10_base = rng.uniform(40, 90)
        no2_base = rng.uniform(30, 60)
        ozone_base = rng.uniform(20, 50)
        temp_base = rng.uniform(15, 30)
        humidity_base = rng.uniform(40, 65)
    else:
        pm25_base = rng.uniform(5, 20)
        pm10_base = rng.uniform(10, 35)
        no2_base = rng.uniform(8, 25)
        ozone_base = rng.uniform(30, 70)
        temp_base = rng.uniform(12, 28)
        humidity_base = rng.uniform(50, 80)

    # ── Diurnal cycles (24h) ─────────────────────────────────────────
    # Rush-hour peaks for PM and NO2 (7-9am, 5-7pm)
    rush_factor = np.where(
        ((hours >= 7) & (hours <= 9)) | ((hours >= 17) & (hours <= 19)),
        rng.uniform(1.3, 1.8), 1.0
    )
    # Ozone peaks midday (UV driven)
    ozone_diurnal = 1 + 0.4 * np.sin(np.pi * (hours - 6) / 12)
    ozone_diurnal = np.clip(ozone_diurnal, 0.6, 1.5)

    # ── Seasonal cycles (365-day) ────────────────────────────────────
    # Winter inversion (higher PM in winter months)
    winter_factor = 1 + 0.35 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    # Summer ozone boost
    summer_ozone = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # Temperature seasonal
    temp_seasonal = 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    # Humidity inverse to temperature
    humidity_seasonal = -10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # ── Generate values ──────────────────────────────────────────────
    n = HOURS_IN_YEAR
    noise_scale = 0.15

    pm25 = (pm25_base * rush_factor * winter_factor
            + rng.normal(0, pm25_base * noise_scale, n))
    pm10 = (pm10_base * rush_factor * winter_factor
            + rng.normal(0, pm10_base * noise_scale, n))
    no2 = (no2_base * rush_factor * winter_factor
           + rng.normal(0, no2_base * noise_scale, n))
    ozone = (ozone_base * ozone_diurnal * summer_ozone
             + rng.normal(0, ozone_base * noise_scale, n))
    temp = (temp_base + temp_seasonal
            + rng.normal(0, 3, n))
    humidity = (humidity_base + humidity_seasonal
                + rng.normal(0, 5, n))

    # ── Extreme pollution events (spikes) ────────────────────────────
    if is_industrial:
        n_spikes = rng.integers(5, 25)
        spike_indices = rng.choice(n, size=n_spikes, replace=False)
        spike_magnitudes = rng.uniform(150, 350, size=n_spikes)
        pm25[spike_indices] += spike_magnitudes
        pm10[spike_indices] += spike_magnitudes * rng.uniform(0.8, 1.5, size=n_spikes)
    else:
        n_spikes = rng.integers(0, 5)
        if n_spikes > 0:
            spike_indices = rng.choice(n, size=n_spikes, replace=False)
            pm25[spike_indices] += rng.uniform(50, 120, size=n_spikes)

    # ── Clamp to physical ranges ─────────────────────────────────────
    pm25 = np.clip(pm25, 0, 500)
    pm10 = np.clip(pm10, 0, 600)
    no2 = np.clip(no2, 0, 300)
    ozone = np.clip(ozone, 0, 250)
    humidity = np.clip(humidity, 5, 100)

    return pd.DataFrame({
        "timestamp": timestamps,
        "station_id": station_meta["station_id"],
        "zone": zone,
        "region": station_meta["region"],
        "population_density": station_meta["population_density"],
        "PM2.5": np.round(pm25, 2),
        "PM10": np.round(pm10, 2),
        "NO2": np.round(no2, 2),
        "Ozone": np.round(ozone, 2),
        "Temperature": np.round(temp, 2),
        "Humidity": np.round(humidity, 2),
    })


def generate_synthetic_data():
    """Generate full synthetic dataset for 100 stations × 8760 hours."""
    logger.info("Generating synthetic data for %d stations...", NUM_STATIONS)
    rng = np.random.default_rng(RANDOM_SEED)

    station_meta = _assign_station_metadata(rng)
    chunks = []

    for _, row in station_meta.iterrows():
        df = _generate_hourly_series(row, rng)
        chunks.append(df)
        if len(chunks) % 20 == 0:
            logger.info("  Generated %d / %d stations", len(chunks), NUM_STATIONS)

    full_df = pd.concat(chunks, ignore_index=True)

    # Add derived time features
    full_df["hour"] = full_df["timestamp"].dt.hour
    full_df["month"] = full_df["timestamp"].dt.month
    full_df["day_of_week"] = full_df["timestamp"].dt.dayofweek
    full_df["date"] = full_df["timestamp"].dt.date

    logger.info("Synthetic data generated: %s rows, %s columns",
                f"{len(full_df):,}", full_df.shape[1])
    return full_df, station_meta


# ═══════════════════════════════════════════════════════════════════════════════
# Data I/O (Parquet for Big Data)
# ═══════════════════════════════════════════════════════════════════════════════

def save_data(df, path=None):
    """Save DataFrame to Parquet with compression."""
    path = path or DATA_FILE
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info("Data saved to %s (%.1f MB)", path, size_mb)


def load_data(path=None):
    """Load DataFrame from Parquet."""
    path = path or DATA_FILE
    if not os.path.exists(path):
        logger.warning("Data file not found: %s. Generating synthetic data...", path)
        df, _ = generate_synthetic_data()
        save_data(df, path)
        return df
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Data loaded from %s: %s rows", path, f"{len(df):,}")
    return df


def load_data_chunked(path=None, chunk_size=100_000):
    """Generator: yield chunks of data for memory-efficient processing."""
    path = path or DATA_FILE
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunk_size):
        yield batch.to_pandas()


# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess(df):
    """Clean and validate data with intelligent handling of missing values."""
    initial_len = len(df)

    # Ensure correct dtypes first
    for col in ENVIRONMENTAL_VARS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by timestamp to enable interpolation
    df = df.sort_values(['station_id', 'timestamp']).reset_index(drop=True)

    # Per-station interpolation (preserves time-series integrity)
    df_processed = []
    for station_id in df['station_id'].unique():
        station_data = df[df['station_id'] == station_id].copy()
        
        # Forward-fill/backward-fill short gaps (up to 3 hours)
        for col in ENVIRONMENTAL_VARS:
            station_data[col] = station_data[col].interpolate(
                method='linear', limit=3, fill_value='extrapolate'
            )
            station_data[col] = station_data[col].bfill().ffill()
        
        # Clip to valid physical ranges
        station_data['PM2.5'] = station_data['PM2.5'].clip(0, 500)
        station_data['PM10'] = station_data['PM10'].clip(0, 600)
        station_data['NO2'] = station_data['NO2'].clip(0, 300)
        station_data['Ozone'] = station_data['Ozone'].clip(0, 250)
        station_data['Temperature'] = station_data['Temperature'].clip(-50, 60)
        station_data['Humidity'] = station_data['Humidity'].clip(0, 100)
        
        df_processed.append(station_data)
    
    df = pd.concat(df_processed, ignore_index=True)

    # Drop only rows where ALL environmental vars are NaN (complete loss)
    df = df.dropna(subset=ENVIRONMENTAL_VARS, how='all')

    rows_lost = initial_len - len(df)
    if rows_lost > 0:
        logger.info("Preprocessing: removed %d rows with complete missing data (%.1f%% retention)",
                     rows_lost, 100 * len(df) / initial_len)
    else:
        logger.info("Preprocessing: no complete data loss. Interpolation preserved %.1f%% of data",
                     100 * len(df) / initial_len)
    return df




# ═══════════════════════════════════════════════════════════════════════════════
# Master Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(force_regenerate=False, use_live_data=False):
    """
    Master data pipeline:
    1. Check for existing parquet data
    2. Generate synthetic data OR fetch live data
    3. Preprocess and return
    """
    if os.path.exists(DATA_FILE) and not force_regenerate and not use_live_data:
        logger.info("Loading existing data...")
        df = load_data()
    else:
        if use_live_data:
            logger.info("Running LIVE DATA pipeline (OpenAQ)...")
            df, station_meta = fetch_real_data()
        else:
            logger.info("Running SYNTHETIC data pipeline...")
            df, station_meta = generate_synthetic_data()
            
        save_data(df)

        # Save station metadata separately
        meta_path = os.path.join(DATA_DIR, "station_metadata.parquet")
        station_meta.to_parquet(meta_path, index=False)
        logger.info("Station metadata saved to %s", meta_path)

    df = preprocess(df)
    logger.info(f"Final dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Example usage: python data_pipeline.py --live
    import sys
    use_live = "--live" in sys.argv
    df = run_pipeline(force_regenerate=True, use_live_data=use_live)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nZone distribution:\n{df['zone'].value_counts()}")
    print(f"\nSample:\n{df.head()}")
    print(f"\nDescriptive stats:\n{df[ENVIRONMENTAL_VARS].describe()}")
