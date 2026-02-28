"""
task3_distribution.py — Distribution modeling and tail integrity analysis.
Produces KDE (peak-optimized) and log-histogram (tail-optimized) plots.
Computes 99th percentile and analyzes extreme hazard probability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import (
    PM25_HEALTH_THRESHOLD, PM25_EXTREME_HAZARD,
    PLOT_STYLE, OUTPUT_DIR
)
import os


def select_industrial_zone(df):
    """Select an industrial station with extreme PM2.5 events for analysis."""
    industrial = df[df["zone"] == "Industrial"]

    # Pick station with most extreme events (PM2.5 > 200)
    extreme_counts = industrial[industrial["PM2.5"] > PM25_EXTREME_HAZARD]\
        .groupby("station_id").size()

    if len(extreme_counts) > 0:
        best_station = extreme_counts.idxmax()
    else:
        # Fallback: station with highest max PM2.5
        best_station = industrial.groupby("station_id")["PM2.5"].max().idxmax()

    station_data = df[df["station_id"] == best_station]["PM2.5"].values
    return station_data, best_station


def plot_kde_peaks(pm25_data, station_id, save_path=None):
    """
    KDE plot optimized to reveal distribution PEAKS/MODES.
    Uses linear scale with tight bandwidth for mode detection.
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    # KDE with optimized bandwidth
    kde = stats.gaussian_kde(pm25_data, bw_method="silverman")
    x_range = np.linspace(0, min(pm25_data.max() * 1.1, 500), 1000)
    density = kde(x_range)

    ax.fill_between(x_range, density, alpha=0.3, color="#3498db")
    ax.plot(x_range, density, color="#3498db", lw=2.5, label="KDE Density")

    # Mark peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(density, height=np.max(density) * 0.05, distance=30)
    for p in peaks:
        ax.axvline(x_range[p], color="#f39c12", ls=":", lw=1.5, alpha=0.7)
        ax.annotate(f"Mode: {x_range[p]:.0f}",
                    xy=(x_range[p], density[p]),
                    xytext=(x_range[p] + 10, density[p] * 1.1),
                    fontsize=9, color="#f39c12",
                    arrowprops=dict(arrowstyle="->", color="#f39c12"))

    # Threshold lines
    ax.axvline(PM25_HEALTH_THRESHOLD, color="#e74c3c", ls="--", lw=2,
               label=f"Health Threshold ({PM25_HEALTH_THRESHOLD})")
    ax.axvline(PM25_EXTREME_HAZARD, color="#e74c3c", ls="-", lw=2,
               label=f"Extreme Hazard ({PM25_EXTREME_HAZARD})")

    ax.set_xlabel("PM2.5 (μg/m³)", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"KDE Distribution — Station {station_id} (Peak-Optimized)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.7, facecolor="#1a1a2e")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_log_histogram_tails(pm25_data, station_id, save_path=None):
    """
    Histogram with LOG-SCALED Y-axis optimized to reveal TAILS.
    Fine bins expose rare extreme events that standard histograms hide.
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Fine-grained bins
    max_val = min(pm25_data.max() * 1.1, 500)
    bins = np.linspace(0, max_val, 150)

    counts, bin_edges, patches = ax.hist(
        pm25_data, bins=bins, color="#2ecc71", alpha=0.7,
        edgecolor="#1a1a2e", linewidth=0.3, label="Frequency"
    )

    # Color extreme tail
    for i, patch in enumerate(patches):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        if bin_center > PM25_EXTREME_HAZARD:
            patch.set_facecolor("#e74c3c")
            patch.set_alpha(0.9)
        elif bin_center > PM25_HEALTH_THRESHOLD:
            patch.set_facecolor("#f39c12")
            patch.set_alpha(0.8)

    # Log scale Y-axis to reveal tails
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)

    # Threshold lines
    ax.axvline(PM25_HEALTH_THRESHOLD, color="#f39c12", ls="--", lw=2,
               label=f"Health Threshold ({PM25_HEALTH_THRESHOLD})")
    ax.axvline(PM25_EXTREME_HAZARD, color="#e74c3c", ls="--", lw=2,
               label=f"Extreme Hazard ({PM25_EXTREME_HAZARD})")

    # 99th percentile
    p99 = np.percentile(pm25_data, 99)
    ax.axvline(p99, color="#9b59b6", ls="-.", lw=2.5,
               label=f"99th Percentile ({p99:.1f})")

    ax.set_xlabel("PM2.5 (μg/m³)", fontsize=13)
    ax.set_ylabel("Frequency (log scale)", fontsize=13)
    ax.set_title(f"Log-Histogram — Station {station_id} (Tail-Optimized)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.7, facecolor="#1a1a2e")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def compute_statistics(pm25_data):
    """Compute distribution statistics including tails."""
    stats_dict = {
        "count": len(pm25_data),
        "mean": np.mean(pm25_data),
        "median": np.median(pm25_data),
        "std": np.std(pm25_data),
        "p95": np.percentile(pm25_data, 95),
        "p99": np.percentile(pm25_data, 99),
        "p99_9": np.percentile(pm25_data, 99.9),
        "max": np.max(pm25_data),
        "violations_pct": 100 * np.sum(pm25_data > PM25_HEALTH_THRESHOLD) / len(pm25_data),
        "extreme_pct": 100 * np.sum(pm25_data > PM25_EXTREME_HAZARD) / len(pm25_data),
        "extreme_count": int(np.sum(pm25_data > PM25_EXTREME_HAZARD)),
    }
    return stats_dict


def generate_analysis_text(stats_dict, station_id):
    """Generate technical justification for distribution plots."""
    analysis = f"""
╔══════════════════════════════════════════════════════════════╗
║         DISTRIBUTION ANALYSIS — Station {station_id:<10}          ║
╠══════════════════════════════════════════════════════════════╣
║ Observations:  {stats_dict['count']:<10,}                            ║
║ Mean PM2.5:    {stats_dict['mean']:<10.1f} μg/m³                     ║
║ Median PM2.5:  {stats_dict['median']:<10.1f} μg/m³                     ║
║ Std Dev:       {stats_dict['std']:<10.1f} μg/m³                     ║
║ 95th pctl:     {stats_dict['p95']:<10.1f} μg/m³                     ║
║ ★ 99th pctl:   {stats_dict['p99']:<10.1f} μg/m³                     ║
║ 99.9th pctl:   {stats_dict['p99_9']:<10.1f} μg/m³                     ║
║ Maximum:       {stats_dict['max']:<10.1f} μg/m³                     ║
╠══════════════════════════════════════════════════════════════╣
║ Violation Rate (>35): {stats_dict['violations_pct']:.2f}% of hours            ║
║ Extreme Events (>200): {stats_dict['extreme_count']} events ({stats_dict['extreme_pct']:.3f}%)     ║
╠══════════════════════════════════════════════════════════════╣
║ TECHNICAL JUSTIFICATION:                                     ║
║                                                              ║
║ ▶ KDE Plot (Peak-Optimized):                                ║
║   • Reveals the modal behavior (where most data lies)        ║
║   • Uses Silverman bandwidth for smooth density estimation    ║
║   • Limitation: compresses the tail → hides extreme events   ║
║                                                              ║
║ ▶ Log-Histogram (Tail-Optimized):                           ║
║   • Log-scale Y-axis exposes rare events (power law tail)    ║
║   • Fine bins (150) prevent over-aggregation of extremes     ║
║   • More HONEST for hazard reporting: shows the true         ║
║     frequency of events that KDE smooths away                ║
║   • Reveals that extreme events, while rare, DO occur and    ║
║     their actual count/frequency is visible                  ║
║                                                              ║
║ ★ VERDICT: The log-histogram provides the more "honest"      ║
║   depiction of extreme hazard events because:                ║
║   1. It preserves individual event counts (no smoothing)     ║
║   2. Log scale prevents visual suppression of rare tails     ║
║   3. Bin edges are fixed — no bandwidth choice bias          ║
║   4. Color coding (red for >200) makes extremes salient      ║
╚══════════════════════════════════════════════════════════════╝
"""
    return analysis


def run_task3(df):
    """Execute complete Task 3 pipeline."""
    print("\n" + "="*60)
    print("  TASK 3: DISTRIBUTION MODELING & TAIL INTEGRITY")
    print("="*60)

    # Select industrial station
    pm25_data, station_id = select_industrial_zone(df)
    print(f"  Selected station: {station_id} ({len(pm25_data):,} hourly readings)")

    # Statistics
    stats_dict = compute_statistics(pm25_data)

    # KDE plot (peaks)
    save_path1 = os.path.join(OUTPUT_DIR, "task3_kde_peaks.png")
    fig_kde = plot_kde_peaks(pm25_data, station_id, save_path1)
    print(f"  KDE plot saved to: {save_path1}")

    # Log-histogram (tails)
    save_path2 = os.path.join(OUTPUT_DIR, "task3_log_histogram.png")
    fig_hist = plot_log_histogram_tails(pm25_data, station_id, save_path2)
    print(f"  Log-histogram saved to: {save_path2}")

    # Analysis
    analysis = generate_analysis_text(stats_dict, station_id)
    print(analysis)

    return fig_kde, fig_hist, stats_dict, analysis


if __name__ == "__main__":
    from data_pipeline import run_pipeline
    df = run_pipeline()
    run_task3(df)
    plt.show()
