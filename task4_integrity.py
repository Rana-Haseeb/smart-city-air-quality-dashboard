"""
task4_integrity.py — Visual integrity audit.
Rejects 3D bar chart proposal, implements Small Multiples alternative.
Justifies sequential color scale based on human luminance perception.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import (
    REGIONS, POPULATION_DENSITY, ZONE_COLORS,
    SEQUENTIAL_CMAP, PLOT_STYLE, OUTPUT_DIR
)
import os


def reject_3d_barchart():
    """
    Generate formal rejection of 3D bar chart proposal with
    Lie Factor and Data-Ink Ratio justifications.
    """
    rejection = """
╔══════════════════════════════════════════════════════════════════╗
║          3D BAR CHART PROPOSAL — REJECTED                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  DECISION: REJECT the 3D bar chart for Pollution vs            ║
║  Population Density vs Region.                                 ║
║                                                                ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ REASON 1: LIE FACTOR VIOLATION                         │   ║
║  │                                                         │   ║
║  │ • 3D perspective causes FORESHORTENING: bars farther    │   ║
║  │   from the viewer appear shorter than they are,         │   ║
║  │   distorting the data-to-visual ratio.                  │   ║
║  │ • Occlusion: front bars hide rear bars, making          │   ║
║  │   comparison impossible for hidden regions.             │   ║
║  │ • Tilt angle changes perceived bar heights — the        │   ║
║  │   Lie Factor = (visual effect size) / (data effect      │   ║
║  │   size) deviates significantly from 1.0.                │   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ REASON 2: LOW DATA-INK RATIO                           │   ║
║  │                                                         │   ║
║  │ • 3D effects (shadows, depth walls, perspective grid)   │   ║
║  │   are "chart junk" — non-data ink that adds no          │   ║
║  │   information but consumes visual bandwidth.            │   ║
║  │ • Tufte's principle: maximize data-ink ratio by         │   ║
║  │   eliminating non-data visual elements.                 │   ║
║  │ • Side walls, floor planes, and shadow projections      │   ║
║  │   are "graphical ducks" that violate assignment rules.  │   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                ║
║  ┌─────────────────────────────────────────────────────────┐   ║
║  │ REASON 3: PERCEPTUAL ISSUES                            │   ║
║  │                                                         │   ║
║  │ • Humans perceive 2D area more accurately than 3D      │   ║
║  │   volume (Stevens' Power Law).                          │   ║
║  │ • 3D adds an extra spatial encoding that competes       │   ║
║  │   with the actual data dimensions.                      │   ║
║  │ • No additional variable is encoded by the 3rd          │   ║
║  │   spatial dimension — it is purely decorative.          │   ║
║  └─────────────────────────────────────────────────────────┘   ║
║                                                                ║
║  ALTERNATIVE: Small Multiples approach (faceted 2D plots)      ║
║  preserves accurate comparison, maximizes data-ink ratio,      ║
║  and enables direct region-by-region comparison.               ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return rejection


def compute_bivariate_data(df):
    """Aggregate mean pollution and population density by region and zone."""
    agg = df.groupby(["region", "zone"]).agg(
        mean_pm25=("PM2.5", "mean"),
        mean_pm10=("PM10", "mean"),
        mean_no2=("NO2", "mean"),
        population_density=("population_density", "first"),
        station_count=("station_id", "nunique")
    ).reset_index()
    return agg


def plot_small_multiples(agg_data, save_path=None):
    """
    Small Multiples: one panel per region.
    X = Population Density, Y = Mean PM2.5, Color = Sequential pollution.
    """
    plt.rcParams.update(PLOT_STYLE)

    regions = sorted(agg_data["region"].unique())
    n_regions = len(regions)
    ncols = 5
    nrows = int(np.ceil(n_regions / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 4.5),
                             sharex=False, sharey=True)
    axes = axes.flatten()

    # Global normalization for color
    vmin = agg_data["mean_pm25"].min()
    vmax = agg_data["mean_pm25"].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(SEQUENTIAL_CMAP)

    for i, region in enumerate(regions):
        ax = axes[i]
        region_data = agg_data[agg_data["region"] == region]

        for _, row in region_data.iterrows():
            color = cmap(norm(row["mean_pm25"]))
            marker = "s" if row["zone"] == "Industrial" else "o"
            edge = ZONE_COLORS[row["zone"]]
            ax.scatter(row["population_density"], row["mean_pm25"],
                       c=[color], s=200, marker=marker,
                       edgecolors=edge, linewidth=2.5, zorder=3)

        ax.set_title(region, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Pop. Density", fontsize=9)
        if i % ncols == 0:
            ax.set_ylabel("Mean PM2.5", fontsize=9)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
               markeredgecolor=ZONE_COLORS["Industrial"], markersize=12,
               markeredgewidth=2, label="Industrial"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor=ZONE_COLORS["Residential"], markersize=12,
               markeredgewidth=2, label="Residential"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=12, framealpha=0.8, facecolor="#1a1a2e",
               edgecolor="#444", bbox_to_anchor=(0.5, -0.02))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:n_regions], shrink=0.6, pad=0.04,
                        aspect=30, location="right")
    cbar.set_label("Mean PM2.5 (μg/m³)", fontsize=12)

    fig.suptitle("Small Multiples: Pollution vs Population Density by Region",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def plot_bivariate_map(agg_data, save_path=None):
    """
    Alternative: Bivariate bubble chart.
    Size = Population Density, Color = PM2.5 (sequential), X = Region.
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(16, 8))

    cmap = plt.cm.get_cmap(SEQUENTIAL_CMAP)
    norm = mcolors.Normalize(vmin=agg_data["mean_pm25"].min(),
                             vmax=agg_data["mean_pm25"].max())

    regions = sorted(agg_data["region"].unique())
    region_idx = {r: i for i, r in enumerate(regions)}

    for _, row in agg_data.iterrows():
        x = region_idx[row["region"]]
        y_offset = 0.15 if row["zone"] == "Industrial" else -0.15
        color = cmap(norm(row["mean_pm25"]))
        size = row["population_density"] * 1.5

        marker = "s" if row["zone"] == "Industrial" else "o"
        ax.scatter(x, row["mean_pm25"] + y_offset, c=[color],
                   s=size, marker=marker,
                   edgecolors="white", linewidth=1, zorder=3, alpha=0.85)

    ax.set_xticks(range(len(regions)))
    ax.set_xticklabels(regions, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean PM2.5 (μg/m³)", fontsize=13)
    ax.set_title("Bivariate Map: Pollution × Population Density × Region",
                 fontsize=14, fontweight="bold", pad=12)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Mean PM2.5 (μg/m³)", fontsize=11)

    # Size legend
    for pd_val in [50, 200, 400]:
        ax.scatter([], [], c="gray", s=pd_val * 1.5, marker="o",
                   edgecolors="white", label=f"Pop. Density: {pd_val}")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.7,
              facecolor="#1a1a2e", edgecolor="#444")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return fig


def justify_color_scale():
    """Justify the selection of Sequential (viridis) over Rainbow colormap."""
    justification = """
╔══════════════════════════════════════════════════════════════╗
║         COLOR SCALE JUSTIFICATION                          ║
╠══════════════════════════════════════════════════════════════╣
║                                                            ║
║  SELECTED: Sequential Colormap (Viridis)                   ║
║  REJECTED: Rainbow (Jet/HSV) Colormap                      ║
║                                                            ║
║  ┌──────────────────────────────────────────────────────┐  ║
║  │ WHY SEQUENTIAL (Viridis)?                           │  ║
║  │                                                      │  ║
║  │ 1. PERCEPTUALLY UNIFORM: luminance increases         │  ║
║  │    monotonically from dark → light, matching human   │  ║
║  │    perception of "more" = "brighter".                │  ║
║  │                                                      │  ║
║  │ 2. COLORBLIND-SAFE: designed to be readable by       │  ║
║  │    all forms of color vision deficiency.             │  ║
║  │                                                      │  ║
║  │ 3. PRINTS WELL: maintains ordering in grayscale,     │  ║
║  │    unlike rainbow which creates banding artifacts.   │  ║
║  │                                                      │  ║
║  │ 4. NO FALSE BOUNDARIES: continuous gradient avoids   │  ║
║  │    artificial perceptual edges that rainbow creates   │  ║
║  │    (e.g., yellow band in jet appears as a boundary). │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                                                            ║
║  ┌──────────────────────────────────────────────────────┐  ║
║  │ WHY NOT RAINBOW (Jet)?                              │  ║
║  │                                                      │  ║
║  │ 1. NON-MONOTONIC LUMINANCE: the bright yellow band   │  ║
║  │    in the middle creates a false "peak" that does    │  ║
║  │    not correspond to data features.                  │  ║
║  │                                                      │  ║
║  │ 2. Borland & Taylor (2007) demonstrated that         │  ║
║  │    rainbow colormaps introduce up to 7 false         │  ║
║  │    boundaries in smooth data fields.                 │  ║
║  │                                                      │  ║
║  │ 3. INEQUITABLE HUE STEPS: perceptual distance        │  ║
║  │    between colors is uneven (green→cyan is smaller   │  ║
║  │    than red→yellow), causing scale distortion.       │  ║
║  └──────────────────────────────────────────────────────┘  ║
║                                                            ║
║  CONCLUSION: Sequential colormaps align with Tufte's       ║
║  principle of "graphical integrity" — the visual           ║
║  representation accurately mirrors the data structure.     ║
╚══════════════════════════════════════════════════════════════╝
"""
    return justification


def run_task4(df):
    """Execute complete Task 4 pipeline."""
    print("\n" + "="*60)
    print("  TASK 4: VISUAL INTEGRITY AUDIT")
    print("="*60)

    # Rejection text
    rejection = reject_3d_barchart()
    print(rejection)

    # Compute aggregated data
    agg_data = compute_bivariate_data(df)
    print(f"  Aggregated data: {len(agg_data)} region-zone combinations")

    # Small Multiples
    save_path1 = os.path.join(OUTPUT_DIR, "task4_small_multiples.png")
    fig_sm = plot_small_multiples(agg_data, save_path1)
    print(f"  Small Multiples saved to: {save_path1}")

    # Bivariate Map
    save_path2 = os.path.join(OUTPUT_DIR, "task4_bivariate_map.png")
    fig_bv = plot_bivariate_map(agg_data, save_path2)
    print(f"  Bivariate Map saved to: {save_path2}")

    # Color scale justification
    color_justification = justify_color_scale()
    print(color_justification)

    return fig_sm, fig_bv, rejection, color_justification


if __name__ == "__main__":
    from data_pipeline import run_pipeline
    df = run_pipeline()
    run_task4(df)
    plt.show()
