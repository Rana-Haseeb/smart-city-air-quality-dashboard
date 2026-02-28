"""
main.py — Main orchestrator for the Urban Environmental Intelligence pipeline.
Runs data generation, all 4 analysis tasks, and saves outputs.
"""

import sys
import io

# Fix Windows encoding issue
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for pipeline mode

import matplotlib.pyplot as plt
import time
import os

from config import OUTPUT_DIR
from data_pipeline import run_pipeline
from task1_dimensionality import run_task1
from task2_temporal import run_task2
from task3_distribution import run_task3
from task4_integrity import run_task4


def main():
    """Run the complete Urban Environmental Intelligence pipeline."""
    start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     URBAN ENVIRONMENTAL INTELLIGENCE ENGINE                ║")
    print("║     100 Sensors × 6 Variables × 8760 Hours (2025)         ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Step 1: Data Pipeline ──────────────────────────────────────
    print("\n▶ Running Data Pipeline...")
    
    import sys
    use_live = "--live" in sys.argv
    if use_live:
        print("  [MODE] Fetching LIVE data from OpenAQ (this may take time)...")
    else:
        print("  [MODE] Using SYNTHETIC data (fast generation)...")
        
    df = run_pipeline(use_live_data=use_live)
    print(f"  Dataset: {len(df):,} rows × {df.shape[1]} columns")

    # ── Step 2: Task 1 — Dimensionality Reduction ─────────────────
    print("\n▶ Running Task 1: Dimensionality Challenge...")
    fig1, loadings1, analysis1 = run_task1(df)
    plt.close(fig1)

    # ── Step 3: Task 2 — Temporal Analysis ────────────────────────
    print("\n▶ Running Task 2: High-Density Temporal Analysis...")
    fig2a, fig2b, analysis2 = run_task2(df)
    plt.close(fig2a)
    plt.close(fig2b)

    # ── Step 4: Task 3 — Distribution Modeling ────────────────────
    print("\n▶ Running Task 3: Distribution Modeling...")
    fig3a, fig3b, stats3, analysis3 = run_task3(df)
    plt.close(fig3a)
    plt.close(fig3b)

    # ── Step 5: Task 4 — Visual Integrity Audit ──────────────────
    print("\n▶ Running Task 4: Visual Integrity Audit...")
    fig4a, fig4b, rejection4, color4 = run_task4(df)
    plt.close(fig4a)
    plt.close(fig4b)

    # ── Summary ───────────────────────────────────────────────────
    elapsed = time.time() - start
    print("\n" + "="*60)
    print(f"  PIPELINE COMPLETE in {elapsed:.1f} seconds")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print("  Files generated:")
    for f in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    • {f} ({size_kb:.0f} KB)")
    print("="*60)


if __name__ == "__main__":
    main()
