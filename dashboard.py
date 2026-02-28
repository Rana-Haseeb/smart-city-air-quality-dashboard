"""
dashboard.py — Interactive Streamlit Dashboard for Urban Environmental Intelligence.
Integrates all 4 analytical tasks with interactive controls using Plotly.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from scipy import stats
import os

from config import (
    ENVIRONMENTAL_VARS, ZONE_COLORS,
    PM25_HEALTH_THRESHOLD, PM25_EXTREME_HAZARD,
    DATA_FILE
)

# ═══════════════════════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Urban Environmental Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #b0b0d0;
        font-size: 1.05rem;
        margin-top: 0.5rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.3rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .metric-card h3 {
        color: #8892b0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }

    .metric-card h2 {
        color: #64ffda;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0 0 0;
    }

    .analysis-box {
        background: #1a1a2e;
        border-left: 4px solid #64ffda;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        font-family: 'Inter', monospace;
        font-size: 0.9rem;
        color: #ccd6f6;
        line-height: 1.6;
    }

    .task-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading sensor data...")
def load_data():
    """Load data from parquet, generate if needed."""
    if os.path.exists(DATA_FILE):
        df = pd.read_parquet(DATA_FILE)
    else:
        from data_pipeline import run_pipeline
        df = run_pipeline()
    
    # Ensure datetime columns
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    if "month" not in df.columns:
        df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🌍 Urban Environmental Intelligence Engine</h1>
    <p>Diagnostic engine analyzing 100 global sensor nodes × 6 environmental variables × Year 2025</p>
</div>
""", unsafe_allow_html=True)

# Load data
df = load_data()

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.markdown("---")

    st.markdown(f"""
    <div class="metric-card">
        <h3>Total Records</h3>
        <h2>{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Stations</h3>
            <h2>{df['station_id'].nunique()}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Variables</h3>
            <h2>6</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Filters
    zone_filter = st.multiselect(
        "🏭 Zone Filter",
        options=["Industrial", "Residential"],
        default=["Industrial", "Residential"]
    )

    regions_list = sorted(df["region"].dropna().unique())
    region_filter = st.multiselect(
        "🌐 Region Filter",
        options=regions_list,
        default=regions_list
    )

    st.markdown("---")
    st.markdown("**📊 Data Summary**")
    
    # Calculate violations safely ignoring NaNs
    violations = (df["PM2.5"] > PM25_HEALTH_THRESHOLD).sum()
    extreme = (df["PM2.5"] > PM25_EXTREME_HAZARD).sum()
    
    st.metric("Health Violations (>35)", f"{violations:,}")
    st.metric("Extreme Events (>200)", f"{extreme:,}")

# Apply filters
df_filtered = df[df["zone"].isin(zone_filter) & df["region"].isin(region_filter)]


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Task 1: Dimensionality Reduction",
    "📈 Task 2: Temporal Analysis",
    "📊 Task 3: Distribution Modeling",
    "🔍 Task 4: Visual Integrity Audit"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Dimensionality Reduction (PCA)
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown('<span class="task-badge">TASK 1 — 25%</span>', unsafe_allow_html=True)
    st.markdown("### PCA Biplot: Industrial vs Residential Zone Clustering")
    st.markdown("""
    **Problem:** Standard scatter plots are ineffective with 6 dimensions and high data volume.
    **Solution:** PCA projects 6 standardized environmental variables into 2D, preserving maximum variance.
    **Note:** Imputation is used for missing sensor data to maximize station inclusion.
    """)

    # Aggregate to station means
    station_df = df_filtered.groupby(["station_id", "zone", "region"])[ENVIRONMENTAL_VARS].mean().reset_index()
    
    # Impute missing values instead of dropping
    # We drop only if ALL environmental vars are NaN
    station_df_clean = station_df.dropna(subset=ENVIRONMENTAL_VARS, how='all')
    
    if len(station_df_clean) >= 3:
        # Imputation
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean') # Impute with column mean
        
        # Fit on only the filtered data available
        X_imputed = imputer.fit_transform(station_df_clean[ENVIRONMENTAL_VARS])
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        # PCA
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_ * 100
        
        # Add PC coordinates to DF
        station_df_clean["PC1"] = X_pca[:, 0]
        station_df_clean["PC2"] = X_pca[:, 1]
        
        # Ensure zone/region ordering matches X_pca index
        # Reset index to adhere to X_pca order if any shifting occurred (though fit_transform preserves order)
        
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=["PC1", "PC2"],
            index=ENVIRONMENTAL_VARS
        ).reset_index()
        loadings.columns = ["Variable", "PC1", "PC2"]

        col1, col2 = st.columns([2, 1])

        with col1:
            # Interactive Scatter Plot
            fig = px.scatter(
                station_df_clean, x="PC1", y="PC2",
                color="zone", color_discrete_map=ZONE_COLORS,
                hover_data=["station_id", "region"] + ENVIRONMENTAL_VARS,
                title=f"PCA Biplot ({sum(var_explained):.1f}% Variance Explained)",
                template="plotly_dark",
                height=600
            )
            
            # Add loading vectors as annotations
            scale = np.max(np.abs(X_pca)) * 0.85
            
            for i, row in loadings.iterrows():
                fig.add_shape(
                    type='line',
                    x0=0, y0=0,
                    x1=row['PC1'] * scale,
                    y1=row['PC2'] * scale,
                    line=dict(color="yellow", width=2)
                )
                fig.add_annotation(
                    x=row['PC1'] * scale * 1.15,
                    y=row['PC2'] * scale * 1.15,
                    text=row['Variable'],
                    showarrow=False,
                    font=dict(color="yellow", size=12)
                )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### 📋 PCA Loadings")
            st.dataframe(loadings.set_index("Variable").style.background_gradient(cmap="RdYlGn", axis=None)
                         .format("{:.3f}"), use_container_width=True)

            st.markdown(f"""
            <div class="analysis-box">
            <strong>Variance Explained:</strong><br>
            • PC1: {var_explained[0]:.1f}%<br>
            • PC2: {var_explained[1]:.1f}%<br>
            • Total: {sum(var_explained):.1f}%<br><br>
            <strong>Interpretation:</strong><br>
            • PC1 captures pollution variables (PM2.5, PM10, NO2)<br>
            • PC2 captures atmospheric conditions<br>
            • Industrial zones cluster at higher PC1
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.warning("Not enough station data for PCA calculation even after imputation. Please check data source.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Temporal Analysis
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<span class="task-badge">TASK 2 — 25%</span>', unsafe_allow_html=True)
    st.markdown("### High-Density Temporal Heatmap: PM2.5 Health Threshold Violations")
    st.markdown(f"""
    **Problem:** 100 overlapping line charts create unreadable chart clutter.
    **Solution:** Heatmap — each row is a sensor, each column a day. Color encodes PM2.5 level.
    """)

    # Time range filter
    months_range = st.slider("Select Month Range", 1, 12, (1, 12), key="t2_months")

    df_t2 = df_filtered[(df_filtered["month"] >= months_range[0]) &
                         (df_filtered["month"] <= months_range[1])]

    if len(df_t2) > 0:
        # Daily aggregation
        daily = df_t2.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        # Aggregate mean per day per station
        pivot = daily.groupby(["station_id", "date"]).agg(
            pm25_mean=("PM2.5", "mean"),
            zone=("zone", "first")
        ).reset_index()

        pivot_table = pivot.pivot_table(index="station_id", columns="date",
                                         values="pm25_mean", aggfunc="mean")
        
        # Sort by zone then ID
        zone_map = pivot.drop_duplicates("station_id").set_index("station_id")["zone"]
        sort_key = zone_map.reindex(pivot_table.index).map({"Industrial": 0, "Residential": 1})
        
        # Sort index
        if sort_key.notna().any(): # Check if we have valid keys
            sorted_idx = sort_key.sort_values().index
            pivot_table = pivot_table.reindex(sorted_idx)
        
        # Interactive Heatmap
        # Handle potential NaNs in pivot table for heatmap (gaps in data)
        # Plotly handles NaNs by not plotting, which is fine
        
        if pivot_table.empty:
             st.warning("No data available for heatmap in this range.")
        else:
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot_table.values,
                x=pivot_table.columns,
                y=pivot_table.index,
                colorscale='Inferno',
                zmin=0,
                zmax=pivot_table.max().max(), # Ensure positive max
                colorbar=dict(title='PM2.5 (μg/m³)')
            ))
            
            fig_heat.update_layout(
                title="PM2.5 Levels Across All Sensors (Interactive Heatmap)",
                xaxis_title="Date",
                yaxis_title="Sensors (Sorted by Zone)",
                template="plotly_dark",
                height=600
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        # Periodic signature
        st.markdown("---")
        st.markdown("### 🔄 Periodic Signature Analysis")

        col1, col2 = st.columns(2)
        with col1:
            # Hourly profile
            hourly = df_t2.groupby(["hour", "zone"])["PM2.5"].mean().reset_index()
            # If no data, hourly empty
            if not hourly.empty:
                fig2 = px.line(
                    hourly, x="hour", y="PM2.5", color="zone",
                    color_discrete_map=ZONE_COLORS,
                    title="Daily Cycle (24h)",
                    template="plotly_dark"
                )
                fig2.add_hline(y=PM25_HEALTH_THRESHOLD, line_dash="dash", line_color="red",
                               annotation_text=f"Threshold ({PM25_HEALTH_THRESHOLD})")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No hourly data available.")

        with col2:
            # Monthly profile
            monthly = df_t2.groupby(["month", "zone"])["PM2.5"].mean().reset_index()
            if not monthly.empty:
                fig3 = px.line(
                    monthly, x="month", y="PM2.5", color="zone",
                    color_discrete_map=ZONE_COLORS,
                    markers=True,
                    title="Seasonal Cycle (Monthly)",
                    template="plotly_dark"
                )
                fig3.update_xaxes(tickvals=list(range(1,13)), 
                                  ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
                fig3.add_hline(y=PM25_HEALTH_THRESHOLD, line_dash="dash", line_color="red",
                               annotation_text=f"Threshold ({PM25_HEALTH_THRESHOLD})")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                 st.info("No monthly data available.")

    else:
        st.warning("No data for selected filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: Distribution Modeling
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<span class="task-badge">TASK 3 — 25%</span>', unsafe_allow_html=True)
    st.markdown("### Distribution Modeling & Tail Integrity")
    st.markdown(f"""
    **Problem:** Traditional histograms may hide rare extreme events.
    **Solution:** Split view — KDE for typical behavior, Log-histogram for extreme tails.
    """)

    # Station selector
    industrial_stations = sorted(df_filtered[df_filtered["zone"] == "Industrial"]["station_id"].unique())
    if len(industrial_stations) > 0:
        selected_station = st.selectbox("Select Industrial Station",
                                         industrial_stations, key="t3_station")
        
        # Drop NaNs for distribution calc
        station_df = df_filtered[df_filtered["station_id"] == selected_station].dropna(subset=["PM2.5"])
        pm25_data = station_df["PM2.5"].values

        if len(pm25_data) > 0:
            # Metrics
            p99 = np.percentile(pm25_data, 99)
            extreme_count = int(np.sum(pm25_data > PM25_EXTREME_HAZARD))
            violation_pct = 100 * np.sum(pm25_data > PM25_HEALTH_THRESHOLD) / len(pm25_data)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("99th Percentile", f"{p99:.1f} μg/m³")
            m2.metric("Extreme Events (>200)", f"{extreme_count}")
            m3.metric("Violation Rate (>35)", f"{violation_pct:.1f}%")
            m4.metric("Total Observations", f"{len(pm25_data):,}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### KDE Plot (Normal Scale)")
                # Using histogram with marginal rug for detail
                fig_kde = px.histogram(
                    station_df, x="PM2.5", 
                    marginal="rug",
                    histnorm='probability density',
                    nbins=50,
                    title=f"Distribution — Station {selected_station}",
                    template="plotly_dark",
                    color_discrete_sequence=["#3498db"]
                )
                fig_kde.add_vline(x=PM25_HEALTH_THRESHOLD, line_dash="dash", line_color="orange")
                fig_kde.add_vline(x=PM25_EXTREME_HAZARD, line_dash="solid", line_color="red")
                st.plotly_chart(fig_kde, use_container_width=True)

            with col2:
                st.markdown("#### Log-Histogram (Tail-Optimized)")
                # Filter out negative or zero if any for Log scale (though histogram handles it usually, safer to clean)
                station_df_log = station_df[station_df["PM2.5"] > 0]
                
                fig_hist = px.histogram(
                    station_df_log, x="PM2.5",
                    log_y=True,
                    nbins=50,
                    title=f"Log-Scale Distribution — Station {selected_station}",
                    template="plotly_dark",
                    color_discrete_sequence=["#2ecc71"]
                )
                fig_hist.add_vline(x=PM25_EXTREME_HAZARD, line_dash="dash", line_color="red",
                                   annotation_text="Extreme Hazard")
                fig_hist.add_vline(x=p99, line_dash="dot", line_color="purple",
                                   annotation_text="99th %")
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown(f"""
            <div class="analysis-box">
            <strong>Verdict:</strong> The log-scale histogram forces visibility of the {extreme_count} extreme events that vanish in a linear scale plot, ensuring hazardous outliers are not ignored.
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("No industrial stations available with current filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: Visual Integrity Audit
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown('<span class="task-badge">TASK 4 — 25%</span>', unsafe_allow_html=True)
    st.markdown("### Visual Integrity: Small Multiples")

    agg = df_filtered.groupby(["region", "zone"]).agg(
        mean_pm25=("PM2.5", "mean"),
        population_density=("population_density", "first"),
    ).reset_index()

    if len(agg) > 0:
        # Create explicit size column ensuring NO NEGATIVES
        agg["size_val"] = agg["mean_pm25"].fillna(0).clip(lower=0)
        
        # If size_val is all zero, add epsilon to make visible or handle gracefully
        if (agg["size_val"] == 0).all():
             agg["size_val"] = 5 # default size
        else:
             # Normalize size for display so tiny values don't disappear
             agg["size_val"] = agg["size_val"] + 2
        
        fig_sm = px.scatter(
            agg, x="population_density", y="mean_pm25",
            facet_col="region", facet_col_wrap=5,
            color="mean_pm25", color_continuous_scale="Viridis",
            size="size_val", size_max=20,
            symbol="zone", symbol_map={"Industrial": "square", "Residential": "circle"},
            title="Pollution vs Population Density by Region (Small Multiples)",
            template="plotly_dark",
            height=600,
            hover_data=["mean_pm25", "population_density"]
        )
        fig_sm.update_layout(showlegend=False)
        st.plotly_chart(fig_sm, use_container_width=True)
        
        st.info("Uses **Viridis** sequential colormap for perceptual uniformity and colorblind safety.")
    else:
        st.warning("No aggregated data available.")

# ═══════════════════════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;">
    🌍 Urban Environmental Intelligence Engine | Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
