# =============================================================================
# app.py — Streamlit Dashboard
# =============================================================================
# RUN AFTER all 4 steps are complete.
# RUN:  streamlit run app.py
# Then open http://localhost:8501 in your browser.
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Accident Analysis",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #378ADD;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 28px; font-weight: 700; color: #1a1a2e; }
    .metric-label { font-size: 13px; color: #666; margin-top: 4px; }
    .section-header {
        font-size: 18px; font-weight: 600;
        margin: 20px 0 10px; color: #1a1a2e;
        border-bottom: 2px solid #378ADD;
        padding-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join("data", "clean_accidents.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, low_memory=False)

@st.cache_resource
def load_model():
    path = os.path.join("models", "xgb_severity.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

df_full  = load_data()
model    = load_model()

# ── Guard: data not found ─────────────────────────────────────────────────────
if df_full is None:
    st.error("❌ data/clean_accidents.csv not found. Run step2_preprocess.py first.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/traffic-jam.png", width=60)
st.sidebar.title("🚦 Filters")

# Year filter
if "year" in df_full.columns:
    years = sorted(df_full["year"].dropna().unique().tolist())
    sel_years = st.sidebar.multiselect("Year", years, default=years)
    df = df_full[df_full["year"].isin(sel_years)].copy()
else:
    df = df_full.copy()

# Severity filter
sev_options = sorted(df["Severity"].unique().tolist())
sel_sev = st.sidebar.multiselect(
    "Severity Level (1=low → 4=high)",
    sev_options,
    default=sev_options
)
df = df[df["Severity"].isin(sel_sev)]

# Day type filter
if "is_weekend" in df.columns:
    day_type = st.sidebar.radio("Day type", ["All", "Weekdays only", "Weekends only"])
    if day_type == "Weekdays only":
        df = df[df["is_weekend"] == 0]
    elif day_type == "Weekends only":
        df = df[df["is_weekend"] == 1]

# Time of day filter
if "is_night" in df.columns:
    time_type = st.sidebar.radio("Time of day", ["All", "Day only", "Night only"])
    if time_type == "Day only":
        df = df[df["is_night"] == 0]
    elif time_type == "Night only":
        df = df[df["is_night"] == 1]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Showing:** {len(df):,} accidents")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🚦 Traffic Accident Analysis Dashboard")
st.caption("US Accidents Dataset | Built with Python + Streamlit")

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Accidents", f"{len(df):,}")
with col2:
    st.metric("Avg Severity", f"{df['Severity'].mean():.2f}")
with col3:
    peak_hr = int(df["hour"].mode()[0]) if "hour" in df.columns else "N/A"
    st.metric("Peak Hour", f"{peak_hr}:00")
with col4:
    if "is_night" in df.columns:
        night_pct = df["is_night"].mean() * 100
        st.metric("Night Accidents", f"{night_pct:.1f}%")
with col5:
    if "is_weekend" in df.columns:
        wknd_pct = df["is_weekend"].mean() * 100
        st.metric("Weekend Accidents", f"{wknd_pct:.1f}%")

st.markdown("---")

# ── Row 1: Time Charts ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⏰ Time-Based Trends</div>', unsafe_allow_html=True)
tc1, tc2 = st.columns(2)

with tc1:
    if "hour" in df.columns:
        hourly = df["hour"].value_counts().sort_index().reset_index()
        hourly.columns = ["Hour", "Accidents"]
        fig = px.bar(
            hourly, x="Hour", y="Accidents",
            title="Accidents by Hour of Day",
            color="Accidents",
            color_continuous_scale=["#378ADD", "#D85A30"],
        )
        fig.update_layout(height=320, showlegend=False,
                          coloraxis_showscale=False,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

with tc2:
    if "day_of_week" in df.columns:
        day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        dow = df["day_of_week"].value_counts().sort_index().reset_index()
        dow.columns = ["Day", "Accidents"]
        dow["Day"] = dow["Day"].apply(lambda x: day_labels[int(x)] if int(x) < 7 else str(x))
        fig2 = px.bar(
            dow, x="Day", y="Accidents",
            title="Accidents by Day of Week",
            color="Accidents",
            color_continuous_scale=["#1D9E75", "#D85A30"],
        )
        fig2.update_layout(height=320, showlegend=False,
                           coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Hour × Day Heatmap + Monthly trend ─────────────────────────────────
hc1, hc2 = st.columns(2)

with hc1:
    if "hour" in df.columns and "day_of_week" in df.columns:
        pivot = df.groupby(["day_of_week","hour"]).size().unstack(fill_value=0)
        pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot)]
        fig3 = px.imshow(
            pivot, title="Accident Heatmap: Day × Hour",
            color_continuous_scale="YlOrRd",
            labels={"x":"Hour", "y":"Day", "color":"Count"},
            aspect="auto"
        )
        fig3.update_layout(height=320,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, use_container_width=True)

with hc2:
    if "month" in df.columns:
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = df["month"].value_counts().sort_index().reset_index()
        monthly.columns = ["Month","Accidents"]
        monthly["Month"] = monthly["Month"].apply(
            lambda x: month_labels[int(x)-1] if 1 <= int(x) <= 12 else str(x)
        )
        fig4 = px.line(
            monthly, x="Month", y="Accidents",
            title="Monthly Accident Trend",
            markers=True, line_shape="spline",
            color_discrete_sequence=["#1D9E75"]
        )
        fig4.update_traces(fill="tozeroy", fillcolor="rgba(29,158,117,0.1)")
        fig4.update_layout(height=320,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Cause Analysis ─────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌧 Cause Analysis</div>', unsafe_allow_html=True)
cc1, cc2 = st.columns(2)

with cc1:
    st.markdown("**Severity Distribution**")
    sev_counts = df["Severity"].value_counts().sort_index().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    sev_counts["Severity"] = sev_counts["Severity"].astype(str)
    fig5 = px.bar(
        sev_counts, x="Severity", y="Count",
        color="Severity",
        color_discrete_sequence=["#1D9E75","#378ADD","#D85A30","#7F77DD"],
        title="Accident Severity Distribution"
    )
    fig5.update_layout(height=300, showlegend=False,
                       plot_bgcolor="rgba(0,0,0,0)",
                       paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig5, use_container_width=True)

with cc2:
    if "Weather_Condition" in df.columns:
        st.markdown("**Top Weather Conditions**")
        weather = df["Weather_Condition"].value_counts().head(10).reset_index()
        weather.columns = ["Weather", "Count"]
        weather["Weather"] = weather["Weather"].astype(str)
        fig6 = px.bar(
            weather.sort_values("Count"), x="Count", y="Weather",
            orientation="h", title="Top 10 Weather Conditions",
            color="Count", color_continuous_scale=["#B5D4F4","#7F77DD"]
        )
        fig6.update_layout(height=300, showlegend=False,
                           coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig6, use_container_width=True)

# ── Row 4: Hotspot Map ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺 Accident Hotspot Map</div>', unsafe_allow_html=True)

if "Start_Lat" in df.columns and "Start_Lng" in df.columns:
    map_sample = (
        df[["Start_Lat","Start_Lng","Severity"]]
        .dropna()
        .sample(min(40_000, len(df)), random_state=42)
    )
    m = folium.Map(location=[37.5, -96.0], zoom_start=4,
                   tiles="CartoDB positron")
    HeatMap(
        map_sample[["Start_Lat","Start_Lng"]].values.tolist(),
        radius=8, blur=10, max_zoom=13,
        gradient={"0.4":"blue","0.65":"lime","1":"red"}
    ).add_to(m)
    st_folium(m, width="100%", height=450, returned_objects=[])
else:
    st.warning("Latitude/Longitude columns not found in dataset.")

# ── Row 5: Feature Importance (if model loaded) ────────────────────────────────
if model is not None:
    st.markdown('<div class="section-header">🤖 Model Feature Importances</div>',
                unsafe_allow_html=True)

    FEATURE_CANDIDATES = [
        "hour","day_of_week","month","is_weekend","is_night",
        "Weather_Condition","Wind_Direction","Sunrise_Sunset",
        "Temperature(F)","Humidity(%)","Pressure(in)",
        "Visibility(mi)","Wind_Speed(mph)","Precipitation(in)",
        "Start_Lat","Start_Lng","Distance(mi)","duration_min"
    ]
    features = [f for f in FEATURE_CANDIDATES if f in df_full.columns]

    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=features)
        imp = imp.sort_values(ascending=True).tail(12).reset_index()
        imp.columns = ["Feature","Importance"]
        fig7 = px.bar(
            imp, x="Importance", y="Feature",
            orientation="h", title="XGBoost Feature Importances",
            color="Importance",
            color_continuous_scale=["#B5D4F4","#D85A30"]
        )
        fig7.update_layout(height=380, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig7, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data: US Accidents (Moosavi et al.) via Kaggle · Built with Streamlit, Folium, Plotly")