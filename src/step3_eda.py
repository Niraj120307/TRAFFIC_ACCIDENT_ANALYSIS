# =============================================================================
# STEP 3 — Exploratory Data Analysis (EDA)
# =============================================================================
# RUN AFTER: python src/step2_preprocess.py
# RUN:        python src/step3_eda.py
# OUTPUTS:    outputs/  → PNG charts + HTML map
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # no display needed — saves files
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import plotly.express as px

print("=" * 60)
print("STEP 3 — Exploratory Data Analysis")
print("=" * 60)

# ── Setup ─────────────────────────────────────────────────────────────────────
CLEAN_PATH = os.path.join("data", "clean_accidents.csv")
OUT_DIR    = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 130, "figure.figsize": (10, 5)})

print(f"\nLoading {CLEAN_PATH} ...")
df = pd.read_csv(CLEAN_PATH, low_memory=False)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

# =============================================================================
# CHART 1 — Accidents by Hour of Day
# =============================================================================
print("[1/7] Chart: accidents by hour of day...")
fig, ax = plt.subplots()
hourly = df["hour"].value_counts().sort_index()
bars = ax.bar(hourly.index, hourly.values, color="#378ADD", edgecolor="white", linewidth=0.5)
ax.set_xlabel("Hour of day (0–23)")
ax.set_ylabel("Number of accidents")
ax.set_title("Accidents by Hour of Day", fontsize=14, fontweight="bold")
ax.set_xticks(range(0, 24))
# Highlight rush hours
for i in [7, 8, 17, 18]:
    if i < len(bars):
        bars[i].set_color("#D85A30")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, color="#378ADD", label="Normal hours"),
    plt.Rectangle((0,0),1,1, color="#D85A30", label="Rush hours"),
], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_hourly_accidents.png"))
plt.close()
print("      Saved → outputs/01_hourly_accidents.png")

# =============================================================================
# CHART 2 — Accidents by Day of Week
# =============================================================================
print("[2/7] Chart: accidents by day of week...")
day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, ax = plt.subplots()
dow = df["day_of_week"].value_counts().sort_index()
colors = ["#D85A30" if i >= 5 else "#378ADD" for i in dow.index]
ax.bar(day_labels[:len(dow)], dow.values, color=colors, edgecolor="white", linewidth=0.5)
ax.set_title("Accidents by Day of Week", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of accidents")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_day_of_week.png"))
plt.close()
print("      Saved → outputs/02_day_of_week.png")

# =============================================================================
# CHART 3 — Hour × Day Heatmap
# =============================================================================
print("[3/7] Chart: hour × day heatmap...")
if "hour" in df.columns and "day_of_week" in df.columns:
    pivot = df.groupby(["hour", "day_of_week"]).size().unstack(fill_value=0)
    pivot.columns = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][:len(pivot.columns)]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.3,
                cbar_kws={"label": "Accident count"})
    ax.set_title("Accident Frequency: Hour × Day of Week", fontsize=14, fontweight="bold")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Hour of day")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_hour_day_heatmap.png"))
    plt.close()
    print("      Saved → outputs/03_hour_day_heatmap.png")

# =============================================================================
# CHART 4 — Accidents by Month
# =============================================================================
print("[4/7] Chart: monthly trend...")
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
fig, ax = plt.subplots()
monthly = df["month"].value_counts().sort_index()
ax.plot(month_labels[:len(monthly)], monthly.values,
        marker="o", color="#1D9E75", linewidth=2, markersize=7)
ax.fill_between(range(len(monthly)), monthly.values, alpha=0.15, color="#1D9E75")
ax.set_xticks(range(len(monthly)))
ax.set_xticklabels(month_labels[:len(monthly)])
ax.set_title("Accidents by Month (Seasonal Trend)", fontsize=14, fontweight="bold")
ax.set_ylabel("Number of accidents")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_monthly_trend.png"))
plt.close()
print("      Saved → outputs/04_monthly_trend.png")

# =============================================================================
# CHART 5 — Severity Distribution
# =============================================================================
print("[5/7] Chart: severity distribution...")
fig, ax = plt.subplots(figsize=(7, 5))
sev = df["Severity"].value_counts().sort_index()
colors_sev = ["#1D9E75", "#378ADD", "#D85A30", "#7F77DD"]
ax.bar(sev.index.astype(str), sev.values, color=colors_sev[:len(sev)],
       edgecolor="white", linewidth=0.5)
ax.set_title("Accident Severity Distribution (1=Low, 4=High)", fontsize=13, fontweight="bold")
ax.set_xlabel("Severity level")
ax.set_ylabel("Count")
for i, (idx, val) in enumerate(sev.items()):
    ax.text(i, val + sev.max()*0.01, f"{val:,}", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_severity_dist.png"))
plt.close()
print("      Saved → outputs/05_severity_dist.png")

# =============================================================================
# CHART 6 — Top 10 Weather Conditions by Accident Count
# =============================================================================
print("[6/7] Chart: top weather conditions...")
if "Weather_Condition" in df.columns:
    # Try to use original string column if available from raw, else use encoded
    weather_col = "Weather_Condition"
    weather_counts = df[weather_col].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    weather_counts.sort_values().plot(kind="barh", ax=ax, color="#7F77DD",
                                      edgecolor="white", linewidth=0.5)
    ax.set_title("Top 10 Weather Conditions in Accidents", fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of accidents")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "06_weather_conditions.png"))
    plt.close()
    print("      Saved → outputs/06_weather_conditions.png")

# =============================================================================
# CHART 7 — Folium Heatmap (Geographical Hotspots)
# =============================================================================
print("[7/7] Map: geographical hotspot heatmap...")
if "Start_Lat" in df.columns and "Start_Lng" in df.columns:
    coords = df[["Start_Lat", "Start_Lng"]].dropna()
    # Sample max 60,000 rows for performance
    sample = coords.sample(min(60000, len(coords)), random_state=42)
    m = folium.Map(
        location=[37.5, -96.0],
        zoom_start=4,
        tiles="CartoDB positron"
    )
    HeatMap(
        sample.values.tolist(),
        radius=8,
        blur=10,
        max_zoom=13,
        gradient={"0.4": "blue", "0.65": "lime", "1": "red"}
    ).add_to(m)
    folium.LayerControl().add_to(m)
    map_path = os.path.join(OUT_DIR, "07_hotspot_map.html")
    m.save(map_path)
    print(f"      Saved → {map_path}")
    print("      Open outputs/07_hotspot_map.html in your browser to view the map!")

# =============================================================================
# QUICK STATS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("EDA QUICK STATS")
print("=" * 60)
print(f"Total accidents          : {len(df):,}")
print(f"Peak hour                : {int(df['hour'].mode()[0])}:00")
print(f"Most dangerous day       : {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(df['day_of_week'].mode()[0])]}")
print(f"Most dangerous month     : {['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][int(df['month'].mode()[0])-1]}")
if "is_night" in df.columns:
    night_pct = df["is_night"].mean() * 100
    print(f"Night accidents          : {night_pct:.1f}%")
if "is_weekend" in df.columns:
    wknd_pct = df["is_weekend"].mean() * 100
    print(f"Weekend accidents        : {wknd_pct:.1f}%")

print("\nStep 3 complete! Now run: python src/step4_model.py")