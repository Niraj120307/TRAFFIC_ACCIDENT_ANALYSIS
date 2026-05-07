# =============================================================================
# STEP 4 — Machine Learning: Clustering + Severity Prediction
# =============================================================================
# RUN AFTER: python src/step3_eda.py
# RUN:        python src/step4_model.py
# OUTPUTS:    models/xgb_severity.pkl
#             outputs/08_feature_importance.png
#             outputs/09_confusion_matrix.png
#             outputs/10_cluster_map.html
# =============================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import folium

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except ImportError:
    XGBOOST_OK = False
    print("XGBoost not found — using RandomForest only. pip install xgboost")

print("=" * 60)
print("STEP 4 — Machine Learning Modelling")
print("=" * 60)

# ── Setup ─────────────────────────────────────────────────────────────────────
CLEAN_PATH  = os.path.join("data", "clean_accidents.csv")
MODELS_DIR  = "models"
OUT_DIR     = "outputs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 130})

print(f"\nLoading {CLEAN_PATH} ...")
df = pd.read_csv(CLEAN_PATH, low_memory=False)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# =============================================================================
# PART A — Severity Prediction (Supervised Classification)
# =============================================================================
print("\n── PART A: Severity Prediction ──────────────────────────────")

FEATURE_CANDIDATES = [
    "hour", "day_of_week", "month", "is_weekend", "is_night",
    "Weather_Condition", "Wind_Direction", "Sunrise_Sunset",
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)",
    "Start_Lat", "Start_Lng", "Distance(mi)", "duration_min"
]

features = [f for f in FEATURE_CANDIDATES if f in df.columns]
target   = "Severity"

print(f"Features used ({len(features)}): {features}")

X = df[features].fillna(0)
y = df[target]

# Make severity 0-indexed for XGBoost (1,2,3,4 → 0,1,2,3)
y_model = y - y.min()

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_model, test_size=0.20, stratify=y_model, random_state=42
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── Train XGBoost (or fallback to RandomForest) ───────────────────────────────
if XGBOOST_OK:
    print("\nTraining XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
else:
    print("\nTraining RandomForest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Evaluation ────────────────────────────────────────────────────────────────
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average="macro")
print(f"\nAccuracy : {acc:.4f}")
print(f"F1 Macro : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=[f"Severity {i+y.min()}" for i in range(y_model.nunique())]))

# ── Save model ────────────────────────────────────────────────────────────────
model_path = os.path.join(MODELS_DIR, "xgb_severity.pkl")
joblib.dump(model, model_path)
print(f"Model saved → {model_path}")

# ── Chart 8: Feature Importances ─────────────────────────────────────────────
print("\nPlotting feature importances...")
importances = pd.Series(model.feature_importances_, index=features)
importances = importances.sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#378ADD" if v < importances.quantile(0.75) else "#D85A30"
          for v in importances.values]
importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white", linewidth=0.4)
ax.set_title("Top 15 Feature Importances (Severity Prediction)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance score")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_feature_importance.png"))
plt.close()
print("      Saved → outputs/08_feature_importance.png")

# ── Chart 9: Confusion Matrix ─────────────────────────────────────────────────
print("Plotting confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=[f"Sev {i+y.min()}" for i in range(cm.shape[1])],
            yticklabels=[f"Sev {i+y.min()}" for i in range(cm.shape[0])],
            linewidths=0.5)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Severity Prediction", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_confusion_matrix.png"))
plt.close()
print("      Saved → outputs/09_confusion_matrix.png")

# =============================================================================
# PART B — Hotspot Clustering (DBSCAN)
# =============================================================================
print("\n── PART B: Hotspot Clustering (DBSCAN) ──────────────────────")

if "Start_Lat" in df.columns and "Start_Lng" in df.columns:
    # Use a manageable sample for DBSCAN
    SAMPLE_SIZE = min(100_000, len(df))
    sample_df = df[["Start_Lat", "Start_Lng", "Severity"]].dropna()
    sample_df = sample_df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    coords = sample_df[["Start_Lat", "Start_Lng"]].values

    print(f"Running DBSCAN on {SAMPLE_SIZE:,} sampled coordinates...")
    print("(eps=0.05 ≈ ~5km,  min_samples=50)")
    db = DBSCAN(eps=0.05, min_samples=50, algorithm="ball_tree",
                metric="haversine", n_jobs=-1)

    # Haversine expects radians
    coords_rad = np.radians(coords)
    labels = db.fit_predict(coords_rad)
    sample_df["cluster"] = labels

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct  = (labels == -1).mean() * 100
    print(f"\nClusters found : {n_clusters}")
    print(f"Noise points   : {noise_pct:.1f}%")

    # Top clusters by size
    cluster_stats = (
        sample_df[sample_df["cluster"] >= 0]
        .groupby("cluster")
        .agg(count=("cluster","size"), avg_severity=("Severity","mean"))
        .sort_values("count", ascending=False)
        .head(10)
    )
    print(f"\nTop 10 clusters:\n{cluster_stats.to_string()}")

    # ── Map clusters ──────────────────────────────────────────────────────────
    print("\nBuilding cluster map...")
    import random
    random.seed(42)
    color_palette = [
        "#E63946","#2A9D8F","#E9C46A","#F4A261","#264653",
        "#7F77DD","#D85A30","#1D9E75","#378ADD","#BA7517"
    ]

    m2 = folium.Map(location=[37.5, -96.0], zoom_start=4,
                    tiles="CartoDB positron")

    # Plot top-5 clusters as coloured circles
    top_clusters = cluster_stats.head(5).index.tolist()
    for cid in top_clusters:
        cluster_pts = sample_df[sample_df["cluster"] == cid]
        center_lat  = cluster_pts["Start_Lat"].mean()
        center_lng  = cluster_pts["Start_Lng"].mean()
        col         = color_palette[cid % len(color_palette)]
        folium.CircleMarker(
            location=[center_lat, center_lng],
            radius=12,
            color=col,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>Cluster #{cid}</b><br>"
                f"Size: {cluster_pts.shape[0]:,} accidents<br>"
                f"Avg severity: {cluster_pts['Severity'].mean():.2f}",
                max_width=200
            )
        ).add_to(m2)

    # Also add a full heatmap layer
    from folium.plugins import HeatMap
    HeatMap(
        sample_df[["Start_Lat","Start_Lng"]].values.tolist(),
        radius=6, blur=8, max_zoom=13
    ).add_to(m2)

    cluster_map_path = os.path.join(OUT_DIR, "10_cluster_map.html")
    m2.save(cluster_map_path)
    print(f"      Saved → {cluster_map_path}")
    print("      Open outputs/10_cluster_map.html in your browser!")

else:
    print("Lat/Lng columns not found — skipping clustering.")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("MODELLING SUMMARY")
print("=" * 60)
print(f"Model           : {'XGBoost' if XGBOOST_OK else 'RandomForest'}")
print(f"Features used   : {len(features)}")
print(f"Test Accuracy   : {acc:.4f}")
print(f"F1 Macro Score  : {f1:.4f}")
if "n_clusters" in dir():
    print(f"DBSCAN clusters : {n_clusters}")
print("\nAll outputs saved in the outputs/ folder.")
print("\nStep 4 complete! Now run: streamlit run app.py")
