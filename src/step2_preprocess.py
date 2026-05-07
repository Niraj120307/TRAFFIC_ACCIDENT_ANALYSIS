# =============================================================================
# STEP 2 — Data Preprocessing & Feature Engineering
# =============================================================================
# RUN AFTER: python src/step1_download.py
# RUN:        python src/step2_preprocess.py
# OUTPUT:     data/clean_accidents.csv
# =============================================================================

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "US_Accidents.csv")

df = pd.read_csv(RAW_PATH, low_memory=False)
import numpy as np
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("STEP 2 — Preprocessing & Feature Engineering")
print("=" * 60)

# ── 1. Load raw data ──────────────────────────────────────────────────────────
RAW_PATH   = os.path.join("data", "raw", "US_Accidents.csv")
CLEAN_PATH = os.path.join("data", "clean_accidents.csv")

print(f"\nLoading {RAW_PATH} ...")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 2. Drop columns with more than 40% missing values ─────────────────────────
print("\n[1/6] Dropping high-missing columns...")
missing_ratio = df.isnull().mean()
drop_cols = missing_ratio[missing_ratio > 0.40].index.tolist()
df.drop(columns=drop_cols, inplace=True)
print(f"      Dropped {len(drop_cols)} columns → {df.shape[1]} remaining")

# ── 3. Drop fully duplicate rows ──────────────────────────────────────────────
print("[2/6] Removing duplicate rows...")
before = len(df)
df.drop_duplicates(inplace=True)
print(f"      Removed {before - len(df):,} duplicates")

# ── 4. Impute missing values ──────────────────────────────────────────────────
print("[3/6] Imputing missing values...")
# Numeric → median
for col in df.select_dtypes(include="number").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
# Categorical → mode
for col in df.select_dtypes(include="object").columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)
print(f"      Remaining nulls: {df.isnull().sum().sum()}")

# ── 5. Parse datetime & engineer time features ────────────────────────────────
print("[4/6] Engineering time features...")
if "Start_Time" in df.columns:
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df["hour"]        = df["Start_Time"].dt.hour
    df["day_of_week"] = df["Start_Time"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]       = df["Start_Time"].dt.month
    df["year"]        = df["Start_Time"].dt.year
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"]    = ((df["hour"] >= 20) | (df["hour"] <= 6)).astype(int)

if "End_Time" in df.columns and "Start_Time" in df.columns:
    df["End_Time"] = pd.to_datetime(df["End_Time"], errors="coerce")
    df["duration_min"] = (
        (df["End_Time"] - df["Start_Time"]).dt.total_seconds() / 60
    ).clip(0, 1440)  # cap at 24 hours

print(f"      Time columns added: hour, day_of_week, month, year, is_weekend, is_night, duration_min")

# ── 6. Encode categorical columns ─────────────────────────────────────────────
print("[5/6] Label-encoding categorical columns...")
cat_cols = ["Weather_Condition", "Wind_Direction", "Sunrise_Sunset",
            "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]
cat_cols = [c for c in cat_cols if c in df.columns]

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
print(f"      Encoded: {cat_cols}")

# ── 7. Save clean data ────────────────────────────────────────────────────────
print("[6/6] Saving clean data...")
df.to_csv(CLEAN_PATH, index=False)
size_mb = os.path.getsize(CLEAN_PATH) / (1024 * 1024)
print(f"      Saved → {CLEAN_PATH}  ({size_mb:.0f} MB)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Final shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Severity dist :\n{df['Severity'].value_counts().to_string()}")
print("\nStep 2 complete! Now run: python src/step3_eda.py")

# Add this at the end of step2_preprocess.py
df.to_parquet("data/clean_accidents.parquet", index=False)
print("Parquet saved — dashboard will now load 10× faster!")