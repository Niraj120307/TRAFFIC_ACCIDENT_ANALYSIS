# =============================================================================
# STEP 1 — Download Dataset from Kaggle
# =============================================================================
# BEFORE RUNNING:
#   1. Go to https://www.kaggle.com/settings → API → Create New Token
#   2. It downloads kaggle.json — place it at:
#      Windows : C:\Users\<YourName>\.kaggle\kaggle.json
#      Mac/Linux: ~/.kaggle/kaggle.json
#   3. Make sure your venv is active and kagglehub is installed:
#      pip install kagglehub
#
# RUN: python src/step1_download.py
# =============================================================================

import os
import shutil
import kagglehub

# ── 1. Download from Kaggle (cached after first run) ─────────────────────────
print("Downloading dataset from Kaggle...")
print("(This may take a few minutes on first run — ~1GB)")

path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
print(f"\nKaggle cache path: {path}")

# ── 2. List what was downloaded ───────────────────────────────────────────────
files = os.listdir(path)
print(f"\nFiles found: {files}")

# ── 3. Copy CSV into project's data/raw/ folder ───────────────────────────────
os.makedirs("data/raw", exist_ok=True)

copied = False
for f in files:
    if f.endswith(".csv"):
        src = os.path.join(path, f)
        dst = os.path.join("data", "raw", "US_Accidents.csv")
        shutil.copy(src, dst)
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"\nSaved → data/raw/US_Accidents.csv  ({size_mb:.0f} MB)")
        copied = True
        break

if not copied:
    print("\nNo CSV found. Files in path:", files)
    print("Try renaming the file manually to data/raw/US_Accidents.csv")

print("\nStep 1 complete! Now run: python src/step2_preprocess.py")