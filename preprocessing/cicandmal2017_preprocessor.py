"""
CIC-AndMal2017 Preprocessor for FL-AdroidMaLD

Step 2 of 2 — Run AFTER reorganize_cicandmal2017.py.

Reads 5 clean CSVs from datasets/CIC-AndMal2017/,
drops CICFlowMeter metadata columns, handles infinities/NaN,
Min-Max normalises, and saves combined_dataset.csv.

Output: normalized_dataset_cicandmal2017/combined_dataset.csv
        Same format as AndMal2020 — drop-in for fl_main_cicandmal2017.py

Run:
    python preprocessing/cicandmal2017_preprocessor.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DATA_DIR  = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\datasets\CIC-AndMal2017'
OUT_DIR   = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\normalized_dataset_cicandmal2017'
LABEL_COL = 'Category'

# Class CSV files
CLASS_FILES = {
    'Adware.csv':      'Adware',
    'Benign.csv':      'Benign',
    'Ransomware.csv':  'Ransomware',
    'SMS_Malware.csv': 'SMS_Malware',
    'Scareware.csv':   'Scareware',
}

# CICFlowMeter metadata columns — not features
DROP_COLS = {
    'Flow ID', ' Flow ID',
    'Source IP', ' Source IP',
    'Source Port', ' Source Port',
    'Destination IP', ' Destination IP',
    'Destination Port', ' Destination Port',
    'Timestamp', ' Timestamp',
    'Label', ' Label',
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _load_class(csv_path: str, label: str) -> pd.DataFrame:
    """Load one class CSV, drop metadata, keep numeric features."""
    print(f"  Loading {os.path.basename(csv_path)} ...", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  [{label:<12}] {len(df):>9,} rows, {len(df.columns)} cols raw", flush=True)

    # Drop metadata columns
    drop = [c for c in df.columns if c.strip() in {d.strip() for d in DROP_COLS}]
    df   = df.drop(columns=drop, errors='ignore')

    # Keep only numeric
    df = df.select_dtypes(include=[np.number])

    # Drop all-NaN columns
    df = df.dropna(axis=1, how='all')

    df[LABEL_COL] = label
    print(f"  [{label:<12}] {len(df.columns)-1} feature columns kept", flush=True)
    return df


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def preprocess(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Load all class CSVs ----
    frames       = []
    feature_sets = []

    for fname, label in CLASS_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            print(f"WARNING: {fpath} not found — skipping {label}")
            continue
        df = _load_class(fpath, label)
        feature_sets.append(set(c for c in df.columns if c != LABEL_COL))
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No class CSVs found in {data_dir}")

    # ---- 2. Align on common feature columns ----
    common_features = sorted(feature_sets[0].intersection(*feature_sets[1:]))
    print(f"\nCommon features across all classes: {len(common_features)}", flush=True)

    combined = pd.concat(
        [df[common_features + [LABEL_COL]] for df in frames],
        ignore_index=True,
    )

    print(f"\nTotal samples : {len(combined):,}")
    print(f"Total features: {len(common_features)}")
    print(f"\nClass distribution:")
    for cls, cnt in combined[LABEL_COL].value_counts().sort_index().items():
        print(f"  {cls:<15}: {cnt:>9,}")

    # ---- 3. Handle infinities and NaN ----
    print("\nImputing missing values ...", flush=True)
    X     = combined[common_features].replace([np.inf, -np.inf], np.nan)
    imp   = SimpleImputer(strategy='mean')
    X_imp = imp.fit_transform(X)

    # ---- 4. Min-Max normalise to [0, 1] ----
    print("Normalising to [0, 1] ...", flush=True)
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_imp)

    # ---- 5. Save ----
    out_df = pd.DataFrame(X_norm, columns=common_features)
    out_df[LABEL_COL] = combined[LABEL_COL].values

    out_path = os.path.join(out_dir, 'combined_dataset.csv')
    print(f"Saving to {out_path} ...", flush=True)
    out_df.to_csv(out_path, index=False)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nDone.")
    print(f"  Output : {out_path}")
    print(f"  Shape  : {out_df.shape}")
    print(f"  Size   : {size_mb:.1f} MB")
    print(f"\nNext step: python fl_main_cicandmal2017.py")
    return out_path


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=DATA_DIR)
    parser.add_argument('--out_dir',  default=OUT_DIR)
    args = parser.parse_args()
    preprocess(args.data_dir, args.out_dir)
