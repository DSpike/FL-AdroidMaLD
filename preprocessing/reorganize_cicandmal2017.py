"""
CIC-AndMal2017 Reorganization Script

Step 1 of 2 — Run this BEFORE the preprocessor.

What it does:
    1. Reads each zip from Downloads (no full extraction to disk)
    2. Flattens all family subfolders into one class label
    3. Merges all CSVs per class into a single CSV file
    4. Saves 5 clean files to datasets/CIC-AndMal2017/

Result:
    datasets/CIC-AndMal2017/
        Adware.csv
        Benign.csv
        Ransomware.csv
        SMS_Malware.csv
        Scareware.csv

Run:
    python preprocessing/reorganize_cicandmal2017.py
"""

import os
import io
import zipfile
import pandas as pd

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

DOWNLOADS_DIR = r'C:\Users\Dspike\Downloads'
OUT_DIR       = r'C:\Users\Dspike\Documents\FL-AdroidMaLD\datasets\CIC-AndMal2017'

ZIP_CLASS_MAP = {
    'Adware-CSVs.zip':     'Adware',
    'Benign-CSVs.zip':     'Benign',
    'Ransomware-CSVs.zip': 'Ransomware',
    'SMSmalware-CSVs.zip': 'SMS_Malware',
    'Scareware-CSVs.zip':  'Scareware',
}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def reorganize(downloads_dir: str = DOWNLOADS_DIR, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    for zip_name, label in ZIP_CLASS_MAP.items():
        zip_path = os.path.join(downloads_dir, zip_name)
        out_path = os.path.join(out_dir, f'{label}.csv')

        if not os.path.exists(zip_path):
            print(f"WARNING: {zip_path} not found — skipping")
            continue

        if os.path.exists(out_path):
            print(f"[{label}] already exists — skipping (delete to rerun)")
            continue

        print(f"\n[{label}] Processing {zip_name} ...", flush=True)

        frames = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            total     = len(csv_names)
            print(f"  {total} CSV files found", flush=True)

            for i, name in enumerate(csv_names):
                try:
                    with zf.open(name) as f:
                        df = pd.read_csv(io.TextIOWrapper(f), low_memory=False)
                        frames.append(df)
                except Exception as e:
                    print(f"  WARNING: skipped {name}: {e}")

                if (i + 1) % 200 == 0 or (i + 1) == total:
                    print(f"  {i+1}/{total} files read ...", flush=True)

        if not frames:
            print(f"  ERROR: no data loaded for {label}")
            continue

        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(out_path, index=False)

        print(f"  Saved: {out_path}")
        print(f"  Shape: {merged.shape}  ({len(merged):,} rows, {len(merged.columns)} cols)")

    print("\n" + "="*60)
    print("Reorganization complete.")
    print(f"Output directory: {out_dir}")
    print("Files created:")
    for f in sorted(os.listdir(out_dir)):
        size_mb = os.path.getsize(os.path.join(out_dir, f)) / 1e6
        print(f"  {f:<20} {size_mb:>8.1f} MB")
    print("\nNext step: python preprocessing/cicandmal2017_preprocessor.py")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloads_dir', default=DOWNLOADS_DIR)
    parser.add_argument('--out_dir',       default=OUT_DIR)
    args = parser.parse_args()
    reorganize(args.downloads_dir, args.out_dir)
