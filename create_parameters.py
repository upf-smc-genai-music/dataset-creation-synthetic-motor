"""
create_parameters.py  –  Create parameters.json and print dataset statistics
=============================================================================

Usage
-----
    python create_parameters.py path/to/glissandi/

Creates  glissandi/parameters.json  and prints summary stats.
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd

def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "glissandi"

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)

    # ── Load all CSVs ─────────────────────────────────────────────────────────
    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    csv_files = [c for c in csv_files
                 if not os.path.basename(c).startswith("parameters")]

    if not csv_files:
        print("No CSV files found.")
        return

    durations = []
    hz_all    = []

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "time_sec" not in df.columns or "frequency_hz" not in df.columns:
            continue
        duration = df["time_sec"].iloc[-1] - df["time_sec"].iloc[0]
        durations.append(duration)
        hz_all.extend(df["frequency_hz"].dropna().tolist())

    durations = np.array(durations)
    hz_all    = np.array(hz_all)

    # ── parameters.json  (min/max from actual data) ───────────────────────────
    parameters = {
        "parameter_1": {
            "name": "frequency_hz",
            "type": "continuous",
            "unit": "Hz",
            "min": round(float(hz_all.min()), 4),
            "max": round(float(hz_all.max()), 4)
        }
    }

    out_path = os.path.join(folder, "parameters.json")
    with open(out_path, "w") as f:
        json.dump(parameters, f, indent=4)
    print(f"Written: {out_path}\n")

    # ── Dataset statistics ────────────────────────────────────────────────────
    total_sec = durations.sum()

    print("── Dataset statistics ───────────────────────────────")
    print(f"  Glissandi:        {len(durations)}")
    print(f"  Total duration:   {total_sec:.1f}s  ({total_sec/60:.2f} min)")
    print(f"  Average duration: {durations.mean():.2f}s")
    print(f"  Min / Max:        {durations.min():.2f}s  /  {durations.max():.2f}s")
    print(f"  Std duration:     {durations.std():.2f}s")
    print(f"  f0 range:         {hz_all.min():.1f} – {hz_all.max():.1f} Hz")
    print(f"  f0 mean:          {hz_all.mean():.1f} Hz")
    print("─────────────────────────────────────────────────────")

if __name__ == "__main__":
    main()