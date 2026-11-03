#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 01:20:07 2026

@author: harshita
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Folder containing the five extracted FDA experimental files
folder = Path("/Users/harshita/Downloads/experiment")

files = sorted(folder.glob("PIV_Sudden_Expansion_500_*.txt"))

def extract_block(filepath, block_name):
    lines = filepath.read_text(errors="ignore").splitlines()

    for i, line in enumerate(lines):
        if line.strip() == block_name:
            n = int(lines[i + 1].strip())
            data = []

            for row in lines[i + 2:i + 2 + n]:
                parts = row.split()
                if len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1])])

            return np.array(data)

    raise ValueError(f"{block_name} not found in {filepath.name}")


all_runs = []

for f in files:
    data = extract_block(f, "plot-z-distribution-axial-velocity")

    df = pd.DataFrame(
        data,
        columns=["z_m", "axial_velocity_mps"]
    )

    df["dataset"] = f.stem.split("_")[-1]
    all_runs.append(df)

combined = pd.concat(all_runs, ignore_index=True)

summary = (
    combined
    .groupby("z_m")["axial_velocity_mps"]
    .agg(
        experimental_mean_mps="mean",
        experimental_std_mps="std",
        experimental_min_mps="min",
        experimental_max_mps="max",
        n="count"
    )
    .reset_index()
)

combined.to_csv(
    folder / "Re500_all_experimental_centreline_runs.csv",
    index=False
)

summary.to_csv(
    folder / "Re500_experimental_centreline_summary.csv",
    index=False
)

print("\nExperimental centreline summary:\n")
print(summary.to_string(index=False))

print("\nCreated:")
print("Re500_all_experimental_centreline_runs.csv")
print("Re500_experimental_centreline_summary.csv")