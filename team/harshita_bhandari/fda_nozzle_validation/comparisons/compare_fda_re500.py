from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

EXP_FILE = (
    ROOT
    / "reference_data"
    / "processed"
    / "Re500_experimental_centreline_summary.csv"
)

CFD_FILE = (
    ROOT
    / "our_results"
    / "cfd_centreline_Re500.csv"
)

RESULT_FILE = ROOT / "comparisons" / "centreline_velocity_Re500_final.csv"
METRIC_FILE = ROOT / "comparisons" / "comparison_metrics_Re500.csv"
PLOT_FILE = ROOT / "plots" / "FDA_Re500_centreline_velocity.png"


exp = pd.read_csv(EXP_FILE)
cfd = pd.read_csv(CFD_FILE)

df = pd.merge(exp, cfd, on="z_m", how="left")

# Error calculations only where CFD values exist
valid = df["cfd_velocity_mps"].notna()

df.loc[valid, "absolute_error_mps"] = np.abs(
    df.loc[valid, "cfd_velocity_mps"]
    - df.loc[valid, "experimental_mean_mps"]
)

df.loc[valid, "percentage_error"] = (
    df.loc[valid, "absolute_error_mps"]
    / np.abs(df.loc[valid, "experimental_mean_mps"])
    * 100
)

df.to_csv(RESULT_FILE, index=False)

if valid.any():

    error = (
        df.loc[valid, "cfd_velocity_mps"]
        - df.loc[valid, "experimental_mean_mps"]
    )

    rmse = np.sqrt(np.mean(error**2))

    mae = np.mean(np.abs(error))

    mape = np.mean(
        np.abs(error)
        / np.abs(df.loc[valid, "experimental_mean_mps"])
    ) * 100

    metrics = pd.DataFrame({
        "metric": [
            "RMSE",
            "MAE",
            "MAPE",
            "number_of_comparison_points"
        ],
        "value": [
            rmse,
            mae,
            mape,
            valid.sum()
        ],
        "units": [
            "m/s",
            "m/s",
            "%",
            "-"
        ]
    })

    metrics.to_csv(METRIC_FILE, index=False)

    print("\nFDA Re=500 validation metrics")
    print(metrics.to_string(index=False))

else:
    print("\nNo CFD values present yet.")
    print("Experimental reference plot will still be generated.")


# Plot experimental mean ± standard deviation
plt.figure(figsize=(8, 5))

plt.errorbar(
    exp["z_m"],
    exp["experimental_mean_mps"],
    yerr=exp["experimental_std_mps"],
    marker="o",
    capsize=3,
    label="FDA experiment: mean ± SD"
)

if valid.any():
    plt.plot(
        df.loc[valid, "z_m"],
        df.loc[valid, "cfd_velocity_mps"],
        marker="s",
        label="CFD"
    )

plt.axvline(
    x=0,
    linestyle="--",
    label="Sudden expansion"
)

plt.xlabel("Axial position, z [m]")
plt.ylabel("Centreline axial velocity [m/s]")
plt.title("FDA Sudden-Expansion Nozzle — Re = 500")
plt.grid(True)
plt.legend()
plt.tight_layout()

PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOT_FILE, dpi=300)
plt.close()

print(f"\nSaved plot: {PLOT_FILE}")
print(f"Saved comparison table: {RESULT_FILE}")
