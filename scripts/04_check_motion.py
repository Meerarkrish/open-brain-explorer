from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "development_fmri"
OUTPUT = ROOT / "outputs" / "motion_qc"
OUTPUT.mkdir(parents=True, exist_ok=True)


def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


fd_names = {
    "framewisedisplacement",
    "framedisplacement",
}

records = []
schemas = {}

# Search both formats, since filenames can vary by dataset version.
files = sorted(set(DATA.rglob("*.tsv")) | set(DATA.rglob("*.csv")))

for path in files:
    match = re.search(r"sub-pixar\d+", str(path))
    if not match:
        continue

    participant_id = match.group()

    try:
        table = pd.read_csv(
            path,
            sep="," if path.suffix == ".csv" else "\t",
            na_values=["n/a", "NA"],
        )

        # Skip unrelated participant-level tables.
        normalized = {normalize(c): c for c in table.columns}
        looks_like_confounds = (
            "confound" in path.name.lower()
            or "regressor" in path.name.lower()
            or any(c in normalized for c in fd_names)
        )
        if not looks_like_confounds:
            continue

        schemas[str(path)] = list(table.columns)

        fd_column = next(
            (c for c in table.columns if normalize(c) in fd_names),
            None,
        )

        record = {
            "participant_id": participant_id,
            "confound_file": str(path),
            "rows": len(table),
        }

        if fd_column is None:
            record["status"] = "FD column missing"
            records.append(record)
            continue

        fd = pd.to_numeric(table[fd_column], errors="coerce")
        fd = fd.replace([np.inf, -np.inf], np.nan)
        valid = fd.dropna()

        record.update({
            "fd_column": fd_column,
            "valid_fd_values": len(valid),
            "missing_fd_values": int(fd.isna().sum()),
            "negative_fd_values": int((valid < 0).sum()),
            "mean_fd_mm": valid.mean(),
            "median_fd_mm": valid.median(),
            "max_fd_mm": valid.max(),
            "percent_valid_fd_over_0.2mm": (
                100 * (valid > 0.2).mean()
            ),
            "percent_valid_fd_over_0.5mm": (
                100 * (valid > 0.5).mean()
            ),
            "status": (
                "Review: no valid FD"
                if valid.empty
                else "Review: negative FD"
                if (valid < 0).any()
                else "Summarized"
            ),
        })

        records.append(record)

    except Exception as error:
        records.append({
            "participant_id": participant_id,
            "confound_file": str(path),
            "status": "Read error",
            "error": str(error),
        })

if not records:
    raise RuntimeError(
        "No confound tables identified. "
        "Please share the filenames inside data/development_fmri."
    )

report = pd.DataFrame(records)
report.to_csv(OUTPUT / "motion_summary.csv", index=False)

with open(OUTPUT / "confound_columns.txt", "w", encoding="utf-8") as file:
    for filename, columns in schemas.items():
        file.write(f"{filename}\n")
        file.write(", ".join(columns) + "\n\n")

print("\nConfound tables processed:", len(report))
print("Unique participants:", report["participant_id"].nunique())
print("\nStatus:")
print(report["status"].value_counts().to_string())

duplicates = report["participant_id"].duplicated(keep=False)
if duplicates.any():
    print("\nWARNING: Multiple confound tables for some participants.")
    print(report.loc[duplicates, "participant_id"].unique())

good = report.loc[report["status"] == "Summarized"].copy()

if not good.empty:
    print("\nMean FD across participants (mm):")
    print(good["mean_fd_mm"].describe().round(3).to_string())

    print("\nTen participants with the highest mean FD:")
    print(
        good.nlargest(10, "mean_fd_mm")[
            ["participant_id", "mean_fd_mm",
             "percent_valid_fd_over_0.5mm"]
        ].round(3).to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        good["mean_fd_mm"],
        bins=25,
        color="#3478a5",
        edgecolor="white",
    )
    ax.set(
        xlabel="Mean framewise displacement (mm)",
        ylabel="Number of confound tables",
        title="Head motion across participants",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT / "mean_fd_distribution.png", dpi=180)
    plt.close(fig)

print(f"\nReports saved in: {OUTPUT}")