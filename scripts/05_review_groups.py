from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "outputs" / "motion_qc"

participants = pd.read_csv(ROOT / "participants.csv")
motion = pd.read_csv(OUTPUT / "motion_summary.csv")

# Validate IDs before joining.
if participants["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs in participants.csv")

if motion["participant_id"].duplicated().any():
    raise ValueError("Multiple motion records for a participant")

merged = participants.merge(
    motion,
    on="participant_id",
    how="outer",
    validate="one_to_one",
    indicator=True,
)

print("\nParticipant/motion matching:")
print(merged["_merge"].value_counts().to_string())

if not merged["_merge"].eq("both").all():
    merged.to_csv(OUTPUT / "matching_review.csv", index=False)
    raise ValueError("Unmatched records found. Review matching_review.csv")

merged = merged.drop(columns="_merge")
merged["group"] = (
    merged["Child_Adult"].astype(str).str.strip().str.lower()
)

if not merged["group"].isin(["child", "adult"]).all():
    raise ValueError("Unexpected participant group labels")

merged.to_csv(OUTPUT / "participants_motion.csv", index=False)

print("\nMean FD by group (mm):")
summary = merged.groupby("group")["mean_fd_mm"].agg(
    ["count", "mean", "median", "min", "max"]
)
print(summary.round(3).to_string())
summary.to_csv(OUTPUT / "motion_by_group.csv")

# Plot every participant rather than just group averages.
fig, ax = plt.subplots(figsize=(8, 5))

for position, group in enumerate(["child", "adult"]):
    values = (
        merged.loc[merged["group"] == group, "mean_fd_mm"]
        .dropna()
        .sort_values()
    )
    # Small deterministic offsets make overlapping points visible.
    offsets = [
        ((i % 9) - 4) * 0.025
        for i in range(len(values))
    ]
    ax.scatter(
        [position + offset for offset in offsets],
        values,
        alpha=0.65,
        s=28,
        label=group.capitalize(),
    )

ax.set_xticks([0, 1], ["Children", "Adults"])
ax.set_ylabel("Mean framewise displacement (mm)")
ax.set_title("Head motion by participant group")
fig.tight_layout()
fig.savefig(OUTPUT / "motion_by_group.png", dpi=180)
plt.close(fig)

# Find one low-motion example from each group.
# These are visualization examples, not representative group estimates.
examples = []

for group in ["child", "adult"]:
    candidates = merged.loc[
        (merged["group"] == group)
        & (merged["status"] == "Summarized")
        & (merged["negative_fd_values"] == 0)
    ].dropna(subset=["mean_fd_mm"])

    if candidates.empty:
        raise ValueError(f"No usable motion summary for {group}")

    selected = candidates.nsmallest(1, "mean_fd_mm").iloc[0]
    examples.append(selected)

    confound_path = Path(selected["confound_file"])
    separator = "," if confound_path.suffix == ".csv" else "\t"
    confounds = pd.read_csv(confound_path, sep=separator)

    print(f"\nSelected {group} example:")
    print("Participant:", selected["participant_id"])
    print("Age:", round(float(selected["Age"]), 2))
    print("Mean FD:", round(float(selected["mean_fd_mm"]), 3))
    print("Confound rows:", len(confounds))
    print("Available confound columns:")
    print(", ".join(confounds.columns))

pd.DataFrame(examples).to_csv(
    OUTPUT / "visualization_examples.csv",
    index=False,
)

print(f"\nOutputs saved in: {OUTPUT}")