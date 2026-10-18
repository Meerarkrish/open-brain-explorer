from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "all_participants_v1"
OUTPUT = SOURCE / "equal_volume_sensitivity"
OUTPUT.mkdir(parents=True, exist_ok=True)

N_VOLUMES = 120
N_REPEATS = 50
SEED = 2026

summary = pd.read_csv(SOURCE / "processing_summary.csv")
included = (
    summary.loc[summary["status"] == "Processed"]
    .sort_values("participant_id")
    .reset_index(drop=True)
)

if included["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs.")

if not included["group"].isin(["child", "adult"]).all():
    raise ValueError("Unexpected group labels.")

adult_mask = included["group"].eq("adult").to_numpy()
child_mask = included["group"].eq("child").to_numpy()

if not adult_mask.any() or not child_mask.any():
    raise ValueError("Both groups must be present.")

signals_by_person = []
indices_by_person = []
labels = None

for _, person in included.iterrows():
    participant_id = person["participant_id"]
    table = pd.read_csv(
        SOURCE / f"{participant_id}_signals.csv"
    )

    if "original_volume_index" not in table.columns:
        raise ValueError(f"Missing volume indices: {participant_id}")

    indices = table["original_volume_index"].to_numpy(dtype=int)
    regional = table.drop(columns="original_volume_index")

    if labels is None:
        labels = list(regional.columns)
    elif list(regional.columns) != labels:
        raise ValueError(f"Region order mismatch: {participant_id}")

    values = regional.to_numpy(dtype=float)

    if len(values) != int(person["retained_volumes"]):
        raise ValueError(f"Volume count mismatch: {participant_id}")

    if len(values) < N_VOLUMES:
        raise ValueError(f"Too few retained volumes: {participant_id}")

    if not np.isfinite(values).all():
        raise ValueError(f"Invalid signals: {participant_id}")

    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate volume indices: {participant_id}")

    signals_by_person.append(values)
    indices_by_person.append(indices)

n_regions = len(labels)
upper = np.triu_indices(n_regions, k=1)
n_edges = len(upper[0])


def fisher_edges(values):
    if (values.std(axis=0) < 1e-8).any():
        raise ValueError("Constant regional signal in selected volumes.")

    correlation = np.corrcoef(values, rowvar=False)

    if not np.isfinite(correlation).all():
        raise ValueError("Invalid correlation matrix.")

    return np.arctanh(
        np.clip(correlation[upper], -0.999999, 0.999999)
    )


# Reference: all retained observations for each participant.
baseline = np.stack([
    fisher_edges(values)
    for values in signals_by_person
])

baseline_difference = (
    baseline[adult_mask].mean(axis=0)
    - baseline[child_mask].mean(axis=0)
)

rng = np.random.default_rng(SEED)
differences = []
selection_records = []

for repeat in range(N_REPEATS):
    participant_edges = []

    for i, person in included.iterrows():
        values = signals_by_person[i]

        # Sampling without replacement; keep original temporal order.
        positions = np.sort(
            rng.choice(
                len(values),
                size=N_VOLUMES,
                replace=False,
            )
        )

        participant_edges.append(
            fisher_edges(values[positions])
        )

        selection_records.append({
            "repeat": repeat + 1,
            "participant_id": person["participant_id"],
            "original_volume_indices": json.dumps(
                indices_by_person[i][positions].tolist()
            ),
        })

    participant_edges = np.stack(participant_edges)

    differences.append(
        participant_edges[adult_mask].mean(axis=0)
        - participant_edges[child_mask].mean(axis=0)
    )

    print(f"Completed repeat {repeat + 1}/{N_REPEATS}")

differences = np.stack(differences)
average_difference = differences.mean(axis=0)

# These describe variability across volume selections, not uncertainty
# from sampling participants.
lower = np.quantile(differences, 0.025, axis=0)
upper_bound = np.quantile(differences, 0.975, axis=0)

sign_agreement = (
    np.sign(differences)
    == np.sign(baseline_difference)[None, :]
).mean(axis=0)

results = pd.DataFrame({
    "region_1": [labels[i] for i in upper[0]],
    "region_2": [labels[j] for j in upper[1]],
    "all_retained_adult_minus_child_z": baseline_difference,
    "equal_volume_mean_adult_minus_child_z": average_difference,
    "selection_percentile_2.5": lower,
    "selection_percentile_97.5": upper_bound,
    "fraction_same_sign_as_all_retained": sign_agreement,
})

results.to_csv(
    OUTPUT / "edge_sensitivity.csv",
    index=False,
)

pd.DataFrame(selection_records).to_csv(
    OUTPUT / "selected_volume_indices.csv",
    index=False,
)

np.save(
    OUTPUT / "repeat_group_differences_z.npy",
    differences,
)

# Save a descriptive difference matrix in Fisher-z units.
matrix = np.zeros((n_regions, n_regions))
matrix[upper] = average_difference
matrix[(upper[1], upper[0])] = average_difference

pd.DataFrame(
    matrix,
    index=labels,
    columns=labels,
).to_csv(
    OUTPUT / "equal_volume_difference_z.csv",
    index_label="region",
)

# Compare the edge-difference patterns.
pattern_correlation = np.corrcoef(
    baseline_difference,
    average_difference,
)[0, 1]

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    baseline_difference,
    average_difference,
    alpha=0.5,
    s=14,
)

limit = max(
    float(np.abs(baseline_difference).max()),
    float(np.abs(average_difference).max()),
    0.01,
)

ax.plot(
    [-limit, limit],
    [-limit, limit],
    linestyle="--",
    color="black",
    linewidth=1,
)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)

ax.set(
    xlabel="Adult minus child mean Fisher-z: all retained volumes",
    ylabel="Adult minus child mean Fisher-z: 120-volume selections",
    title="Sensitivity of descriptive group differences",
)
fig.tight_layout()
fig.savefig(
    OUTPUT / "sensitivity_comparison.png",
    dpi=180,
)
plt.close(fig)

settings = {
    "participants": len(included),
    "adults": int(adult_mask.sum()),
    "children": int(child_mask.sum()),
    "volumes_per_participant": N_VOLUMES,
    "repeats": N_REPEATS,
    "seed": SEED,
    "selection": "Without replacement from retained volumes",
    "cleaning": "Reuses signals cleaned by script 07; not refitted",
    "difference_units": "Adult minus child mean Fisher-z",
    "percentiles": "Volume-selection variability, not confidence intervals",
    "statistical_significance_testing": False,
}

with open(
    OUTPUT / "settings.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(settings, file, indent=2)

print("\nSENSITIVITY CHECK COMPLETE")
print(f"Participants: {len(included)}")
print(f"Adults: {adult_mask.sum()}")
print(f"Children: {child_mask.sum()}")
print(f"Connections: {n_edges}")
print(f"Volumes per participant: {N_VOLUMES}")
print(f"Repeated selections: {N_REPEATS}")
print(
    "Correlation between all-retained and equal-volume "
    f"edge-difference patterns: {pattern_correlation:.3f}"
)
print(
    "Median fraction of selections retaining the original "
    f"difference direction: {np.median(sign_agreement):.3f}"
)
print(f"\nOutputs saved in:\n{OUTPUT}")