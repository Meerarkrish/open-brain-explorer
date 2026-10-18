import truststore
truststore.inject_into_ssl()

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nilearn import datasets, plotting


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "all_participants_v1"
OUTPUT = SOURCE / "group_summaries"
OUTPUT.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(SOURCE / "processing_summary.csv")

if summary["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs in processing summary.")

included = summary.loc[summary["status"] == "Processed"].copy()

# Summarize the retained samples before comparing connectivity.
qc_columns = [
    "age",
    "mean_fd_before_censoring",
    "mean_fd_retained",
    "retained_volumes",
    "percent_retained",
]

qc_summary = included.groupby("group")[qc_columns].agg(
    ["count", "mean", "median", "min", "max"]
)
qc_summary.to_csv(OUTPUT / "retained_sample_summary.csv")

print("\nRETAINED SAMPLE SUMMARY:")
print(qc_summary.round(3).to_string())

atlas = datasets.fetch_atlas_msdl(
    data_dir=str(ROOT / "data" / "atlases")
)

labels = [
    item.decode() if isinstance(item, bytes) else str(item)
    for item in atlas.labels
]
coordinates = np.asarray(atlas.region_coords, dtype=float)
n_regions = len(labels)

group_matrices = {}

for group in ["child", "adult"]:
    members = included.loc[included["group"] == group]

    if members.empty:
        raise ValueError(f"No processed participants in group: {group}")

    transformed = []

    for participant_id in members["participant_id"]:
        path = SOURCE / f"{participant_id}_correlations.csv"
        table = pd.read_csv(path, index_col=0)

        # Verify identical region order for every participant.
        if list(table.index) != labels or list(table.columns) != labels:
            raise ValueError(f"Atlas label mismatch: {participant_id}")

        matrix = table.to_numpy(dtype=float)

        if matrix.shape != (n_regions, n_regions):
            raise ValueError(f"Unexpected matrix shape: {participant_id}")

        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite correlations: {participant_id}")

        if (
            not np.allclose(matrix, matrix.T, atol=1e-6)
            or np.max(np.abs(matrix)) > 1 + 1e-6
        ):
            raise ValueError(f"Invalid correlation matrix: {participant_id}")

        # Diagonal values are not used in connectivity summaries.
        matrix = matrix.copy()
        np.fill_diagonal(matrix, 0)

        # Clipping prevents infinite values in arctanh.
        transformed.append(
            np.arctanh(np.clip(matrix, -0.999999, 0.999999))
        )

    # Equal weight for each participant.
    mean_z = np.mean(np.stack(transformed), axis=0)
    mean_r = np.tanh(mean_z)
    np.fill_diagonal(mean_r, 0)

    group_matrices[group] = mean_r

    pd.DataFrame(
        mean_r,
        index=labels,
        columns=labels,
    ).to_csv(
        OUTPUT / f"{group}_group_correlations.csv",
        index_label="region",
    )

    viewer = plotting.view_connectome(
        mean_r,
        coordinates,
        node_labels=labels,
        node_size=5,
        edge_threshold=0.4,
        symmetric_cmap=True,
        title=f"{group.capitalize()} group — n={len(members)}",
    )
    viewer.save_as_html(
        str(OUTPUT / f"{group}_group_3d.html")
    )

    print(f"\nCreated {group} summary: {len(members)} participants")

# Descriptive difference between the group correlation summaries.
difference = group_matrices["adult"] - group_matrices["child"]

pd.DataFrame(
    difference,
    index=labels,
    columns=labels,
).to_csv(
    OUTPUT / "adult_minus_child_correlations.csv",
    index_label="region",
)

# Common scales make the two group matrices comparable.
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

panels = [
    ("Children", group_matrices["child"], 1.0),
    ("Adults", group_matrices["adult"], 1.0),
    (
        "Adults minus children: descriptive",
        difference,
        max(float(np.max(np.abs(difference))), 0.01),
    ),
]

for ax, (title, matrix, limit) in zip(axes, panels):
    masked = np.ma.array(
        matrix,
        mask=np.eye(n_regions, dtype=bool),
    )

    image = ax.imshow(
        masked,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_title(title)
    ax.set_xlabel("Atlas region index")
    ax.set_ylabel("Atlas region index")
    fig.colorbar(image, ax=ax, shrink=0.7)

fig.suptitle(
    "Movie-watching connectivity — exploratory group summaries",
    fontsize=15,
)
fig.tight_layout()
fig.savefig(OUTPUT / "group_comparison.png", dpi=180)
plt.close(fig)

pd.DataFrame({
    "region_index": range(n_regions),
    "region_label": labels,
}).to_csv(OUTPUT / "region_labels.csv", index=False)

print(f"\nFinished! Outputs saved in:\n{OUTPUT}")