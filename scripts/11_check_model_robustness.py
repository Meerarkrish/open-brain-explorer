from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "all_participants_v1"
RESULTS = SOURCE / "adjusted_group_comparison"
OUTPUT = SOURCE / "model_robustness"
OUTPUT.mkdir(parents=True, exist_ok=True)

people = pd.read_csv(SOURCE / "processing_summary.csv")
people = (
    people.loc[people["status"] == "Processed"]
    .sort_values("participant_id")
    .reset_index(drop=True)
)

edges = pd.read_csv(RESULTS / "all_edge_results.csv")

if people["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs.")

if not people["group"].isin(["child", "adult"]).all():
    raise ValueError("Unexpected group labels.")

design = pd.DataFrame({
    "intercept": np.ones(len(people)),
    "adult": people["group"].eq("adult").astype(float),
})

for column in ["mean_fd_retained", "retained_volumes"]:
    values = pd.to_numeric(people[column], errors="raise")

    if not np.isfinite(values).all() or values.std(ddof=0) == 0:
        raise ValueError(f"Invalid covariate: {column}")

    design[column] = (
        values - values.mean()
    ) / values.std(ddof=0)

X = design.to_numpy(dtype=float)

if np.linalg.matrix_rank(X) != X.shape[1]:
    raise ValueError("Model design is not full rank.")

# Load participant matrices once.
matrices = []
labels = None

for participant_id in people["participant_id"]:
    table = pd.read_csv(
        SOURCE / f"{participant_id}_correlations.csv",
        index_col=0,
    )

    if labels is None:
        labels = list(table.columns)

    if list(table.columns) != labels or list(table.index) != labels:
        raise ValueError(f"Region mismatch: {participant_id}")

    matrix = table.to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(f"Invalid matrix: {participant_id}")

    matrices.append(matrix)

matrices = np.stack(matrices)

# Build outcomes in the same order as all_edge_results.csv.
outcomes = []

for _, edge in edges.iterrows():
    i = int(edge["region_1_index"])
    j = int(edge["region_2_index"])

    if labels[i] != edge["region_1"] or labels[j] != edge["region_2"]:
        raise ValueError("Result-table region labels do not match matrices.")

    outcomes.append(
        np.arctanh(
            np.clip(matrices[:, i, j], -0.999999, 0.999999)
        )
    )

Y = np.column_stack(outcomes)

# Fit all edges simultaneously for coefficient diagnostics.
full_beta = np.linalg.lstsq(X, Y, rcond=None)[0][1]

X_motion_only = design[
    ["intercept", "adult", "mean_fd_retained"]
].to_numpy()

motion_only_beta = np.linalg.lstsq(
    X_motion_only, Y, rcond=None
)[0][1]

if not np.allclose(
    full_beta,
    edges["adult_minus_child_beta_z"].to_numpy(),
    atol=1e-7,
):
    raise ValueError("Recalculated coefficients differ from script 10.")

# Leave one participant out each time.
loo_beta = []

for omitted in range(len(people)):
    keep = np.arange(len(people)) != omitted

    if np.linalg.matrix_rank(X[keep]) != X.shape[1]:
        raise ValueError("A leave-one-out design is not full rank.")

    coefficients = np.linalg.lstsq(
        X[keep], Y[keep], rcond=None
    )[0][1]

    loo_beta.append(coefficients)

loo_beta = np.stack(loo_beta)
absolute_changes = np.abs(loo_beta - full_beta[None, :])
most_influential = absolute_changes.argmax(axis=0)

report = edges.copy()
report["motion_only_beta_z"] = motion_only_beta
report["change_without_volume_covariate"] = (
    motion_only_beta - full_beta
)
report["loo_min_beta_z"] = loo_beta.min(axis=0)
report["loo_max_beta_z"] = loo_beta.max(axis=0)
report["loo_max_absolute_change"] = absolute_changes.max(axis=0)
report["loo_sign_change"] = (
    np.sign(loo_beta) != np.sign(full_beta)[None, :]
).any(axis=0)
report["largest_change_when_omitting"] = (
    people["participant_id"].to_numpy()[most_influential]
)

report.to_csv(
    OUTPUT / "edge_robustness.csv",
    index=False,
)

# Identify participants that change coefficients most.
participant_influence = pd.DataFrame({
    "participant_id": people["participant_id"],
    "group": people["group"],
    "maximum_absolute_beta_change": absolute_changes.max(axis=1),
    "median_absolute_beta_change": np.median(
        absolute_changes, axis=1
    ),
}).sort_values(
    "maximum_absolute_beta_change",
    ascending=False,
)

participant_influence.to_csv(
    OUTPUT / "participant_influence.csv",
    index=False,
)

# Compare coefficient estimates across adjustment models.
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(full_beta, motion_only_beta, s=14, alpha=0.5)

limit = max(
    float(np.abs(full_beta).max()),
    float(np.abs(motion_only_beta).max()),
    0.01,
)
ax.plot(
    [-limit, limit],
    [-limit, limit],
    "--",
    color="black",
    linewidth=1,
)
ax.set(
    xlabel="Group coefficient: motion + retained-volume adjustment",
    ylabel="Group coefficient: motion adjustment only",
    title="Sensitivity to retained-volume adjustment",
)
fig.tight_layout()
fig.savefig(OUTPUT / "covariate_sensitivity.png", dpi=180)
plt.close(fig)

# Diagnostic plots for five selected edges.
top_positions = np.argsort(
    edges["p_adjusted_bh"].to_numpy()
)[:5]

for position in top_positions:
    edge = edges.iloc[position]
    y = Y[:, position]

    fit = sm.OLS(y, design).fit()
    influence = fit.get_influence()
    cooks_distance = influence.cooks_distance[0]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for group, colour in [
        ("child", "#3478a5"),
        ("adult", "#d66a37"),
    ]:
        keep = people["group"].eq(group).to_numpy()
        axes[0].scatter(
            people.loc[keep, "mean_fd_retained"],
            y[keep],
            label=group,
            color=colour,
            alpha=0.65,
            s=22,
        )

    axes[0].set(
        xlabel="Retained mean FD (mm)",
        ylabel="Individual connectivity (Fisher-z)",
        title="Connectivity versus motion",
    )
    axes[0].legend()

    axes[1].scatter(
        fit.fittedvalues,
        fit.resid,
        alpha=0.6,
        s=22,
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        xlabel="Fitted connectivity",
        ylabel="Residual",
        title="Residuals versus fitted values",
    )

    axes[2].scatter(
        np.arange(len(people)),
        cooks_distance,
        s=20,
    )
    axes[2].set(
        xlabel="Participant row in sorted analysis sample",
        ylabel="Cook's distance",
        title="Participant influence",
    )

    fig.suptitle(f"{edge['region_1']} ↔ {edge['region_2']}")
    fig.tight_layout()
    fig.savefig(
        OUTPUT / f"diagnostic_edge_{position:03d}.png",
        dpi=180,
    )
    plt.close(fig)

selected = report.loc[report["p_adjusted_bh"] < 0.05]

print("\nROBUSTNESS CHECK COMPLETE")
print(f"Connections examined: {len(report)}")
print(
    "Previously BH-selected connections with a leave-one-out "
    f"sign change: {selected['loo_sign_change'].sum()}"
)
print(
    "Largest leave-one-out coefficient change across all edges: "
    f"{report['loo_max_absolute_change'].max():.4f}"
)
print(
    "Correlation of coefficients between adjustment models: "
    f"{np.corrcoef(full_beta, motion_only_beta)[0, 1]:.4f}"
)

print("\nFive participants with largest coefficient changes:")
print(participant_influence.head(5).to_string(index=False))
print(f"\nOutputs saved in:\n{OUTPUT}")