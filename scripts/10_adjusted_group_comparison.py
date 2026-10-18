from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "all_participants_v1"
OUTPUT = SOURCE / "adjusted_group_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

people = pd.read_csv(SOURCE / "processing_summary.csv")
people = (
    people.loc[people["status"] == "Processed"]
    .sort_values("participant_id")
    .reset_index(drop=True)
)

if people["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs.")

if not people["group"].isin(["child", "adult"]).all():
    raise ValueError("Unexpected group labels.")

if people["group"].nunique() != 2:
    raise ValueError("Both groups are required.")

# Adult = 1, child = 0.
# Standardize nuisance covariates for numerical stability.
design = pd.DataFrame({
    "intercept": np.ones(len(people)),
    "adult": people["group"].eq("adult").astype(float),
})

scaling = {}

for column in ["mean_fd_retained", "retained_volumes"]:
    values = pd.to_numeric(people[column], errors="raise")

    if not np.isfinite(values).all():
        raise ValueError(f"Invalid values in {column}")

    mean = float(values.mean())
    std = float(values.std(ddof=0))

    if std == 0:
        raise ValueError(f"No variation in {column}")

    design[column] = (values - mean) / std
    scaling[column] = {"mean": mean, "std": std}

if np.linalg.matrix_rank(design.to_numpy()) != design.shape[1]:
    raise ValueError("The model design is not full rank.")

design_output = design.copy()
design_output.insert(0, "participant_id", people["participant_id"])
design_output.to_csv(OUTPUT / "model_design.csv", index=False)

labels = None
edge_values = []

for participant_id in people["participant_id"]:
    table = pd.read_csv(
        SOURCE / f"{participant_id}_correlations.csv",
        index_col=0,
    )

    if labels is None:
        labels = list(table.columns)
        upper = np.triu_indices(len(labels), k=1)

    if list(table.columns) != labels or list(table.index) != labels:
        raise ValueError(f"Region label mismatch: {participant_id}")

    matrix = table.to_numpy(dtype=float)

    if (
        matrix.shape != (len(labels), len(labels))
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, atol=1e-6)
        or np.max(np.abs(matrix)) > 1 + 1e-6
    ):
        raise ValueError(f"Invalid matrix: {participant_id}")

    edge_values.append(
        np.arctanh(
            np.clip(matrix[upper], -0.999999, 0.999999)
        )
    )

edge_values = np.stack(edge_values)
results = []

for edge_number, (i, j) in enumerate(zip(*upper)):
    outcome = edge_values[:, edge_number]

    if outcome.std() < 1e-10:
        raise ValueError(f"Constant outcome for connection {i}, {j}")

    fit = sm.OLS(outcome, design).fit(
        cov_type="HC3",
        use_t=True,
    )

    interval = fit.conf_int(alpha=0.05).loc["adult"]

    results.append({
        "region_1": labels[i],
        "region_2": labels[j],
        "region_1_index": int(i),
        "region_2_index": int(j),
        "adult_minus_child_beta_z": float(fit.params["adult"]),
        "robust_standard_error": float(fit.bse["adult"]),
        "ci_95_lower": float(interval.iloc[0]),
        "ci_95_upper": float(interval.iloc[1]),
        "p_value": float(fit.pvalues["adult"]),
        "n_participants": int(fit.nobs),
    })

    if (edge_number + 1) % 100 == 0:
        print(f"Fitted {edge_number + 1}/{edge_values.shape[1]} connections")

results = pd.DataFrame(results)

if not np.isfinite(results["p_value"]).all():
    raise ValueError("Non-finite p-values found.")

reject, adjusted_p, _, _ = multipletests(
    results["p_value"].to_numpy(),
    alpha=0.05,
    method="fdr_bh",
)

results["p_adjusted_bh"] = adjusted_p
results["passes_bh_0.05"] = reject

results = results.sort_values("p_adjusted_bh")
results.to_csv(OUTPUT / "all_edge_results.csv", index=False)

selected = results.loc[results["passes_bh_0.05"]]
selected.to_csv(OUTPUT / "bh_selected_edges.csv", index=False)

# Plot every estimated group coefficient.
n_regions = len(labels)
effect_matrix = np.zeros((n_regions, n_regions))
selected_matrix = np.zeros_like(effect_matrix)

for _, row in results.iterrows():
    i = int(row["region_1_index"])
    j = int(row["region_2_index"])
    value = row["adult_minus_child_beta_z"]

    effect_matrix[i, j] = effect_matrix[j, i] = value

    if row["passes_bh_0.05"]:
        selected_matrix[i, j] = selected_matrix[j, i] = value

limit = max(float(np.abs(effect_matrix).max()), 0.01)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, matrix, title in [
    (axes[0], effect_matrix, "All adjusted group coefficients"),
    (axes[1], selected_matrix, "Connections passing BH-adjusted p < 0.05"),
]:
    masked = np.ma.array(
        matrix,
        mask=np.eye(n_regions, dtype=bool),
    )
    display = ax.imshow(
        masked,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    ax.set_title(title)
    ax.set_xlabel("Atlas region index")
    ax.set_ylabel("Atlas region index")
    fig.colorbar(
        display,
        ax=ax,
        shrink=0.8,
        label="Adult minus child coefficient: Fisher-z units",
    )

fig.tight_layout()
fig.savefig(OUTPUT / "adjusted_group_effects.png", dpi=180)
plt.close(fig)

settings = {
    "analysis": "Exploratory edge-wise participant-level OLS",
    "outcome": "Fisher-z transformed Pearson correlation",
    "group_coding": "Adult=1; child=0",
    "covariates": ["mean_fd_retained", "retained_volumes"],
    "covariate_scaling": scaling,
    "covariance": "HC3",
    "use_t": True,
    "multiple_testing": "Benjamini-Hochberg across all tested edges",
    "confidence_intervals": "Pointwise 95%; not multiplicity-adjusted",
    "age_adjustment": False,
    "sex_adjustment": False,
    "statsmodels_version": statsmodels.__version__,
    "limitations": [
        "Associational comparison of selected participants",
        "Residual motion and selection effects may remain",
        "Assumes independent participants",
        "Assumes linear additive covariate effects",
        "No adjustment for sex or other unmeasured differences",
    ],
}

with open(OUTPUT / "model_settings.json", "w", encoding="utf-8") as file:
    json.dump(settings, file, indent=2)

print("\nADJUSTED COMPARISON COMPLETE")
print(f"Participants: {len(people)}")
print(people["group"].value_counts().to_string())
print(f"Connections tested: {len(results)}")
print(f"Connections passing BH-adjusted p < 0.05: {len(selected)}")
print(f"Minimum adjusted p-value: {results['p_adjusted_bh'].min():.6g}")

print("\nFirst ten connections ordered by adjusted p-value:")
print(
    results[
        ["region_1", "region_2", "adult_minus_child_beta_z",
         "p_adjusted_bh"]
    ].head(10).to_string(index=False)
)

print(f"\nOutputs saved in:\n{OUTPUT}")