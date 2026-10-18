import truststore
truststore.inject_into_ssl()

from pathlib import Path
from html import escape
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nilearn import datasets


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "outputs" / "all_participants_v1"
OUTPUT = SOURCE / "results_explorer"
OUTPUT.mkdir(parents=True, exist_ok=True)

results = pd.read_csv(
    SOURCE / "adjusted_group_comparison" / "all_edge_results.csv"
)

participants = pd.read_csv(SOURCE / "processing_summary.csv")
included = participants.loc[participants["status"] == "Processed"]

n_adults = int(included["group"].eq("adult").sum())
n_children = int(included["group"].eq("child").sum())

atlas = datasets.fetch_atlas_msdl(
    data_dir=str(ROOT / "data" / "atlases")
)

labels = [
    value.decode() if isinstance(value, bytes) else str(value)
    for value in atlas.labels
]
coordinates = np.asarray(atlas.region_coords, dtype=float)

required = [
    "region_1", "region_2",
    "region_1_index", "region_2_index",
    "adult_minus_child_beta_z",
    "ci_95_lower", "ci_95_upper",
    "p_adjusted_bh",
]

if not set(required).issubset(results.columns):
    raise ValueError("The results file is missing required columns.")

numeric = results[
    ["adult_minus_child_beta_z", "ci_95_lower",
     "ci_95_upper", "p_adjusted_bh"]
].to_numpy(dtype=float)

if not np.isfinite(numeric).all():
    raise ValueError("Non-finite result values found.")

fig = go.Figure()
edge_metadata = []

# One trace per connection enables independent filtering.
for _, row in results.iterrows():
    i = int(row["region_1_index"])
    j = int(row["region_2_index"])

    if labels[i] != row["region_1"] or labels[j] != row["region_2"]:
        raise ValueError("Atlas labels do not match the results.")

    beta = float(row["adult_minus_child_beta_z"])
    adjusted_p = float(row["p_adjusted_bh"])

    # Include a midpoint to make hovering on edges easier.
    points = np.vstack([
        coordinates[i],
        (coordinates[i] + coordinates[j]) / 2,
        coordinates[j],
    ])

    detail = (
        f"<b>{escape(labels[i])} ↔ {escape(labels[j])}</b>"
        f"<br>Adult–child coefficient: {beta:.3f} Fisher-z"
        f"<br>Pointwise 95% CI: "
        f"[{row['ci_95_lower']:.3f}, {row['ci_95_upper']:.3f}]"
        f"<br>BH-adjusted p: {adjusted_p:.3g}"
    )

    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="lines",
        line={
            "color": "#dc654b" if beta > 0 else "#438ac5",
            "width": 2 + min(abs(beta) * 12, 5),
        },
        text=[detail] * 3,
        hovertemplate="%{text}<extra></extra>",
        visible=adjusted_p < 0.05,
        showlegend=False,
    ))

    edge_metadata.append({
        "beta": beta,
        "p": adjusted_p,
    })

# Region positions are MNI coordinates, not a reconstructed brain surface.
fig.add_trace(go.Scatter3d(
    x=coordinates[:, 0],
    y=coordinates[:, 1],
    z=coordinates[:, 2],
    mode="markers",
    marker={
        "size": 5,
        "color": "#d9e8f5",
        "line": {"color": "#20344d", "width": 1},
    },
    text=labels,
    hovertemplate="<b>%{text}</b><extra></extra>",
    showlegend=False,
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="#101a2b",
    scene={
        "xaxis": {"title": "MNI x (mm)"},
        "yaxis": {"title": "MNI y (mm)"},
        "zaxis": {"title": "MNI z (mm)"},
        "aspectmode": "data",
    },
    margin={"l": 0, "r": 0, "t": 10, "b": 0},
    height=720,
    uirevision="preserve-camera",
)

# Controls are placed before the plot so they exist when this runs.
post_script = """
const plot = document.getElementById('{plot_id}');
const edges = __EDGE_DATA__;

function applyFilters() {
    const pLimit = Number(document.getElementById('pFilter').value);
    const minimum = Number(document.getElementById('effectFilter').value);
    const direction = document.getElementById('directionFilter').value;

    let count = 0;
    const visible = edges.map(edge => {
        const matches = edge.p < pLimit
            && Math.abs(edge.beta) >= minimum
            && (direction === 'both'
                || (direction === 'positive' && edge.beta > 0)
                || (direction === 'negative' && edge.beta < 0));
        if (matches) count++;
        return matches;
    });

    visible.push(true); // Keep region markers visible.

    Plotly.restyle(plot, {visible: visible});
    document.getElementById('connectionCount').textContent =
        count + ' of ' + edges.length + ' connections displayed';
}

['pFilter', 'effectFilter', 'directionFilter'].forEach(id => {
    document.getElementById(id).addEventListener('change', applyFilters);
});
applyFilters();
""".replace("__EDGE_DATA__", json.dumps(edge_metadata))

plot_html = fig.to_html(
    full_html=False,
    include_plotlyjs=True,
    post_script=post_script,
    config={"responsive": True, "displaylogo": False},
)

header = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open Brain Explorer</title>
<style>
body {
    margin: 0;
    background: #101a2b;
    color: #e5edf7;
    font-family: Arial, sans-serif;
}
main { max-width: 1400px; margin: auto; padding: 28px; }
h1 { margin-bottom: 8px; }
p { line-height: 1.6; max-width: 1050px; }
.controls {
    display: flex; flex-wrap: wrap; gap: 20px;
    padding: 18px; background: #1b2940; border-radius: 12px;
}
label { display: grid; gap: 8px; }
select {
    padding: 8px; background: #101a2b; color: white;
    border: 1px solid #63748b; border-radius: 5px;
}
.note { color: #b8c6d9; font-size: 14px; }
</style>
</head>
<body>
<main>
<h1>Open Brain Explorer</h1>
<p>Exploratory adult–child differences in movie-watching connectivity.
MSDL atlas: 39 regions and 741 unique connections.</p>
"""

header += (
    f"<p><b>Analysis sample:</b> {n_children} children and "
    f"{n_adults} adults. "
    "Group coefficients are adjusted for retained mean framewise "
    "displacement and retained-volume count.</p>"
)

header += """
<div class="controls">
<label>BH-adjusted p below
<select id="pFilter">
<option value="0.01">0.01</option>
<option value="0.05" selected>0.05</option>
<option value="0.10">0.10</option>
<option value="1.000001">Show all tested connections</option>
</select>
</label>

<label>Minimum absolute coefficient
<select id="effectFilter">
<option value="0" selected>0.00</option>
<option value="0.1">0.10 Fisher-z</option>
<option value="0.2">0.20 Fisher-z</option>
<option value="0.3">0.30 Fisher-z</option>
</select>
</label>

<label>Direction
<select id="directionFilter">
<option value="both" selected>Both directions</option>
<option value="positive">Higher in adults</option>
<option value="negative">Lower in adults</option>
</select>
</label>
</div>

<p id="connectionCount"></p>
<p class="note">
Drag to rotate; scroll to zoom. Hover over a line or its midpoint
for its coefficient, confidence interval, and adjusted p-value.
Orange: higher adjusted connectivity in adults.
Blue: lower adjusted connectivity in adults.
Node positions use MNI coordinates; no anatomical surface is shown.
</p>
"""

footer = """
<p class="note">
<b>Interpretation:</b> Edges show group differences in Fisher-z
connectivity, not anatomical tracts or causal effects.
Confidence intervals are pointwise and are not multiplicity-adjusted.
BH-adjusted p-values were calculated across all 741 tests and are
not recalculated when display filters change.
</p>
<p class="note">
<b>Limitations:</b> Exploratory analysis of a selected sample.
Residual motion, unequal data retention, and other confounding may
remain. Sex was not included in the model. Cross-sectional group
differences do not demonstrate developmental change.
</p>
<p class="note">
Data source: OpenNeuro ds000228, using the preprocessed,
downsampled data distributed through Nilearn.
</p>
</main>
</body>
</html>
"""

html_file = OUTPUT / "open_brain_explorer.html"
html_file.write_text(header + plot_html + footer, encoding="utf-8")

print("\nEXPLORER CREATED")
print(f"Children: {n_children}")
print(f"Adults: {n_adults}")
print(f"Connections available: {len(results)}")
print(f"\nOpen this file in your browser:\n{html_file}")