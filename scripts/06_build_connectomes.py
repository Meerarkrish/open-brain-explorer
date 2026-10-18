import truststore
truststore.inject_into_ssl()

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "development_fmri"
OUTPUT = ROOT / "outputs" / "connectomes"
OUTPUT.mkdir(parents=True, exist_ok=True)

EXAMPLES_FILE = (
    ROOT / "outputs" / "motion_qc" / "visualization_examples.csv"
)

if not EXAMPLES_FILE.exists():
    raise FileNotFoundError(
        f"Missing {EXAMPLES_FILE}. Run 05_review_groups.py first."
    )

examples = pd.read_csv(EXAMPLES_FILE)

# Download the atlas once; later runs reuse the local copy.
atlas = datasets.fetch_atlas_msdl(
    data_dir=str(ROOT / "data" / "atlases")
)

labels = [
    label.decode() if isinstance(label, bytes) else str(label)
    for label in atlas.labels
]
coordinates = np.asarray(atlas.region_coords, dtype=float)

if coordinates.shape != (len(labels), 3):
    raise ValueError("Atlas coordinates and labels do not match.")

# Prototype nuisance regression:
# six motion parameters, CSF, white matter, and supplied cosine terms.
BASE_CONFOUNDS = [
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "csf",
    "white_matter",
]

DISPLAY_THRESHOLD = 0.4
scan_files = sorted(DATA.rglob("*.nii.gz"))

for _, participant in examples.iterrows():
    participant_id = str(participant["participant_id"])

    print(f"\nProcessing {participant_id}...")

    matching_scans = [
        path
        for path in scan_files
        if path.name.startswith(participant_id + "_")
        or participant_id in path.parts
    ]

    if len(matching_scans) != 1:
        raise ValueError(
            f"{participant_id}: expected one scan, "
            f"found {len(matching_scans)}."
        )

    scan_path = matching_scans[0]
    scan = nib.load(scan_path)

    if scan.ndim != 4:
        raise ValueError(
            f"{participant_id}: expected a 4D functional scan, "
            f"found shape {scan.shape}."
        )

    confound_path = Path(participant["confound_file"])

    if not confound_path.exists():
        raise FileNotFoundError(confound_path)

    separator = "," if confound_path.suffix.lower() == ".csv" else "\t"

    confounds = pd.read_csv(
        confound_path,
        sep=separator,
        na_values=["n/a", "NA"],
    )

    if len(confounds) != scan.shape[3]:
        raise ValueError(
            f"{participant_id}: {scan.shape[3]} scan volumes "
            f"but {len(confounds)} confound rows."
        )

    missing_columns = set(BASE_CONFOUNDS) - set(confounds.columns)

    if missing_columns:
        raise ValueError(
            f"{participant_id}: missing cleaning columns: "
            f"{sorted(missing_columns)}"
        )

    cosine_columns = sorted(
        column
        for column in confounds.columns
        if column.startswith("cosine")
    )

    cleaning_columns = BASE_CONFOUNDS + cosine_columns

    nuisance = confounds[cleaning_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Stop rather than silently filling missing cleaning values.
    if not np.isfinite(nuisance.to_numpy()).all():
        problematic = [
            column
            for column in nuisance.columns
            if not np.isfinite(nuisance[column].to_numpy()).all()
        ]
        raise ValueError(
            f"{participant_id}: missing or infinite values in "
            f"{problematic}"
        )

    print(f"Scan shape: {scan.shape}")
    print(f"Cleaning variables: {', '.join(cleaning_columns)}")

    masker = NiftiMapsMasker(
        maps_img=atlas.maps,
        resampling_target="data",
        standardize="zscore_sample",
        standardize_confounds=True,
        detrend=True,
        smoothing_fwhm=None,
        high_pass=None,
        low_pass=None,
        reports=False,
    )

    signals = masker.fit_transform(
        scan,
        confounds=nuisance.to_numpy(),
    )

    if signals.shape != (scan.shape[3], len(labels)):
        raise ValueError(
            f"{participant_id}: unexpected signal shape "
            f"{signals.shape}."
        )

    if not np.isfinite(signals).all():
        raise ValueError(
            f"{participant_id}: non-finite regional signals."
        )

    if (signals.std(axis=0) < 1e-8).any():
        raise ValueError(
            f"{participant_id}: one or more regional signals "
            "are constant."
        )

    # Pearson correlation between every pair of regional signals.
    correlation = np.corrcoef(signals, rowvar=False)

    if not np.isfinite(correlation).all():
        raise ValueError(
            f"{participant_id}: invalid correlation matrix."
        )

    # Save cleaned regional time series.
    pd.DataFrame(
        signals,
        columns=labels,
    ).to_csv(
        OUTPUT / f"{participant_id}_regional_signals.csv",
        index=False,
    )

    # Save full correlations, including diagonal self-correlations.
    pd.DataFrame(
        correlation,
        index=labels,
        columns=labels,
    ).to_csv(
        OUTPUT / f"{participant_id}_correlations.csv",
        index_label="region",
    )

    # Static matrix with a fixed scale for both participants.
    fig, ax = plt.subplots(figsize=(11, 9))

    matrix_image = ax.imshow(
        correlation,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=90,
        fontsize=6,
    )
    ax.set_yticks(
        range(len(labels)),
        labels=labels,
        fontsize=6,
    )
    ax.set_title(
        f"{participant_id} — regional Pearson correlations"
    )

    fig.colorbar(
        matrix_image,
        ax=ax,
        label="Pearson r",
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT / f"{participant_id}_matrix.png",
        dpi=180,
    )
    plt.close(fig)

    # Hide diagonal self-connections for the interactive network.
    network = correlation.copy()
    np.fill_diagonal(network, 0)

    # view_connectome does not accept vmin/vmax or edge_vmin/edge_vmax.
    viewer = plotting.view_connectome(
        network,
        coordinates,
        node_labels=labels,
        node_size=5,
        edge_threshold=DISPLAY_THRESHOLD,
        edge_cmap="RdBu_r",
        symmetric_cmap=True,
        title=f"{participant_id} — exploratory connectivity",
    )

    html_file = OUTPUT / f"{participant_id}_3d_network.html"
    viewer.save_as_html(str(html_file))

    # Record the analysis settings for reproducibility.
    metadata = {
        "participant_id": participant_id,
        "scan_file": str(scan_path),
        "confound_file": str(confound_path),
        "atlas": "MSDL",
        "timepoints": int(signals.shape[0]),
        "regions": int(signals.shape[1]),
        "connectivity_measure": "Pearson correlation",
        "regressed_columns": cleaning_columns,
        "detrend": True,
        "standardization": "zscore_sample",
        "additional_spatial_smoothing": False,
        "additional_temporal_filtering": False,
        "supplied_cosine_regressors_used": cosine_columns,
        "volume_censoring": False,
        "global_signal_regression": False,
        "display_absolute_r_threshold": DISPLAY_THRESHOLD,
        "html_colour_limits": "Automatically chosen per participant",
        "matrix_png_colour_limits": [-1, 1],
        "purpose": "Exploratory individual visualization",
        "versions": {
            "nilearn": __import__("nilearn").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "nibabel": nib.__version__,
        },
    }

    with open(
        OUTPUT / f"{participant_id}_methods.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"Extracted signals: {signals.shape}")
    print(f"Saved interactive network: {html_file}")

print(f"\nFinished! Results are in:\n{OUTPUT}")