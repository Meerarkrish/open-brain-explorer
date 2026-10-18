import truststore
truststore.inject_into_ssl()

from pathlib import Path
import json

import nibabel as nib
import nilearn
import numpy as np
import pandas as pd

from nilearn import datasets, plotting
from nilearn.maskers import NiftiMapsMasker


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "development_fmri"
OUTPUT = ROOT / "outputs" / "all_participants_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

FD_THRESHOLD = 0.5
MIN_VOLUMES = 120
DISPLAY_THRESHOLD = 0.4

participants = pd.read_csv(
    ROOT / "outputs" / "motion_qc" / "participants_motion.csv"
)

if participants["participant_id"].duplicated().any():
    raise ValueError("Duplicate participant IDs.")

atlas = datasets.fetch_atlas_msdl(
    data_dir=str(ROOT / "data" / "atlases")
)

labels = [
    x.decode() if isinstance(x, bytes) else str(x)
    for x in atlas.labels
]
coordinates = np.asarray(atlas.region_coords, dtype=float)
scan_files = sorted(DATA.rglob("*.nii.gz"))

base_columns = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "csf", "white_matter",
]

settings = {
    "purpose": "Exploratory movie-watching connectivity",
    "atlas": "MSDL",
    "fd_threshold_mm": FD_THRESHOLD,
    "minimum_retained_volumes": MIN_VOLUMES,
    "missing_fd": "Censor",
    "non_steady_state_volumes": "Censor when flags are supplied",
    "neighbouring_volume_censoring": False,
    "nuisance_regressors": base_columns + ["supplied cosine terms"],
    "global_signal_regression": False,
    "detrend": True,
    "additional_temporal_filter": None,
    "additional_spatial_smoothing": None,
    "connectivity": "Pearson correlation",
    "display_absolute_r_threshold": DISPLAY_THRESHOLD,
    "html_colour_limits": "Automatic per participant",
    "nilearn_version": nilearn.__version__,
    "numpy_version": np.__version__,
}

with open(OUTPUT / "settings.json", "w", encoding="utf-8") as file:
    json.dump(settings, file, indent=2)

records = []

for number, (_, person) in enumerate(participants.iterrows(), start=1):
    participant_id = str(person["participant_id"])
    print(f"\n[{number}/{len(participants)}] {participant_id}")

    record = {
        "participant_id": participant_id,
        "group": person["group"],
        "age": person["Age"],
        "mean_fd_before_censoring": person["mean_fd_mm"],
    }

    try:
        matches = [
            path for path in scan_files
            if path.name.startswith(participant_id + "_")
            or participant_id in path.parts
        ]

        if len(matches) != 1:
            raise ValueError(f"Expected one scan, found {len(matches)}")

        scan = nib.load(matches[0])

        path = Path(person["confound_file"])
        separator = "," if path.suffix.lower() == ".csv" else "\t"
        confounds = pd.read_csv(
            path,
            sep=separator,
            na_values=["n/a", "NA"],
        )

        if scan.ndim != 4 or scan.shape[3] != len(confounds):
            raise ValueError("Scan/confound dimensions do not match")

        n_volumes = scan.shape[3]

        fd = pd.to_numeric(
            confounds["framewise_displacement"],
            errors="coerce",
        ).to_numpy()

        if (fd[np.isfinite(fd)] < 0).any():
            raise ValueError("Negative FD values found")

        keep = np.isfinite(fd) & (fd <= FD_THRESHOLD)

        # Respect non-steady-state flags if present.
        flag_columns = [
            c for c in confounds.columns
            if c.lower().startswith("non_steady_state_outlier")
        ]

        if flag_columns:
            flags = confounds[flag_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy()

            if not np.isfinite(flags).all():
                raise ValueError("Missing non-steady-state flags")

            keep &= ~np.any(flags != 0, axis=1)

        retained = np.flatnonzero(keep)

        record.update({
            "original_volumes": n_volumes,
            "retained_volumes": len(retained),
            "censored_volumes": n_volumes - len(retained),
            "percent_retained": 100 * len(retained) / n_volumes,
            "mean_fd_retained": (
                float(fd[keep].mean()) if keep.any() else None
            ),
            "scan_file": str(matches[0]),
            "confound_file": str(path),
        })

        if len(retained) < MIN_VOLUMES:
            record["status"] = "Excluded: insufficient retained volumes"
            print(record["status"], f"({len(retained)} retained)")

        else:
            cosine_columns = sorted(
                c for c in confounds.columns
                if c.startswith("cosine")
            )
            cleaning_columns = base_columns + cosine_columns

            missing = set(cleaning_columns) - set(confounds.columns)
            if missing:
                raise ValueError(f"Missing columns: {sorted(missing)}")

            nuisance = confounds[cleaning_columns].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy()

            if not np.isfinite(nuisance).all():
                raise ValueError("Missing/infinite nuisance values")

            # Check the retained nuisance design, including an intercept
            # and the linear trend used by detrending.
            design = np.column_stack([
                np.ones(n_volumes),
                np.linspace(-1, 1, n_volumes),
                nuisance,
            ])
            design_rank = np.linalg.matrix_rank(design[retained])
            record["retained_design_rank"] = int(design_rank)
            record["residual_dimension"] = int(
                len(retained) - design_rank
            )

            if len(retained) - design_rank < 30:
                raise ValueError("Too little residual dimension")

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
                confounds=nuisance,
                sample_mask=retained,
            )

            if signals.shape != (len(retained), len(labels)):
                raise ValueError(
                    f"Unexpected extracted shape: {signals.shape}"
                )

            if (
                not np.isfinite(signals).all()
                or (signals.std(axis=0) < 1e-8).any()
            ):
                raise ValueError("Invalid or constant regional signals")

            correlation = np.corrcoef(signals, rowvar=False)

            if not np.isfinite(correlation).all():
                raise ValueError("Invalid correlation matrix")

            # Preserve original volume indices after censoring.
            signal_table = pd.DataFrame(signals, columns=labels)
            signal_table.insert(0, "original_volume_index", retained)
            signal_table.to_csv(
                OUTPUT / f"{participant_id}_signals.csv",
                index=False,
            )

            pd.DataFrame(
                correlation,
                index=labels,
                columns=labels,
            ).to_csv(
                OUTPUT / f"{participant_id}_correlations.csv",
                index_label="region",
            )

            network = correlation.copy()
            np.fill_diagonal(network, 0)

            viewer = plotting.view_connectome(
                network,
                coordinates,
                node_labels=labels,
                node_size=5,
                edge_threshold=DISPLAY_THRESHOLD,
                symmetric_cmap=True,
                title=f"{participant_id} — motion-censored connectivity",
            )
            viewer.save_as_html(
                str(OUTPUT / f"{participant_id}_3d_network.html")
            )

            record["cleaning_columns"] = ", ".join(cleaning_columns)
            record["status"] = "Processed"
            print(f"Processed: {len(retained)}/{n_volumes} volumes")

    except Exception as error:
        record["status"] = "Error"
        record["error"] = str(error)
        print(f"ERROR: {error}")

    records.append(record)

    # Save progress after each participant.
    pd.DataFrame(records).to_csv(
        OUTPUT / "processing_summary.csv",
        index=False,
    )

summary = pd.DataFrame(records)

print("\nFINAL STATUS:")
print(summary["status"].value_counts().to_string())

print("\nSTATUS BY GROUP:")
print(
    pd.crosstab(summary["group"], summary["status"]).to_string()
)

print(f"\nOutputs saved in:\n{OUTPUT}")