from pathlib import Path

import nibabel as nib
from nilearn import image, plotting

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "development_fmri"
output_folder = project_folder / "outputs" / "previews"
output_folder.mkdir(parents=True, exist_ok=True)

participants = {
    "sub-pixar123": "Adult, age 27",
    "sub-pixar001": "Child, age 4",
}

for participant_id, description in participants.items():
    matches = sorted(data_folder.rglob(f"*{participant_id}*.nii.gz"))

    if len(matches) != 1:
        print(
            f"{participant_id}: expected one scan, "
            f"found {len(matches)}. Skipping."
        )
        continue

    scan = nib.load(matches[0])

    # Compute the average image across all timepoints.
    mean_scan = image.mean_img(scan)

    viewer = plotting.view_img(
        mean_scan,
        bg_img=None,
        title=f"{participant_id}: {description} — mean fMRI",
        cmap="gray",
        symmetric_cmap=False,
        threshold=None,
    )

    html_file = output_folder / f"{participant_id}_preview.html"
    viewer.save_as_html(str(html_file))

    plotting.plot_epi(
        mean_scan,
        title=f"{participant_id}: mean fMRI",
        display_mode="ortho",
        output_file=str(output_folder / f"{participant_id}_preview.png"),
    )

    print(f"\n{participant_id}")
    print(f"Dimensions: {scan.shape}")
    print(f"Viewer: {html_file}")

print("\nPreview generation complete.")