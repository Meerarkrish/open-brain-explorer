from pathlib import Path

import nibabel as nib
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data" / "development_fmri"
output_folder = project_folder / "outputs"
output_folder.mkdir(exist_ok=True)

scan_files = sorted(data_folder.rglob("*.nii.gz"))

print(f"Found {len(scan_files)} compressed NIfTI files.")

records = []

for scan_path in scan_files:
    try:
        scan = nib.load(scan_path)

        records.append({
            "file": str(scan_path),
            "dimensions": str(scan.shape),
            "voxel_size_mm": str(scan.header.get_zooms()[:3]),
            "timepoints": scan.shape[3] if scan.ndim == 4 else None,
            "status": "Header readable",
            "error": "",
        })

    except Exception as error:
        records.append({
            "file": str(scan_path),
            "status": "ERROR",
            "error": str(error),
        })

report = pd.DataFrame(
    records,
    columns=[
        "file", "dimensions", "voxel_size_mm",
        "timepoints", "status", "error",
    ],
)

report_path = output_folder / "scan_inventory.csv"
report.to_csv(report_path, index=False)

print("\nResults:")
print(report["status"].value_counts())
print(f"\nReport saved: {report_path}")