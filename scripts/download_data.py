import truststore
truststore.inject_into_ssl()

from pathlib import Path
import time

from nilearn.datasets import fetch_development_fmri
from requests.exceptions import ConnectionError, HTTPError, Timeout

project_folder = Path(__file__).resolve().parent.parent
data_folder = project_folder / "data"

max_attempts = 6

for attempt in range(1, max_attempts + 1):
    try:
        print(f"\nDownload attempt {attempt}/{max_attempts}")

        dataset = fetch_development_fmri(
            n_subjects=None,
            age_group="both",
            data_dir=str(data_folder),
            reduce_confounds=False,
            resume=True,
        )
        break

    except HTTPError as error:
        status = (
            error.response.status_code
            if error.response is not None
            else None
        )

        # Retry temporary server errors only.
        if status not in {500, 502, 503, 504}:
            raise

        if attempt == max_attempts:
            raise

        print(f"Server returned HTTP {status}. Retrying in 30 seconds.")
        time.sleep(30)

    except (ConnectionError, Timeout) as error:
        if attempt == max_attempts:
            raise

        print(f"Connection interrupted: {error}")
        print("Retrying in 30 seconds.")
        time.sleep(30)

participants_file = project_folder / "participants.csv"
dataset.phenotypic.to_csv(participants_file, index=False)

print("\nDownload complete!")
print(f"Participants: {len(dataset.phenotypic)}")
print(f"Brain scan files: {len(dataset.func)}")
print(f"Confound files: {len(dataset.confounds)}")
print(f"\nData saved in: {data_folder}")