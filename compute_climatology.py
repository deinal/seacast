# Standard library
import argparse
import glob
import os
from datetime import datetime

# Third-party
import numpy as np
from tqdm import tqdm


def compute_climatology(dataset):
    """
    Compute climatology for each day-of-year.
    """
    reanalysis_folder = os.path.join("data", dataset, "raw", "reanalysis")
    analysis_folder = os.path.join("data", dataset, "raw", "analysis")

    output_folder = os.path.join("data", dataset, "climatology")
    os.makedirs(output_folder, exist_ok=True)

    # Gather all files from both folders
    all_files = sorted(
        glob.glob(os.path.join(reanalysis_folder, "*.npy"))
        + glob.glob(os.path.join(analysis_folder, "*.npy"))
    )

    # Group files by doy
    files_by_doy = {}
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        date_str = os.path.splitext(file_name)[0]
        dt = datetime.strptime(date_str, "%Y%m%d")
        doy = dt.timetuple().tm_yday  # doy (1 to 365/366)
        files_by_doy.setdefault(doy, []).append(file_path)

    # Process one doy at a time
    for doy in tqdm(sorted(files_by_doy.keys()), desc="Processing day-of-year"):
        file_list = files_by_doy[doy]
        sum_data = None
        count = 0
        for file_path in file_list:
            data = np.load(file_path)  # (n_grid, features)
            if sum_data is None:
                sum_data = np.zeros_like(data, dtype=np.float32)
            sum_data += data
            count += 1
        # Compute doy mean
        daily_mean = sum_data / count

        # Save each day's climatology in the form doyXXX
        output_filename = f"doy{doy:03d}.npy"
        output_path = os.path.join(output_folder, output_filename)
        np.save(output_path, daily_mean)

    print(f"Daily climatology computed and saved to {output_folder}")


def main():
    """
    Compute climatology.
    """
    parser = argparse.ArgumentParser(description="Compute climatology.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
    )
    args = parser.parse_args()

    compute_climatology(args.dataset)


if __name__ == "__main__":
    main()
