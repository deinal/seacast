# Standard library
import argparse
import os
from datetime import datetime, timedelta
from glob import glob

# Third-party
import numpy as np


def prepare_states(
    in_directory, out_directory, n_states, prefix, start_date, end_date
):
    """
    Processes and concatenates state sequences from numpy files.

    Args:
        in_directory (str): Directory containing the .npy files.
        out_directory (str): Directory to store the concatenated files.
        n_states (int): Number of consecutive states to concatenate.
        prefix (str): Prefix for naming the output files.
        start_date (str): Start date.
        end_date (str): End date.
    """
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    print(start_dt, end_dt)

    # Get all numpy files sorted by date
    all_files = sorted(glob(os.path.join(in_directory, "*.npy")))
    files = [
        f
        for f in all_files
        if start_dt
        <= datetime.strptime(os.path.basename(f)[:8], "%Y%m%d")
        <= end_dt
    ]

    # Ensure output directory exists
    os.makedirs(out_directory, exist_ok=True)

    # Process each file, concatenate with the next t-1 files
    for i in range(len(files) - n_states + 1):
        # Name as today's date
        out_filename = f"{prefix}_{os.path.basename(files[i+1])}"
        out_file = os.path.join(out_directory, out_filename)

        if os.path.isfile(out_file):
            continue

        state_sequence = []

        # Load each state to concatenate
        for j in range(n_states):
            state = np.load(files[i + j])
            state_sequence.append(state)

        # Concatenate along new axis (time axis)
        full_state = np.stack(state_sequence, axis=0)

        # Save concatenated data to the output directory
        np.save(out_file, full_state)
        print(f"Saved states to: {out_file}")


def prepare_forcing(in_directory, out_directory, prefix, start_date, end_date):
    """
    Prepare atmospheric forcing data from forecasts.
    """
    forecast_dir = in_directory

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    os.makedirs(out_directory, exist_ok=True)

    # Get files sorted by date
    forecast_files = sorted(
        glob(os.path.join(forecast_dir, "*.npy")),
        key=lambda x: datetime.strptime(os.path.basename(x)[:8], "%Y%m%d"),
    )
    forecast_files = [
        f
        for f in forecast_files
        if start_dt
        <= datetime.strptime(os.path.basename(f)[:8], "%Y%m%d")
        <= end_dt
    ]

    for forecast_file in forecast_files:
        forecast_date = datetime.strptime(
            os.path.basename(forecast_file)[:8], "%Y%m%d"
        )
        # Get files for the preceding day
        preceding_day_file = os.path.join(
            forecast_dir,
            (forecast_date - timedelta(days=1)).strftime("%Y%m%d") + ".npy",
        )
        preceding_day_data = np.load(preceding_day_file)[0:1]

        # Load the current forecast data
        current_forecast_data = np.load(forecast_file)[:15]

        print(preceding_day_data.shape, current_forecast_data.shape)

        # Concatenate all data along the time axis
        concatenated_forcing = np.concatenate(
            [preceding_day_data, current_forecast_data], axis=0
        )

        # Save concatenated data
        out_filename = f"{prefix}_{os.path.basename(forecast_file)}"
        out_file = os.path.join(out_directory, out_filename)
        np.save(out_file, concatenated_forcing)
        print(f"Saved forcing states to: {out_file}")


def prepare_aifs_forcing(
    in_directory, out_directory, prefix, start_date, end_date
):
    """
    Prepare atmospheric forcing data from AIFS forecasts (add SSR from ENS).
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    os.makedirs(out_directory, exist_ok=True)

    # Get files sorted by date
    forecast_files = sorted(
        glob(os.path.join(in_directory, "*.npy")),
        key=lambda x: datetime.strptime(os.path.basename(x)[:8], "%Y%m%d"),
    )
    forecast_files = [
        f
        for f in forecast_files
        if start_dt
        <= datetime.strptime(os.path.basename(f)[:8], "%Y%m%d")
        <= end_dt
    ]

    ifs_variables = ["u10", "v10", "t2m", "msl", "ssr", "tp"]

    for forecast_file in forecast_files:
        forecast_date = datetime.strptime(
            os.path.basename(forecast_file)[:8], "%Y%m%d"
        )

        # Load the preceding day's data
        preceding_day_file = os.path.join(
            in_directory,
            (forecast_date - timedelta(days=1)).strftime("%Y%m%d") + ".npy",
        )
        preceding_day_data = np.load(preceding_day_file)[0:1]

        # Insert SSR from ENS data
        preceding_day_ens_data = np.load(
            preceding_day_file.replace("aifs", "ens")
        )[0:1]
        preceding_day_ssr_data = preceding_day_ens_data[
            ..., ifs_variables.index("ssr")
        ]
        preceding_day_data = np.insert(
            preceding_day_data,
            ifs_variables.index("ssr"),
            preceding_day_ssr_data,
            axis=-1,
        )

        # Load the current forecast data
        current_forecast_data = np.load(forecast_file)[:15]

        # Insert SSR from ENS data
        ens_data = np.load(forecast_file.replace("aifs", "ens"))[:15]
        ssr_data = ens_data[..., ifs_variables.index("ssr")]
        current_forecast_data = np.insert(
            current_forecast_data, ifs_variables.index("ssr"), ssr_data, axis=-1
        )

        # Concatenate preceding day data with current forecast data
        aifs_data = np.concatenate(
            [preceding_day_data, current_forecast_data], axis=0
        )

        # Save combined data
        out_filename = f"{prefix}_{os.path.basename(forecast_file)}"
        out_file = os.path.join(out_directory, out_filename)
        np.save(out_file, aifs_data)
        print(f"Saved forcing states to: {out_file}")


def main():
    """
    Main function to parse arguments and prepare state sequences.
    """
    parser = argparse.ArgumentParser(
        description="Prepare state sequences from Baltic Sea data files."
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing .npy files",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for concatenated files",
    )
    parser.add_argument(
        "-n",
        "--n_states",
        type=int,
        default=6,
        help="Number of states to concatenate",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        required="ana_data",
        help="Prefix for the output files",
    )
    parser.add_argument(
        "-s",
        "--start_date",
        type=str,
        default="1987-01-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "-e",
        "--end_date",
        type=str,
        default="2024-05-25",
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--forecast_forcing",
        action="store_true",
    )
    args = parser.parse_args()

    if args.forecast_forcing:
        if args.data_dir.endswith("aifs"):
            prepare_aifs_forcing(
                args.data_dir,
                args.out_dir,
                args.prefix,
                args.start_date,
                args.end_date,
            )
        else:
            prepare_forcing(
                args.data_dir,
                args.out_dir,
                args.prefix,
                args.start_date,
                args.end_date,
            )
    else:
        prepare_states(
            args.data_dir,
            args.out_dir,
            args.n_states,
            args.prefix,
            args.start_date,
            args.end_date,
        )


if __name__ == "__main__":
    main()
