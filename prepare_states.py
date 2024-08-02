# Standard library
import argparse
import os
from datetime import datetime, timedelta
from glob import glob

# Third-party
import numpy as np

# First-party
from neural_lam import constants


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


def prepare_states_with_boundary(
    in_directory,
    static_dir_path,
    out_directory,
    n_states,
    prefix,
    start_date,
    end_date,
):
    """
    Concatenat analysis states and include forecast boundary.

    Args:
        in_directory (str): Directory containing the analysis .npy files.
        static_dir_path (str): Directory containing the static files.
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

    # Load boundary mask
    boundary_mask = np.load(
        os.path.join(static_dir_path, "boundary_mask.npy")
    )  # (depths, h, w)
    sea_mask = np.load(
        os.path.join(static_dir_path, "sea_mask.npy")
    )  # (depths, h, w)

    surface_mask = sea_mask[0]
    boundary_mask = boundary_mask[:, surface_mask]

    border_mask = []
    for level_applies in constants.LEVELS:
        if level_applies:
            border_mask.append(boundary_mask)
        else:
            border_mask.append(boundary_mask[0][np.newaxis, :])

    border_mask = np.concatenate(border_mask, axis=0).transpose(1, 0)[
        np.newaxis, :, :
    ]
    print("border mask", border_mask.shape)  # 1, N_grid, d_features

    # Get all analysis files sorted by date
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
        today = os.path.basename(files[i + 1])
        out_filename = f"{prefix}_{today}"
        out_file = os.path.join(out_directory, out_filename)

        if os.path.isfile(out_file):
            continue

        # Stack analysis states
        state_sequence = [np.load(files[i + j]) for j in range(n_states)]
        full_state = np.stack(state_sequence, axis=0)
        print("full state", full_state.shape)  # (n_states, N_grid, d_features)

        forecast_file = files[i + 1].replace("analysis", "forecast")
        forecast_data = np.load(forecast_file)
        forecast_len = forecast_data.shape[0]
        print(
            "forecast before", forecast_data.shape
        )  # (forecast_len, N_grid, d_features)

        assert n_states >= forecast_len, "n_states less than forecast length"
        extra_states = n_states - 1 - forecast_data.shape[0]
        last_forecast_state = forecast_data[-1]
        repeated_forecast_states = np.repeat(
            last_forecast_state[np.newaxis, ...], extra_states, axis=0
        )
        forecast_data = np.concatenate(
            [forecast_data, repeated_forecast_states], axis=0
        )
        print(
            "forecast after", forecast_data.shape
        )  # (n_states - 1, N_grid, d_features)

        # Concatenate preceding day analysis state with forecast data
        forecast_data = np.concatenate(
            (state_sequence[:1], forecast_data), axis=0
        )  # (n_states, N_grid, d_features)

        full_state = (
            full_state * (1 - border_mask) + forecast_data * border_mask
        )

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

        preceding_day_data = preceding_day_data[:, :, :4]
        current_forecast_data = current_forecast_data[:, :, :4]

        # Concatenate all data along the time axis
        concatenated_forcing = np.concatenate(
            [preceding_day_data, current_forecast_data], axis=0
        )

        # Save concatenated data
        out_filename = f"{prefix}_{os.path.basename(forecast_file)}"
        out_file = os.path.join(out_directory, out_filename)
        np.save(out_file, concatenated_forcing)
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
        "-m",
        "--static_dir",
        type=str,
        default="data/mediterranean/static",
        help="Directory containing static files",
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
        "--forecast",
        action="store_true",
    )
    args = parser.parse_args()

    if args.forecast:
        if args.data_dir.endswith("aifs") or args.data_dir.endswith("ens"):
            prepare_forcing(
                args.data_dir,
                args.out_dir,
                args.prefix,
                args.start_date,
                args.end_date,
            )
        else:
            prepare_states_with_boundary(
                args.data_dir,
                args.static_dir,
                args.out_dir,
                args.n_states,
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
