# Standard library
import argparse
import glob
import os

# Third-party
import numpy as np

# First-party
from neural_lam import constants


def compute_wind_stress(u_a, v_a, temp_a, temp_s, rho_a=1.2):
    """
    Compute wind stress components using the
    Hellerman–Rosenstein (1983) formulation.

    Parameters:
        u_a: 10-meter zonal wind component (m/s)
        v_a: 10-meter meridional wind component (m/s)
        temp_a: 2-meter air temperature (K)
        temp_s: Sea surface temperature (K) from simulation (thetao_1)
        rho_a: Air density (kg/m^3), default 1.2.

    Returns:
        tau_u, tau_v: Wind stress components (N/m^2)
    """
    m = np.sqrt(u_a**2 + v_a**2)
    delta_temp = temp_a - temp_s

    # Hellerman–Rosenstein drag coefficient
    cd = (
        0.934e-3
        + 0.788e-4 * m
        + 0.868e-4 * delta_temp
        - 0.616e-6 * m**2
        - 0.120e-5 * delta_temp**2
        - 0.214e-5 * m * delta_temp
    )

    tau_u = rho_a * cd * u_a * m
    tau_v = rho_a * cd * v_a * m

    return tau_u, tau_v


def process_forcing(sim_file, forcing_file):
    """
    Process a single forcing file by replacing u10 and v10
    with computed wind stress, then save the resulting file
    under a new name with a "stress_" prefix.

    Parameters:
        sim_file (str): Path to the simulation file for SST.
        forcing_file (str): Path to the corresponding forcing file.
    """
    # Load simulation data: assumed shape (N_t, N_grid, d_features)
    sim_data = np.load(sim_file)
    # Get the index for sea surface temperature
    sst_idx = constants.EXP_PARAM_NAMES_SHORT.index("thetao_1")
    # Extract SST (N_t, N_grid)
    sst = sim_data[:, :, sst_idx]

    # Load forcing data (N_t, N_grid, d_atm) where d_atm=4: [u10, v10, t2m, msl]
    forcing_data = np.load(forcing_file)

    # Extract wind and temperature variables
    u10 = forcing_data[..., 0]
    v10 = forcing_data[..., 1]
    t2m = forcing_data[..., 2]

    # Compute wind stress (tau_x, tau_y)
    tau_u, tau_v = compute_wind_stress(u10, v10, t2m, sst)

    # Replace u10 and v10 in the forcing file with wind stress components
    forcing_data[..., 0] = tau_u
    forcing_data[..., 1] = tau_v

    # Create new filename (here replace current file)
    directory = os.path.dirname(forcing_file)
    basename = os.path.basename(forcing_file)
    new_filename = os.path.join(directory, basename)

    # Save the updated forcing file under the new filename
    np.save(new_filename, forcing_data)
    print(f"Processed and saved forcing file as: {new_filename}")


def main():
    """
    Convert 10m wind components to wind stress in the atmospheric forcing.
    """
    parser = argparse.ArgumentParser(
        description="Arguments for wind stress computation."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
        help="Dataset name (default: mediterranean)",
    )
    args = parser.parse_args()

    dataset = args.dataset
    base_dir = os.path.join("data", dataset, "samples")
    splits = ["train", "val", "test"]

    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist, skipping.")
            continue

        if split == "test":
            sim_pattern = "for_data_*.npy"
        else:
            sim_pattern = "*_data_*.npy"  # Match rea_data or ana_data

        sim_files = glob.glob(os.path.join(split_dir, sim_pattern))

        for sim_file in sim_files:
            base_name = os.path.basename(sim_file)
            date_str = base_name.split("_")[-1].split(".")[0]

            if split != "test":
                forcing_file = os.path.join(
                    split_dir, f"forcing_{date_str}.npy"
                )
                if os.path.exists(forcing_file):
                    process_forcing(sim_file, forcing_file)
                else:
                    print(
                        f"Forcing file {forcing_file} not found in {split_dir}."
                    )
            else:
                # For test split, process both ENS and AIFS forcings
                for prefix in ["ens_forcing", "aifs_forcing"]:
                    forcing_file = os.path.join(
                        split_dir, f"{prefix}_{date_str}.npy"
                    )
                    process_forcing(sim_file, forcing_file)


if __name__ == "__main__":
    main()
