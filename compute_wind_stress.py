#!/usr/bin/env python
import argparse
import glob
import os
import numpy as np
import datetime as dt

# First-party import; adjust the import path as needed.
from neural_lam import constants

def compute_wind_stress(ua, va, Ta, Ts, rho=1.2):
    """
    Compute wind stress components using the Hellerman–Rosenstein (1983) formulation.
    
    Parameters:
        u10: 10-meter zonal wind component (m/s)
        v10: 10-meter meridional wind component (m/s)
        t2m: 2-meter air temperature (K)
        Ts: Sea surface temperature (K) from simulation (thetao_1)
        rho: Air density (kg/m^3), default 1.2.
        
    Returns:
        tau_x, tau_y (ndarray, ndarray): Wind stress components (N/m^2)
    """
    M = np.sqrt(ua**2 + va**2)
    deltaT = Ta - Ts

    # Hellerman–Rosenstein drag coefficient:
    CD = (0.934e-3 +
          0.788e-4 * M +
          0.868e-4 * deltaT -
          0.616e-6 * M**2 -
          0.120e-5 * deltaT**2 -
          0.214e-5 * M * deltaT)
    
    tau_x = rho * CD * ua * M
    tau_y = rho * CD * va * M

    return tau_x, tau_y

def process_forcing(sim_file, forcing_file):
    """
    Process a single forcing file by replacing u10 and v10 with computed wind stress,
    then save the resulting file under a new name with a "stress_" prefix.
    
    Parameters:
        sim_file (str): Path to the simulation file (rea_data, ana_data, or for_data) for SST.
        forcing_file (str): Path to the corresponding forcing file.
    """
    # Load simulation data: assumed shape (N_t, N_grid, d_features)
    sim_data = np.load(sim_file)
    # Get the index for sea surface temperature ("thetao_1")
    sst_idx = constants.EXP_PARAM_NAMES_SHORT.index("thetao_1")
    # Assume simulation's first time index aligns with forcing; extract Ts (shape: (N_t, N_grid))
    Ts = sim_data[:, :, sst_idx]
    
    # Load forcing data: assumed shape (N_t, N_grid, d_atm) where d_atm=4: [u10, v10, t2m, msl]
    forcing_data = np.load(forcing_file)
    
    # Extract wind and temperature variables
    u10 = forcing_data[..., 0]
    v10 = forcing_data[..., 1]
    t2m = forcing_data[..., 2]
    
    # Compute wind stress (tau_x, tau_y)
    tau_x, tau_y = compute_wind_stress(u10, v10, t2m, Ts)
    
    # Replace u10 and v10 in the forcing file with computed wind stress components
    forcing_data[..., 0] = tau_x
    forcing_data[..., 1] = tau_y
    
    # Create new filename with "stress_" prefix
    directory = os.path.dirname(forcing_file)
    basename = os.path.basename(forcing_file)
    new_filename = os.path.join(directory, "stress_" + basename)
    
    # Save the updated forcing file under the new filename
    np.save(new_filename, forcing_data)
    print(f"Processed and saved forcing file as: {new_filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert 10-m wind components to wind stress in atmospheric forcing files."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
        help="Dataset name (default: mediterranean)"
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
        
        # For test split, simulation files are the forecast files: "for_data_*.npy"
        if split == "test":
            sim_pattern = "for_data_*.npy"
        else:
            sim_pattern = "*_data_*.npy"  # This will match rea_data or ana_data
        
        sim_files = glob.glob(os.path.join(split_dir, sim_pattern))
        
        for sim_file in sim_files:
            base_name = os.path.basename(sim_file)
            # Assume the file is named like "rea_data_YYYYMMDD.npy", "ana_data_YYYYMMDD.npy", or "for_data_YYYYMMDD.npy"
            try:
                date_str = base_name.split("_")[-1].split(".")[0]
            except Exception as e:
                print(f"Could not extract date from {base_name}, skipping.")
                continue

            if split != "test":
                forcing_file = os.path.join(split_dir, f"forcing_{date_str}.npy")
                if os.path.exists(forcing_file):
                    process_forcing(sim_file, forcing_file)
                else:
                    print(f"Forcing file {forcing_file} not found in {split_dir}.")
            else:
                # For test split, process both "ens_forcing_*.npy" and "aifs_forcing_*.npy"
                for prefix in ["ens_forcing", "aifs_forcing"]:
                    forcing_file = os.path.join(split_dir, f"{prefix}_{date_str}.npy")
                    if os.path.exists(forcing_file):
                        process_forcing(sim_file, forcing_file)
                    else:
                        print(f"Forcing file {forcing_file} not found in test directory.")

if __name__ == "__main__":
    main()
