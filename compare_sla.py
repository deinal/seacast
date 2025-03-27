# Standard library
import argparse
import json
import os
from datetime import datetime, timedelta
from glob import glob

# Third-party
import numpy as np
import xarray as xr

# First-party
from neural_lam import constants


def read_npy_to_xarray_sla(file_path, sea_mask, init=False):
    """
    Reads a forecast file and reconstructs a full grid for the model zos.
    """
    # Get start date from filename
    filename = os.path.basename(file_path)
    date_str = filename.split(".")[0]
    start_date = datetime.strptime(date_str[-8:], "%Y%m%d")

    # Load forecast data (time, n_grid, n_features)
    if init:
        data = np.load(file_path)[2:]
    else:
        data = np.load(file_path)
    n_time, _, _ = data.shape

    # Full horizontal grid dimensions
    n_lat, n_lon = constants.GRID_SHAPE

    # Get indices from the sea mask
    surface_mask = sea_mask.isel(depth=0).values
    lat_idx, lon_idx = np.where(surface_mask == 1)

    # Feature index for zos (surface height above geoid)
    feature_idx = constants.EXP_PARAM_NAMES_SHORT.index("zos")

    # Extract zos forecast (n_time, n_grid)
    var_forecast = data[:, :, feature_idx]

    # Create full grid array and assign values where sea
    var_array = np.full((n_time, n_lat, n_lon), np.nan, dtype=data.dtype)
    var_array[:, lat_idx, lon_idx] = var_forecast

    data_vars = {"zos": (("time", "latitude", "longitude"), var_array)}

    # Create coordinates
    time_coords = [start_date + timedelta(days=i) for i in range(n_time)]
    coords = {
        "time": time_coords,
        "latitude": sea_mask.latitude,
        "longitude": sea_mask.longitude,
    }
    ds = xr.Dataset(data_vars, coords=coords)
    return ds


def compute_rmse_forecast_vs_obs_over_samples_sla(
    forecast_files,
    sla_obs_file,
    mdt_mod_file,
    sea_mask,
    init_flag=False,
    max_lead=None,
):
    """
    Computes RMSE between model SLA and observed SLA.
    """
    # Load observed SLA
    ds_obs = xr.open_dataset(sla_obs_file)
    sla_obs = ds_obs["sla"]

    # Load model MDT and align
    ds_mdt = xr.open_dataset(mdt_mod_file)
    ds_mdt = ds_mdt.where(sea_mask.isel(depth=0), drop=True)
    mdt = ds_mdt["mdt"]

    lead_sq_errors_list = []  # (n_lead,) per sample
    forecast_times = None

    for forecast_file in forecast_files:
        print(f"Processing: {forecast_file}", flush=True)
        # Read forecast sample and compute model SLA as (zos - mdt)
        ds_forecast = read_npy_to_xarray_sla(
            forecast_file, sea_mask, init=init_flag
        )
        sla_mod = ds_forecast["zos"] - mdt

        if max_lead is not None:
            sla_mod = sla_mod.isel(time=slice(0, max_lead))

        # Use the same forecast times for all files
        if forecast_times is None:
            forecast_times = sla_mod.time.values
        n_lead = sla_mod.time.size

        # Array to accumulate SE for each lead time in this sample
        lead_sq = np.zeros(n_lead)

        for i, t in enumerate(sla_mod.time.values):
            # Get observation SLA for time t
            o_t = sla_obs.sel(time=t)

            # Get model SLA for time t and interpolate onto the obs points
            f_t = sla_mod.sel(time=t)
            f_t_interp = f_t.interp(
                latitude=o_t.latitude, longitude=o_t.longitude
            )

            # Create a mask where the interpolation produced valid values
            valid_mask = ~np.isnan(f_t_interp)

            # Compute daily mean for obs only where the interpolation was valid
            o_mean = o_t.where(valid_mask).mean(dim="obs", skipna=True)
            o_anom = o_t - o_mean

            # Compute daily mean for the model only over valid locations
            f_mean = f_t_interp.where(valid_mask).mean(dim="obs", skipna=True)
            f_anom = f_t_interp - f_mean

            # Compute SE field of anomaly differences
            err_sq = (f_anom - o_anom) ** 2

            # Compute spatial mean SE over the observation dimension
            mse = err_sq.mean(dim="obs", skipna=True).values
            lead_sq[i] = mse

        lead_sq_errors_list.append(lead_sq)

    # Convert the list of lead SE into an array (n_samples, n_lead)
    lead_sq_errors_arr = np.stack(lead_sq_errors_list, axis=0)
    mean_lead_mse = np.mean(lead_sq_errors_arr, axis=0)
    rmse_lead = np.sqrt(mean_lead_mse)

    # Bootstrapping 1000 iterations to estimate the 50% confidence interval
    bootstrap_iterations = 1000
    n_samples, n_lead = lead_sq_errors_arr.shape
    bootstrap_rmse = np.zeros((bootstrap_iterations, n_lead))
    for i in range(bootstrap_iterations):
        sample_indices = np.random.choice(
            n_samples, size=n_samples, replace=True
        )
        sample_mean_mse = np.mean(lead_sq_errors_arr[sample_indices, :], axis=0)
        bootstrap_rmse[i, :] = np.sqrt(sample_mean_mse)
    ci_lower = np.percentile(bootstrap_rmse, 25, axis=0)
    ci_upper = np.percentile(bootstrap_rmse, 75, axis=0)

    # Build dictionary with lead-wise RMSE and confidence intervals
    rmse_dict = {
        "rmse": rmse_lead.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
    }

    return rmse_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mediterranean")
    parser.add_argument("--forecast", default="seacast")
    args = parser.parse_args()

    # Determine forecast files for SLA error calculation
    if args.forecast == "med_phy":
        forecast_dir = os.path.join("data", args.dataset, "samples", "test")
        forecast_files = sorted(
            glob(os.path.join(forecast_dir, "for_data_*.npy"))
        )
        init_flag = True
        max_lead = 10
    else:
        forecast_dir = os.path.join(
            "data", args.dataset, "predictions", args.forecast
        )
        forecast_files = sorted(glob(os.path.join(forecast_dir, "*.npy")))
        init_flag = False
        max_lead = 15

    # Define file paths for observed SLA and model MDT
    sla_obs_file = os.path.join("data", args.dataset, "observations", "sla.nc")
    mdt_mod_file = os.path.join(
        "data", args.dataset, "observations", "cmems_mod_med_phy_anfc_mdt.nc"
    )

    # Load sea mask from the bathy file
    bathy_path = os.path.join("data", args.dataset, "static", "bathy_mask.nc")
    bathy_data = xr.load_dataset(bathy_path)
    mask = bathy_data.where(bathy_data.mask, drop=True).mask

    # Compute SLA RMSE over forecast samples
    rmse_dict = compute_rmse_forecast_vs_obs_over_samples_sla(
        forecast_files=forecast_files,
        sla_obs_file=sla_obs_file,
        mdt_mod_file=mdt_mod_file,
        sea_mask=mask,
        init_flag=init_flag,
        max_lead=max_lead,
    )

    # Build output directory path
    out_dir = os.path.join("data", args.dataset, "metrics", args.forecast)
    os.makedirs(out_dir, exist_ok=True)

    # Save SLA RMSE
    json_path = os.path.join(out_dir, "sla_rmse.json")
    with open(json_path, "w") as jf:
        json.dump(rmse_dict, jf, indent=2)
    print(f"Saved lead-wise SLA RMSE with CI to {json_path}")
