# Standard library
import argparse
import json
import os
from datetime import datetime, timedelta
from glob import glob

# Third-party
import numpy as np
import pandas as pd
import xarray as xr

# First-party
from neural_lam import constants


def read_npy_to_xarray(file_path, sea_mask, init=False):
    """
    Reads forecast file and reconstructs a full grid for SST.
    """
    # Get start date from filename
    filename = os.path.basename(file_path)
    date_str = filename.split(".")[0]
    start_date = datetime.strptime(date_str[-8:], "%Y%m%d")

    data = np.load(file_path)  # (time, n_grid, n_features)
    if init:
        data = data[2:]
    n_time, _, _ = data.shape

    # Full horizontal grid dimensions
    n_lat, n_lon = constants.GRID_SHAPE

    # Get indices from the surface mask
    surface_mask = sea_mask.isel(depth=0).values
    lat_idx, lon_idx = np.where(surface_mask == 1)

    # Feature index for SST
    feature_idx = constants.EXP_PARAM_NAMES_SHORT.index("thetao_1")

    # Extract SST forecast (n_time, n_grid)
    var_forecast = data[:, :, feature_idx]

    # Create a full grid array and assign values where sea
    var_array = np.full((n_time, n_lat, n_lon), np.nan, dtype=data.dtype)
    var_array[:, lat_idx, lon_idx] = var_forecast

    data_vars = {"sst": (("time", "latitude", "longitude"), var_array)}

    # Create coordinates
    time_coords = [start_date + timedelta(days=i) for i in range(n_time)]
    coords = {
        "time": time_coords,
        "latitude": sea_mask.latitude,
        "longitude": sea_mask.longitude,
    }
    ds = xr.Dataset(data_vars, coords=coords)
    return ds


def compute_sst_error(
    forecast_files, sst_obs_file, sea_mask, init_flag=False, max_lead=None
):
    """
    Computes RMSE between the forecast SST and observed SST.

    Returns:
      - dict with lead-wise SST RMSE and 50% CI using bootstrapping.
      - data array for the spatial SST RMSE for each lead time.
    """

    # Open the observed SST dataset and match the time coordinate
    ds_obs = xr.open_dataset(sst_obs_file)
    new_times = ds_obs["time"] - pd.Timedelta(days=1)
    ds_obs = ds_obs.assign_coords(time=new_times)

    # Accumulate errors over samples
    lead_sq_errors_list = []  # list of arrays (n_lead,)
    spatial_sq_errors_by_lead_list = (
        []
    )  # list (over samples) of lists (over lead time) of error fields

    forecast_times = None

    for forecast_file in forecast_files:
        print(forecast_file, flush=True)
        # Read forecast sample and interpolate its SST onto the observation grid
        ds_forecast = read_npy_to_xarray(
            forecast_file, sea_mask, init=init_flag
        )
        forecast_sst = ds_forecast["sst"].interp(
            latitude=ds_obs.latitude, longitude=ds_obs.longitude
        )

        if max_lead is not None:
            forecast_sst = forecast_sst.isel(time=slice(0, max_lead))

        # Use the same forecast times for all files
        if forecast_times is None:
            forecast_times = forecast_sst.time.values
        n_lead = forecast_sst.time.size

        # Array to accumulate scalar SE per lead for this sample
        lead_sq = np.zeros(n_lead)
        # Store list of error fields per lead for this sample
        sample_spatial_sq_by_lead = []

        for i, t in enumerate(forecast_sst.time.values):
            f_t = forecast_sst.sel(time=t)
            # Select SST and convert from Kelvin to Celsius
            obs = ds_obs["sea_surface_temperature"].sel(time=t) - 273.15

            # Compute SE field for this lead time
            err_sq = (f_t - obs) ** 2

            # Average SE over grid
            mse = err_sq.mean(dim=["latitude", "longitude"], skipna=True).values
            lead_sq[i] = mse

            # Spatial SE by lead
            sample_spatial_sq_by_lead.append(err_sq)

        # Append this sample's errors
        lead_sq_errors_list.append(lead_sq)
        spatial_sq_errors_by_lead_list.append(sample_spatial_sq_by_lead)

    # Convert lead squared errors list to an array of shape (n_samples, n_lead)
    lead_sq_errors_arr = np.stack(lead_sq_errors_list, axis=0)
    mean_lead_mse = np.mean(lead_sq_errors_arr, axis=0)
    rmse_lead = np.sqrt(mean_lead_mse)

    # Bootstrap 1000 iterations to estimate 50% CI per lead time
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

    # For spatial RMSE by lead time, average over samples for each lead time
    spatial_sq_by_lead_samples = []
    for i in range(n_lead):
        # Collect the error field for lead time i from each sample
        sample_fields = [
            spatial_sq_errors_by_lead_list[sample_idx][i]
            for sample_idx in range(n_samples)
        ]
        # Concatenate along a new sample dimension
        da = xr.concat(sample_fields, dim="sample")
        # Average over samples
        mean_da = da.mean(dim="sample", skipna=True)
        spatial_sq_by_lead_samples.append(mean_da)
    # Stack the resulting DataArrays along a new lead time dimension
    spatial_rmse = xr.concat(
        [np.sqrt(da) for da in spatial_sq_by_lead_samples], dim="time"
    )
    # Assign the lead times from forecast_times as the new time coordinate
    spatial_rmse = spatial_rmse.assign_coords(time=forecast_times)

    # Build a dictionary for lead-wise RMSE with CI keys
    rmse_dict = {
        "rmse": rmse_lead.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
    }

    return rmse_dict, spatial_rmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mediterranean")
    parser.add_argument("--forecast", default="seacast")
    args = parser.parse_args()

    # Determine forecast files
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

    # Path to observed SST file
    sst_obs_file = os.path.join("data", args.dataset, "observations", "sst.nc")

    # Load sea mask from the bathy file
    bathy_path = os.path.join("data", args.dataset, "static", "bathy_mask.nc")
    bathy_data = xr.load_dataset(bathy_path)
    mask = bathy_data.where(bathy_data.mask, drop=True).mask

    # Compute RMSE over all forecast sample files
    rmse_dict, spatial_rmse = compute_sst_error(
        forecast_files=forecast_files,
        sst_obs_file=sst_obs_file,
        sea_mask=mask,
        init_flag=init_flag,
        max_lead=max_lead,
    )

    # Build output directory path
    out_dir = os.path.join("data", args.dataset, "metrics", args.forecast)
    os.makedirs(out_dir, exist_ok=True)

    # Save lead-wise RMSE with confidence intervals
    json_path = os.path.join(out_dir, "sst_rmse.json")
    with open(json_path, "w") as jf:
        json.dump(rmse_dict, jf, indent=2)
    print(f"Saved lead-wise RMSE with CI to {json_path}")

    # Save spatial RMSE for each lead time
    ds_out_lead = xr.Dataset({"rmse": spatial_rmse})
    nc_lead_path = os.path.join(out_dir, "sst_spatial_rmse.nc")
    ds_out_lead.to_netcdf(nc_lead_path)
    print(f"Saved spatial RMSE by lead time to {nc_lead_path}")


if __name__ == "__main__":
    main()
