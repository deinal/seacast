# Standard library
import argparse
import json
import os
from datetime import datetime, timedelta
from glob import glob

# Third-party
import numpy as np
import xarray as xr
from dask import compute, delayed
from dask.diagnostics import ProgressBar

# First-party
from neural_lam import constants


def read_npy_to_xarray(file_path, sea_mask, init=False):
    """Reads forecast file and reconstructs the full grid using a sea mask."""
    # Get start date from filename
    filename = os.path.basename(file_path)
    date_str = filename.split(".")[0]
    start_date = datetime.strptime(date_str[-8:], "%Y%m%d")

    # Load forecast
    data = np.load(file_path)  # (time, n_grid, n_features)
    if init:
        data = data[2:]

    n_time, _, _ = data.shape

    # Full horizontal grid dimensions
    n_lat, n_lon = constants.GRID_SHAPE

    # Get the indices (row, col) belonging to the sea
    surface_mask = sea_mask.isel(depth=0).values
    lat_idx, lon_idx = np.where(surface_mask == 1)

    data_vars = {}
    feature_idx = 0

    # Loop over the parameters
    for param_short, has_depth in zip(
        constants.PARAM_NAMES_SHORT,
        constants.LEVELS,
    ):
        if has_depth:
            n_depths = len(constants.DEPTHS)
            # Extract forecast data for this variable (n_time, n_grid, n_depths)
            var_forecast = data[:, :, feature_idx : feature_idx + n_depths]
            feature_idx += n_depths

            # Initialize a full array (time, lat, lon, depth)
            var_array = np.full(
                (n_time, n_lat, n_lon, n_depths), np.nan, dtype=data.dtype
            )

            # Assign forecast data only where sea_mask is water at that depth
            for d in range(n_depths):
                mask_d = sea_mask.isel(depth=d).values  # (n_lat, n_lon)
                valid_indices = np.where(mask_d[lat_idx, lon_idx] == 1)[0]
                if valid_indices.size > 0:
                    var_array[
                        :, lat_idx[valid_indices], lon_idx[valid_indices], d
                    ] = var_forecast[:, valid_indices, d]
            # Add variable to the dataset
            data_vars[param_short] = (
                ("time", "latitude", "longitude", "depth"),
                var_array,
            )
        else:
            # Non-depth variable (n_time, n_grid)
            var_forecast = data[:, :, feature_idx]
            feature_idx += 1

            # Create a full grid (time, lat, lon) and fill with forecast values
            var_array = np.full(
                (n_time, n_lat, n_lon), np.nan, dtype=data.dtype
            )
            var_array[:, lat_idx, lon_idx] = var_forecast
            data_vars[param_short] = (
                ("time", "latitude", "longitude"),
                var_array,
            )

    # Create coordinates
    time_coords = [start_date + timedelta(days=i) for i in range(n_time)]
    coords = {
        "time": time_coords,
        "latitude": sea_mask.latitude,
        "longitude": sea_mask.longitude,
        "depth": sea_mask.depth,
    }
    ds = xr.Dataset(data_vars, coords=coords)

    return ds


@delayed
def process_forecast_file(
    forecast_file,
    obs_data,
    sim_var,
    obs_var,
    sea_mask,
    init,
    max_lead,
    extrapolate_depth,
):
    ds_sim = read_npy_to_xarray(forecast_file, sea_mask, init=init)
    ds_sim = ds_sim[sim_var]
    if max_lead is not None:
        ds_sim = ds_sim.isel(time=slice(0, max_lead))
    n_lead = ds_sim.time.size

    lead_err = np.zeros(n_lead)
    for i, t in enumerate(ds_sim.time.values):
        # Select in-situ coordinates for time t
        daily_obs_data = obs_data.where(
            obs_data["time"] == np.datetime64(t), drop=True
        )

        # Interpolate simulation field to in-situ latitude, longitude
        s_t = ds_sim.sel(time=t)  # (latitude, longitude, depth)
        s_interp_spatial = s_t.interp(
            latitude=daily_obs_data["latitude"],
            longitude=daily_obs_data["longitude"],
        )  # (depth, obs)

        # Interpolate along depth to the observation depths.
        if extrapolate_depth:
            s_interp = s_interp_spatial.interp(
                depth=daily_obs_data["depth"],
                kwargs={"fill_value": None, "bounds_error": False},
            )  # (obs)
        else:
            s_interp = s_interp_spatial.interp(depth=daily_obs_data["depth"])

        # Compute MSE over valid points
        err_sq = (s_interp - daily_obs_data[obs_var]) ** 2
        err = err_sq.mean(dim="obs", skipna=True).values

        lead_err[i] = err

    return lead_err


def compute_in_situ_error(
    sim_var,
    obs_var,
    forecast_files,
    in_situ_file,
    sea_mask,
    init=False,
    max_lead=None,
    extrapolate_depth=False,
):
    """
    Computes RMSE between simulation forecasts and in-situ observations.

    The simulation field is interpolated in two steps:
      1. Horizontal (lat–lon) bilinear interpolation using in-situ coordinates.
      2. Vertical (depth) linear interpolation to the in-situ depths.

    Returns:
      rmse_dict: dict with keys rmse, ci_lower, ci_upper
    """
    # Load in-situ dataset
    ds_obs = xr.load_dataset(in_situ_file)

    if "index" in ds_obs.dims:
        ds_obs = ds_obs.rename({"index": "obs"})
    if "date" in ds_obs:
        ds_obs = ds_obs.rename({"date": "time"})
    # Set the time as a coordinate
    ds_obs = ds_obs.set_coords("time")

    # Drop where obs_var is NaN
    ds_obs = ds_obs.dropna(dim="obs", subset=[obs_var])

    # Get the observation variable with dims (obs,)
    obs_data = ds_obs[[obs_var, "latitude", "longitude", "depth"]]

    delayed_lead_errors_list = []
    for f in forecast_files:
        delayed_lead_errors_list.append(
            process_forecast_file(
                f,
                obs_data,
                sim_var,
                obs_var,
                sea_mask,
                init,
                max_lead,
                extrapolate_depth,
            )
        )

    with ProgressBar():
        lead_errors_list = compute(
            *delayed_lead_errors_list, scheduler="processes"
        )

    lead_errors_arr = np.stack(lead_errors_list, axis=0)
    mean_lead_errors = np.nanmean(lead_errors_arr, axis=0)
    mean_lead_errors = np.sqrt(mean_lead_errors)

    # Bootstrap 1000 iterations for 50% CI
    bootstrap_iterations = 1000
    n_samples, n_lead = lead_errors_arr.shape
    bootstrap_error = np.zeros((bootstrap_iterations, n_lead))
    for i in range(bootstrap_iterations):
        sample_indices = np.random.choice(
            n_samples, size=n_samples, replace=True
        )
        sample_mean_err = np.nanmean(lead_errors_arr[sample_indices, :], axis=0)
        bootstrap_error[i, :] = np.sqrt(sample_mean_err)
    ci_lower = np.percentile(bootstrap_error, 25, axis=0)
    ci_upper = np.percentile(bootstrap_error, 75, axis=0)

    rmse_dict = {
        "rmse": mean_lead_errors.tolist(),
        "ci_lower": ci_lower.tolist(),
        "ci_upper": ci_upper.tolist(),
    }
    return rmse_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mediterranean")
    parser.add_argument("--forecast", default="seacast")
    args = parser.parse_args()

    # Determine forecast files for in-situ evaluation
    if args.forecast == "med_phy":
        forecast_dir = os.path.join("data", args.dataset, "samples", "test")
        forecast_files = sorted(
            glob(os.path.join(forecast_dir, "for_data_*.npy"))
        )
        init = True
        max_lead = 10
    else:
        forecast_dir = os.path.join(
            "data", args.dataset, "predictions", args.forecast
        )
        forecast_files = sorted(glob(os.path.join(forecast_dir, "*.npy")))
        init = False
        max_lead = 15

    # File paths for in-situ data and sea mask
    in_situ_file = os.path.join(
        "data", args.dataset, "observations", "in_situ.nc"
    )
    bathy_path = os.path.join("data", args.dataset, "static", "bathy_mask.nc")
    bathy_data = xr.load_dataset(bathy_path)
    mask = bathy_data.where(bathy_data.mask, drop=True).mask

    # Mapping between simulation variable names and in-situ variables
    var_mapping = {
        "uo": {"obs": "ewct", "extrapolate_depth": True},
        "vo": {"obs": "nsct", "extrapolate_depth": True},
        "so": {"obs": "psal", "extrapolate_depth": False},
        "thetao": {"obs": "temp", "extrapolate_depth": False},
    }

    out_dir = os.path.join("data", args.dataset, "metrics", args.forecast)
    os.makedirs(out_dir, exist_ok=True)

    for var, var_dict in var_mapping.items():
        print(f"Evaluating variable {var}")
        rmse_dict = compute_in_situ_error(
            sim_var=var,
            obs_var=var_dict["obs"],
            forecast_files=forecast_files,
            in_situ_file=in_situ_file,
            sea_mask=mask,
            init=init,
            max_lead=max_lead,
            extrapolate_depth=var_dict["extrapolate_depth"],
        )
        json_path = os.path.join(out_dir, f"in_situ_{var}_rmse.json")
        with open(json_path, "w") as jf:
            json.dump(rmse_dict, jf, indent=2)
        print(f"Saved RMSE for {var} to {json_path}")
