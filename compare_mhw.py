# Standard library
import argparse
import json
import os
from datetime import datetime, timedelta
from glob import glob

# Third-party
import numba as nb
import numpy as np
import pandas as pd
import xarray as xr

# First-party
from neural_lam import constants


@nb.njit
def compute_confusion(forecast_slice, observed_slice):
    """
    Given two 1D boolean arrays (flattened forecast and observed masks),
    compute True Positives (TP), False Positives (FP) and False Negatives (FN).
    """
    TP = 0
    FP = 0
    FN = 0
    n = forecast_slice.shape[0]
    for i in range(n):
        if forecast_slice[i]:
            if observed_slice[i]:
                TP += 1
            else:
                FP += 1
        else:
            if observed_slice[i]:
                FN += 1
    return TP, FP, FN


def compute_bootstrap_ci(values, num_bootstrap=1000, ci_percent=50):
    """Compute bootstrap confidence interval for an array of values."""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan

    boot_means = []
    for _ in range(num_bootstrap):
        resampled = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(resampled))

    lower_percentile = (100 - ci_percent) / 2
    upper_percentile = 100 - lower_percentile
    return np.percentile(boot_means, lower_percentile), np.percentile(
        boot_means, upper_percentile
    )


def evaluate_detection_metrics(
    forecast_files,
    sst_obs_file,
    mhw_thresholds_file,
    sea_mask,
    init_flag=False,
    max_lead=None,
):
    """
    For each forecast file, compute detection metrics
    (SR, POD, TS, HSS, Bias, and ETS) for each lead time.
    """
    # Load observed SST and thresholds
    ds_obs = xr.open_dataset(sst_obs_file)
    sst_obs = ds_obs["sea_surface_temperature"]

    ds_threshold = xr.open_dataset(mhw_thresholds_file)
    doy_obs = ds_obs["time"].dt.dayofyear
    # Interpolate thresholds to the observation time coordinate
    threshold = ds_threshold["sst_threshold"].interp(dayofyear=doy_obs)

    # Observed binary mask True when SST > threshold
    obs_mask = sst_obs > threshold

    # Define forecast period length
    T = max_lead
    L_max = T

    # Initialize dictionary to collect binary metrics per lead time
    metrics_by_lead = {
        L: {"sr": [], "pod": [], "ts": [], "hss": [], "bias": [], "ets": []}
        for L in range(L_max)
    }

    n_files = 0
    for forecast_file in forecast_files:
        print("Processing forecast file:", forecast_file, flush=True)
        n_files += 1

        # Read forecast file and interpolate onto observation grid
        ds_forecast = read_npy_to_xarray(
            forecast_file, sea_mask, init=init_flag
        )
        forecast_sst = ds_forecast["sst"].interp(
            latitude=ds_obs.latitude, longitude=ds_obs.longitude
        )
        if max_lead is not None:
            forecast_sst = forecast_sst.sel(
                time=slice(
                    forecast_sst.time.values[0],
                    forecast_sst.time.values[0] + np.timedelta64(max_lead, "D"),
                )
            )

        # Compute forecast thresholds
        doy_forecast = forecast_sst["time"].dt.dayofyear
        threshold_forecast = ds_threshold["sst_threshold"].interp(
            dayofyear=doy_forecast
        )
        forecast_mask = forecast_sst > threshold_forecast
        mod_mask_arr = forecast_mask.values.astype(np.bool_)

        # Restrict observed data to the same period
        forecast_start = pd.to_datetime(forecast_sst.time.values[0])
        obs_start_time = forecast_start
        obs_end_time = forecast_start + np.timedelta64(max_lead, "D")
        obs_restricted_mask = obs_mask.sel(
            time=slice(obs_start_time, obs_end_time)
        )
        obs_mask_arr = obs_restricted_mask.values.astype(np.bool_)

        # Loop over lead times
        for L in range(L_max):
            # Compare forecast and observed masks at the same time index
            forecast_slice = mod_mask_arr[L, :, :]
            observed_slice = obs_mask_arr[L, :, :]
            f_flat = forecast_slice.flatten()
            o_flat = observed_slice.flatten()
            TP, FP, FN = compute_confusion(f_flat, o_flat)
            total = f_flat.size
            TN = total - (TP + FP + FN)

            # Success ratio
            sr = TP / (TP + FP) if (TP + FP) > 0 else np.nan
            # Probability of detection
            pod = TP / (TP + FN) if (TP + FN) > 0 else np.nan

            # Threat score
            ts = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else np.nan

            # Heidke skill score
            denominator_hss = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
            hss = (
                2 * (TP * TN - FP * FN) / denominator_hss
                if denominator_hss > 0
                else np.nan
            )

            # Bias
            bias = (TP + FP) / (TP + FN) if (TP + FN) > 0 else np.nan

            # Equitable threat score
            denominator_total = TP + FP + FN + TN
            if denominator_total > 0:
                CH = ((TP + FP) * (TP + FN)) / denominator_total
            else:
                CH = 0.0
            ets = (
                (TP - CH) / (TP + FP + FN - CH)
                if (TP + FP + FN - CH) > 0
                else np.nan
            )

            metrics_by_lead[L]["sr"].append(sr)
            metrics_by_lead[L]["pod"].append(pod)
            metrics_by_lead[L]["ts"].append(ts)
            metrics_by_lead[L]["hss"].append(hss)
            metrics_by_lead[L]["bias"].append(bias)
            metrics_by_lead[L]["ets"].append(ets)

    results = {}
    for L in range(L_max):
        sr_vals = np.array(metrics_by_lead[L]["sr"])
        pod_vals = np.array(metrics_by_lead[L]["pod"])
        ts_vals = np.array(metrics_by_lead[L]["ts"])
        hss_vals = np.array(metrics_by_lead[L]["hss"])
        bias_vals = np.array(metrics_by_lead[L]["bias"])
        ets_vals = np.array(metrics_by_lead[L]["ets"])

        mean_sr = np.nanmean(sr_vals)
        mean_pod = np.nanmean(pod_vals)
        mean_ts = np.nanmean(ts_vals)
        mean_hss = np.nanmean(hss_vals)
        mean_bias = np.nanmean(bias_vals)
        mean_ets = np.nanmean(ets_vals)

        ci_sr_lower, ci_sr_upper = compute_bootstrap_ci(
            sr_vals, num_bootstrap=1000, ci_percent=50
        )
        ci_pod_lower, ci_pod_upper = compute_bootstrap_ci(
            pod_vals, num_bootstrap=1000, ci_percent=50
        )
        ci_ts_lower, ci_ts_upper = compute_bootstrap_ci(
            ts_vals, num_bootstrap=1000, ci_percent=50
        )
        ci_hss_lower, ci_hss_upper = compute_bootstrap_ci(
            hss_vals, num_bootstrap=1000, ci_percent=50
        )
        ci_bias_lower, ci_bias_upper = compute_bootstrap_ci(
            bias_vals, num_bootstrap=1000, ci_percent=50
        )
        ci_ets_lower, ci_ets_upper = compute_bootstrap_ci(
            ets_vals, num_bootstrap=1000, ci_percent=50
        )

        # Store results with lead time keys starting at 1
        results[str(L + 1)] = {
            "sr": mean_sr,
            "sr_ci_lower": ci_sr_lower,
            "sr_ci_upper": ci_sr_upper,
            "pod": mean_pod,
            "pod_ci_lower": ci_pod_lower,
            "pod_ci_upper": ci_pod_upper,
            "ts": mean_ts,
            "ts_ci_lower": ci_ts_lower,
            "ts_ci_upper": ci_ts_upper,
            "hss": mean_hss,
            "hss_ci_lower": ci_hss_lower,
            "hss_ci_upper": ci_hss_upper,
            "bias": mean_bias,
            "bias_ci_lower": ci_bias_lower,
            "bias_ci_upper": ci_bias_upper,
            "ets": mean_ets,
            "ets_ci_lower": ci_ets_lower,
            "ets_ci_upper": ci_ets_upper,
        }

    return results


# ----------------------------------------------------
def read_npy_to_xarray(file_path, sea_mask, init=False):
    """
    Reads a forecast file and reconstructs a full grid for SST.
    """
    filename = os.path.basename(file_path)
    date_str = filename.split(".")[0]
    start_date = datetime.strptime(date_str[-8:], "%Y%m%d")
    data = np.load(file_path)
    if init:
        data = data[2:]
    n_time, _, _ = data.shape
    n_lat, n_lon = constants.GRID_SHAPE
    surface_mask = sea_mask.isel(depth=0).values
    lat_idx, lon_idx = np.where(surface_mask == 1)
    feature_idx = constants.EXP_PARAM_NAMES_SHORT.index("thetao_1")
    var_forecast = data[:, :, feature_idx]
    var_array = np.full((n_time, n_lat, n_lon), np.nan, dtype=data.dtype)
    var_array[:, lat_idx, lon_idx] = var_forecast
    data_vars = {"sst": (("time", "latitude", "longitude"), var_array)}
    time_coords = [start_date + timedelta(days=i) for i in range(n_time)]
    coords = {
        "time": time_coords,
        "latitude": sea_mask.latitude,
        "longitude": sea_mask.longitude,
    }
    ds = xr.Dataset(data_vars, coords=coords)
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mediterranean")
    parser.add_argument("--forecast", type=str, default="seacast")
    parser.add_argument("--max_lead", type=int, default=None)
    args = parser.parse_args()

    if args.forecast == "med_phy":
        forecast_dir = os.path.join("data", args.dataset, "samples", "test")
        forecast_files = sorted(
            glob(os.path.join(forecast_dir, "for_data_*.npy"))
        )
        init_flag = True
        if args.max_lead is None:
            max_lead = 10
            suffix = ""
        else:
            max_lead = args.max_lead
            suffix = f"_{max_lead}"
    else:
        forecast_dir = os.path.join(
            "data", args.dataset, "predictions", args.forecast
        )
        forecast_files = sorted(glob(os.path.join(forecast_dir, "*.npy")))
        init_flag = False
        if args.max_lead is None:
            max_lead = 15
            suffix = ""
        else:
            max_lead = args.max_lead
            suffix = f"_{max_lead}"

    sst_obs_file = os.path.join("data", args.dataset, "observations", "sst.nc")
    mhw_thresholds_file = os.path.join(
        "data", args.dataset, "observations", "mhw_thresholds.nc"
    )
    bathy_path = os.path.join("data", args.dataset, "static", "bathy_mask.nc")
    bathy_data = xr.load_dataset(bathy_path)
    mask = bathy_data.where(bathy_data.mask, drop=True).mask

    metrics = evaluate_detection_metrics(
        forecast_files,
        sst_obs_file,
        mhw_thresholds_file,
        mask,
        init_flag=init_flag,
        max_lead=max_lead,
    )

    out_json = os.path.join(
        "data",
        args.dataset,
        "metrics",
        args.forecast,
        f"mhw_detection_metrics{suffix}.json",
    )
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Detection metrics saved in:", out_json)


if __name__ == "__main__":
    main()
