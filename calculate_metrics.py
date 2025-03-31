# Standard library
import argparse
import glob
import json
import os
from datetime import datetime, timedelta

# Third-party
import numpy as np
from dask import compute, delayed
from dask.distributed import Client

# First-party
from neural_lam import constants


def load_static_data(dataset_name):
    """
    Load static data for dataset.
    """
    static_dir_path = os.path.join("data", dataset_name, "static")

    # Load boundary mask, 1. if node is part of boundary, else 0
    boundary_mask_np = np.load(
        os.path.join(static_dir_path, "boundary_mask.npy")
    )  # (depths, h, w)

    # Load sea mask, 1. if node is part of the sea, else 0
    sea_mask_np = np.load(
        os.path.join(static_dir_path, "sea_mask.npy")
    )  # (depths, h, w)

    grid_weights_np = np.load(
        os.path.join(static_dir_path, "grid_weights.npy")
    )  # (h, w)

    # Mask for the surface grid
    surface_mask_np = sea_mask_np[0]

    # Grid mask for all depth levels to be multiplied with output states
    boundary_mask = boundary_mask_np[:, surface_mask_np]  # (depths, n_grid)
    border_mask = []
    for level_applies in constants.LEVELS:
        if level_applies:
            border_mask.append(boundary_mask)  # Multi level
        else:
            border_mask.append(boundary_mask[0][np.newaxis, :])  # Single level

    border_mask = np.concatenate(border_mask, axis=0).transpose(
        1, 0
    )  # (n_grid, d_features)

    # Grid mask for all depth levels to be multiplied with output states
    grid_mask = sea_mask_np[:, surface_mask_np]  # (depths, n_grid)
    sea_mask = []
    for level_applies in constants.LEVELS:
        if level_applies:
            sea_mask.append(grid_mask)  # Multi level
        else:
            sea_mask.append(grid_mask[0][np.newaxis, :])  # Single level

    sea_mask = np.concatenate(sea_mask, axis=0).transpose(
        1, 0
    )  # n_grid, d_features

    grid_weights = grid_weights_np[surface_mask_np]  # (n_grid,)

    return {
        "border_mask": border_mask,
        "sea_mask": sea_mask,
        "grid_weights": grid_weights,
    }


def load_forecast_analysis(forecast_file, analysis_file, is_med_phy):
    """
    Load forecast and analysis arrays, ignoring init states.
    """
    if is_med_phy:
        forecast = np.load(forecast_file)[2:12]
        analysis = np.load(analysis_file)[2:12]
    else:
        forecast = np.load(forecast_file)
        analysis = np.load(analysis_file)[2:]
    return forecast, analysis


def fisher_ci(acc_values):
    """
    Compute the ci for correlation coefficients using Fisher's z-transform.
    """
    acc_values = np.array(acc_values)
    acc_values = acc_values[~np.isnan(acc_values)]
    n = len(acc_values)
    if n < 2:
        return (np.nan, np.nan)
    # Clip values if needed to avoid infinities
    acc_values = np.clip(acc_values, -0.9999, 0.9999)
    z = np.arctanh(acc_values)
    mean_z = np.mean(z)
    se_z = np.std(z, ddof=1) / np.sqrt(n)
    # 50% CI using standard normal quantiles
    z_lower = mean_z - 0.6745 * se_z
    z_upper = mean_z + 0.6745 * se_z
    return (np.tanh(z_lower), np.tanh(z_upper))


def compute_mse_per_feature(
    forecast, analysis, feature_names, interior_mask, grid_weights
):
    """
    Compute per-feature MSE for each lead time.
    Returns a dict keyed by lead time with a dict mapping feature name to MSE.
    """
    lead_times = forecast.shape[0]
    mse_dict = {}
    for t in range(lead_times):
        diff = (
            forecast[t] - analysis[t]
        ) * interior_mask  # shape: (n_grid, n_features)
        w_sq_diff = diff**2 * grid_weights[:, np.newaxis]
        mse = np.sum(w_sq_diff, axis=0) / np.sum(interior_mask, axis=0)
        mse_dict[str(t)] = {
            fname: float(mse[i]) for i, fname in enumerate(feature_names)
        }
    return mse_dict


def compute_acc_per_feature(
    forecast,
    analysis,
    feature_names,
    interior_mask,
    grid_weights,
    climatology,
    validity_dates,
):
    """
    Compute per-feature ACC for each lead time.
    Returns a dict keyed by lead time with a dict mapping feature name to ACC.
    """
    lead_times = forecast.shape[0]
    acc_dict = {}
    for t in range(lead_times):
        # Determine day-of-year from the validity date
        date = validity_dates[t]
        doy = date.timetuple().tm_yday
        # Build the file path for the daily climatology
        clim_file = os.path.join(climatology, f"doy{doy:03d}.npy")
        daily_clim = np.load(clim_file)  # shape: (n_grid, features)

        acc_dict[str(t)] = {}
        for i, fname in enumerate(feature_names):
            f = forecast[t, :, i] - daily_clim[:, i]
            a = analysis[t, :, i] - daily_clim[:, i]
            mask = interior_mask[:, i].astype(bool)
            f_valid = f[mask]
            a_valid = a[mask]
            weights_valid = grid_weights[mask]

            numerator = np.sum(weights_valid * f_valid * a_valid)

            var_f = np.sum(weights_valid * f_valid**2)
            var_a = np.sum(weights_valid * a_valid**2)
            denominator = np.sqrt(var_f) * np.sqrt(var_a)

            acc = numerator / denominator

            acc_dict[str(t)][fname] = acc
    return acc_dict


def compute_avg_group_mse(
    forecast, analysis, feature_names, interior_mask, grid_weights, t_idx=None
):
    """
    Compute the average group RMSE for groups: uo, vo, so, thetao.

    Returns:
      A dict with keys for each group mapping to a dict with:
        - mse: a 1D numpy array (n_timesteps,) of MSE values.
    """
    groups = {"uo": [], "vo": [], "so": [], "thetao": []}
    for depth in constants.DEPTHS:
        groups["uo"].append(feature_names.index(f"uo_{round(depth)}"))
        groups["vo"].append(feature_names.index(f"vo_{round(depth)}"))
        groups["so"].append(feature_names.index(f"so_{round(depth)}"))
        groups["thetao"].append(feature_names.index(f"thetao_{round(depth)}"))

    if t_idx is None:
        t_idx = range(forecast.shape[0])

    results = {}
    for group, indices in groups.items():
        mse_list = []
        for t in t_idx:
            # error: shape (n_grid, n_group_features)
            error = forecast[t][:, indices] - analysis[t][:, indices]
            mask = interior_mask[:, indices]
            error = np.where(mask, error, np.nan)
            # Average over group indices (depth)
            cell_sq_error = np.nanmean(error**2, axis=1)  # (n_grid,)
            # Compute weighted average over grid cells:
            mse_t = np.nansum(cell_sq_error * grid_weights) / np.nansum(
                grid_weights
            )
            mse_list.append(mse_t)
        results[group] = {"mse": np.array(mse_list)}
    return results


def compute_avg_group_acc(
    forecast,
    analysis,
    feature_names,
    interior_mask,
    grid_weights,
    climatology,
    validity_dates,
    t_idx=None,
):
    """
    Compute the average group ACC for groups: uo, vo, so, thetao.

    Returns:
      A dict with keys for each group mapping to:
        - acc: a 1D numpy array (n_timesteps,) of ACC values.
    """
    if t_idx is None:
        t_idx = range(forecast.shape[0])

    groups = {"uo": [], "vo": [], "so": [], "thetao": []}
    for depth in constants.DEPTHS:
        groups["uo"].append(feature_names.index(f"uo_{round(depth)}"))
        groups["vo"].append(feature_names.index(f"vo_{round(depth)}"))
        groups["so"].append(feature_names.index(f"so_{round(depth)}"))
        groups["thetao"].append(feature_names.index(f"thetao_{round(depth)}"))

    agg_acc = {}
    for group, indices in groups.items():
        n_timesteps = len(t_idx)
        group_acc = np.zeros(n_timesteps)
        ci_lower = np.zeros(n_timesteps)
        ci_upper = np.zeros(n_timesteps)

        for j, t in enumerate(t_idx):
            # Get the validity date for time step t and load daily climatology
            date = validity_dates[t]
            doy = date.timetuple().tm_yday
            clim_file = os.path.join(climatology, f"doy{doy:03d}.npy")
            daily_clim = np.load(clim_file)  # (n_grid, n_features)

            # Compute anomalies for each feature in the group
            f_anom = forecast[t][:, indices] - daily_clim[:, indices]
            a_anom = analysis[t][:, indices] - daily_clim[:, indices]

            # Compute group anomaly by averaging over features per grid cell
            f_group = np.nanmean(f_anom, axis=1)  # (n_grid,)
            a_group = np.nanmean(a_anom, axis=1)  # (n_grid,)

            # Define a valid mask
            valid_mask = np.all(interior_mask[:, indices], axis=1)
            f_valid = f_group[valid_mask]
            a_valid = a_group[valid_mask]
            weights_valid = grid_weights[valid_mask]

            numerator = np.nansum(weights_valid * f_valid * a_valid)
            denominator = np.sqrt(
                np.nansum(weights_valid * f_valid**2)
            ) * np.sqrt(np.nansum(weights_valid * a_valid**2))
            if denominator == 0:
                acc_t = np.nan
            else:
                acc_t = numerator / denominator
            group_acc[j] = acc_t
            ci_lower[j] = acc_t
            ci_upper[j] = acc_t

        agg_acc[group] = {
            "acc": group_acc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    return agg_acc


def compute_group_bias(forecast, analysis, feature_names, sea_mask, t_idx):
    """
    For each feature group (uo, vo, so, thetao and zos),
    compute bias (forecast - analysis) averaged over group features.
    Returns group_bias: {group: (T, n_grid)}
    """
    groups = {"uo": [], "vo": [], "so": [], "thetao": [], "zos": []}
    for depth in constants.DEPTHS:
        groups["uo"].append(feature_names.index(f"uo_{round(depth)}"))
        groups["vo"].append(feature_names.index(f"vo_{round(depth)}"))
        groups["so"].append(feature_names.index(f"so_{round(depth)}"))
        groups["thetao"].append(feature_names.index(f"thetao_{round(depth)}"))
    groups["zos"].append(feature_names.index("zos"))

    n_grid = forecast.shape[1]

    group_bias = {}
    for group_name, indices in groups.items():
        sea_mask_group = sea_mask[:, indices]  # (n_grid, len(indices))
        bias_array = np.zeros((len(t_idx), n_grid))
        for i, t in enumerate(t_idx):
            error = (
                forecast[t][:, indices] - analysis[t][:, indices]
            )  # (n_grid, len(indices))
            # Where mask is 0 replace error with nan
            error[~sea_mask_group] = np.nan
            # Mean ignoring nans
            bias_array[i] = np.nanmean(error, axis=1)
        group_bias[group_name] = bias_array
    return group_bias


def compute_group_rmse(
    forecast,
    analysis,
    feature_names,
    sea_mask,
    grid_weights,
    t_idx=None,
    max_lead_time=10,
):
    """
    For each feature group, compute per-file MSE.
    """
    if t_idx is None:
        t_idx = list(range(min(max_lead_time, forecast.shape[0])))
    else:
        t_idx = [t for t in t_idx if t < max_lead_time]

    groups = {"uo": [], "vo": [], "so": [], "thetao": []}
    for depth in constants.DEPTHS:
        groups["uo"].append(feature_names.index(f"uo_{round(depth)}"))
        groups["vo"].append(feature_names.index(f"vo_{round(depth)}"))
        groups["so"].append(feature_names.index(f"so_{round(depth)}"))
        groups["thetao"].append(feature_names.index(f"thetao_{round(depth)}"))

    group_mse = {}
    for group_name, indices in groups.items():
        time_sq_err_avg = []
        time_w_sq_err_all = []

        for t in t_idx:
            error = forecast[t][:, indices] - analysis[t][:, indices]
            mask_group = sea_mask[:, indices]  # (n_grid, n_group_features)
            error = np.where(mask_group, error, np.nan)

            # Average error over the group features for each grid cell
            error_avg = np.nanmean(error, axis=1)  # (n_grid,)
            # Compute squared error fields for this time step
            sq_err_avg = (
                error_avg**2
            )  # (n_grid,) no weights needed for lat, lon errors
            w_sq_err_all = (
                error**2 * grid_weights[:, np.newaxis]
            )  # (n_grid, n_group_features)

            time_sq_err_avg.append(sq_err_avg)
            time_w_sq_err_all.append(w_sq_err_all)

        mse_avg = np.nanmean(
            np.stack(time_sq_err_avg, axis=0), axis=0
        )  # (n_grid,)
        mse_all = np.nanmean(
            np.stack(time_w_sq_err_all, axis=0), axis=0
        )  # (n_grid, len(indices))

        group_mse[group_name] = {"mse_avg": mse_avg, "mse_all": mse_all}

    return group_mse


def process_file(
    forecast_file,
    analysis_file,
    feature_names,
    sea_mask,
    interior_mask,
    grid_weights,
    climatology,
    is_med_phy,
):
    """
    Load forecast and analysis arrays and compute metrics.
    """
    print(forecast_file, flush=True)
    forecast, analysis = load_forecast_analysis(
        forecast_file, analysis_file, is_med_phy
    )

    base_name = os.path.basename(forecast_file)
    date_str = base_name[-12:-4]
    lead_time_date = datetime.strptime(date_str, "%Y%m%d")

    # Generate validity dates for each forecast time step.
    lead_times = forecast.shape[0]
    validity_dates = [
        lead_time_date + timedelta(days=t) for t in range(lead_times)
    ]

    mse_per_feature = compute_mse_per_feature(
        forecast,
        analysis,
        feature_names,
        interior_mask,
        grid_weights,
    )
    acc_per_feature = compute_acc_per_feature(
        forecast,
        analysis,
        feature_names,
        interior_mask,
        grid_weights,
        climatology,
        validity_dates,
    )
    avg_group_mse = compute_avg_group_mse(
        forecast,
        analysis,
        feature_names,
        interior_mask,
        grid_weights,
    )
    avg_group_acc = compute_avg_group_acc(
        forecast,
        analysis,
        feature_names,
        interior_mask,
        grid_weights,
        climatology,
        validity_dates,
    )
    group_mse = compute_group_rmse(
        forecast,
        analysis,
        feature_names,
        sea_mask,
        grid_weights,
    )
    group_bias = compute_group_bias(
        forecast,
        analysis,
        feature_names,
        interior_mask,
        t_idx=([0, 4, 9] if is_med_phy else [0, 4, 9, 14]),
    )
    return {
        "mse_per_feature": mse_per_feature,
        "acc_per_feature": acc_per_feature,
        "avg_group_mse": avg_group_mse,
        "avg_group_acc": avg_group_acc,
        "group_mse": group_mse,
        "group_bias": group_bias,
    }


def aggregate_mse_per_feature(metrics_list, feature_names, n_bootstrap=1000):
    """
    For each lead time and each feature, compute per-file MSE,
    then return the aggregated RMSE and ci via bootstrapping.
    """
    lead_times = len(metrics_list[0]["mse_per_feature"])
    aggregated = {}
    for t in range(lead_times):
        t_str = str(t)
        mse_vals = {fname: [] for fname in feature_names}
        for metrics in metrics_list:
            mse_per_feature = metrics["mse_per_feature"]
            for fname in feature_names:
                mse_vals[fname].append(mse_per_feature[t_str][fname])
        aggregated[t_str] = {}
        for fname in feature_names:
            # Convert list of MSE values to array and compute aggregated MSE
            mse_array = np.array(mse_vals[fname])
            aggregated_mse = np.mean(mse_array)
            aggregated_rmse = np.sqrt(aggregated_mse)

            # Bootstrap on the MSE values
            boot_mses = []
            n = len(mse_array)
            for _ in range(n_bootstrap):
                sample = np.random.choice(mse_array, size=n, replace=True)
                boot_mses.append(np.mean(sample))
            boot_mses = np.array(boot_mses)
            boot_rmses = np.sqrt(boot_mses)
            ci_lower = np.percentile(boot_rmses, 25)
            ci_upper = np.percentile(boot_rmses, 75)

            aggregated[t_str][fname] = {
                "rmse": aggregated_rmse,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
    return aggregated


def aggregate_acc_per_feature(metrics_list, feature_names):
    """
    For each lead time and each feature, compute per-file ACC,
    and its 95% confidence interval computed via Fisher's z-transform.
    """
    lead_times = len(metrics_list[0]["acc_per_feature"])
    aggregated = {}
    for t in range(lead_times):
        t_str = str(t)
        acc_vals = {fname: [] for fname in feature_names}
        for metrics in metrics_list:
            acc_per_feature = metrics["acc_per_feature"]
            for fname in feature_names:
                acc_vals[fname].append(acc_per_feature[t_str][fname])
        aggregated[t_str] = {}
        for fname in feature_names:
            acc_array = np.array(acc_vals[fname])
            mean_acc = float(np.nanmean(acc_array))
            # Compute confidence interval using Fisher's z-transformation
            ci_lower, ci_upper = fisher_ci(acc_vals[fname])
            aggregated[t_str][fname] = {
                "acc": mean_acc,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
    return aggregated


def aggregate_avg_group_rmse(metrics_list, n_bootstrap=1000):
    """
    Aggregate per-file average group RMSE for each timestep.

    Returns:
      A dict with keys for each group mapping to a dict with:
        - rmse: numpy array of aggregated RMSE values per time step.
        - ci_lower: numpy array of lower bounds per time step.
        - ci_upper: numpy array of upper bounds per time step.
    """
    aggregated = {}
    groups = ["uo", "vo", "so", "thetao"]
    for group in groups:
        # Stack MSE arrays from each file shape (n_files, n_timesteps)
        mse_array = np.stack(
            [
                file_metric["avg_group_mse"][group]["mse"]
                for file_metric in metrics_list
            ],
            axis=0,
        )
        n_files, n_timesteps = mse_array.shape
        agg_rmse = np.zeros(n_timesteps)
        ci_lower = np.zeros(n_timesteps)
        ci_upper = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            mse_vals = mse_array[:, t]
            agg_mse = np.mean(mse_vals)
            agg_rmse[t] = np.sqrt(agg_mse)

            # Bootstrap the mean MSE for time step t
            boot_mses = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(mse_vals, size=n_files, replace=True)
                boot_mses.append(np.mean(sample))
            boot_mses = np.array(boot_mses)
            boot_rmses = np.sqrt(boot_mses)
            ci_lower[t] = np.percentile(boot_rmses, 25)
            ci_upper[t] = np.percentile(boot_rmses, 75)

        aggregated[group] = {
            "rmse": agg_rmse.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
        }
    return aggregated


def aggregate_avg_group_acc(metrics_list):
    """
    Aggregate per-file average group ACC metrics for each timestep.

    Returns:
      A dict with keys for each group mapping to a dict with:
        - "acc": numpy array of aggregated ACC values per time step.
        - "ci_lower": numpy array of lower bounds (per time step).
        - "ci_upper": numpy array of upper bounds (per time step).
    """
    aggregated = {}
    groups = ["uo", "vo", "so", "thetao"]
    for group in groups:
        # Stack ACC arrays from each file (n_files, n_timesteps)
        acc_array = np.stack(
            [
                file_metric["avg_group_acc"][group]["acc"]
                for file_metric in metrics_list
            ],
            axis=0,
        )
        _, n_timesteps = acc_array.shape
        agg_acc = np.zeros(n_timesteps)
        ci_lower = np.zeros(n_timesteps)
        ci_upper = np.zeros(n_timesteps)
        for t in range(n_timesteps):
            acc_vals = acc_array[:, t]
            agg_acc[t] = np.nanmean(acc_vals)
            # Compute CI using Fisher's z-transform for time step t
            ci_low, ci_up = fisher_ci(acc_vals)
            ci_lower[t] = ci_low
            ci_upper[t] = ci_up
        aggregated[group] = {
            "acc": agg_acc.tolist(),
            "ci_lower": ci_lower.tolist(),
            "ci_upper": ci_upper.tolist(),
        }
    return aggregated


def aggregate_group_rmse(metrics_list):
    """
    Given a list of per-file group MSE metrics this function averages
    the MSE arrays elementwise over files and then takes the square root
    to obtain the aggregated RMSE.

    Returns:
       aggregated_group_rmse : dictionary with keys for each group
        - rmse_avg: (array of shape (n_grid,))
        - mse_all: (n_grid, n_group_features)
    """
    aggregated = {}
    groups = metrics_list[0]["group_mse"].keys()
    # Loop over each group
    for group in groups:
        # Stack mse_avg arrays from all files, each (n_grid,)
        mse_avg_stack = np.stack(
            [
                file_metric["group_mse"][group]["mse_avg"]
                for file_metric in metrics_list
            ],
            axis=0,
        )
        mean_mse_avg = np.nanmean(mse_avg_stack, axis=0)  # (n_grid,)
        rmse_avg = np.sqrt(mean_mse_avg)

        # Stack mse_all arrays from all files, each (n_grid, n_group_features)
        mse_all_stack = np.stack(
            [
                file_metric["group_mse"][group]["mse_all"]
                for file_metric in metrics_list
            ],
            axis=0,
        )
        mean_mse_all = np.nanmean(
            mse_all_stack, axis=0
        )  # (n_grid, n_group_features)

        aggregated[group] = {"rmse_avg": rmse_avg, "mse_all": mean_mse_all}
    return aggregated


def aggregate_group_bias(metrics_list):
    """
    Given a list of group metrics, average the group bias over files.
    """
    groups = metrics_list[0]["group_bias"].keys()
    aggregated_group_bias = {}
    for group in groups:
        sum_bias = None
        count = 0
        for metrics in metrics_list:
            group_bias = metrics["group_bias"]
            if group in group_bias:
                if sum_bias is None:
                    sum_bias = np.array(group_bias[group])
                else:
                    sum_bias += group_bias[group]
                count += 1
        if sum_bias is not None:
            aggregated_group_bias[group] = sum_bias / count
    return aggregated_group_bias


@delayed
def aggregate_directory_metrics(
    input_dir,
    analysis_dir,
    output_dir,
    feature_names,
    sea_mask,
    interior_mask,
    grid_weights,
    climatology,
    day_filter=None,
):
    """
    For a given input directory, compute metrics for each file individually,
    then aggregate the computed metrics by averaging over samples.
    """
    is_med_phy = "med_phy" in output_dir
    if is_med_phy:
        flist = glob.glob(os.path.join(input_dir, "for_data_*.npy"))
    else:
        flist = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
        flist = [os.path.join(input_dir, f) for f in flist]

    # Apply filtering by day-of-week
    if day_filter is not None:
        new_flist = []
        for file in flist:
            basename = os.path.basename(file)
            date_str = basename[-12:-4]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date.isoweekday() != day_filter:
                continue
            new_flist.append(file)
        flist = new_flist

    delayed_metrics = []
    for forecast_file in flist:
        date = os.path.basename(forecast_file)[-12:-4]
        analysis_file = os.path.join(analysis_dir, f"ana_data_{date}.npy")
        delayed_metrics.append(
            delayed(process_file)(
                forecast_file,
                analysis_file,
                feature_names,
                sea_mask,
                interior_mask,
                grid_weights,
                climatology,
                is_med_phy,
            )
        )

    # Compute metrics for each file
    metrics_list = compute(*delayed_metrics)

    # Aggregate per-feature metrics
    aggregated_rmse = aggregate_mse_per_feature(metrics_list, feature_names)
    aggregated_acc = aggregate_acc_per_feature(metrics_list, feature_names)
    aggregated_avg_group_rmse = aggregate_avg_group_rmse(metrics_list)
    aggregated_avg_group_acc = aggregate_avg_group_acc(metrics_list)
    aggregated_group_rmse = aggregate_group_rmse(metrics_list)
    aggregated_group_bias = aggregate_group_bias(metrics_list)

    # Save metrics
    json_path_rmse = os.path.join(output_dir, "rmse.json")
    with open(json_path_rmse, "w", encoding="utf8") as jf:
        json.dump(aggregated_rmse, jf, indent=2)

    json_path_acc = os.path.join(output_dir, "acc.json")
    with open(json_path_acc, "w", encoding="utf8") as jf:
        json.dump(aggregated_acc, jf, indent=2)

    json_path_avg_rmse = os.path.join(output_dir, "avg_group_rmse.json")
    with open(json_path_avg_rmse, "w", encoding="utf8") as jf:
        json.dump(aggregated_avg_group_rmse, jf, indent=2)

    json_path_avg_acc = os.path.join(output_dir, "avg_group_acc.json")
    with open(json_path_avg_acc, "w", encoding="utf8") as jf:
        json.dump(aggregated_avg_group_acc, jf, indent=2)

    np.save(os.path.join(output_dir, "group_rmse.npy"), aggregated_group_rmse)

    np.save(os.path.join(output_dir, "group_bias.npy"), aggregated_group_bias)

    return f"Aggregated metrics saved for {input_dir} in {output_dir}"


def main():
    """
    Calculate forecast metrics.
    """
    parser = argparse.ArgumentParser(
        description="Compute metrics per file and average using dask."
    )
    parser.add_argument(
        "--dataset", type=str, default="mediterranean", help="Dataset name"
    )
    parser.add_argument("--forecast", type=str, help="Forecast directory")
    parser.add_argument(
        "--day",
        type=int,
        default=None,
        help="Filter files by day of week (1-7)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=1, help="Number of processors to use"
    )
    args = parser.parse_args()
    dataset = args.dataset

    feature_names = constants.EXP_PARAM_NAMES_SHORT

    c = load_static_data(args.dataset)
    sea_mask = c["sea_mask"]
    interior_mask = c["sea_mask"] ^ c["border_mask"]
    grid_weights = c["grid_weights"]

    climatology = os.path.join("data", args.dataset, "climatology")

    # Setup Dask distributed client
    client = Client(n_workers=args.n_workers)
    print(f"Dask client created with {args.n_workers} workers.")

    base_dir = os.path.join("data", dataset)
    predictions_dir = os.path.join(base_dir, "predictions")
    samples_test_dir = os.path.join(base_dir, "samples", "test")
    metrics_dir = os.path.join(base_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    if args.day is None:
        subdir_name = args.forecast
    else:
        subdir_name = f"{args.forecast}_{args.day}"

    if args.forecast == "med_phy":
        output_subdir = os.path.join(metrics_dir, subdir_name)
        os.makedirs(output_subdir, exist_ok=True)
        result = aggregate_directory_metrics(
            samples_test_dir,
            samples_test_dir,
            output_subdir,
            feature_names,
            sea_mask,
            interior_mask,
            grid_weights,
            climatology,
            day_filter=args.day,
        ).compute()
        print(result)
    else:
        subdir_path = os.path.join(predictions_dir, args.forecast)
        output_subdir = os.path.join(metrics_dir, subdir_name)
        os.makedirs(output_subdir, exist_ok=True)
        result = aggregate_directory_metrics(
            subdir_path,
            samples_test_dir,
            output_subdir,
            feature_names,
            sea_mask,
            interior_mask,
            grid_weights,
            climatology,
            day_filter=args.day,
        ).compute()
        print(result)

    client.close()


if __name__ == "__main__":
    main()
