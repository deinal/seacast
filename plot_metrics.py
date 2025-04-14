# Standard library
import argparse
import json
import os
import re
from datetime import datetime, timedelta

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import FuncFormatter, MultipleLocator

# First-party
from neural_lam import constants


def read_npy_to_xarray(file_path, sea_mask, init=False):
    """Reads forecast file and reconstructs the full grid using a sea mask."""
    # Get start date from filename
    filename = os.path.basename(file_path)
    date_str = filename.split(".")[0]
    start_date = datetime.strptime(date_str[-8:], "%Y%m%d")
    if init:
        start_date -= timedelta(days=2)

    # Load forecast
    data = np.load(file_path)  # (time, n_grid, n_features)
    n_time, _, _ = data.shape

    # Full horizontal grid dimensions
    n_lat, n_lon = constants.GRID_SHAPE

    # Get the indices (row, col) belonging to the sea
    surface_mask = sea_mask.isel(depth=0).values
    lat_idx, lon_idx = np.where(surface_mask == 1)

    data_vars = {}
    feature_idx = 0

    # Loop over the parameters
    for param_name, param_short, has_depth, unit, colormap, diverging in zip(
        constants.PARAM_NAMES,
        constants.PARAM_NAMES_SHORT,
        constants.LEVELS,
        constants.PARAM_UNITS,
        constants.PARAM_COLORMAPS,
        constants.DIVERGING,
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

    # Add metadata attributes for each variable
    for param_short, param_name, unit, colormap, diverging in zip(
        constants.PARAM_NAMES_SHORT,
        constants.PARAM_NAMES,
        constants.PARAM_UNITS,
        constants.PARAM_COLORMAPS,
        constants.DIVERGING,
    ):
        ds[param_short].attrs = {
            "description": param_name,
            "unit": unit,
            "colormap": colormap,
            "diverging": diverging,
        }

    return ds


def plot_forecast(
    ds, analysis_ds, param, lead_indices, depth_index=2, model="seacast", fs=18
):
    """
    Plot a stack of forecast fields and bias to analysis.
    """
    unit = ds[param].attrs.get("unit", "")
    cmap = ds[param].attrs.get("colormap", "viridis")
    diverging = ds[param].attrs.get("diverging", False)

    lons = ds["longitude"].values
    lats = ds["latitude"].values
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]

    fc_vals = []
    diff_vals = []
    for idx in lead_indices:
        fc = ds[param].isel(time=idx)
        if "depth" in fc.dims:
            fc = fc.isel(depth=depth_index)
        fc_vals.append(fc.values)

        ana = analysis_ds[param].sel(time=ds.time[idx])
        if "depth" in ana.dims:
            ana = ana.isel(depth=depth_index)
        diff = fc - ana
        diff_vals.append(diff.values)

    fc_all = np.stack(fc_vals)
    diff_all = np.stack(diff_vals)
    if diverging:
        fc_abs_max = np.nanmax(np.abs(fc_all))
        fc_vmin, fc_vmax = -fc_abs_max, fc_abs_max
    else:
        fc_vmin, fc_vmax = np.nanmin(fc_all), np.nanmax(fc_all)

    diff_abs_max = np.nanmax(np.abs(diff_all))
    diff_vmin, diff_vmax = -diff_abs_max, diff_abs_max

    n_rows = len(lead_indices)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(18, 3.1 * n_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(lead_indices):
        # Forecast panel left
        fc_data = ds[param].isel(time=idx)
        if "depth" in fc_data.dims:
            fc_data = fc_data.isel(depth=depth_index)
        ax_fc = axes[row, 0]
        im_fc = ax_fc.imshow(
            fc_data,
            origin="lower",
            cmap=cmap,
            extent=extent,
            vmin=fc_vmin,
            vmax=fc_vmax,
        )
        ax_fc.set_ylabel(
            f"{idx+1}d", fontsize=fs, rotation=0, labelpad=7, ha="right"
        )
        if row == 0:
            ax_fc.set_title("SeaCast", fontsize=fs)
        ax_fc.coastlines(resolution="10m", linewidth=1)
        ax_fc.add_feature(cfeature.LAND, facecolor="whitesmoke")
        cbar_fc = fig.colorbar(
            im_fc,
            ax=ax_fc,
            orientation="vertical",
            shrink=0.8,
            aspect=22,
            pad=0.02,
        )
        cbar_fc.ax.set_ylabel(unit, fontsize=fs - 4)
        cbar_fc.ax.tick_params(labelsize=fs - 4)

        # Bias panel right
        ana_data = analysis_ds[param].sel(time=ds.time[idx])
        if "depth" in ana_data.dims:
            ana_data = ana_data.isel(depth=depth_index)
        diff_data = fc_data - ana_data
        ax_diff = axes[row, 1]
        im_diff = ax_diff.imshow(
            diff_data,
            origin="lower",
            cmap="RdBu_r",
            extent=extent,
            vmin=diff_vmin,
            vmax=diff_vmax,
        )
        if row == 0:
            ax_diff.set_title("Bias", fontsize=fs)
        ax_diff.coastlines(resolution="10m", linewidth=1)
        ax_diff.add_feature(cfeature.LAND, facecolor="whitesmoke")
        cbar_diff = fig.colorbar(
            im_diff,
            ax=ax_diff,
            orientation="vertical",
            shrink=0.8,
            aspect=22,
            pad=0.02,
        )
        cbar_diff.ax.set_ylabel(unit)
        cbar_diff.ax.set_ylabel(unit, fontsize=fs - 4)
        cbar_diff.ax.tick_params(labelsize=fs - 4)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    save_dir = os.path.join("figures", "metrics", "forecast")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model}_{param}.png")
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(
        os.path.join("figures", "results", f"{model}_{param}.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_forecast_vertical(
    ds,
    analysis_ds,
    param,
    lead_indices,
    direction="zonal",
    suffix="forecast",
    fs=14,
):
    """
    Plot average over zonal or meridional direction.
    """
    unit = ds[param].attrs.get("unit", "")
    cmap = ds[param].attrs.get("colormap", "viridis")
    cmap.set_bad(color="lightgray")
    cmap_bias = plt.get_cmap("RdBu_r").copy()
    cmap_bias.set_bad("lightgray")
    diverging = ds[param].attrs.get("diverging", False)

    if direction == "zonal":
        horiz_dim = "longitude"
        x_coord_name = "latitude"
        xlabel = "Latitude (°)"
        fig_width = 9
        locator = MultipleLocator(3)
    elif direction == "meridional":
        horiz_dim = "latitude"
        x_coord_name = "longitude"
        xlabel = "Longitude (°)"
        fig_width = 19
        locator = MultipleLocator(5)
    else:
        raise ValueError("direction must be 'zonal' or 'meridional'")

    x_coord = ds[x_coord_name].values

    depth_vals = ds["depth"].values
    n_depths = len(depth_vals)
    y_positions = np.arange(n_depths)
    yticklabels = [str(int(d)) for d in depth_vals]

    # Precompute forecast and bias sections for each lead time
    fc_sections = []
    bias_sections = []
    for idx in lead_indices:
        fc = ds[param].isel(time=idx)
        ana = analysis_ds[param].sel(time=ds.time[idx])
        # Average horizontally
        fc_sec = fc.mean(dim=horiz_dim, skipna=True)
        ana_sec = ana.mean(dim=horiz_dim, skipna=True)
        # Ensure depth is the first dimension
        if "depth" in fc_sec.dims and x_coord_name in fc_sec.dims:
            fc_sec = fc_sec.transpose("depth", x_coord_name)
            ana_sec = ana_sec.transpose("depth", x_coord_name)
        fc_sections.append(fc_sec.values)  # (n_depths, len(x_coord))
        bias_sections.append(fc_sec.values - ana_sec.values)

    # Determine global color limits
    fc_all = np.concatenate([s.flatten() for s in fc_sections])
    bias_all = np.concatenate([s.flatten() for s in bias_sections])
    if diverging:
        fc_abs_max = np.nanmax(np.abs(fc_all))
        fc_vmin, fc_vmax = -fc_abs_max, fc_abs_max
    else:
        fc_vmin, fc_vmax = np.nanmin(fc_all), np.nanmax(fc_all)
    bias_abs_max = np.nanmax(np.abs(bias_all))
    bias_vmin, bias_vmax = -bias_abs_max, bias_abs_max

    n_rows = len(lead_indices)
    n_cols = 2

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(fig_width, 4 * n_rows),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    extent = [x_coord.min(), x_coord.max(), n_depths - 1, 0]

    for i, idx in enumerate(lead_indices):
        ax_fc = axes[i, 0]
        im_fc = ax_fc.imshow(
            fc_sections[i],
            origin="upper",
            cmap=cmap,
            vmin=fc_vmin,
            vmax=fc_vmax,
            extent=extent,
        )
        ax_fc.set_title(
            f"Forecasted {constants.PLOT_NAMES_MAP[param]}, day {idx+1}",
            fontsize=fs,
        )
        ax_fc.set_ylabel("Depth (m)", fontsize=fs)
        ax_fc.xaxis.set_major_locator(locator)
        if i == n_rows - 1:
            ax_fc.set_xlabel(xlabel, fontsize=fs)
        else:
            ax_fc.set_xticklabels([])

        ax_fc.set_yticks(y_positions)
        ax_fc.set_yticklabels(yticklabels)
        cbar_fc = fig.colorbar(
            im_fc,
            ax=ax_fc,
            orientation="vertical",
            shrink=0.855,
            aspect=25,
            pad=0.02,
        )
        cbar_fc.ax.set_ylabel(unit)

        # Bias panel
        ax_bias = axes[i, 1]
        im_bias = ax_bias.imshow(
            bias_sections[i],
            origin="upper",
            cmap=cmap_bias,
            vmin=bias_vmin,
            vmax=bias_vmax,
            extent=extent,
        )
        ax_bias.set_title(
            f"Bias of {constants.PLOT_NAMES_MAP[param]}, day {idx+1}",
            fontsize=fs,
        )
        ax_bias.xaxis.set_major_locator(locator)
        if i == n_rows - 1:
            ax_bias.set_xlabel(xlabel, fontsize=fs)
        else:
            ax_bias.set_xticklabels([])
        ax_bias.set_yticks(y_positions)
        ax_bias.set_yticklabels(yticklabels)
        cbar_bias = fig.colorbar(
            im_bias,
            ax=ax_bias,
            orientation="vertical",
            shrink=0.855,
            aspect=25,
            pad=0.02,
        )
        cbar_bias.ax.set_ylabel(unit)

    formatter = FuncFormatter(lambda x, pos: f"{x:.0f}")
    for ax in axes[-1, :]:
        ax.xaxis.set_major_formatter(formatter)

    save_dir = os.path.join("figures", "metrics", "forecast_vertical")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{param}_{direction}_{suffix}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_group_bias(var, model, sea_mask, dataset="mediterranean"):
    """
    Plot group bias for a given variable.
    """
    bias_path = os.path.join(
        "data", dataset, "metrics", model, "group_bias.npy"
    )
    data = np.load(bias_path, allow_pickle=True).item()
    if var == "zos":
        bias = data[var]
    else:
        bias = data[var]  # (4, n_grid)

    unit = constants.PARAM_UNITS[constants.PARAM_NAMES_SHORT.index(var)]

    sea_surface_mask = sea_mask.isel(depth=0)  # (lat, lon)
    n_time, _ = bias.shape

    # Reconstruct full grid
    full_grid_bias = np.full((n_time, *sea_surface_mask.shape), np.nan)
    for t in range(n_time):
        full_grid_bias[t][sea_surface_mask == 1] = bias[t]

    lon = sea_mask.longitude.values
    lat = sea_mask.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    extent = constants.GRID_LIMITS

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    vabs = np.nanmax(np.abs(full_grid_bias))
    if var == "so":
        vabs = np.percentile(
            np.abs(full_grid_bias[~np.isnan(full_grid_bias)]), 99.7
        )

    lead_days = [1, 5, 10, 15]

    for i, ax in enumerate(axes):
        pcm = ax.pcolormesh(
            lon2d,
            lat2d,
            full_grid_bias[i],
            cmap="RdBu_r",
            vmin=-vabs,
            vmax=vabs,
            shading="auto",
        )
        ax.set_title(f"{var} t={lead_days[i]}")
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.coastlines(resolution="10m", linewidth=0.5)

        ax.set_xticks(
            np.linspace(extent[0], extent[1], 5), crs=ccrs.PlateCarree()
        )
        ax.set_yticks(
            np.linspace(extent[2], extent[3], 5), crs=ccrs.PlateCarree()
        )
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(3))
        ax.tick_params(labelsize=8)

        if i % 2 == 0:
            ax.set_ylabel("Latitude (°)", fontsize=9)
        if i // 2 == 1:
            ax.set_xlabel("Longitude (°)", fontsize=9)

    # Shared colorbar
    if var == "so":
        extend = "both"
    else:
        extend = "neither"
    cbar = fig.colorbar(
        pcm,
        ax=axes,
        orientation="vertical",
        shrink=0.6,
        extend=extend,
        pad=0.02,
    )
    cbar.set_label(f"Bias ({unit})", fontsize=10)

    save_dir = os.path.join("figures", "metrics", "group_bias")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model}_{var}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def load_metric_std_steps(json_path, n_steps=15, metric="rmse"):
    """
    Loads a JSON file and returns
      - metric_matrix (n_steps, d_features)
      - ci_data, dict with ci_lower and ci_upper, each (n_steps, d_features)
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    param_names = constants.EXP_PARAM_NAMES_SHORT
    n_features = len(param_names)
    metric_matrix = np.empty((n_steps, n_features))
    ci_lower_matrix = np.empty((n_steps, n_features))
    ci_upper_matrix = np.empty((n_steps, n_features))
    metric_matrix[:] = np.nan
    ci_lower_matrix[:] = np.nan
    ci_upper_matrix[:] = np.nan

    for step in range(n_steps):
        step_key = str(step)
        if step_key in data:
            cycle_data = data[step_key]
            for j, param in enumerate(param_names):
                if param in cycle_data:
                    metric_matrix[step, j] = cycle_data[param][metric]
                    ci_lower_matrix[step, j] = cycle_data[param]["ci_lower"]
                    ci_upper_matrix[step, j] = cycle_data[param]["ci_upper"]
                else:
                    metric_matrix[step, j] = np.nan
                    ci_lower_matrix[step, j] = np.nan
                    ci_upper_matrix[step, j] = np.nan
        else:
            metric_matrix[step, :] = np.nan
            ci_lower_matrix[step, :] = np.nan
            ci_upper_matrix[step, :] = np.nan
    ci_data = {"ci_lower": ci_lower_matrix, "ci_upper": ci_upper_matrix}
    return metric_matrix, ci_data


def plot_metric_by_depth(
    variable,
    output_dir,
    metric_std_all,
    model_labels,
    metric="rmse",
    n_steps=15,
    depths=None,
    fs=12,
    fill_between=False,
    color_map=None,
    legend_ncol=None,
):
    """
    Create one figure for a given variable with a grid of subplots.
    Each subplot shows the chosen metric vs. lead time at a depth level.
    """

    os.makedirs(output_dir, exist_ok=True)

    param_names = constants.EXP_PARAM_NAMES_SHORT
    var_indices = []
    depths_list = []
    for i, pname in enumerate(param_names):
        if pname.startswith(variable + "_"):
            d_val = int(pname.split("_")[1])
            var_indices.append(i)
            depths_list.append(d_val)

    unit = constants.PARAM_UNITS[constants.PARAM_NAMES_SHORT.index(variable)]

    # Sort by depth (ascending)
    sorted_pairs = sorted(zip(depths_list, var_indices), key=lambda x: x[0])
    sorted_depths, sorted_var_indices = zip(*sorted_pairs)

    # Use only every other depth
    sorted_depths = sorted_depths[::2]
    sorted_var_indices = sorted_var_indices[::2]

    if depths is None:
        depths = list(sorted_depths)

    n_subplots = len(sorted_depths)
    ncols = 3
    nrows = int(np.ceil(n_subplots / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3, nrows * 2.6),
        sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(hspace=0.1)
    axes = axes.flatten()
    x = np.arange(1, n_steps + 1)

    # One subplot per depth
    for i in range(n_subplots):
        ax = axes[i]
        row_index = i // ncols
        col_index = i % ncols
        for model, (metric_matrix, ci_data) in metric_std_all.items():
            ax.axvline(x=10, color="lightgray", ls="--", zorder=0)
            # Get metric values for all lead times at this depth
            y = metric_matrix[:, sorted_var_indices[i]]
            if fill_between:
                ci_lower = ci_data["ci_lower"][:, sorted_var_indices[i]]
                ci_upper = ci_data["ci_upper"][:, sorted_var_indices[i]]
                (line,) = ax.plot(
                    x,
                    y,
                    linewidth=2,
                    linestyle="-",
                    label=model_labels.get(model, model),
                    color=color_map[model],
                )
                ax.fill_between(
                    x, ci_lower, ci_upper, color=line.get_color(), alpha=0.3
                )
            else:
                ax.plot(
                    x,
                    y,
                    linewidth=2,
                    linestyle="-",
                    label=model_labels.get(model, model),
                    color=color_map[model],
                )

        ax.set_xticks(np.arange(1, n_steps + 1), minor=True)
        major_ticks = range(1, 16, 2)
        ax.set_xticks(major_ticks, minor=False)
        ax.set_xticklabels(major_ticks)
        ax.tick_params(axis="both", which="major", labelsize=fs)

        letter = chr(97 + i)
        var_name = constants.PLOT_NAMES_MAP[variable]
        ax.set_title(
            f"{letter}) {var_name}, {sorted_depths[i]}m",
            fontsize=fs,
        )

        if row_index == nrows - 1:
            ax.set_xlabel("Lead time (days)", fontsize=fs)
        if col_index == 0:
            if metric == "rmse":
                ax.set_ylabel(f"{metric.upper()} ({unit})", fontsize=fs)
            else:
                ax.set_ylabel(metric.upper(), fontsize=fs)

    # Create a common legend
    handles, labels = axes[0].get_legend_handles_labels()
    if legend_ncol is None:
        legend_ncol = len(metric_std_all)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.02),
        ncol=legend_ncol,
        fontsize=fs,
        frameon=False,
    )

    save_path = os.path.join(output_dir, f"{variable}_{metric}.png")
    fig.savefig(save_path, bbox_inches="tight")
    pdf_name = f"{variable}_{metric}_{output_dir.split('_')[-1]}.pdf"
    fig.savefig(
        os.path.join("figures", "results", pdf_name), bbox_inches="tight"
    )
    plt.close(fig)


def plot_metric_single(
    variable,
    metric_std_all,
    model_labels,
    metric="rmse",
    n_steps=15,
    fs=12,
    output_dir="zos_metric_plots",
    fill_between=False,
    color_map=None,
    legend_ncol=None,
):
    """
    Plot the chosen metric vs. lead time.
    """

    os.makedirs(output_dir, exist_ok=True)

    param_names = constants.EXP_PARAM_NAMES_SHORT
    var_idx = param_names.index(variable)

    unit = constants.EXP_PARAM_UNITS[var_idx]

    x = np.arange(1, n_steps + 1)
    plt.figure(figsize=(6, 5))
    plt.axvline(x=10, color="lightgray", ls="--", zorder=0)
    for model, (metric_matrix, ci_data) in metric_std_all.items():
        y = metric_matrix[:, var_idx]
        if fill_between:
            ci_lower = ci_data["ci_lower"][:, var_idx]
            ci_upper = ci_data["ci_upper"][:, var_idx]
            (line,) = plt.plot(
                x,
                y,
                linewidth=2,
                linestyle="-",
                label=model_labels.get(model, model),
                color=color_map[model],
            )
            plt.fill_between(
                x, ci_lower, ci_upper, color=line.get_color(), alpha=0.3
            )
        else:
            plt.plot(
                x,
                y,
                linewidth=2,
                linestyle="-",
                label=model_labels.get(model, model),
                color=color_map[model],
            )
    plt.xlabel("Lead time (days)", fontsize=fs)
    if metric == "rmse":
        plt.ylabel(f"{metric.upper()} ({unit})", fontsize=fs)
    else:
        plt.ylabel(metric.upper(), fontsize=fs)

    ax = plt.gca()
    ax.set_xticks(np.arange(1, n_steps + 1), minor=True)
    major_ticks = range(1, 16, 2)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_xticklabels(major_ticks)
    ax.tick_params(axis="both", which="major", labelsize=fs)

    handles, labels = ax.get_legend_handles_labels()
    if legend_ncol is None:
        legend_ncol = len(metric_std_all)
    plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.15),
        ncol=legend_ncol,
        fontsize=fs,
        frameon=False,
    )
    save_path = os.path.join(output_dir, f"{variable}_{metric}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_avg_group_metric(
    agg_group_metrics_dict,
    metric="rmse",
    n_steps=15,
    fs=12,
    output_dir="avg_group_metric",
    fill_between=True,
    model_labels=None,
    color_map=None,
    legend_ncol=None,
):
    """
    Plot aggregated average group metrics vs. lead time.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(
        2, 2, figsize=(2 * 3.2, 2 * 2.8), constrained_layout=True
    )
    fig.set_constrained_layout_pads(hspace=0.1)
    axes = axes.flatten()
    x = np.arange(1, n_steps + 1)

    groups = ["uo", "vo", "so", "thetao"]
    # Loop over the four groups
    for i, group in enumerate(groups):
        ax = axes[i]
        ax.axvline(x=10, color="lightgray", ls="--", zorder=0)
        unit = constants.PARAM_UNITS[constants.PARAM_NAMES_SHORT.index(group)]
        for model, group_data in agg_group_metrics_dict.items():
            y = np.array(group_data[group][metric])
            ci_lower = np.array(group_data[group]["ci_lower"])
            ci_upper = np.array(group_data[group]["ci_upper"])
            steps = np.arange(1, len(y) + 1)
            (line,) = ax.plot(
                steps,
                y,
                linewidth=2,
                linestyle="-",
                label=model_labels.get(model, model) if model_labels else model,
                color=color_map[model],
            )
            if fill_between:
                ax.fill_between(
                    steps, ci_lower, ci_upper, color=line.get_color(), alpha=0.3
                )
        ax.set_title(
            f"{chr(97+i)}) {constants.PLOT_NAMES_MAP[group]}", fontsize=fs
        )

        ax.set_xticks(x, minor=True)
        major_ticks = range(1, 16, 2)
        ax.set_xticks(major_ticks, minor=False)
        ax.set_xticklabels(major_ticks)
        ax.set_xlabel("Lead time (days)", fontsize=fs)
        ax.tick_params(axis="both", which="major", labelsize=fs)

        if metric == "rmse":
            ax.set_ylabel(f"{metric.upper()} ({unit})", fontsize=fs)
        else:
            ax.set_ylabel(metric.upper(), fontsize=fs)
        ax.tick_params(labelsize=fs)

    # Create a common legend across all subplots.
    handles, labels = ax.get_legend_handles_labels()
    if legend_ncol is None:
        legend_ncol = len(agg_group_metrics_dict)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.05),
        ncol=legend_ncol,
        fontsize=fs,
        frameon=False,
    )

    save_path = os.path.join(output_dir, f"{metric}.png")
    fig.savefig(save_path, bbox_inches="tight")
    save_path = os.path.join("figures", "results", f"{metric}_avg.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_norm_rmse_diff_by_depth(
    variable,
    baseline_json,
    model_jsons,
    baseline_label,
    model_labels,
    output_dir,
    n_steps=15,
    fs=12,
    fill_between=False,
    color_map=None,
    legend_ncol=None,
):
    """
    Plot normalized RMSE diff computed as
    (model_rmse - baseline_rmse) / baseline_rmse.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Extract indices and depth values for the chosen variable
    param_names = constants.EXP_PARAM_NAMES_SHORT
    var_indices = []
    depths_list = []
    for i, pname in enumerate(param_names):
        if pname.startswith(variable + "_"):
            d_val = int(pname.split("_")[1])
            var_indices.append(i)
            depths_list.append(d_val)

    # Sort by depth (ascending)
    sorted_pairs = sorted(zip(depths_list, var_indices), key=lambda x: x[0])
    sorted_depths, sorted_var_indices = zip(*sorted_pairs)

    # Use only every other depth
    sorted_depths = sorted_depths[::2]
    sorted_var_indices = sorted_var_indices[::2]

    n_subplots = len(sorted_depths)
    ncols = 3
    nrows = int(np.ceil(n_subplots / ncols))

    # Create the figure grid
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 3, nrows * 2.7),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(hspace=0.1)
    axes = axes.flatten()
    x = np.arange(1, n_steps + 1)

    # Load baseline metrics for the variable
    baseline_metric, baseline_ci = load_metric_std_steps(
        baseline_json, n_steps=n_steps, metric="rmse"
    )

    # Loop over each depth
    for i in range(n_subplots):
        ax = axes[i]
        ax.axvline(x=10, color="lightgray", ls="--", zorder=0)

        # Extract the baseline values for this depth
        baseline_vals = baseline_metric[:, sorted_var_indices[i]]
        if fill_between:
            baseline_ci_lower = baseline_ci["ci_lower"][
                :, sorted_var_indices[i]
            ]
            baseline_ci_upper = baseline_ci["ci_upper"][
                :, sorted_var_indices[i]
            ]

        # For each model, compute the normalized difference
        for model, json_path in model_jsons.items():
            model_metric, model_ci = load_metric_std_steps(
                json_path, n_steps=n_steps, metric="rmse"
            )
            model_vals = model_metric[:, sorted_var_indices[i]]
            # Compute normalized difference
            norm_diff = (model_vals - baseline_vals) / baseline_vals
            if fill_between:
                model_ci_lower = model_ci["ci_lower"][:, sorted_var_indices[i]]
                model_ci_upper = model_ci["ci_upper"][:, sorted_var_indices[i]]
                # Compute normalized CI bounds conservatively
                norm_diff_lower = (
                    model_ci_lower - baseline_ci_upper
                ) / baseline_vals
                norm_diff_upper = (
                    model_ci_upper - baseline_ci_lower
                ) / baseline_vals
                (line,) = ax.plot(
                    x,
                    100 * norm_diff,
                    linewidth=2,
                    linestyle="-",
                    label=model_labels.get(model, model),
                    color=color_map[model],
                )
                ax.fill_between(
                    x,
                    100 * norm_diff_lower,
                    100 * norm_diff_upper,
                    color=line.get_color(),
                    alpha=0.3,
                )
            else:
                ax.plot(
                    x,
                    100 * norm_diff,
                    linewidth=2,
                    linestyle="-",
                    label=model_labels.get(model, model),
                    color=color_map[model],
                )

        # Plot the baseline horizontal line
        ax.plot(
            x,
            np.zeros_like(x),
            linestyle="-",
            linewidth=2,
            label=model_labels[baseline_label],
            color=color_map[baseline_label],
            zorder=0,
        )

        ax.set_xticks(np.arange(1, n_steps + 1), minor=True)
        major_ticks = range(1, 16, 2)
        ax.set_xticks(major_ticks, minor=False)
        ax.set_xticklabels(major_ticks)
        ax.tick_params(axis="both", which="major", labelsize=fs)

        letter = chr(97 + i)
        var_name = constants.PLOT_NAMES_MAP[variable]
        ax.set_title(
            f"{letter}) {var_name}, {sorted_depths[i]}m",
            fontsize=fs,
        )

        if (i // ncols) == nrows - 1:
            ax.set_xlabel("Lead time (days)", fontsize=fs)
        if (i % ncols) == 0:
            ax.set_ylabel("Norm. RMSE diff. (%)", fontsize=fs)

    handles, labels = axes[0].get_legend_handles_labels()
    if legend_ncol is None:
        legend_ncol = len(model_jsons) + 1
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.02),
        ncol=legend_ncol,
        fontsize=fs,
        frameon=False,
    )
    handles, labels = axes[0].get_legend_handles_labels()

    save_path = os.path.join(output_dir, f"{variable}_norm_rmse_diff.png")
    fig.savefig(save_path, bbox_inches="tight")
    pdf_name = f"{variable}_norm_rmse_diff_{output_dir.split('_')[-1]}.pdf"
    fig.savefig(
        os.path.join("figures", "results", pdf_name), bbox_inches="tight"
    )
    plt.close(fig)


def plot_norm_rmse_diff_single(
    variable,
    baseline_json,
    model_jsons,
    baseline_label,
    model_labels,
    output_dir,
    n_steps=15,
    fs=12,
    fill_between=False,
    color_map=None,
    legend_ncol=None,
):
    """
    Plot normalized RMSE diff computed as
    (model_rmse - baseline_rmse) / baseline_rmse.
    """

    os.makedirs(output_dir, exist_ok=True)

    param_names = constants.EXP_PARAM_NAMES_SHORT
    var_index = param_names.index(variable)

    x = np.arange(1, n_steps + 1)

    # Load baseline metrics
    baseline_metric, baseline_ci = load_metric_std_steps(
        baseline_json, n_steps=n_steps, metric="rmse"
    )
    baseline_vals = baseline_metric[:, var_index]
    if fill_between:
        baseline_ci_lower = baseline_ci["ci_lower"][:, var_index]
        baseline_ci_upper = baseline_ci["ci_upper"][:, var_index]

    plt.figure(figsize=(6, 5))
    plt.axvline(x=10, color="lightgray", ls="--", zorder=0)
    # Loop over each model
    for model, json_path in model_jsons.items():
        model_metric, model_ci = load_metric_std_steps(
            json_path, n_steps=n_steps, metric="rmse"
        )
        model_vals = model_metric[:, var_index]
        norm_diff = (model_vals - baseline_vals) / baseline_vals
        if fill_between:
            model_ci_lower = model_ci["ci_lower"][:, var_index]
            model_ci_upper = model_ci["ci_upper"][:, var_index]
            norm_diff_lower = (
                model_ci_lower - baseline_ci_upper
            ) / baseline_vals
            norm_diff_upper = (
                model_ci_upper - baseline_ci_lower
            ) / baseline_vals
            (line,) = plt.plot(
                x,
                100 * norm_diff,
                linewidth=2,
                linestyle="-",
                label=model_labels.get(model, model),
                color=color_map[model],
            )
            plt.fill_between(
                x,
                100 * norm_diff_lower,
                100 * norm_diff_upper,
                color=line.get_color(),
                alpha=0.3,
            )
        else:
            plt.plot(
                x,
                100 * norm_diff,
                linewidth=2,
                linestyle="-",
                label=model_labels.get(model, model),
                color=color_map[model],
            )
    # Plot the baseline horizontal line
    plt.plot(
        x,
        np.zeros_like(x),
        color=color_map[baseline_label],
        linestyle="-",
        linewidth=2,
        label=model_labels[baseline_label],
    )
    plt.xlabel("Lead time (days)", fontsize=fs)
    plt.ylabel("Norm. RMSE diff. (%)", fontsize=fs)

    ax = plt.gca()
    ax.set_xticks(np.arange(1, n_steps + 1), minor=True)
    major_ticks = range(1, 16, 2)
    ax.set_xticks(major_ticks, minor=False)
    ax.set_xticklabels(major_ticks)
    ax.tick_params(axis="both", which="major", labelsize=fs)

    handles, labels = ax.get_legend_handles_labels()
    if legend_ncol is None:
        legend_ncol = len(model_jsons) + 1
    plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.15),
        ncol=legend_ncol,
        fontsize=fs,
        frameon=False,
    )
    save_path = os.path.join(output_dir, f"{variable}_norm_rmse_diff.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def load_rmse_steps(json_path, n_steps=10):
    """
    Loads an rmse.json file and returns an array of shape (n_steps, n_features).
    Only steps from 0 to n_steps-1 are loaded.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    param_names = constants.EXP_PARAM_NAMES_SHORT
    n_features = len(param_names)
    rmse_matrix = np.empty((n_steps, n_features))
    rmse_matrix[:] = np.nan

    for step in range(n_steps):
        step_key = str(step)
        if step_key in data:
            cycle_data = data[step_key]
            for j, param in enumerate(param_names):
                if param in cycle_data:
                    rmse_matrix[step, j] = cycle_data[param]["rmse"]
                else:
                    rmse_matrix[step, j] = np.nan
        else:
            rmse_matrix[step, :] = np.nan
    return rmse_matrix


def plot_scorecard(norm_rmse_diff, depths, base_save_name):
    """
    Plot scorecard of normalized RMSE differences.
    """
    param_names = constants.EXP_PARAM_NAMES_SHORT
    n_steps = norm_rmse_diff.shape[0]

    norm_rmse_diff_by_variable = {
        "uo": np.zeros((len(depths), n_steps)),
        "vo": np.zeros((len(depths), n_steps)),
        "so": np.zeros((len(depths), n_steps)),
        "thetao": np.zeros((len(depths), n_steps)),
    }

    # Populate the matrices for each variable
    for idx, param in enumerate(param_names):
        match = re.match(r"(uo|vo|so|thetao)_(\d+)", param)
        if match:
            variable = match.group(1)
            depth = int(match.group(2))
            if variable in norm_rmse_diff_by_variable and depth in depths:
                depth_index = depths.index(depth)
                norm_rmse_diff_by_variable[variable][depth_index, :] = (
                    norm_rmse_diff[:, idx]
                )

    # Compute a global limit over all values for a common vmin/vmax
    global_limit = np.max(
        [
            np.max(np.abs(matrix))
            for matrix in norm_rmse_diff_by_variable.values()
        ]
    )

    # Create a 2x2 figure with shared x and y axes
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 4.3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ims = []  # List to store the image objects

    # Loop over each variable to create subplots
    for i, (ax, (variable, norm_rmse_diff_matrix)) in enumerate(
        zip(axes.ravel(), norm_rmse_diff_by_variable.items())
    ):
        im = ax.imshow(
            norm_rmse_diff_matrix,
            cmap="bwr",
            aspect="auto",
            vmin=-global_limit,
            vmax=global_limit,
        )
        ims.append(im)
        ax.set_title(
            f"{chr(97+i)}) {constants.PLOT_NAMES_MAP[variable]}", fontsize=14
        )
        ax.set_xlabel("Lead time (days)", fontsize=12)
        if i == 0:
            ax.set_ylabel("Depth (m)", fontsize=12)
        if n_steps > 10:
            major_ticks = list(range(0, n_steps, 2))
            minor_ticks = list(range(n_steps))
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_xticks(major_ticks)
            ax.set_xticklabels([str(i + 1) for i in major_ticks])
            ax.tick_params(axis="both", which="major", labelsize=11)
            ax.tick_params(axis="x", which="minor", length=3)
        else:
            ax.set_xticks(range(n_steps))
            ax.set_xticklabels(range(1, n_steps + 1))
            ax.tick_params(axis="both", labelsize=11)
        ax.set_yticks(np.arange(len(depths)))
        ax.set_yticklabels(depths)

    cbar = fig.colorbar(
        ims[0],
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        shrink=0.3,
        aspect=30,
    )
    cbar.set_label("Norm. RMSE diff. (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=11)

    # Save the figure
    save_dir = os.path.join("figures", "metrics", "scorecards")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, base_save_name)
    fig.savefig(f"{save_path}.png", bbox_inches="tight")
    save_path = os.path.join("figures", "results", base_save_name)
    fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rmse_vs_depth(
    rmse_std_all,
    variable,
    model_labels,
    n_steps=15,
    fs=12,
    output_dir="rmse_depths",
    color_map=None,
    legend_ncol=None,
):
    """
    Plots RMSE (x-axis) vs depth (y-axis) for every other lead time.
    """
    os.makedirs(output_dir, exist_ok=True)

    param_names = constants.EXP_PARAM_NAMES_SHORT
    var_indices = []
    depths = []
    for i, pname in enumerate(param_names):
        if pname.startswith(variable + "_"):
            d_val = int(pname.split("_")[1])
            var_indices.append(i)
            depths.append(d_val)

    unit = constants.PARAM_UNITS[constants.PARAM_NAMES_SHORT.index(variable)]

    # Sort by depth (ascending)
    sorted_pairs = sorted(zip(depths, var_indices), key=lambda x: x[0])
    depths, var_indices = zip(*sorted_pairs)

    ncols = 4
    nrows = 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 2.5, nrows * 2.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.flatten()

    # Loop over lead times
    selected_lead_times = list(range(0, n_steps, 2))
    for idx, t in enumerate(selected_lead_times):
        ax = axes[idx]
        for model, (rmse_matrix, _) in rmse_std_all.items():
            rmse_depth = rmse_matrix[t, list(var_indices)]
            ax.plot(
                rmse_depth,
                depths,
                label=model_labels[model],
                color=color_map[model],
                linewidth=2,
            )
        ax.set_title(
            f"{chr(97+idx)}) {constants.PLOT_NAMES_MAP[variable]}, {t+1}d",
            fontsize=fs,
        )
        if idx // ncols == nrows - 1:
            ax.set_xlabel(f"RMSE ({unit})", fontsize=fs)
        if idx % ncols == 0:
            ax.set_ylabel("Depth (m)", fontsize=fs)
        ax.set_ylim(200, -7)
        ax.tick_params(axis="both", which="major", labelsize=fs)

    handles, labels = axes[0].get_legend_handles_labels()
    if legend_ncol is None:
        legend_ncol = len(rmse_std_all)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.02),
        ncol=legend_ncol,
        fontsize=fs,
        frameon=False,
    )
    save_path = os.path.join(output_dir, f"{variable}_rmse_depth.png")
    fig.savefig(save_path, bbox_inches="tight")
    save_path = os.path.join("figures", "results", f"{variable}_rmse_depth.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_rmse_diff(
    model,
    baseline,
    sea_mask,
    dataset="mediterranean",
    output_dir="figures/metrics/spatial_rmse_diff",
    fs=14,
):
    """
    Load aggregated group RMSE files for model and baseline and plot
    for each group the difference in spatially averaged RMSE.
    """
    model_file = os.path.join(
        "data", dataset, "metrics", model, "group_rmse.npy"
    )
    baseline_file = os.path.join(
        "data", dataset, "metrics", baseline, "group_rmse.npy"
    )
    model_data = np.load(model_file, allow_pickle=True).item()
    baseline_data = np.load(baseline_file, allow_pickle=True).item()

    # Use the surface slice of the sea mask
    surface_mask = sea_mask.isel(depth=0).values  # (lat, lon)
    lats = sea_mask.latitude.values
    lons = sea_mask.longitude.values

    groups = ["uo", "vo", "so", "thetao"]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(20, 8),
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    # Loop over each group and compute normalized difference
    for i, group in enumerate(groups):
        unit = constants.PARAM_UNITS[constants.PARAM_NAMES_SHORT.index(group)]
        rmse_model = model_data[group]["rmse_avg"]  # (n_grid,)
        rmse_baseline = baseline_data[group]["rmse_avg"]  # (n_grid,)
        rmse_diff = rmse_model - rmse_baseline
        vlim = np.percentile(np.abs(rmse_diff[~np.isnan(rmse_diff)]), 99)

        # Reconstruct full grid (n_lat, n_lon) using the surface mask:
        full_grid_norm_diff = np.full(surface_mask.shape, np.nan)
        full_grid_norm_diff[surface_mask == 1] = rmse_diff

        ax = axes[i]

        im = ax.pcolormesh(
            lons,
            lats,
            full_grid_norm_diff,
            cmap="RdBu_r",
            shading="auto",
            vmin=-vlim,
            vmax=vlim,
        )
        ax.coastlines(resolution="10m", linewidth=0.5)
        ax.add_feature(cfeature.LAND, facecolor="whitesmoke")
        ax.set_title(f"{group} RMSE diff. ({unit})", fontsize=fs)

        # Set ticks and format them as degrees
        ax.set_xticks(
            np.linspace(lons.min(), lons.max(), 5), crs=ccrs.PlateCarree()
        )
        ax.set_yticks(
            np.linspace(lats.min(), lats.max(), 5), crs=ccrs.PlateCarree()
        )
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(3))

        for tick in ax.get_xticklabels():
            tick.set_fontsize(fs)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(fs)

        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            shrink=0.7,
            pad=0.02,
            extend="both",
        )

    n_rows = 2
    n_cols = 2
    for i, ax in enumerate(axes):
        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Longitude (°)", fontsize=fs)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel("Latitude (°)", fontsize=fs)
        else:
            ax.tick_params(labelleft=False)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model}_vs_{baseline}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_vertical_rmse_diff(
    model,
    baseline,
    sea_mask,
    dataset="mediterranean",
    output_dir="figures/metrics/vertical_rmse_diff",
    fs=14,
):
    """
    Load aggregated group RMSE npy files for model and baseline and plot,
    for each group (uo, vo, so, thetao), the normalized difference in rmse.
    """
    # Load aggregated group RMSE dictionaries
    model_file = os.path.join(
        "data", dataset, "metrics", model, "group_rmse.npy"
    )
    baseline_file = os.path.join(
        "data", dataset, "metrics", baseline, "group_rmse.npy"
    )
    model_data = np.load(model_file, allow_pickle=True).item()
    baseline_data = np.load(baseline_file, allow_pickle=True).item()

    # Get horizontal grid from the surface sea mask.
    surface_mask = sea_mask.isel(depth=0).values  # (lat, lon)
    lons = sea_mask.longitude.values

    # Get depth values from constants (rounded)
    depth_vals = [round(d) for d in constants.DEPTHS]
    n_depths = len(depth_vals)

    groups = ["uo", "vo", "so", "thetao"]

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgray")

    # Create figure with 2x2 subplots (one for each group)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    axes = axes.flatten()

    # For each group, process mse_all (n_grid, n_depths)
    for i, group in enumerate(groups):
        unit = constants.PARAM_UNITS[constants.PARAM_NAMES_SHORT.index(group)]
        mse_all_model = model_data[group]["mse_all"]  # (n_grid, n_depths)
        mse_all_baseline = baseline_data[group]["mse_all"]  # (n_grid, n_depths)

        # Reconstruct a full grid for each depth level and compute RMSE diff
        section_list = []
        for d in range(n_depths):
            # Reconstruct full grid, avg over latitude, then take sqrt of MSE
            full_grid_model = np.full(surface_mask.shape, np.nan)
            full_grid_model[surface_mask == 1] = mse_all_model[:, d]
            avg_over_lat_model = np.nanmean(full_grid_model, axis=0)
            rmse_model = np.sqrt(avg_over_lat_model)

            # Reconstruct full grid, avg over latitude, then take sqrt of MSE
            full_grid_baseline = np.full(surface_mask.shape, np.nan)
            full_grid_baseline[surface_mask == 1] = mse_all_baseline[:, d]
            avg_over_lat_baseline = np.nanmean(full_grid_baseline, axis=0)
            rmse_baseline = np.sqrt(avg_over_lat_baseline)

            # model RMSE - baseline RMSE
            diff_rmse = rmse_model - rmse_baseline
            section_list.append(diff_rmse)

        # Stack sections to form a 2D array with shape (n_depths, n_lon)
        vertical_section = np.stack(section_list, axis=0)

        # Update vlim
        vlim = np.percentile(
            np.abs(vertical_section[~np.isnan(vertical_section)]), 99
        )

        ax = axes[i]
        extent = [lons.min(), lons.max(), max(depth_vals), min(depth_vals)]

        im = ax.imshow(
            vertical_section,
            origin="upper",
            cmap=cmap,
            extent=extent,
            aspect="auto",
            vmin=-vlim,
            vmax=vlim,
        )
        ax.set_title(f"{group} RMSE diff. ({unit})", fontsize=fs)
        fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            shrink=0.7,
            pad=0.02,
            extend="both",
        )

        for tick in ax.get_xticklabels():
            tick.set_fontsize(fs)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(fs)

    n_rows = 2
    n_cols = 2
    for i, ax in enumerate(axes):
        row = i // n_cols
        col = i % n_cols
        if row == n_rows - 1:
            ax.set_xlabel("Longitude (°)", fontsize=fs)
        if col == 0:
            ax.set_ylabel("Depth (m)", fontsize=fs)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{model}_vs_{baseline}.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    """
    Script for plotting results.
    """
    parser = argparse.ArgumentParser(description="Plot forecast metrics.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
        help="Dataset name",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to the forecast file",
    )
    parser.add_argument(
        "--var",
        type=str,
        nargs="+",
        default=["uo", "vo", "so", "thetao", "zos"],
        help="Variable(s) to plot for forecast",
    )
    parser.add_argument(
        "--plot_forecast", action="store_true", help="Plot forecast"
    )
    parser.add_argument(
        "--plot_forecast_vertical",
        action="store_true",
        help="Plot vertical sections of potential variables",
    )
    parser.add_argument(
        "--plot_rmse", action="store_true", help="Plot RMSE metrics"
    )
    parser.add_argument(
        "--plot_acc", action="store_true", help="Plot ACC metrics"
    )
    parser.add_argument(
        "--plot_scorecard",
        action="store_true",
        help="Plot heatmap scorecards comparing metrics to a baseline",
    )
    parser.add_argument(
        "--plot_rmse_depth",
        action="store_true",
        help="Plot RMSE metrics with depth axis",
    )
    parser.add_argument(
        "--plot_norm_rmse_diff",
        action="store_true",
        help="Plot normalized RMSE diff wrt to a baseline",
    )
    parser.add_argument(
        "--plot_group_bias",
        action="store_true",
        help="Plot spatial bias for each feature group",
    )
    parser.add_argument(
        "--plot_spatial_rmse_diff",
        action="store_true",
        help="Plot normalized group rmse_avg diff",
    )
    parser.add_argument(
        "--plot_vertical_rmse_diff",
        action="store_true",
        help="Plot normalized group rmse_all diff",
    )
    args = parser.parse_args()

    color_map = {
        "seacast": "tab:blue",
        "med_phy": "tab:orange",
        "persistence": "tab:gray",
        "seacast_base": "#aec7e8",
        "seacast_10y": "tab:green",
        "seacast_10y_base": "#98df8a",
        "t2m_permuted": "tab:red",
        "tau_permuted": "tab:purple",
        "msl_permuted": "tab:olive",
        "all_permuted": "tab:brown",
        "seacast_analysis": "tab:cyan",
        "seacast_2": "#aec7e8",
        "med_phy_2": "#ffbb78",
    }
    model_labels = {
        "seacast": "SeaCast",
        "med_phy": "MedFS",
        "persistence": "Persistence",
        "seacast_base": "SeaCast (w/o finetuning)",
        "seacast_10y": "SeaCast (10y)",
        "seacast_10y_base": "SeaCast (10y, w/o finetuning)",
        "t2m_permuted": "2m-temp. perm.",
        "tau_permuted": "Wind stress perm.",
        "msl_permuted": "MSLP perm.",
        "all_permuted": "All perm.",
        "seacast_analysis": "SeaCast (analysis init.)",
        "seacast_2": "SeaCast (Tuesdays)",
        "med_phy_2": "MedFS (Tuesdays)",
    }

    if args.plot_group_bias:
        bathy_path = os.path.join(
            "data", args.dataset, "static", "bathy_mask.nc"
        )
        bathy_data = xr.load_dataset(bathy_path)
        sea_mask = bathy_data.mask

        models = ["seacast"]
        variables = ["uo", "vo", "so", "thetao", "zos"]
        for model in models:
            for var in variables:
                plot_group_bias(var, model, sea_mask, dataset=args.dataset)

    # Plot forecast
    if args.plot_forecast:
        bathy_path = os.path.join(
            "data", args.dataset, "static", "bathy_mask.nc"
        )
        bathy_data = xr.load_dataset(bathy_path)
        mask = bathy_data.where(bathy_data.mask, drop=True).mask
        ds = read_npy_to_xarray(args.file, mask)

        basename = os.path.basename(args.file)
        file_date = basename.split(".")[0][-8:]
        analysis_file = os.path.join(
            "data", args.dataset, "samples", "test", f"ana_data_{file_date}.npy"
        )
        analysis_ds = read_npy_to_xarray(analysis_file, mask, init=True)
        for var in args.var:
            plot_forecast(
                ds, analysis_ds, var, lead_indices=[0, 7, 14], model="seacast"
            )

    if args.plot_forecast_vertical:
        bathy_path = os.path.join(
            "data", args.dataset, "static", "bathy_mask.nc"
        )
        bathy_data = xr.load_dataset(bathy_path)
        mask = bathy_data.where(bathy_data.mask, drop=True).mask
        ds = read_npy_to_xarray(args.file, mask)
        basename = os.path.basename(args.file)
        file_date = basename.split(".")[0][-8:]
        analysis_file = os.path.join(
            "data", args.dataset, "samples", "test", f"ana_data_{file_date}.npy"
        )
        analysis_ds = read_npy_to_xarray(analysis_file, mask, init=True)
        for var in ["uo", "vo", "so", "thetao"]:
            for direction in ["zonal", "meridional"]:
                plot_forecast_vertical(
                    ds,
                    analysis_ds,
                    var,
                    [0, 7, 14],
                    direction,
                    suffix="seacast",
                )

    # Plot RMSE / ACC w.r.t. lead time
    if args.plot_rmse:
        models = [
            "seacast",
            "seacast_2",
            "med_phy",
            "med_phy_2",
            "persistence",
        ]
        output_dir = os.path.join("figures", "metrics", "avg_group_metric")
        agg_group_rmse_all = {}
        for model in models:
            json_path_avg_rmse = os.path.join(
                "data", args.dataset, "metrics", model, "avg_group_rmse.json"
            )
            with open(json_path_avg_rmse, "r", encoding="utf8") as jf:
                agg_group_rmse_all[model] = json.load(jf)
        # Plot aggregated group RMSE for all models in one figure.
        plot_avg_group_metric(
            agg_group_rmse_all,
            metric="rmse",
            n_steps=15,
            fs=11,
            output_dir=output_dir,
            fill_between=True,
            model_labels=model_labels,
            color_map=color_map,
            legend_ncol=3,
        )

    if args.plot_acc:
        models = [
            "seacast",
            "seacast_2",
            "med_phy",
            "med_phy_2",
            "persistence",
        ]
        output_dir = os.path.join("figures", "metrics", "avg_group_metric")
        agg_group_acc_all = {}
        for model in models:
            json_path_avg_acc = os.path.join(
                "data", args.dataset, "metrics", model, "avg_group_acc.json"
            )
            with open(json_path_avg_acc, "r", encoding="utf8") as jf:
                agg_group_acc_all[model] = json.load(jf)
        # Plot aggregated group ACC for all models in one figure.
        plot_avg_group_metric(
            agg_group_acc_all,
            metric="acc",
            n_steps=15,
            fs=11,
            output_dir=output_dir,
            fill_between=False,
            model_labels=model_labels,
            color_map=color_map,
            legend_ncol=3,
        )

    if args.plot_rmse:
        output_dir = os.path.join("figures", "metrics", "rmse_models")
        models = [
            "seacast",
            "seacast_2",
            "med_phy",
            "med_phy_2",
            "persistence",
        ]
        metric_std_all = {}
        for model in models:
            json_path = os.path.join(
                "data", args.dataset, "metrics", model, "rmse.json"
            )
            metric_matrix, std_matrix = load_metric_std_steps(
                json_path, n_steps=15, metric="rmse"
            )
            metric_std_all[model] = (metric_matrix, std_matrix)
        for variable in ["uo", "vo", "so", "thetao"]:
            plot_metric_by_depth(
                variable,
                output_dir,
                metric_std_all,
                model_labels,
                color_map=color_map,
                metric="rmse",
                n_steps=15,
                fs=12,
                fill_between=True,
                legend_ncol=3,
            )
        plot_metric_single(
            "zos",
            metric_std_all,
            model_labels,
            color_map=color_map,
            metric="rmse",
            n_steps=15,
            fs=12,
            output_dir=output_dir,
            fill_between=True,
            legend_ncol=3,
        )

    if args.plot_acc:
        output_dir = os.path.join("figures", "metrics", "acc_models")
        models = [
            "seacast",
            "seacast_2",
            "med_phy",
            "med_phy_2",
            "persistence",
        ]
        metric_std_all = {}
        for model in models:
            json_path = os.path.join(
                "data", args.dataset, "metrics", model, "acc.json"
            )
            metric_matrix, std_matrix = load_metric_std_steps(
                json_path, n_steps=15, metric="acc"
            )
            metric_std_all[model] = (metric_matrix, std_matrix)
        for variable in ["uo", "vo", "so", "thetao"]:
            plot_metric_by_depth(
                variable,
                output_dir,
                metric_std_all,
                model_labels,
                color_map=color_map,
                metric="acc",
                n_steps=15,
                fs=12,
                legend_ncol=3,
            )
        plot_metric_single(
            "zos",
            metric_std_all,
            model_labels,
            color_map=color_map,
            metric="acc",
            n_steps=15,
            fs=12,
            output_dir=output_dir,
            legend_ncol=3,
        )

    if args.plot_rmse:
        output_dir = os.path.join("figures", "metrics", "rmse_forcings")
        models = [
            "seacast",
            "t2m_permuted",
            "tau_permuted",
            "msl_permuted",
            "all_permuted",
        ]
        metric_std_all = {}
        for model in models:
            json_path = os.path.join(
                "data", args.dataset, "metrics", model, "rmse.json"
            )
            metric_matrix, std_matrix = load_metric_std_steps(
                json_path, n_steps=15, metric="rmse"
            )
            metric_std_all[model] = (metric_matrix, std_matrix)
        for variable in ["uo", "vo", "so", "thetao"]:
            plot_metric_by_depth(
                variable,
                output_dir,
                metric_std_all,
                model_labels,
                color_map=color_map,
                metric="rmse",
                n_steps=15,
                fs=12,
            )
        plot_metric_single(
            "zos",
            metric_std_all,
            model_labels,
            color_map=color_map,
            metric="rmse",
            n_steps=15,
            fs=12,
            output_dir=output_dir,
            fill_between=False,
        )

    if args.plot_acc:
        output_dir = os.path.join("figures", "metrics", "acc_forcings")
        models = [
            "seacast",
            "t2m_permuted",
            "tau_permuted",
            "msl_permuted",
            "all_permuted",
        ]
        metric_std_all = {}
        for model in models:
            json_path = os.path.join(
                "data", args.dataset, "metrics", model, "acc.json"
            )
            metric_matrix, std_matrix = load_metric_std_steps(
                json_path, n_steps=15, metric="acc"
            )
            metric_std_all[model] = (metric_matrix, std_matrix)
        for variable in ["uo", "vo", "so", "thetao"]:
            plot_metric_by_depth(
                variable,
                output_dir,
                metric_std_all,
                model_labels,
                color_map=color_map,
                metric="acc",
                n_steps=15,
                fs=12,
            )
        plot_metric_single(
            "zos",
            metric_std_all,
            model_labels,
            color_map=color_map,
            metric="acc",
            n_steps=15,
            fs=12,
            output_dir=output_dir,
        )

    # Plot normalized RMSE difference
    if args.plot_norm_rmse_diff:
        output_dir = os.path.join(
            "figures", "metrics", "norm_rmse_diff_forcing"
        )
        baseline_label = "seacast"
        baseline_json = os.path.join(
            "data", args.dataset, "metrics", baseline_label, "rmse.json"
        )
        models = [
            "t2m_permuted",
            "tau_permuted",
            "msl_permuted",
            "all_permuted",
        ]
        model_jsons = {}
        for model in models:
            json_path = os.path.join(
                "data", args.dataset, "metrics", model, "rmse.json"
            )
            model_jsons[model] = json_path
        for variable in ["uo", "vo", "so", "thetao"]:
            plot_norm_rmse_diff_by_depth(
                variable,
                baseline_json,
                model_jsons,
                baseline_label,
                model_labels,
                output_dir,
                color_map=color_map,
                n_steps=15,
                fs=12,
                fill_between=False,
            )
        plot_norm_rmse_diff_single(
            "zos",
            baseline_json,
            model_jsons,
            baseline_label,
            model_labels,
            output_dir,
            color_map=color_map,
            n_steps=15,
            fs=12,
            fill_between=False,
        )

    if args.plot_norm_rmse_diff:
        output_dir = os.path.join("figures", "metrics", "norm_rmse_diff")
        baseline_label = "med_phy"
        baseline_json = os.path.join(
            "data", args.dataset, "metrics", baseline_label, "rmse.json"
        )
        models = [
            "seacast",
            "seacast_base",
            "seacast_10y",
            "seacast_10y_base",
        ]
        model_jsons = {}
        for model in models:
            json_path = os.path.join(
                "data", args.dataset, "metrics", model, "rmse.json"
            )
            model_jsons[model] = json_path
        for variable in ["uo", "vo", "so", "thetao"]:
            plot_norm_rmse_diff_by_depth(
                variable,
                baseline_json,
                model_jsons,
                baseline_label,
                model_labels,
                output_dir,
                color_map=color_map,
                n_steps=10,
                fs=12,
                fill_between=False,
                legend_ncol=3,
            )
        plot_norm_rmse_diff_single(
            "zos",
            baseline_json,
            model_jsons,
            baseline_label,
            model_labels,
            output_dir,
            color_map=color_map,
            n_steps=10,
            fs=12,
            fill_between=False,
            legend_ncol=3,
        )

        if args.plot_norm_rmse_diff:
            output_dir = os.path.join(
                "figures", "metrics", "norm_rmse_diff_persistence"
            )
            baseline_label = "persistence"
            baseline_json = os.path.join(
                "data", args.dataset, "metrics", baseline_label, "rmse.json"
            )
            models = [
                "seacast",
                "seacast_base",
                "seacast_10y",
                "seacast_10y_base",
                "med_phy",
            ]
            model_jsons = {}
            for model in models:
                json_path = os.path.join(
                    "data", args.dataset, "metrics", model, "rmse.json"
                )
                model_jsons[model] = json_path
            for variable in ["uo", "vo", "so", "thetao"]:
                plot_norm_rmse_diff_by_depth(
                    variable,
                    baseline_json,
                    model_jsons,
                    baseline_label,
                    model_labels,
                    output_dir,
                    color_map=color_map,
                    n_steps=15,
                    fs=12,
                    fill_between=False,
                    legend_ncol=3,
                )
            plot_norm_rmse_diff_single(
                "zos",
                baseline_json,
                model_jsons,
                baseline_label,
                model_labels,
                output_dir,
                color_map=color_map,
                n_steps=15,
                fs=12,
                fill_between=False,
                legend_ncol=3,
            )

    # Scorecards
    if args.plot_scorecard:
        n_steps = 10
        baseline_label = "med_phy"
        base_path = os.path.join("data", args.dataset, "metrics")
        med_phy_path = os.path.join(base_path, baseline_label, "rmse.json")
        med_phy_rmse = load_rmse_steps(med_phy_path, n_steps=n_steps)
        depths = [round(d) for d in constants.DEPTHS]
        compare_models = ["seacast"]
        for model in compare_models:
            model_path = os.path.join(base_path, model, "rmse.json")
            model_rmse = load_rmse_steps(model_path, n_steps=n_steps)
            norm_rmse_diff = 100 * (model_rmse - med_phy_rmse) / med_phy_rmse
            base_save_name = f"{model}_vs_med_phy"
            plot_scorecard(norm_rmse_diff, depths, base_save_name)

    if args.plot_scorecard:
        n_steps = 15
        base_path = os.path.join("data", args.dataset, "metrics")
        aifs_path = os.path.join(base_path, "seacast_ens", "rmse.json")
        aifs_rmse = load_rmse_steps(aifs_path, n_steps=n_steps)
        depths = [round(d) for d in constants.DEPTHS]
        compare_models = [
            "seacast",
        ]
        for model in compare_models:
            model_path = os.path.join(base_path, model, "rmse.json")
            model_rmse = load_rmse_steps(model_path, n_steps=n_steps)
            norm_rmse_diff = 100 * (model_rmse - aifs_rmse) / aifs_rmse
            base_save_name = "seacast_aifs_vs_seacast_ens"
            plot_scorecard(norm_rmse_diff, depths, base_save_name)

    # RMSE at depth
    if args.plot_rmse_depth:
        output_dir = os.path.join("figures", "metrics", "rmse_depths")
        models = ["seacast", "seacast_analysis", "med_phy", "persistence"]
        rmse_std_all = {}
        for model in models:
            json_path = os.path.join(
                "data", args.dataset, "metrics", model, "rmse.json"
            )
            rmse_matrix, std_matrix = load_metric_std_steps(
                json_path, n_steps=15, metric="rmse"
            )
            rmse_std_all[model] = (rmse_matrix, std_matrix)
        for variable in ["uo", "vo", "so", "thetao"]:
            plot_rmse_vs_depth(
                rmse_std_all,
                variable,
                model_labels,
                color_map=color_map,
                n_steps=15,
                fs=12,
                output_dir=output_dir,
            )

    # Spatially averaged RMSE diff
    if args.plot_spatial_rmse_diff:
        bathy_path = os.path.join(
            "data", args.dataset, "static", "bathy_mask.nc"
        )
        bathy_data = xr.load_dataset(bathy_path)
        sea_mask = bathy_data.mask
        plot_spatial_rmse_diff(
            "seacast", "med_phy", sea_mask, dataset=args.dataset
        )

    if args.plot_vertical_rmse_diff:
        bathy_path = os.path.join(
            "data", args.dataset, "static", "bathy_mask.nc"
        )
        bathy_data = xr.load_dataset(bathy_path)
        sea_mask = bathy_data.mask
        plot_vertical_rmse_diff(
            "seacast", "med_phy", sea_mask, dataset=args.dataset
        )


if __name__ == "__main__":
    main()
