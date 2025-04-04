# Standard library
import argparse
import json
import os

# Third-party
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import MultipleLocator

# First-party
from neural_lam import constants


def create_in_situ_rmse_plot(
    dataset,
    color_map,
    label_map,
    fs=11,
    plot_ci=True,
    out_dir="figures/results",
):
    """
    Plot RMSE for in-situ evaluation.
    """

    models_rmse = ["seacast", "med_phy", "persistence"]
    variables = ["in_situ_uo", "in_situ_vo", "in_situ_so", "in_situ_thetao"]
    titles = [
        "a) Zonal current",
        "b) Meridional current",
        "c) Salinity",
        "d) Temperature",
    ]
    title_map = dict(zip(variables, titles))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(2 * 3.7, 2 * 3),
        sharex=True,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(hspace=0.1)

    for idx, var in enumerate(variables):
        ax = axes.flatten()[idx]

        ax.axvline(x=10, color="lightgray", linestyle="--", zorder=0)

        for model in models_rmse:
            json_path = os.path.join(
                "data", dataset, "metrics", model, f"{var}_rmse.json"
            )
            try:
                with open(json_path, "r") as jf:
                    data = json.load(jf)
            except FileNotFoundError:
                print(f"File not found: {json_path}")
                continue

            rmse = data["rmse"]
            n_lead = min(len(rmse), 15)
            x = np.arange(1, n_lead + 1)
            y = np.array(rmse[:n_lead])
            if model == "persistence":
                line = ax.plot(
                    x,
                    y,
                    lw=2,
                    color=color_map[model],
                    label=label_map.get(model, model),
                    zorder=0,
                )[0]
            else:
                line = ax.plot(
                    x,
                    y,
                    lw=2,
                    color=color_map[model],
                    label=label_map.get(model, model),
                )[0]

            ci_lower = np.array(data["ci_lower"][:n_lead])
            ci_upper = np.array(data["ci_upper"][:n_lead])
            if model == "persistence":
                ax.fill_between(
                    x,
                    ci_lower,
                    ci_upper,
                    alpha=0.3,
                    color=line.get_color(),
                    zorder=0,
                )
            else:
                ax.fill_between(
                    x, ci_lower, ci_upper, alpha=0.3, color=line.get_color()
                )

        ax.set_xticks(np.arange(1, n_lead + 1, 2))
        ax.set_xticks(x, minor=True)
        ax.tick_params(axis="both", which="major", labelsize=fs)
        ax.tick_params(axis="both", which="minor", labelsize=0)

        param = var.split("_")[-1]
        idx_param = constants.PARAM_NAMES_SHORT.index(param)
        unit = constants.PARAM_UNITS[idx_param]

        ax.set_ylabel(f"RMSE ({unit})", fontsize=fs)
        ax.set_xlabel("Lead time (days)", fontsize=fs)
        ax.set_title(title_map[var], fontsize=fs)

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(handles),
        fontsize=fs,
        frameon=False,
    )

    for ext in ["png", "pdf"]:
        output_path = os.path.join(out_dir, f"in_situ_rmse.{ext}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_norm_rmse_diff(
    dataset,
    color_map,
    label_map,
    title_map,
    fs=11,
    n_steps=15,
    out_dir="figures/results",
):
    """
    Plot normalized RMSE difference toward persistence.
    """

    os.makedirs(out_dir, exist_ok=True)

    baseline_label = "persistence"
    models = [
        "seacast",
        "seacast_base",
        "seacast_10y",
        "seacast_10y_base",
        "med_phy",
    ]
    groups = ["uo", "vo", "so", "thetao", "sst", "sla"]

    baseline_in_situ_path = os.path.join(
        "data", dataset, "metrics", baseline_label, "avg_group_rmse.json"
    )
    with open(baseline_in_situ_path, "r") as jf:
        baseline_in_situ_data = json.load(jf)

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(3 * 3.3, 2 * 3), constrained_layout=True
    )
    fig.set_constrained_layout_pads(hspace=0.1)
    x = np.arange(1, n_steps + 1)
    axes_flat = axes.flatten()

    global_y_values = []

    for idx, group in enumerate(groups):
        ax = axes_flat[idx]
        if group in ["uo", "vo", "so", "thetao"]:
            baseline_rmse = np.array(baseline_in_situ_data[group]["rmse"])[
                :n_steps
            ]
        else:
            baseline_file = os.path.join(
                "data", dataset, "metrics", baseline_label, f"{group}_rmse.json"
            )
            with open(baseline_file, "r") as jf:
                group_data = json.load(jf)
            baseline_rmse = np.array(group_data["rmse"])[:n_steps]

        for model in models:
            if group in ["uo", "vo", "so", "thetao"]:
                model_file = os.path.join(
                    "data", dataset, "metrics", model, "avg_group_rmse.json"
                )
                with open(model_file, "r") as jf:
                    model_group_data = json.load(jf)
                rmse_model = np.array(model_group_data[group]["rmse"])[:n_steps]
            else:
                model_file = os.path.join(
                    "data", dataset, "metrics", model, f"{group}_rmse.json"
                )
                with open(model_file, "r") as jf:
                    model_group_data = json.load(jf)
                rmse_model = np.array(model_group_data["rmse"])[:n_steps]

            if model == "med_phy":
                norm_diff = (rmse_model - baseline_rmse[:10]) / baseline_rmse[
                    :10
                ]
            else:
                norm_diff = (rmse_model - baseline_rmse) / baseline_rmse
            y_values = 100 * norm_diff
            global_y_values.append(y_values)
            x_vals = np.arange(1, len(y_values) + 1)
            ax.plot(
                x_vals,
                y_values,
                lw=2,
                label=label_map.get(model, model),
                color=color_map.get(model),
            )

        ax.axvline(x=10, color="lightgray", linestyle="--", lw=2, zorder=0)
        ax.plot(
            x,
            np.zeros(len(x)),
            color="tab:gray",
            linestyle="-",
            lw=2,
            label="Persistence",
            zorder=0,
        )
        global_y_values.append(np.zeros(len(x)))

        ax.set_ylabel("Norm. RMSE diff. (%)", fontsize=fs)
        ax.set_xlabel("Lead time (days)", fontsize=fs)
        ax.set_xticks(np.arange(1, n_steps + 1, 2))
        ax.set_xticks(x, minor=True)
        ax.tick_params(axis="both", which="major", labelsize=fs)
        ax.tick_params(axis="both", which="minor", labelsize=0)
        ax.set_title(title_map[group], fontsize=fs)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        fontsize=fs,
        frameon=False,
    )

    for ext in ["png", "pdf"]:
        output_path = os.path.join(out_dir, f"norm_rmse_diff.{ext}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_norm_rmse_diff_forcing(
    dataset,
    color_map,
    label_map,
    title_map,
    fs=11,
    n_steps=15,
    out_dir="figures/results",
):
    """
    Plot normalized RMSE difference for different forcings toward SeaCast.
    """

    os.makedirs(out_dir, exist_ok=True)

    baseline_label = "seacast"
    forced_models = [
        "t2m_permuted",
        "tau_permuted",
        "msl_permuted",
        "all_permuted",
    ]

    sim_groups = ["uo", "vo", "so", "thetao"]
    remote_vars = ["sst", "sla"]
    groups = sim_groups + remote_vars

    baseline_rmse = {}
    baseline_avg_path = os.path.join(
        "data", dataset, "metrics", baseline_label, "avg_group_rmse.json"
    )
    with open(baseline_avg_path, "r") as jf:
        baseline_avg_data = json.load(jf)
    for group in sim_groups:
        baseline_rmse[group] = np.array(baseline_avg_data[group]["rmse"])[
            :n_steps
        ]
    for var in remote_vars:
        baseline_path = os.path.join(
            "data", dataset, "metrics", baseline_label, f"{var}_rmse.json"
        )
        with open(baseline_path, "r") as jf:
            data = json.load(jf)
        baseline_rmse[var] = np.array(data["rmse"])[:n_steps]

    forced_data = {}
    for model in forced_models:
        forced_data[model] = {}
        avg_path = os.path.join(
            "data", dataset, "metrics", model, "avg_group_rmse.json"
        )
        with open(avg_path, "r") as jf:
            avg_data = json.load(jf)
        for group in sim_groups:
            forced_data[model][group] = np.array(avg_data[group]["rmse"])[
                :n_steps
            ]
        for var in remote_vars:
            file_path = os.path.join(
                "data", dataset, "metrics", model, f"{var}_rmse.json"
            )
            with open(file_path, "r") as jf:
                var_data = json.load(jf)
            forced_data[model][var] = np.array(var_data["rmse"])[:n_steps]

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(3 * 3.2, 2 * 3), constrained_layout=True
    )
    fig.set_constrained_layout_pads(hspace=0.1)
    x = np.arange(1, n_steps + 1)
    axes_flat = axes.flatten()

    for idx, group in enumerate(groups):
        ax = axes_flat[idx]
        ax.plot(
            x,
            np.zeros(n_steps),
            color="tab:blue",
            linestyle="-",
            lw=2,
            label="SeaCast",
            zorder=0,
        )
        for model in forced_models:
            model_rmse = forced_data[model][group]
            norm_diff = (model_rmse - baseline_rmse[group]) / baseline_rmse[
                group
            ]
            ax.plot(
                x,
                100 * norm_diff,
                lw=2,
                label=label_map.get(model, model),
                color=color_map.get(model),
            )
        ax.set_xlabel("Lead time (days)", fontsize=fs)
        ax.set_ylabel("Norm. RMSE diff. (%)", fontsize=fs)
        ax.set_xticks(np.arange(1, n_steps + 1, 2))
        ax.set_xticks(x, minor=True)
        ax.tick_params(axis="both", which="major", labelsize=fs)
        ax.tick_params(axis="both", which="minor", labelsize=0)
        ax.set_title(title_map[group], fontsize=fs)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(unique),
        fontsize=fs,
        frameon=False,
    )

    for ext in ["png", "pdf"]:
        output_path = os.path.join(out_dir, f"norm_rmse_diff_forcing.{ext}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rmse(
    dataset,
    color_map,
    label_map,
    title_map,
    fs=11,
    n_steps=15,
    out_dir="figures/results",
):
    """
    Plot RMSE vs lead time.
    """

    os.makedirs(out_dir, exist_ok=True)

    models_rmse = ["seacast", "med_phy", "persistence"]

    sim_groups = ["uo", "vo", "so", "thetao"]
    remote_vars = ["sst", "sla"]
    all_vars = sim_groups + remote_vars

    remote_units = {"sst": "°C", "sla": "m"}

    fig, axes = plt.subplots(
        nrows=2, ncols=3, figsize=(3 * 3.4, 2 * 3), constrained_layout=True
    )
    fig.set_constrained_layout_pads(hspace=0.1)
    x = np.arange(1, n_steps + 1)
    axes_flat = axes.flatten()

    for idx, var in enumerate(all_vars):
        ax = axes_flat[idx]
        for model in models_rmse:
            if var in sim_groups:
                json_path = os.path.join(
                    "data", dataset, "metrics", model, "avg_group_rmse.json"
                )
                with open(json_path, "r") as jf:
                    data = json.load(jf)
                rmse = np.array(data[var]["rmse"])[:n_steps]
                ci_lower = np.array(data[var]["ci_lower"])[:n_steps]
                ci_upper = np.array(data[var]["ci_upper"])[:n_steps]
                idx_param = constants.PARAM_NAMES_SHORT.index(var)
                unit = constants.PARAM_UNITS[idx_param]
            else:
                json_path = os.path.join(
                    "data", dataset, "metrics", model, f"{var}_rmse.json"
                )
                with open(json_path, "r") as jf:
                    data = json.load(jf)
                rmse = np.array(data["rmse"])[:n_steps]
                ci_lower = np.array(data["ci_lower"])[:n_steps]
                ci_upper = np.array(data["ci_upper"])[:n_steps]
                unit = remote_units[var]
            x_model = range(1, len(rmse) + 1)
            ax.plot(
                x_model,
                rmse,
                lw=2,
                label=label_map.get(model, model),
                color=color_map[model],
            )
            ax.fill_between(
                x_model, ci_lower, ci_upper, alpha=0.3, color=color_map[model]
            )
        ax.axvline(x=10, color="lightgray", linestyle="--", lw=2, zorder=0)
        ax.set_xlabel("Lead time (days)", fontsize=fs)
        ax.set_ylabel(f"RMSE ({unit})", fontsize=fs)
        ax.set_xticks(np.arange(1, n_steps + 1, 2))
        ax.set_xticks(x, minor=True)
        ax.tick_params(axis="both", which="major", labelsize=fs)
        ax.tick_params(axis="both", which="minor", labelsize=0)
        ax.set_title(title_map[var], fontsize=fs)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=len(unique),
        fontsize=fs,
        frameon=False,
    )

    for ext in ["png", "pdf"]:
        output_path = os.path.join(out_dir, f"rmse.{ext}")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_spatial_rmse_diff(
    model, baseline, dataset, out_dir, leads=[1, 4, 7, 10], fs=12
):
    """
    Plot normalized RMSE difference for SST spatially.
    """
    nc_model = os.path.join(
        "data", dataset, "metrics", model, "sst_spatial_rmse.nc"
    )
    nc_baseline = os.path.join(
        "data", dataset, "metrics", baseline, "sst_spatial_rmse.nc"
    )
    ds_model = xr.load_dataset(nc_model)
    ds_baseline = xr.load_dataset(nc_baseline)

    da_model = ds_model["rmse"]
    da_baseline = ds_baseline["rmse"]

    diff_list = []
    for lead in leads:
        idx = lead - 1
        diff = (
            da_model.isel(time=idx) - da_baseline.isel(time=idx)
        ) / da_baseline.isel(time=idx)
        diff_list.append(diff)

    all_vals = np.concatenate([np.ravel(diff.values) for diff in diff_list])
    max_abs = np.nanpercentile(np.abs(all_vals), 99.99)
    vmin, vmax = -max_abs, max_abs

    proj = ccrs.PlateCarree()
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 6),
        subplot_kw={"projection": proj},
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    fig.set_constrained_layout_pads(hspace=0.1)
    axes = axes.flatten()

    letters = ["a)", "b)", "c)", "d)"]

    for i, (ax, lead, diff) in enumerate(zip(axes, leads, diff_list)):
        row, col = i // ncols, i % ncols
        letter = letters[i]

        ax.coastlines(resolution="10m", linewidth=1)
        ax.add_feature(cfeature.LAND, facecolor="whitesmoke")

        im = ax.pcolormesh(
            diff.longitude,
            diff.latitude,
            diff,
            cmap="RdBu_r",
            shading="auto",
            transform=proj,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_title(f"{letter} {lead} day lead", fontsize=fs + 2)

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color="lightgray",
            alpha=0.5,
            linestyle="--",
        )
        gl.xlocator = MultipleLocator(5)
        gl.ylocator = MultipleLocator(3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": fs}
        gl.ylabel_style = {"size": fs}
        gl.bottom_labels = row == nrows - 1
        gl.left_labels = col == 0

        if col == 0:
            ax.set_ylabel("Latitude (°)", fontsize=fs)
        if row == nrows - 1:
            ax.set_xlabel("Longitude (°)", fontsize=fs)

    cbar = fig.colorbar(
        im, ax=axes, orientation="horizontal", shrink=0.3, aspect=30
    )
    cbar.set_label("Norm. RMSE diff. (%)", fontsize=fs)
    cbar.ax.tick_params(labelsize=fs)

    for ext in ["png", "pdf"]:
        save_path = os.path.join(
            out_dir, f"sst_rmse_diff_{model}_{baseline}.{ext}"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_region(dataset, fs=11, out_dir="figures/results"):
    bathy_data = xr.load_dataset(
        os.path.join("data", dataset, "static", "bathy_mask.nc")
    )
    surface_mask = bathy_data.mask.isel(depth=0)
    selected_data = bathy_data.isel(depth=0).where(surface_mask, drop=True)

    sea_depth_np = np.load(
        os.path.join("data", dataset, "static", "sea_depth.npy")
    )
    sea_depth = xr.DataArray(
        data=sea_depth_np,
        dims=("latitude", "longitude"),
        coords={
            "latitude": selected_data.latitude,
            "longitude": selected_data.longitude,
        },
        name="depth",
    )
    sea_depth = sea_depth.where(selected_data.mask == 1)
    selected_data["depth"] = sea_depth

    depth_surface = selected_data["depth"]
    mask_surface = selected_data["mask"]

    sea_mask = mask_surface.where(mask_surface == 1)

    # Gibraltar
    gibraltar_mask = sea_mask.where(selected_data["longitude"] < -5.2)

    # Dardanelles
    dardanelles_mask = sea_mask.where(
        (selected_data["latitude"] >= 39.9)
        & (selected_data["latitude"] <= 40.4)
        & (selected_data["longitude"] >= 25.9)
        & (selected_data["longitude"] <= 26.4)
    )

    fig, ax = plt.subplots(
        figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot bathymetry
    depth_surface.plot(
        ax=ax,
        cmap="Blues",
        cbar_kwargs={
            "label": "Depth (m)",
            "shrink": 0.725,
            "aspect": 20,
            "pad": 0.02,
        },
        rasterized=True,
    )

    # Overlay Gibraltar
    gibraltar_mask.plot(ax=ax, cmap="Reds", add_colorbar=False, rasterized=True)

    # Overlay Dardanelles
    dardanelles_mask.plot(
        ax=ax, cmap="Reds", add_colorbar=False, rasterized=True
    )

    ax.set_xlabel("Longitude", fontsize=fs)
    ax.set_ylabel("Latitude", fontsize=fs)

    gl = ax.gridlines(
        draw_labels=True, linestyle="--", color="lightgray", alpha=0.5
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = MultipleLocator(5)
    gl.ylocator = MultipleLocator(3)
    gl.xlabel_style = {"size": fs}
    gl.ylabel_style = {"size": fs}

    ax.coastlines(resolution="10m", linewidth=1)
    ax.add_feature(cfeature.LAND, facecolor="whitesmoke")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(
            os.path.join(out_dir, f"region.{ext}"), bbox_inches="tight", dpi=300
        )

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot forecast results.")
    parser.add_argument(
        "--dataset", type=str, default="mediterranean", help="Dataset name"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="figures/results",
        help="Output directory",
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
    }

    label_map = {
        "seacast": "SeaCast",
        "med_phy": "MedFS",
        "persistence": "Persistence",
        "seacast_base": "SeaCast (w/o finetuning)",
        "seacast_10y": "SeaCast (10y)",
        "seacast_10y_base": "SeaCast (10y, w/o finetuning)",
        "t2m_permuted": "2m-temp. perm.",
        "tau_permuted": "Wind stress perm.",
        "msl_permuted": "MSL perm.",
        "all_permuted": "All perm.",
    }

    title_map = {
        "uo": "a) Zonal current",
        "vo": "b) Meridional current",
        "so": "c) Salinity",
        "thetao": "d) Temperature",
        "sst": "e) SST",
        "sla": "f) SLA",
    }

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    create_in_situ_rmse_plot(args.dataset, color_map, label_map)
    plot_rmse(args.dataset, color_map, label_map, title_map)
    plot_norm_rmse_diff(args.dataset, color_map, label_map, title_map)
    plot_norm_rmse_diff_forcing(args.dataset, color_map, label_map, title_map)

    for model, baseline in [("seacast", "med_phy"), ("seacast", "seacast_ens")]:
        plot_spatial_rmse_diff(model, baseline, args.dataset, out_dir)

    plot_region(args.dataset)
