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


def plot_lead_rmse(var, models, dataset, out_dir, suffix, plot_ci=False):
    """
    Plot RMSE for each lead time.
    """
    plt.figure(figsize=(5.5, 4))
    for model in models:
        json_path = os.path.join(
            "data", dataset, "metrics", model, f"{var}_rmse.json"
        )
        with open(json_path, "r") as jf:
            data = json.load(jf)

        rmse = data["rmse"]
        n_lead = min(len(rmse), 15)
        x = np.arange(1, n_lead + 1)
        y = np.array(rmse[:n_lead])
        plt.plot(x, y, label=model)

        if plot_ci and "ci_lower" in data and "ci_upper" in data:
            ci_lower = np.array(data["ci_lower"][:n_lead])
            ci_upper = np.array(data["ci_upper"][:n_lead])
            plt.fill_between(x, ci_lower, ci_upper, alpha=0.3)

    plt.axvline(x=10, color="gray", linestyle="--", zorder=0)
    plt.xticks(np.arange(1, n_lead + 1, 2))
    plt.xticks(np.arange(1, n_lead), minor=True)
    plt.xlabel("Lead time (days)")
    if var == "sst":
        plt.ylabel("RMSE (°C)")
    else:
        plt.ylabel("RMSE (m)")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{var}_rmse_{suffix}.png")
    plt.savefig(save_path)
    plt.close()


def plot_spatial_rmse_diff(
    var, model, baseline, dataset, out_dir, leads=[1, 5, 10]
):
    """
    Load the spatial RMSE for model and baseline from their NetCDF files,
    and plot the normalized difference for the selected lead times.
    """
    nc_model = os.path.join(
        "data", dataset, "metrics", model, f"{var}_spatial_rmse.nc"
    )
    nc_baseline = os.path.join(
        "data", dataset, "metrics", baseline, f"{var}_spatial_rmse.nc"
    )

    ds_model = xr.load_dataset(nc_model)
    ds_baseline = xr.load_dataset(nc_baseline)

    da_model = ds_model["rmse"]
    da_baseline = ds_baseline["rmse"]

    # Compute normalized difference for each lead
    diff_list = []
    for lead in leads:
        idx = lead - 1
        diff = (
            da_model.isel(time=idx) - da_baseline.isel(time=idx)
        ) / da_baseline.isel(time=idx)
        diff_list.append(diff)

    # Determine global symmetric limits across the selected leads
    all_vals = np.concatenate([np.ravel(diff.values) for diff in diff_list])
    if var == "sla":
        max_abs = np.nanpercentile(np.abs(all_vals), 95)
    else:
        max_abs = np.nanmax(np.abs(all_vals))
    vmin, vmax = -max_abs, max_abs

    # Create a figure with one row per lead time
    proj = ccrs.PlateCarree()
    n_rows = len(leads)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=1,
        figsize=(8, 4 * n_rows),
        subplot_kw={"projection": proj},
        constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]

    for ax, lead, diff in zip(axes, leads, diff_list):
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
        )
        ax.set_title(f"Lead Time {lead} days", fontsize=12)
        cb = fig.colorbar(
            im, ax=ax, orientation="vertical", shrink=0.7, pad=0.02
        )
        cb.set_label("Norm. RMSE Diff. (%)", fontsize=10)

    save_path = os.path.join(out_dir, f"{var}_rmse_diff_{model}_{baseline}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot forecast results.")
    parser.add_argument(
        "--dataset", default="mediterranean", help="Dataset name"
    )
    parser.add_argument(
        "--var",
        nargs="+",
        choices=["sst", "sla"],
        required=True,
    )
    parser.add_argument(
        "--leads",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Lead times to plot",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join("figures", "observations"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--plot_ci",
        action="store_true",
        help="Whether to plot CI for RMSE",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    models = [
        "seacast",
        "med_phy",
        "seacast_ens",
        "seacast_analysis",
        "persistence",
    ]

    for var in args.var:
        plot_lead_rmse(
            var,
            models,
            args.dataset,
            out_dir,
            suffix="models",
            plot_ci=args.plot_ci,
        )

    models = [
        "seacast",
        "t2m_permuted",
        "tau_permuted",
        "msl_permuted",
        "all_permuted",
    ]

    for var in args.var:
        plot_lead_rmse(
            var,
            models,
            args.dataset,
            out_dir,
            suffix="forcing",
            plot_ci=args.plot_ci,
        )

    for model, baseline in [("seacast", "med_phy"), ("seacast", "seacast_ens")]:
        plot_spatial_rmse_diff(
            "sst", model, baseline, args.dataset, out_dir, leads=args.leads
        )
