# Standard library
import argparse
import json
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np

# First-party
from neural_lam import constants


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
        plt.plot(x, y, lw=2, label=model)

        if plot_ci and "ci_lower" in data and "ci_upper" in data:
            ci_lower = np.array(data["ci_lower"][:n_lead])
            ci_upper = np.array(data["ci_upper"][:n_lead])
            plt.fill_between(x, ci_lower, ci_upper, alpha=0.3)

    plt.axvline(x=10, color="gray", linestyle="--", zorder=0)
    plt.xticks(np.arange(1, n_lead + 1, 2))
    plt.xticks(np.arange(1, n_lead), minor=True)
    plt.xlabel("Lead time (days)")
    if var == "sst":
        unit = "°C"
    elif var == "sla":
        unit = "m"
    else:
        param_idx = constants.PARAM_NAMES_SHORT.index(var.split("_")[-1])
        unit = constants.PARAM_UNITS[param_idx]
    plt.ylabel(f"RMSE ({unit})")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{var}_rmse_{suffix}.png")
    plt.savefig(save_path)
    plt.close()


def plot_norm_rmse_diff(
    var, models, baseline, dataset, out_dir, suffix, plot_ci=False
):
    """
    Plot normalized RMSE diff computed as
    (model_rmse - baseline_rmse) / baseline_rmse.
    """

    baseline_json_path = os.path.join(
        "data", dataset, "metrics", baseline, f"{var}_rmse.json"
    )
    with open(baseline_json_path, "r") as jf:
        baseline_data = json.load(jf)
    baseline_rmse = np.array(baseline_data["rmse"])
    n_lead = min(len(baseline_rmse), 15)
    x = np.arange(1, n_lead + 1)

    plt.figure(figsize=(5.5, 4))

    plt.plot(
        x,
        np.zeros(len(x)),
        color="gray",
        linestyle="--",
        label=baseline,
        zorder=0,
    )

    for model in models:
        model_json_path = os.path.join(
            "data", dataset, "metrics", model, f"{var}_rmse.json"
        )
        with open(model_json_path, "r") as jf:
            model_data = json.load(jf)
        model_rmse = np.array(model_data["rmse"][:n_lead])
        norm_diff = (model_rmse - baseline_rmse[:n_lead]) / baseline_rmse[
            :n_lead
        ]
        plt.plot(x, 100 * norm_diff, lw=2, label=model)

        if plot_ci and "ci_lower" in model_data and "ci_upper" in model_data:
            ci_lower = np.array(model_data["ci_lower"][:n_lead])
            ci_upper = np.array(model_data["ci_upper"][:n_lead])
            norm_ci_lower = (ci_lower - baseline_rmse[:n_lead]) / baseline_rmse[
                :n_lead
            ]
            norm_ci_upper = (ci_upper - baseline_rmse[:n_lead]) / baseline_rmse[
                :n_lead
            ]
            plt.fill_between(
                x, 100 * norm_ci_lower, 100 * norm_ci_upper, alpha=0.3
            )

    plt.xticks(np.arange(1, n_lead + 1, 2))
    plt.xlabel("Lead time (days)")
    plt.ylabel("Norm. RMSE diff. (%)")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(out_dir, f"{var}_norm_rmse_diff.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot forecast results.")
    parser.add_argument(
        "--dataset", default="mediterranean", help="Dataset name"
    )
    parser.add_argument(
        "--var",
        nargs="+",
        choices=[
            "sst",
            "sla",
            "in_situ_uo",
            "in_situ_vo",
            "in_situ_so",
            "in_situ_thetao",
        ],
        required=True,
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

    baseline = "seacast"
    forcing_models = [
        "t2m_permuted",
        "tau_permuted",
        "msl_permuted",
        "all_permuted",
    ]
    for var in args.var:
        plot_norm_rmse_diff(
            var,
            forcing_models,
            baseline,
            args.dataset,
            out_dir,
            suffix="forcing",
        )
