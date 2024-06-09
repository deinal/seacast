# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# First-party
from neural_lam import constants, utils
from neural_lam.weather_dataset import WeatherDataset


def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
        help="Dataset to compute weights for (default: mediterranean)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--step_length",
        type=int,
        default=1,
        help="Step length in days to consider single time step (default: 1)",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=32,
        help="Number of workers in data loader (default: 32)",
    )
    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")

    expanded_mask = utils.load_mask(args.dataset)["interior_mask"].unsqueeze(
        0
    )  # 1, 1, N_grid, d_features

    w_list = np.ones(len(constants.EXP_PARAM_NAMES_SHORT))
    print("Saving parameter weights...")
    np.save(
        os.path.join(static_dir_path, "parameter_weights.npy"),
        w_list.astype("float32"),
    )

    # Load dataset without any subsampling
    ds = WeatherDataset(
        args.dataset,
        split="train",
        subsample_step=1,
        pred_length=4,
        standardize=False,
    )  # Without standardization
    loader = torch.utils.data.DataLoader(
        ds, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    # Compute mean and std.-dev. of each parameter
    # across full dataset
    print("Computing mean and std.-dev. for parameters...")
    means = []
    squares = []
    forcing_means = []
    forcing_squares = []
    for init_batch, target_batch, forcing_batch in tqdm(loader):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t, N_grid, d_features)
        masked_mean = torch.sum(expanded_mask * batch, dim=2) / torch.sum(
            expanded_mask, dim=2
        )  # (N_batch, N_t, d_features)
        masked_squares = torch.sum(expanded_mask * batch**2, dim=2) / torch.sum(
            expanded_mask, dim=2
        )  # (N_batch, N_t, d_features)
        means.append(torch.mean(masked_mean, dim=1))  # (N_batch, d_features,)
        squares.append(
            torch.mean(masked_squares, dim=1)
        )  # (N_batch, d_features,)

        # Atmospheric forcing at 1st windowed position
        forcing_batch = forcing_batch[:, :, :, :6]
        forcing_means.append(torch.mean(forcing_batch))  # (,)
        forcing_squares.append(torch.mean(forcing_batch**2))  # (,)

    mean = torch.mean(torch.cat(means, dim=0), dim=0)  # (d_features)
    second_moment = torch.mean(torch.cat(squares, dim=0), dim=0)
    std = torch.sqrt(second_moment - mean**2)  # (d_features)

    forcing_mean = torch.mean(torch.stack(forcing_means))  # (,)
    forcing_second_moment = torch.mean(torch.stack(forcing_squares))  # (,)
    forcing_std = torch.sqrt(forcing_second_moment - forcing_mean**2)  # (,)
    forcing_stats = torch.stack((forcing_mean, forcing_std))

    print("Saving mean, std.-dev...")
    torch.save(mean, os.path.join(static_dir_path, "parameter_mean.pt"))
    torch.save(std, os.path.join(static_dir_path, "parameter_std.pt"))
    torch.save(forcing_stats, os.path.join(static_dir_path, "forcing_stats.pt"))

    # Compute mean and std.-dev. of one-step differences across the dataset
    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = WeatherDataset(
        args.dataset,
        split="train",
        subsample_step=1,
        pred_length=4,
        standardize=True,
    )  # Re-load with standardization
    loader_standard = torch.utils.data.DataLoader(
        ds_standard, args.batch_size, shuffle=False, num_workers=args.n_workers
    )
    used_subsample_len = (6 // args.step_length) * args.step_length

    diff_means = []
    diff_squares = []
    for init_batch, target_batch, _ in tqdm(loader_standard):
        batch = torch.cat(
            (init_batch, target_batch), dim=1
        )  # (N_batch, N_t', N_grid, d_features)
        # Note: batch contains only 1h-steps
        stepped_batch = torch.cat(
            [
                batch[:, ss_i : used_subsample_len : args.step_length]
                for ss_i in range(args.step_length)
            ],
            dim=0,
        )
        # (N_batch', N_t, N_grid, d_features),
        # N_batch' = args.step_length*N_batch

        batch_diffs = stepped_batch[:, 1:] - stepped_batch[:, :-1]
        # (N_batch', N_t-1, N_grid, d_features)
        masked_diff_mean = torch.sum(
            expanded_mask * batch_diffs, dim=2
        ) / torch.sum(
            expanded_mask, dim=2
        )  # (N_batch', N_t-1, d_features)
        masked_diff_squares = torch.sum(
            expanded_mask * batch_diffs**2, dim=2
        ) / torch.sum(
            expanded_mask, dim=2
        )  # (N_batch', N_t-1, d_features)
        diff_means.append(
            torch.mean(masked_diff_mean, dim=1)
        )  # (N_batch', d_features,)
        diff_squares.append(
            torch.mean(masked_diff_squares, dim=1)
        )  # (N_batch', d_features,)

    diff_mean = torch.mean(torch.cat(diff_means, dim=0), dim=0)  # (d_features)
    diff_second_moment = torch.mean(torch.cat(diff_squares, dim=0), dim=0)
    diff_std = torch.sqrt(diff_second_moment - diff_mean**2)  # (d_features)

    print("Saving one-step difference mean and std.-dev...")
    torch.save(diff_mean, os.path.join(static_dir_path, "diff_mean.pt"))
    torch.save(diff_std, os.path.join(static_dir_path, "diff_std.pt"))


if __name__ == "__main__":
    main()
