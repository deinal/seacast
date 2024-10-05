# Standard library
import os
from argparse import ArgumentParser

# Third-party
import numpy as np
import torch


def main():
    """
    Pre-compute all static features related to the grid nodes
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
        help="Dataset to compute weights for (default: mediterranean)",
    )
    args = parser.parse_args()

    static_dir_path = os.path.join("data", args.dataset, "static")

    # -- Static grid node features --
    coordinates = torch.tensor(
        np.load(os.path.join(static_dir_path, "coordinates.npy"))
    )  # (2, N_x, N_y)
    coordinates = coordinates.flatten(1, 2).T  # (N_grid_full, 2)
    pos_max = torch.max(torch.abs(coordinates))
    coordinates = coordinates / pos_max  # Divide by maximum coordinate

    sea_depth = torch.tensor(
        np.load(os.path.join(static_dir_path, "sea_depth.npy"))
    )  # (N_x, N_y)
    sea_depth = sea_depth.flatten(0, 1).unsqueeze(1)  # (N_grid_full, 1)
    gp_min = torch.min(sea_depth)
    gp_max = torch.max(sea_depth)
    # Rescale sea_depth to [0,1]
    sea_depth = (sea_depth - gp_min) / (gp_max - gp_min)  # (N_grid_full, 1)

    sea_topography = torch.tensor(
        np.load(os.path.join(static_dir_path, "sea_topography.npy"))
    )  # (N_x, N_y)
    sea_topography = sea_topography.flatten(0, 1).unsqueeze(
        1
    )  # (N_grid_full, 1)
    gp_min = torch.min(sea_topography)
    gp_max = torch.max(sea_topography)
    # Rescale sea_topography to [0,1]
    sea_topography = (sea_topography - gp_min) / (
        gp_max - gp_min
    )  # (N_grid_full, 1)

    sea_mask = torch.tensor(
        np.load(os.path.join(static_dir_path, "sea_mask.npy"))[0],
        dtype=torch.int64,
    )  # (N_x, N_y)
    sea_mask = sea_mask.flatten(0, 1).to(torch.bool)  # (N_grid_full,)

    # Concatenate grid features
    grid_features = torch.cat(
        (coordinates, sea_depth, sea_topography), dim=1
    )  # (N_grid_full, 4)
    grid_features = grid_features[sea_mask]  # (N_grid, 4)

    torch.save(grid_features, os.path.join(static_dir_path, "grid_features.pt"))


if __name__ == "__main__":
    main()
