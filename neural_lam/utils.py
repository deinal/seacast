# Standard library
import os

# Third-party
import numpy as np
import torch
from torch import nn
from tueplots import bundles, figsizes

# First-party
from neural_lam import constants


def load_dataset_stats(dataset_name, device="cpu"):
    """
    Load arrays with stored dataset statistics from pre-processing
    """
    static_dir_path = os.path.join("data", dataset_name, "static")

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    forcing_stats = loads_file("forcing_stats.pt")  # (d_atm,)
    forcing_mean, forcing_std = forcing_stats

    return {
        "data_mean": data_mean,
        "data_std": data_std,
        "forcing_mean": forcing_mean,
        "forcing_std": forcing_std,
    }


def load_mask(dataset_name, device="cpu"):
    """
    Load interior mask for dataset
    """
    static_dir_path = os.path.join("data", dataset_name, "static")

    # Load sea mask, 1. if node is part of the sea, else 0.
    sea_mask_np = np.load(
        os.path.join(static_dir_path, "sea_mask.npy")
    )  # (depths, h, w)

    # Mask for the surface grid
    surface_mask_np = sea_mask_np[0]

    # Grid mask for all depth levels to be multiplied with output states
    grid_mask = torch.tensor(
        sea_mask_np[:, surface_mask_np],
        dtype=torch.float32,
        device=device,
    )  # (depths, N_grid)
    interior_mask = []
    for level_applies in constants.LEVELS:
        if level_applies:
            interior_mask.append(grid_mask)  # Multi level
        else:
            interior_mask.append(grid_mask[0].unsqueeze(0))  # Single level
    interior_mask = (
        torch.cat(interior_mask, dim=0).transpose(0, 1).unsqueeze(0)
    )  # 1, N_grid, d_features

    return {"interior_mask": interior_mask}


def load_static_data(dataset_name, device="cpu"):
    """
    Load static files related to dataset
    """
    static_dir_path = os.path.join("data", dataset_name, "static")

    def loads_file(fn):
        return torch.load(
            os.path.join(static_dir_path, fn), map_location=device
        )

    # Load sea mask, 1. if node is part of the sea, else 0.
    sea_mask_np = np.load(
        os.path.join(static_dir_path, "sea_mask.npy")
    )  # (depths, h, w)

    # Mask for the surface grid
    surface_mask_np = sea_mask_np[0]

    # Grid mask for all depth levels to be multiplied with output states
    grid_mask = torch.tensor(
        sea_mask_np[:, surface_mask_np],
        dtype=torch.float32,
        device=device,
    )  # (depths, N_grid)
    interior_mask = []
    for level_applies in constants.LEVELS:
        if level_applies:
            interior_mask.append(grid_mask)  # Multi level
        else:
            interior_mask.append(grid_mask[0].unsqueeze(0))  # Single level
    interior_mask = (
        torch.cat(interior_mask, dim=0).transpose(0, 1).unsqueeze(0)
    )  # 1, N_grid, d_features

    # Full grid mask, for plotting purposes
    full_grid_mask = torch.tensor(
        sea_mask_np, dtype=torch.float32, device=device
    ).flatten(
        1, 2
    )  # (depths, N_grid_full)
    full_mask = []
    for level_applies in constants.LEVELS:
        if level_applies:
            full_mask.append(full_grid_mask)  # Multi level
        else:
            full_mask.append(full_grid_mask[0].unsqueeze(0))  # Single level
    full_mask = (
        torch.cat(full_mask, dim=0).transpose(0, 1).to(torch.bool)
    )  # N_grid_full, d_features

    # Load cell weights
    grid_weights_np = np.load(
        os.path.join(static_dir_path, "grid_weights.npy")
    )  # (h, w)
    grid_weights = torch.tensor(
        grid_weights_np[surface_mask_np], dtype=torch.float32, device=device
    ).unsqueeze(
        1
    )  # (N_grid, 1)

    grid_static_features = loads_file(
        "grid_features.pt"
    )  # (N_grid, d_grid_static)

    # Load step diff stats
    step_diff_mean = loads_file("diff_mean.pt")  # (d_f,)
    step_diff_std = loads_file("diff_std.pt")  # (d_f,)

    # Load parameter std for computing validation errors in original data scale
    data_mean = loads_file("parameter_mean.pt")  # (d_features,)
    data_std = loads_file("parameter_std.pt")  # (d_features,)

    # Load loss weighting vectors
    param_weights = torch.tensor(
        np.load(os.path.join(static_dir_path, "parameter_weights.npy")),
        dtype=torch.float32,
        device=device,
    )  # (d_f,)

    return {
        "interior_mask": interior_mask,
        "full_mask": full_mask,
        "grid_weights": grid_weights,
        "grid_static_features": grid_static_features,
        "step_diff_mean": step_diff_mean,
        "step_diff_std": step_diff_std,
        "data_mean": data_mean,
        "data_std": data_std,
        "param_weights": param_weights,
    }


class BufferList(nn.Module):
    """
    A list of torch buffer tensors that sit together as a Module with no
    parameters and only buffers.

    This should be replaced by a native torch BufferList once implemented.
    See: https://github.com/pytorch/pytorch/issues/37386
    """

    def __init__(self, buffer_tensors, persistent=True):
        super().__init__()
        self.n_buffers = len(buffer_tensors)
        for buffer_i, tensor in enumerate(buffer_tensors):
            self.register_buffer(f"b{buffer_i}", tensor, persistent=persistent)

    def __getitem__(self, key):
        return getattr(self, f"b{key}")

    def __len__(self):
        return self.n_buffers

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def load_graph(graph_name, device="cpu"):
    """
    Load all tensors representing the graph
    """
    # Define helper lambda function
    graph_dir_path = os.path.join("graphs", graph_name)

    def loads_file(fn):
        return torch.load(os.path.join(graph_dir_path, fn), map_location=device)

    # Load edges (edge_index)
    m2m_edge_index = BufferList(
        loads_file("m2m_edge_index.pt"), persistent=False
    )  # List of (2, M_m2m[l])
    g2m_edge_index = loads_file("g2m_edge_index.pt")  # (2, M_g2m)
    m2g_edge_index = loads_file("m2g_edge_index.pt")  # (2, M_m2g)

    n_levels = len(m2m_edge_index)
    hierarchical = n_levels > 1  # Nor just single level mesh graph

    # Load static edge features
    m2m_features = loads_file("m2m_features.pt")  # List of (M_m2m[l], d_edge_f)
    g2m_features = loads_file("g2m_features.pt")  # (M_g2m, d_edge_f)
    m2g_features = loads_file("m2g_features.pt")  # (M_m2g, d_edge_f)

    # Normalize by dividing with longest edge (found in m2m)
    longest_edge = max(
        torch.max(level_features[:, 0]) for level_features in m2m_features
    )  # Col. 0 is length
    m2m_features = BufferList(
        [level_features / longest_edge for level_features in m2m_features],
        persistent=False,
    )
    g2m_features = g2m_features / longest_edge
    m2g_features = m2g_features / longest_edge

    # Load static node features
    mesh_static_features = loads_file(
        "mesh_features.pt"
    )  # List of (N_mesh[l], d_mesh_static)

    # Some checks for consistency
    assert (
        len(m2m_features) == n_levels
    ), "Inconsistent number of levels in mesh"
    assert (
        len(mesh_static_features) == n_levels
    ), "Inconsistent number of levels in mesh"

    if hierarchical:
        # Load up and down edges and features
        mesh_up_edge_index = BufferList(
            loads_file("mesh_up_edge_index.pt"), persistent=False
        )  # List of (2, M_up[l])
        mesh_down_edge_index = BufferList(
            loads_file("mesh_down_edge_index.pt"), persistent=False
        )  # List of (2, M_down[l])

        mesh_up_features = loads_file(
            "mesh_up_features.pt"
        )  # List of (M_up[l], d_edge_f)
        mesh_down_features = loads_file(
            "mesh_down_features.pt"
        )  # List of (M_down[l], d_edge_f)

        # Rescale
        mesh_up_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_up_features
            ],
            persistent=False,
        )
        mesh_down_features = BufferList(
            [
                edge_features / longest_edge
                for edge_features in mesh_down_features
            ],
            persistent=False,
        )

        mesh_static_features = BufferList(
            mesh_static_features, persistent=False
        )
    else:
        # Extract single mesh level
        m2m_edge_index = m2m_edge_index[0]
        m2m_features = m2m_features[0]
        mesh_static_features = mesh_static_features[0]

        (
            mesh_up_edge_index,
            mesh_down_edge_index,
            mesh_up_features,
            mesh_down_features,
        ) = ([], [], [], [])

    return hierarchical, {
        "g2m_edge_index": g2m_edge_index,
        "m2g_edge_index": m2g_edge_index,
        "m2m_edge_index": m2m_edge_index,
        "mesh_up_edge_index": mesh_up_edge_index,
        "mesh_down_edge_index": mesh_down_edge_index,
        "g2m_features": g2m_features,
        "m2g_features": m2g_features,
        "m2m_features": m2m_features,
        "mesh_up_features": mesh_up_features,
        "mesh_down_features": mesh_down_features,
        "mesh_static_features": mesh_static_features,
    }


def make_mlp(blueprint, layer_norm=True):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    return nn.Sequential(*layers)


def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in constants.VAL_STEP_LOG_ERRORS:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")


def get_ar_steps(total_epochs, max_steps, change_point=0.6):
    """
    Calculate progressively increasing steps and change points
    """
    start_epoch = int(change_point * total_epochs)
    interval = (total_epochs - start_epoch) // (max_steps - 1)
    change_epochs = [start_epoch + i * interval for i in range(max_steps - 1)]

    if change_epochs[-1] >= total_epochs:
        change_epochs[-1] = total_epochs - 1

    ar_steps = [i + 2 for i in range(max_steps - 1)]

    return change_epochs, ar_steps
