# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# First-party
from neural_lam import constants, utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, title=None, step_length=3):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15, 20))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in days
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (days)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{name} ({unit})"
        for name, unit in zip(
            constants.EXP_PARAM_NAMES_SHORT, constants.EXP_PARAM_UNITS
        )
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    pred,
    target,
    interior_mask,
    obs_mask,
    title=None,
    colormap="viridis",
    vrange=None,
):
    """
    Plot example prediction and grond truth.
    pred: (N_grid,)
    target: (N_grid,)
    interior_mask (N_grid,)
    obs_mask (N_grid_full,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Map pred and target back onto the full grid
    obs_mask = obs_mask.cpu().numpy()
    original_shape = obs_mask.shape
    full_pred = np.full(original_shape, np.nan)
    full_target = np.full(original_shape, np.nan)
    full_pred[obs_mask] = pred[interior_mask].cpu().numpy()
    full_target[obs_mask] = target[interior_mask].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    # Plot pred and target
    for ax, data in zip(axes, (full_target, full_pred)):
        data_grid = data.reshape(*constants.GRID_SHAPE)
        im = ax.imshow(
            data_grid,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=colormap,
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, obs_mask, title=None, vrange=None):
    """
    Plot errors over spatial map
    error: (N_grid,)
    obs_mask: (N_grid_full,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Map error onto the full grid
    obs_mask = obs_mask.cpu().numpy()
    original_shape = obs_mask.shape
    full_error = np.full(original_shape, np.nan)
    full_error[obs_mask] = error.cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 4.8))

    error_grid = full_error.reshape(*constants.GRID_SHAPE)

    im = ax.imshow(
        error_grid,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig
