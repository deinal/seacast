# Standard library
import argparse
from pathlib import Path

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import xarray as xr


def load_mask(path, factor=2):
    """
    Load bathymetry mask.

    Args:
    path (str): Path to the bathymetry mask file.
    factor (ing): Coarsening factor.

    Returns:
    mask (xarray.DataArray): Coarsened bathymetry mask.
    """
    ds = xr.load_dataset(path)

    ds = ds.coarsen(longitude=factor, boundary="pad").mean()
    ds = ds.coarsen(latitude=factor, boundary="pad").mean()

    return ds.mask


def create_3d_scatter_plot(mask, show_axis):
    """
    Create a 3D scatter plot of the bathymetry mask.

    Args:
    mask (xarray.DataArray): Bathymetry mask.
    show_axis (bool): Flag to show or hide the axes.

    Returns:
    fig (plotly.graph_objects.Figure): Plotly 3D scatter plot.
    """

    mask_true = mask.where(mask, drop=True)

    # Get the 3D coordinates of the sea mask
    coords = (
        mask_true.stack(z=("depth", "latitude", "longitude"))
        .dropna(dim="z")
        .coords
    )

    lats_valid = coords["latitude"].values
    lons_valid = coords["longitude"].values
    depths_valid = coords["depth"].values / 70
    min_depth = depths_valid.min()

    # Define colors
    norm = mpl.colors.Normalize(
        vmin=depths_valid.min(), vmax=depths_valid.max()
    )
    blues = plt.colormaps.get_cmap("Blues")(
        norm(depths_valid) * (1 - 0.5) + 0.5
    )
    n = []
    for b in blues:
        b_tuple = tuple(b)  # Convert the RGBA array to a tuple
        if b_tuple not in n:
            n.append(b_tuple)
            print(mpl.colors.to_hex(b))

    colors = []
    for depth, lon, blue in zip(depths_valid, lons_valid, blues):
        if lon < -5.2:
            colors.append("maroon")
        elif depth == min_depth:
            colors.append("seagreen")
        else:
            color = mpl.colors.to_hex(blue)
            colors.append(color)

    # Create 3D scatter plot
    data_objs = [
        go.Scatter3d(
            x=lats_valid,
            y=lons_valid,
            z=-depths_valid,
            mode="markers",
            marker={"size": 1.4, "color": colors, "opacity": 1},
        )
    ]

    fig = go.Figure(data=data_objs)

    fig.update_layout(
        scene_aspectmode="data",
        scene={
            "xaxis": {"visible": bool(show_axis), "autorange": "reversed"},
            "yaxis": {"visible": bool(show_axis)},
            "zaxis": {"visible": bool(show_axis)},
            "camera": {"eye": {"x": 1, "y": 0, "z": 2}},
        },
        width=1400,
        height=700,
        margin={"l": 0, "r": 0, "b": 50, "t": 0},
        showlegend=False,
    )

    fig.update_traces(connectgaps=False)

    if not show_axis:
        fig.update_layout(
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            }
        )

    return fig


def main():
    """
    Plot data grid
    """
    parser = argparse.ArgumentParser(
        description="Plot bathymetry mask as 3D scatter plot"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="data/mediterranean/static/bathy_mask.nc",
        help="Path to the bathymetry mask file",
    )
    parser.add_argument(
        "--show_axis", action="store_true", help="Show axis in the plot"
    )
    parser.add_argument("--save", type=str, help="Filename to save as")
    args = parser.parse_args()

    mask = load_mask(args.path)
    fig = create_3d_scatter_plot(mask, args.show_axis)

    if args.save:
        save_path = Path("figures")
        save_path.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path / f"{args.save}.html", include_plotlyjs="cdn")
        fig.write_image(save_path / f"{args.save}.pdf", scale=1, engine="orca")
    else:
        fig.show()


if __name__ == "__main__":
    main()
