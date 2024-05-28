# Standard library
from argparse import ArgumentParser
from pathlib import Path

# Third-party
import numpy as np
import plotly.graph_objects as go
import torch_geometric as pyg

# First-party
from neural_lam import utils

MESH_HEIGHT = 0.1
MESH_LEVEL_DIST = 0.03
GRID_HEIGHT = 0


def main():
    """
    Plot graph structure in 3D using plotly
    """
    parser = ArgumentParser(description="Plot graph")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mediterranean",
        help="Datast to load grid coordinates from (default: mediterranean)",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="hierarchical",
        help="Graph to plot (default: hierarchical)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="hi_graph",
        help="Name of .html file to save interactive plot to (default: None)",
    )
    parser.add_argument(
        "--show_axis",
        type=int,
        default=0,
        help="If the axis should be displayed (default: 0 (No))",
    )
    parser.add_argument(
        "--plot_grid",
        type=int,
        default=0,
        help="If the grid should be plotted (default: 0 (No))",
    )
    parser.add_argument(
        "--plot_intra_level_edges",
        type=int,
        default=1,
        help="If the grid should be plotted (default: 1 (Yes))",
    )

    args = parser.parse_args()

    # Load graph data
    hierarchical, graph_ldict = utils.load_graph(args.graph)
    (
        g2m_edge_index,
        m2g_edge_index,
        m2m_edge_index,
    ) = (
        graph_ldict["g2m_edge_index"],
        graph_ldict["m2g_edge_index"],
        graph_ldict["m2m_edge_index"],
    )
    mesh_up_edge_index, mesh_down_edge_index = (
        graph_ldict["mesh_up_edge_index"],
        graph_ldict["mesh_down_edge_index"],
    )
    mesh_static_features = graph_ldict["mesh_static_features"]

    grid_static_features = utils.load_static_data(args.dataset)[
        "grid_static_features"
    ]

    # Extract values needed, turn to numpy
    grid_pos = grid_static_features[:, [1, 0]].numpy()
    # Add in z-dimension
    z_grid = GRID_HEIGHT * np.ones((grid_pos.shape[0],))
    grid_pos = np.concatenate(
        (grid_pos, np.expand_dims(z_grid, axis=1)), axis=1
    )

    # List of edges to plot, (edge_index, color, line_width, label)
    if args.plot_grid:
        edge_plot_list = [
            (m2g_edge_index.numpy(), "black", 0.4, "M2G"),
            (g2m_edge_index.numpy(), "black", 0.4, "G2M"),
        ]
    else:
        edge_plot_list = []

    # Mesh positioning and edges to plot differ if we have a hierarchical graph
    if hierarchical:
        mesh_level_pos = [
            np.concatenate(
                (
                    level_static_features.numpy(),
                    MESH_HEIGHT
                    + MESH_LEVEL_DIST
                    * height_level
                    * np.ones((level_static_features.shape[0], 1)),
                ),
                axis=1,
            )
            for height_level, level_static_features in enumerate(
                mesh_static_features, start=1
            )
        ]
        mesh_pos = np.concatenate(mesh_level_pos, axis=0)
        mesh_pos = mesh_pos[:, [1, 0, 2]]

        # Add inter-level mesh edges
        edge_plot_list += [
            (level_ei.numpy(), "navy", 1, f"M2M Level {level}")
            for level, level_ei in enumerate(m2m_edge_index)
        ]

        # Add intra-level mesh edges
        if args.plot_intra_level_edges:
            up_edges_ei = np.concatenate(
                [level_up_ei.numpy() for level_up_ei in mesh_up_edge_index],
                axis=1,
            )
            down_edges_ei = np.concatenate(
                [
                    level_down_ei.numpy()
                    for level_down_ei in mesh_down_edge_index
                ],
                axis=1,
            )
            edge_plot_list.append((up_edges_ei, "gold", 1, "Mesh up"))
            edge_plot_list.append((down_edges_ei, "gold", 1, "Mesh down"))

        mesh_node_size = 1
    else:
        mesh_pos = mesh_static_features.numpy()

        mesh_degrees = pyg.utils.degree(m2m_edge_index[1]).numpy()
        z_mesh = MESH_HEIGHT + 0.01 * mesh_degrees
        mesh_node_size = mesh_degrees / 2

        mesh_pos = np.concatenate(
            (mesh_pos, np.expand_dims(z_mesh, axis=1)), axis=1
        )

        edge_plot_list.append((m2m_edge_index.numpy(), "navy", 1, "M2M"))

    # All node positions in one array
    node_pos = np.concatenate((mesh_pos, grid_pos), axis=0)

    # Add edges
    data_objs = []
    for (
        ei,
        col,
        width,
        label,
    ) in edge_plot_list:
        edge_start = node_pos[ei[0]]  # (M, 2)
        edge_end = node_pos[ei[1]]  # (M, 2)
        n_edges = edge_start.shape[0]

        x_edges = np.stack(
            (edge_start[:, 0], edge_end[:, 0], np.full(n_edges, None)), axis=1
        ).flatten()
        y_edges = np.stack(
            (edge_start[:, 1], edge_end[:, 1], np.full(n_edges, None)), axis=1
        ).flatten()
        z_edges = np.stack(
            (edge_start[:, 2], edge_end[:, 2], np.full(n_edges, None)), axis=1
        ).flatten()

        scatter_obj = go.Scatter3d(
            x=x_edges,
            y=y_edges,
            z=z_edges,
            mode="lines",
            line={"color": col, "width": width},
            name=label,
        )
        data_objs.append(scatter_obj)

    # Add node objects
    if args.plot_grid:
        data_objs.append(
            go.Scatter3d(
                x=grid_pos[:, 0],
                y=grid_pos[:, 1],
                z=grid_pos[:, 2],
                mode="markers",
                marker={"color": "black", "size": 1},
                name="Grid nodes",
            )
        )
    data_objs.append(
        go.Scatter3d(
            x=mesh_pos[:, 0],
            y=mesh_pos[:, 1],
            z=mesh_pos[:, 2],
            mode="markers",
            marker={"color": "navy", "size": mesh_node_size},
            name="Mesh nodes",
        )
    )

    fig = go.Figure(data=data_objs)

    fig.update_layout(
        scene_aspectmode="data",
        scene={
            "xaxis": {"visible": bool(args.show_axis), "autorange": "reversed"},
            "yaxis": {"visible": bool(args.show_axis)},
            "zaxis": {"visible": bool(args.show_axis)},
            "camera": {"eye": {"x": 1, "y": 0, "z": 2}},
        },
        width=1400,
        height=700,
        margin={"l": 0, "r": 0, "b": 50, "t": 0},
        showlegend=False,
    )

    fig.update_traces(connectgaps=False)

    if not args.show_axis:
        # Hide axis
        fig.update_layout(
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
            }
        )

    if args.save:
        fig.write_html(
            Path("figures", f"{args.save}.html"), include_plotlyjs="cdn"
        )
        fig.write_image(Path("figures", f"{args.save}.pdf"), scale=1)
    else:
        fig.show()


if __name__ == "__main__":
    main()
