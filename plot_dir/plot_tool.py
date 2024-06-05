import numpy as np
import plotly.graph_objects as go


def plot_root_aabb(root_list):
    fig = go.Figure()

    for root in root_list:
        s, x, y, z = root
        x_min = x - s * 0.5
        x_max = x + s * 0.5
        y_min = y - s * 0.5
        y_max = y + s * 0.5
        z_min = z - s * 0.5
        z_max = z + s * 0.5

        # Define the 8 vertices of the cube
        vertices = np.asarray([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ])

        # Define the 12 lines that make up the cube (by indices in the vertices list)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Plot each edge
        for edge in edges:
            start, end = edge
            x_coords = [vertices[start][0], vertices[end][0]]
            y_coords = [vertices[start][1], vertices[end][1]]
            z_coords = [vertices[start][2], vertices[end][2]]
            fig.add_trace(go.Scatter3d(x=x_coords, y=y_coords, z=z_coords, mode='lines', line=dict(color='red'))) 

    fig.show()
