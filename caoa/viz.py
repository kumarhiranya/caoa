"""Plotly helpers for `notebooks/single_scene_pipeline.ipynb` (extra `notebook`). Everything is drawn in the scan's own
(`.ply`, z-up) coordinates; world-frame (Scan2CAD) geometry is mapped back with the scan's `trs`."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from caoa.data import scan2cad
from caoa.geometry import transform_points

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
           "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
BOX_EDGES = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def world_to_scan(trs: dict) -> np.ndarray:
    return np.linalg.inv(scan2cad.trs_matrix(trs))


def subsample(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).choice(n, max_points, replace=False) if n > max_points else np.arange(n)


def points(xyz, rgb=None, name="", size=1.5, max_points=60_000, uniform=None, opacity=1.0, showlegend=True) -> go.Scatter3d:
    """`rgb`: (N, 3) uint8 per point, or `uniform`: one CSS colour."""
    idx = subsample(len(xyz), max_points)
    xyz = np.asarray(xyz)[idx]
    c = uniform if uniform is not None else [f"rgb({r},{g},{b})" for r, g, b in np.asarray(rgb)[idx]]
    return go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name=name, opacity=opacity,
                        marker={"size": size, "color": c}, showlegend=showlegend, hoverinfo="name")


def box(center, rotation, extents, line_color="black", name="", width=4, dash=None, showlegend=True) -> go.Scatter3d:
    """Oriented box (centre, rotation columns = axes, full side lengths), as a wireframe."""
    corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * np.asarray(extents)
    corners = corners @ np.asarray(rotation).T + np.asarray(center)
    xs, ys, zs = [], [], []
    for a, b in BOX_EDGES:
        xs += [corners[a, 0], corners[b, 0], None]
        ys += [corners[a, 1], corners[b, 1], None]
        zs += [corners[a, 2], corners[b, 2], None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name, line={"color": line_color, "width": width, "dash": dash},
                        showlegend=showlegend, hoverinfo="name")


def mesh(trimesh_mesh, M=None, mesh_color="lightgrey", name="", opacity=1.0, max_faces=40_000, showlegend=True) -> go.Mesh3d:
    """A trimesh, transformed by the 4 x 4 `M` (e.g. an object's `cad_to_scan`)."""
    m = trimesh_mesh
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    if len(f) > max_faces:
        f = f[subsample(len(f), max_faces)]
    if M is not None:
        v = transform_points(v, np.asarray(M))
    return go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color=mesh_color, opacity=opacity,
                     name=name, showlegend=showlegend, flatshading=True, lighting={"ambient": 0.5, "diffuse": 0.8},
                     hoverinfo="name")


def figure(traces, title="", height=650) -> go.Figure:
    fig = go.Figure(traces)
    fig.update_layout(title=title, height=height, margin={"l": 0, "r": 0, "t": 40, "b": 0},
                      scene={"aspectmode": "data", "xaxis": {"visible": False}, "yaxis": {"visible": False},
                             "zaxis": {"visible": False}},
                      legend={"itemsizing": "constant"})
    return fig


def grid(panels, cols=5, titles=None, height_per_row=260, title="") -> go.Figure:
    """`panels`: list of trace lists, one small 3D view each."""
    rows = (len(panels) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, specs=[[{"type": "scene"}] * cols] * rows, subplot_titles=titles,
                        horizontal_spacing=0.01, vertical_spacing=0.06)
    for n, traces in enumerate(panels):
        for t in traces:
            t.showlegend = False
            fig.add_trace(t, row=n // cols + 1, col=n % cols + 1)
    hidden = {"visible": False}
    fig.update_scenes(aspectmode="data", xaxis=hidden, yaxis=hidden, zaxis=hidden)
    fig.update_layout(height=rows * height_per_row, title=title, margin={"l": 0, "r": 0, "t": 60, "b": 0})
    fig.update_annotations(font_size=10)
    return fig
