from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv


def space_profile(
    data: np.ndarray,
    coord: np.ndarray | None = None,
    *,
    color_by: str = "dz",
    cmap: str = "coolwarm",
    factor: float = 25.0,
    projection: str = "ortho",
    clim: tuple[float, float] | None = None,
    select: dict | None = None,
    cell: np.ndarray | None = None,
    title: str = "3D Vector Field",
    show_bounding_box: bool = True,
    show_axes: bool = True,
    plotter: pv.Plotter | None = None,
) -> pv.Plotter:
    """Plot a 3D vector field using PyVista.

    Two input modes are supported:

    * **Grid mode** – *data* has shape ``(nx, ny, nz, 3)``.  Coordinates are
      generated automatically from grid indices and *coord* is ignored.
    * **Point mode** – *data* has shape ``(npoint, 3)`` and *coord* (also
      ``(npoint, 3)``) must be provided.

    Parameters
    ----------
    data : np.ndarray
        Vector data.  ``(nx, ny, nz, 3)`` for grid mode or ``(npoint, 3)``
        for point mode.
    coord : np.ndarray or None
        Cartesian coordinates ``(npoint, 3)``.  Required for point mode,
        ignored for grid mode.
    color_by : str
        Coloring strategy: ``'magnitude'``, ``'dx'``, ``'dy'``, ``'dz'``,
        ``'all'`` (RGB from components), or ``'gradient'`` (grid mode only).
    cmap : str
        Matplotlib colormap name (ignored when *color_by='all'*).
    factor : float
        Arrow scale factor passed to ``pv.PolyData.glyph``.
    projection : str
        ``'ortho'`` for parallel projection, ``'persp'`` for perspective.
    clim : tuple or None
        Colorbar range ``(vmin, vmax)``.  *None* for automatic.
    select : dict or None
        Region filter in **fractional** coordinates.  Keys are ``'x'``,
        ``'y'``, ``'z'``; values are ``[lo, hi]`` or *None* (no filter).
        In grid mode fractions are computed from grid indices; in point
        mode *cell* is required.
    cell : np.ndarray or None
        ``(3, 3)`` lattice matrix (rows = lattice vectors).  Only needed
        when *select* is used in point mode.
    title : str
        Window title.
    show_bounding_box : bool
        Draw a wireframe box around the full point cloud.
    show_axes : bool
        Show orientation axes widget.
    plotter : pv.Plotter or None
        Reuse an existing plotter.  A new one is created when *None*.

    Returns
    -------
    pv.Plotter
        The plotter instance (already shown).
    """
    try:
        import pyvista as _pv
    except ImportError:
        raise ImportError(
            "PyVista is not installed. "
            "Install it with `pip install pyvista` or "
            "`pip install ferrodispcalc[vis]`."
        )

    # --- resolve input mode ---
    grid_data_raw: np.ndarray | None = None  # keep for gradient

    if data.ndim == 4 and data.shape[-1] == 3:
        # Grid mode (nx, ny, nz, 3)
        grid_data_raw = data
        nx, ny, nz, _ = data.shape
        gx, gy, gz = np.mgrid[0:nx, 0:ny, 0:nz]
        coord = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(np.float32)
        data = data.reshape(-1, 3)
    elif data.ndim == 2 and data.shape[-1] == 3:
        # Point mode
        if coord is None:
            raise ValueError(
                "coord is required when data has shape (npoint, 3)."
            )
        if coord.shape != data.shape:
            raise ValueError(
                f"data {data.shape} and coord {coord.shape} shape mismatch."
            )
    else:
        raise ValueError(
            f"Unsupported data shape {data.shape}. "
            "Expected (nx, ny, nz, 3) or (npoint, 3)."
        )

    pts = np.asarray(coord, dtype=np.float64)
    vecs = np.asarray(data, dtype=np.float64)

    # --- filter NaN ---
    mask = ~np.isnan(vecs).any(axis=1)
    full_bounds = [
        pts[mask, 0].min(), pts[mask, 0].max(),
        pts[mask, 1].min(), pts[mask, 1].max(),
        pts[mask, 2].min(), pts[mask, 2].max(),
    ]

    # --- region selection (fractional coords) ---
    if select is not None:
        if grid_data_raw is not None:
            # Grid mode: fractional = index / grid_size
            frac = pts / np.array([nx, ny, nz], dtype=np.float64)
        elif cell is not None:
            frac = pts @ np.linalg.inv(np.asarray(cell, dtype=np.float64))
        else:
            raise ValueError("cell is required when select is used in point mode.")
        for ax_name, ax_idx in (("x", 0), ("y", 1), ("z", 2)):
            rng = select.get(ax_name)
            if rng is not None:
                lo, hi = rng
                mask &= (frac[:, ax_idx] >= lo) & (frac[:, ax_idx] <= hi)

    pts = pts[mask]
    vecs = vecs[mask]

    # --- build scalars based on color_by ---
    _component = {"dx": 0, "dy": 1, "dz": 2}
    color_by = str(color_by).lower().strip()
    scalars_title: str | None = None
    rgb_scalars: np.ndarray | None = None
    scalar_values: np.ndarray | None = None

    if color_by == "magnitude":
        scalar_values = np.linalg.norm(vecs, axis=1)
        scalars_title = "|d|"
    elif color_by in _component:
        scalar_values = vecs[:, _component[color_by]]
        scalars_title = color_by
    elif color_by == "gradient":
        if grid_data_raw is None:
            raise ValueError(
                "color_by='gradient' requires grid mode data (nx, ny, nz, 3)."
            )
        d3 = grid_data_raw
        grad_mag = np.zeros(d3.shape[:3], dtype=np.float64)
        for ax in range(3):
            for comp in range(3):
                g = np.gradient(d3[..., comp], axis=ax)
                grad_mag += g ** 2
        grad_mag = np.sqrt(grad_mag)
        scalar_values = grad_mag.reshape(-1)[mask]
        scalars_title = "Gradient"
    elif color_by == "all":
        rgb = np.zeros_like(vecs)
        for i in range(3):
            lo, hi = vecs[:, i].min(), vecs[:, i].max()
            rgb[:, i] = (vecs[:, i] - lo) / (hi - lo + 1e-12)
        rgb_scalars = (rgb * 255).astype(np.uint8)
    else:
        raise ValueError(
            f"Unknown color_by='{color_by}'. "
            "Choose from: magnitude, dx, dy, dz, all, gradient."
        )

    # --- pyvista scene ---
    _pv.set_plot_theme("document")
    pl = plotter if plotter is not None else _pv.Plotter(title=title)

    cloud = _pv.PolyData(pts)
    cloud["displacement"] = vecs
    cloud["magnitude"] = np.linalg.norm(vecs, axis=1)

    if rgb_scalars is not None:
        cloud["RGB"] = rgb_scalars
    if scalar_values is not None:
        cloud["scalars"] = scalar_values

    arrows = cloud.glyph(
        orient="displacement", scale="magnitude",
        factor=factor, geom=_pv.Arrow(),
    )

    if rgb_scalars is not None:
        pl.add_mesh(arrows, scalars="RGB", rgb=True)
    else:
        sbar = {
            "title": scalars_title,
            "title_font_size": 14,
            "label_font_size": 12,
            "position_x": 0.82,
            "position_y": 0.25,
            "width": 0.12,
            "height": 0.5,
        }
        pl.add_mesh(
            arrows, scalars="scalars", cmap=cmap, clim=clim,
            scalar_bar_args=sbar,
        )

    if show_bounding_box:
        box = _pv.Box(bounds=full_bounds)
        pl.add_mesh(box.outline(), color="grey", line_width=1)

    if show_axes:
        pl.add_axes(
            xlabel="X", ylabel="Y", zlabel="Z",
            line_width=2, label_size=(0.12, 0.05),
        )

    pl.camera_position = "iso"
    if projection == "ortho":
        pl.enable_parallel_projection()

    if plotter is None:
        pl.show()

    return pl
