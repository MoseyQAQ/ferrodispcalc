from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv


_COMPONENT = {"dx": 0, "dy": 1, "dz": 2}
_COLOR_OPTIONS = "magnitude, dx, dy, dz, all, gradient"


def _import_pyvista():
    try:
        import pyvista as _pv
    except ImportError:
        raise ImportError(
            "PyVista is not installed. "
            "Install it with `pip install pyvista` or "
            "`pip install ferrodispcalc[vis]`."
        )
    return _pv


def _grid_points(
    shape: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    nx, ny, nz = shape
    sx, sy, sz = stride
    gx, gy, gz = np.meshgrid(
        np.arange(0, nx, sx, dtype=np.float32),
        np.arange(0, ny, sy, dtype=np.float32),
        np.arange(0, nz, sz, dtype=np.float32),
        indexing="ij",
    )
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


def _bounds(points: np.ndarray) -> list[float]:
    if points.size == 0:
        raise ValueError("No valid points are available for plotting.")
    return [
        points[:, 0].min(), points[:, 0].max(),
        points[:, 1].min(), points[:, 1].max(),
        points[:, 2].min(), points[:, 2].max(),
    ]


def _apply_select(
    mask: np.ndarray,
    pts: np.ndarray,
    select: dict | None,
    *,
    grid_shape: tuple[int, int, int] | None = None,
    cell: np.ndarray | None = None,
) -> np.ndarray:
    if select is None:
        return mask

    if grid_shape is not None:
        frac = pts / np.array(grid_shape, dtype=np.float64)
    elif cell is not None:
        frac = pts @ np.linalg.inv(np.asarray(cell, dtype=np.float64))
    else:
        raise ValueError("cell is required when select is used in point mode.")

    for ax_name, ax_idx in (("x", 0), ("y", 1), ("z", 2)):
        rng = select.get(ax_name)
        if rng is not None:
            lo, hi = rng
            mask &= (frac[:, ax_idx] >= lo) & (frac[:, ax_idx] <= hi)
    return mask


def _component_rgb(vecs: np.ndarray) -> np.ndarray:
    rgb = np.zeros_like(vecs)
    for i in range(3):
        lo, hi = vecs[:, i].min(), vecs[:, i].max()
        rgb[:, i] = (vecs[:, i] - lo) / (hi - lo + 1e-12)
    return (rgb * 255).astype(np.uint8)


def _color_arrays(
    vecs: np.ndarray,
    color_by: str,
    *,
    grid_data_raw: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[str | None, np.ndarray | None, np.ndarray | None]:
    color_by = str(color_by).lower().strip()
    scalars_title: str | None = None
    rgb_scalars: np.ndarray | None = None
    scalar_values: np.ndarray | None = None

    if color_by == "magnitude":
        scalar_values = np.linalg.norm(vecs, axis=1)
        scalars_title = "|d|"
    elif color_by in _COMPONENT:
        scalar_values = vecs[:, _COMPONENT[color_by]]
        scalars_title = color_by
    elif color_by == "gradient":
        if grid_data_raw is None or mask is None:
            raise ValueError(
                "color_by='gradient' requires grid mode data (nx, ny, nz, 3)."
            )
        grad_mag = np.zeros(grid_data_raw.shape[:3], dtype=np.float64)
        for ax in range(3):
            for comp in range(3):
                g = np.gradient(grid_data_raw[..., comp], axis=ax)
                grad_mag += g ** 2
        scalar_values = np.sqrt(grad_mag).reshape(-1)[mask]
        scalars_title = "Gradient"
    elif color_by == "all":
        rgb_scalars = _component_rgb(vecs)
    else:
        raise ValueError(
            f"Unknown color_by='{color_by}'. Choose from: {_COLOR_OPTIONS}."
        )

    return scalars_title, rgb_scalars, scalar_values


def _scalar_bar_args(title: str | None) -> dict:
    return {
        "title": title,
        "title_font_size": 14,
        "label_font_size": 12,
        "position_x": 0.82,
        "position_y": 0.25,
        "width": 0.12,
        "height": 0.5,
    }


def _set_text(text_actor, text: str, position: str = "upper_left") -> None:
    if hasattr(text_actor, "set_text"):
        text_actor.set_text(position, text)
    elif hasattr(text_actor, "SetInput"):
        text_actor.SetInput(text)
    elif hasattr(text_actor, "SetText"):
        text_actor.SetText(text)
    else:
        raise TypeError(f"Unsupported text actor type: {type(text_actor)!r}.")


def _normalize_grid_stride(stride: int | Sequence[int]) -> tuple[int, int, int]:
    if np.isscalar(stride):
        values = (int(stride),) * 3
    else:
        values = tuple(int(v) for v in stride)
    if len(values) != 3:
        raise ValueError("stride must be an int or a 3-item sequence in grid mode.")
    if any(v <= 0 for v in values):
        raise ValueError("stride values must be positive.")
    return values


def _normalize_point_stride(stride: int | Sequence[int]) -> int:
    if not np.isscalar(stride):
        raise ValueError("stride must be an int in point mode.")
    value = int(stride)
    if value <= 0:
        raise ValueError("stride must be positive.")
    return value


def _frame_indices(
    nframe: int,
    frame_indices: Sequence[int] | slice | None,
    frame_step: int,
) -> list[int]:
    frame_step = int(frame_step)
    if frame_step <= 0:
        raise ValueError("frame_step must be positive.")

    if frame_indices is None:
        indices = list(range(0, nframe, frame_step))
    elif isinstance(frame_indices, slice):
        indices = list(range(nframe))[frame_indices]
        if frame_step != 1:
            indices = indices[::frame_step]
    else:
        indices = [int(i) for i in frame_indices]

    if not indices:
        raise ValueError("No frames were selected for animation.")
    if min(indices) < 0 or max(indices) >= nframe:
        raise ValueError(f"frame_indices must be within [0, {nframe - 1}].")
    return indices


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
    _pv = _import_pyvista()

    # --- resolve input mode ---
    grid_data_raw: np.ndarray | None = None  # keep for gradient
    grid_shape: tuple[int, int, int] | None = None

    if data.ndim == 4 and data.shape[-1] == 3:
        # Grid mode (nx, ny, nz, 3)
        grid_data_raw = data
        grid_shape = tuple(int(v) for v in data.shape[:3])
        coord = _grid_points(grid_shape)
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
    full_bounds = _bounds(pts[mask])

    # --- region selection (fractional coords) ---
    mask = _apply_select(mask, pts, select, grid_shape=grid_shape, cell=cell)

    pts = pts[mask]
    vecs = vecs[mask]

    # --- build scalars based on color_by ---
    scalars_title, rgb_scalars, scalar_values = _color_arrays(
        vecs,
        color_by,
        grid_data_raw=grid_data_raw,
        mask=mask,
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
        pl.add_mesh(
            arrows, scalars="scalars", cmap=cmap, clim=clim,
            scalar_bar_args=_scalar_bar_args(scalars_title),
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


def space_animation(
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
    title: str = "3D Vector Field Animation",
    show_bounding_box: bool = True,
    show_axes: bool = True,
    show_slider: bool = True,
    show_frame_text: bool = True,
    plotter: pv.Plotter | None = None,
    stride: int | Sequence[int] = 1,
    frame_indices: Sequence[int] | slice | None = None,
    frame_step: int = 1,
    fps: float = 20.0,
    autoplay: bool = True,
    loop: bool = True,
    dtype: type | str | np.dtype = np.float32,
) -> pv.Plotter:
    """Play a 3-D vector-field animation in a PyVista window.

    Two time-dependent input modes are supported:

    * **Grid mode** - *data* has shape ``(nframe, nx, ny, nz, 3)``.
      Coordinates are generated automatically from grid indices and *coord*
      is ignored.
    * **Point mode** - *data* has shape ``(nframe, npoint, 3)`` and *coord*
      has shape ``(npoint, 3)``.

    The scene is created once and updated in-place as frames advance. A slider
    widget is shown by default for direct frame selection. Keyboard shortcuts
    are available when more than one frame is selected: space toggles playback,
    and the left/right arrow keys step backward/forward by one displayed frame.

    Parameters
    ----------
    data : np.ndarray
        Vector animation data. Expected shape is ``(nframe, nx, ny, nz, 3)``
        for grid mode or ``(nframe, npoint, 3)`` for point mode.
    coord : np.ndarray or None
        Cartesian coordinates ``(npoint, 3)``. Required for point mode,
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
        Colorbar range ``(vmin, vmax)``. *None* updates the scalar range from
        the current frame.
    select : dict or None
        Region filter in fractional coordinates. Keys are ``'x'``, ``'y'``,
        ``'z'``; values are ``[lo, hi]`` or *None* (no filter). In grid mode
        fractions are computed from grid indices; in point mode *cell* is
        required.
    cell : np.ndarray or None
        ``(3, 3)`` lattice matrix (rows = lattice vectors). Only needed when
        *select* is used in point mode.
    title : str
        Window title.
    show_bounding_box : bool
        Draw a wireframe box around the full point cloud.
    show_axes : bool
        Show orientation axes widget.
    show_slider : bool
        Show a frame slider at the bottom of the window.
    show_frame_text : bool
        Show the current frame index in the upper-left corner.
    plotter : pv.Plotter or None
        Reuse an existing plotter. A new one is created when *None*.
    stride : int or sequence of int
        Spatial down-sampling. In grid mode, an int applies to all axes and a
        3-item sequence applies to ``x``, ``y`` and ``z``. In point mode, only
        an int stride is accepted.
    frame_indices : sequence of int, slice, or None
        Frames to expose in the animation. *None* selects all frames after
        applying *frame_step*.
    frame_step : int
        Temporal down-sampling step used when *frame_indices* is *None*; also
        applied to a slice selection when greater than one.
    fps : float
        Playback rate in displayed frames per second.
    autoplay : bool
        Start playback immediately when the window opens.
    loop : bool
        Restart from the first displayed frame after the last one.
    dtype : type, str, or np.dtype
        Floating dtype used for the per-frame vector arrays passed to PyVista.

    Returns
    -------
    pv.Plotter
        The plotter instance. When *plotter* is *None*, the window is shown
        before returning.
    """
    _pv = _import_pyvista()

    if fps <= 0:
        raise ValueError("fps must be positive.")

    data = np.asarray(data)
    grid_mode = False
    grid_shape: tuple[int, int, int] | None = None
    grid_stride: tuple[int, int, int] | None = None
    point_indices: np.ndarray | None = None

    if data.ndim == 5 and data.shape[-1] == 3:
        grid_mode = True
        nframe = int(data.shape[0])
        grid_shape = tuple(int(v) for v in data.shape[1:4])
        grid_stride = _normalize_grid_stride(stride)
        pts_all = _grid_points(grid_shape, grid_stride)

        def raw_frame(frame_index: int) -> np.ndarray:
            sx, sy, sz = grid_stride
            return data[frame_index, ::sx, ::sy, ::sz, :]

        first_grid = raw_frame(0)
        first_vecs_all = first_grid.reshape(-1, 3)
    elif data.ndim == 3 and data.shape[-1] == 3:
        nframe = int(data.shape[0])
        if coord is None:
            raise ValueError(
                "coord is required when data has shape (nframe, npoint, 3)."
            )
        coord = np.asarray(coord)
        if coord.ndim != 2 or coord.shape != data.shape[1:]:
            raise ValueError(
                f"coord must have shape {data.shape[1:]}; got {coord.shape}."
            )
        point_stride = _normalize_point_stride(stride)
        point_indices = np.arange(0, data.shape[1], point_stride)
        pts_all = coord[point_indices]

        def raw_frame(frame_index: int) -> np.ndarray:
            return data[frame_index, point_indices, :]

        first_grid = None
        first_vecs_all = raw_frame(0)
    else:
        raise ValueError(
            f"Unsupported data shape {data.shape}. "
            "Expected (nframe, nx, ny, nz, 3) or (nframe, npoint, 3)."
        )

    frames = _frame_indices(nframe, frame_indices, frame_step)
    first_frame = frames[0]

    if first_frame != 0:
        if grid_mode:
            first_grid = raw_frame(first_frame)
            first_vecs_all = first_grid.reshape(-1, 3)
        else:
            first_vecs_all = raw_frame(first_frame)

    pts_all = np.asarray(pts_all, dtype=np.float64)
    first_vecs_all = np.asarray(first_vecs_all, dtype=dtype)

    mask = ~np.isnan(first_vecs_all).any(axis=1)
    full_bounds = _bounds(pts_all[mask])
    mask = _apply_select(mask, pts_all, select, grid_shape=grid_shape, cell=cell)
    if not np.any(mask):
        raise ValueError("No points remain after applying select/mask filters.")

    pts = pts_all[mask]

    def vectors_for(frame_index: int) -> tuple[np.ndarray, np.ndarray | None]:
        grid_raw = raw_frame(frame_index) if grid_mode else None
        if grid_raw is None:
            vecs = raw_frame(frame_index)
        else:
            vecs = grid_raw.reshape(-1, 3)
        vecs = np.asarray(vecs[mask], dtype=dtype)
        if not np.isfinite(vecs).all():
            vecs = np.nan_to_num(vecs, copy=True)
        return vecs, grid_raw

    vecs, grid_data_raw = vectors_for(first_frame)
    scalars_title, rgb_scalars, scalar_values = _color_arrays(
        vecs,
        color_by,
        grid_data_raw=grid_data_raw,
        mask=mask,
    )

    _pv.set_plot_theme("document")
    pl = plotter if plotter is not None else _pv.Plotter(title=title)

    cloud = _pv.PolyData(pts)
    cloud["displacement"] = vecs
    cloud["magnitude"] = np.linalg.norm(vecs, axis=1)
    if rgb_scalars is not None:
        cloud["RGB"] = rgb_scalars
    if scalar_values is not None:
        cloud["scalars"] = scalar_values

    arrow_geom = _pv.Arrow()
    arrows = cloud.glyph(
        orient="displacement",
        scale="magnitude",
        factor=factor,
        geom=arrow_geom,
    )

    if rgb_scalars is not None:
        actor = pl.add_mesh(arrows, scalars="RGB", rgb=True)
    else:
        actor = pl.add_mesh(
            arrows,
            scalars="scalars",
            cmap=cmap,
            clim=clim,
            scalar_bar_args=_scalar_bar_args(scalars_title),
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

    state = {
        "slot": 0,
        "playing": bool(autoplay),
        "slider_lock": False,
    }
    slider_widget = None
    frame_text = None
    if show_frame_text:
        frame_text = pl.add_text(
            f"Frame {first_frame} ({1}/{len(frames)})",
            position="upper_left",
            font_size=10,
            color="black",
        )

    def set_slot(slot: int, *, update_slider: bool = True) -> None:
        slot = max(0, min(int(slot), len(frames) - 1))
        state["slot"] = slot
        frame_index = frames[slot]
        vecs, grid_data_raw = vectors_for(frame_index)
        scalars_title, rgb_scalars, scalar_values = _color_arrays(
            vecs,
            color_by,
            grid_data_raw=grid_data_raw,
            mask=mask,
        )

        cloud["displacement"] = vecs
        cloud["magnitude"] = np.linalg.norm(vecs, axis=1)
        if rgb_scalars is not None:
            cloud["RGB"] = rgb_scalars
        if scalar_values is not None:
            cloud["scalars"] = scalar_values
            if clim is None and hasattr(actor, "mapper"):
                actor.mapper.scalar_range = (
                    float(np.nanmin(scalar_values)),
                    float(np.nanmax(scalar_values)),
                )

        new_arrows = cloud.glyph(
            orient="displacement",
            scale="magnitude",
            factor=factor,
            geom=arrow_geom,
        )
        arrows.copy_from(new_arrows)

        if frame_text is not None:
            _set_text(frame_text, f"Frame {frame_index} ({slot + 1}/{len(frames)})")
        if update_slider and slider_widget is not None:
            state["slider_lock"] = True
            slider_widget.GetRepresentation().SetValue(float(slot))
            state["slider_lock"] = False
        pl.render()

    def slider_callback(value: float) -> None:
        if state["slider_lock"]:
            return
        state["playing"] = False
        set_slot(round(value), update_slider=False)

    if show_slider and len(frames) > 1:
        slider_widget = pl.add_slider_widget(
            slider_callback,
            [0, len(frames) - 1],
            value=0,
            title="Frame",
            pointa=(0.20, 0.08),
            pointb=(0.80, 0.08),
            style="modern",
            fmt="%0.0f",
        )

    def step_forward() -> None:
        slot = state["slot"] + 1
        if slot >= len(frames):
            if loop:
                slot = 0
            else:
                state["playing"] = False
                slot = len(frames) - 1
        set_slot(slot)

    def step_backward() -> None:
        slot = state["slot"] - 1
        if slot < 0:
            slot = len(frames) - 1 if loop else 0
        set_slot(slot)

    def toggle_play() -> None:
        state["playing"] = not state["playing"]

    if len(frames) > 1:
        pl.add_key_event("space", toggle_play)
        pl.add_key_event("Right", step_forward)
        pl.add_key_event("Left", step_backward)

        def timer_callback(_step: int) -> None:
            if state["playing"]:
                step_forward()

        pl.add_timer_event(
            max_steps=2_000_000_000,
            duration=int(round(1000.0 / fps)),
            callback=timer_callback,
        )

    if plotter is None:
        pl.show()

    return pl
