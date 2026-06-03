import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from ferrodispcalc.vis import space_animation as exported_space_animation
from ferrodispcalc.vis.space_plot import (
    _activate_point_scalars,
    _mapper_use_point_array,
    space_animation,
)


def test_space_animation_is_exported():
    assert exported_space_animation is space_animation


def test_space_animation_grid_mode_with_slider_and_frame_text():
    data = np.zeros((3, 4, 3, 2, 3), dtype=np.float32)
    data[..., 0] = 0.1
    data[1, ..., 2] = 0.2
    data[2, ..., 2] = 0.4

    plotter = pv.Plotter(off_screen=True)
    result = space_animation(
        data,
        plotter=plotter,
        color_by="dz",
        factor=1.0,
        stride=(2, 1, 1),
        show_axes=False,
        show_bounding_box=False,
        show_slider=True,
        show_frame_text=True,
        autoplay=False,
    )

    assert result is plotter
    assert plotter.actors


def test_glyph_color_scalars_remain_active_after_mapper_update():
    cloud = pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
    arrow_geom = pv.Arrow()

    cloud["displacement"] = np.array([[0.0, 0.0, 1.0]])
    cloud["magnitude"] = np.array([1.0])
    cloud["scalars"] = np.array([1.0])
    arrows = cloud.glyph(
        orient="displacement",
        scale="magnitude",
        factor=1.0,
        geom=arrow_geom,
    )
    _activate_point_scalars(arrows, "scalars")
    plotter = pv.Plotter(off_screen=True)
    actor = plotter.add_mesh(arrows, scalars="scalars", clim=[-2.0, 2.0])

    cloud["displacement"] = np.array([[0.0, 0.0, -1.0]])
    cloud["magnitude"] = np.array([1.0])
    cloud["scalars"] = np.array([-1.0])
    new_arrows = cloud.glyph(
        orient="displacement",
        scale="magnitude",
        factor=1.0,
        geom=arrow_geom,
    )
    _activate_point_scalars(new_arrows, "scalars")
    _mapper_use_point_array(actor, "scalars", new_arrows)

    assert actor.mapper.array_name == "scalars"
    assert actor.mapper.scalar_map_mode == "point"
    assert actor.mapper.dataset.active_scalars_name == "scalars"
    assert np.all(actor.mapper.dataset.active_scalars == -1.0)


def test_space_animation_point_mode_with_stride():
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    data = np.ones((2, 4, 3), dtype=np.float32)
    data[1, :, 2] = 2.0

    plotter = pv.Plotter(off_screen=True)
    result = space_animation(
        data,
        coord=coord,
        plotter=plotter,
        color_by="magnitude",
        factor=1.0,
        stride=2,
        frame_indices=[1],
        show_axes=False,
        show_bounding_box=False,
        show_slider=False,
        show_frame_text=False,
        autoplay=False,
    )

    assert result is plotter
    assert plotter.actors


def test_space_animation_requires_point_coordinates():
    data = np.ones((2, 4, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="coord is required"):
        space_animation(
            data,
            show_slider=False,
            show_frame_text=False,
            autoplay=False,
        )
