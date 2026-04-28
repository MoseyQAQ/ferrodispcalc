3D Space Visualization
======================

``space_profile`` visualizes 3D vector fields (e.g. displacement fields) as
arrows, powered by PyVista.

Installation
------------

.. code-block:: bash

   pip install ferrodispcalc[vis]

Quick Start
-----------

.. code-block:: python

   from ase.io import read
   from ferrodispcalc.compute import calculate_displacement
   from ferrodispcalc.neighborlist import build_neighbor_list
   from ferrodispcalc.vis import grid_data, space_profile

   atoms = read("structure.vasp")
   nl = build_neighbor_list(atoms, ["Pb"], ["O"], 4, 12, False)
   disp = calculate_displacement(atoms, nl)
   d_grid, coord = grid_data(atoms, disp, ["Pb"],
                              target_size=[40, 20, 11],
                              return_coord=True)

   # Grid mode (recommended -- supports gradient coloring)
   space_profile(d_grid, color_by="dz", cmap="coolwarm", factor=25.0)

   # Or point mode
   pts = coord.reshape(-1, 3)
   vecs = d_grid.reshape(-1, 3)
   space_profile(vecs, coord=pts, color_by="magnitude")

.. figure:: /_static/space_profile_example.png
   :align: center
   :width: 80%

   Example output of ``space_profile`` with ``color_by="dz"``.

Input Modes
-----------

``space_profile`` accepts two data layouts:

- **Grid mode** -- ``data`` has shape ``(nx, ny, nz, 3)``. Coordinates are
  generated automatically from grid indices; the ``coord`` argument is ignored.
- **Point mode** -- ``data`` has shape ``(npoint, 3)``. You must also pass
  ``coord`` with the same shape.

Coloring (``color_by``)
-----------------------

Controls how arrows are colored:

- ``'magnitude'`` -- vector norm
- ``'dx'`` / ``'dy'`` / ``'dz'`` -- single component value
- ``'all'`` -- RGB mapping (each component normalized to [0, 1])
- ``'gradient'`` -- gradient field magnitude (grid mode only)

``cmap`` sets the matplotlib colormap (default ``"coolwarm"``; ignored when
``color_by="all"``). ``clim`` overrides the colorbar range, e.g.
``clim=(-0.2, 0.2)``.

Region Selection (``select``)
-----------------------------

Display a sub-region by filtering in fractional coordinates. In grid mode,
fractions are derived from grid indices automatically. In point mode, you
must also pass the ``cell`` parameter (the 3x3 lattice matrix):

.. code-block:: python

   # Grid mode -- no cell needed
   space_profile(d_grid, color_by="dz",
                 select={"x": [0.5, 1.0], "y": None, "z": None})

   # Point mode -- cell required
   import numpy as np
   cell = np.array(atoms.cell)
   space_profile(vecs, coord=pts, color_by="dz",
                 select={"x": [0.5, 1.0], "y": None, "z": None},
                 cell=cell)

Each key (``'x'``, ``'y'``, ``'z'``) maps to a ``[lo, hi]`` range or ``None``
(no filtering on that axis).

Display Options
---------------

- ``factor`` -- arrow scale factor (default ``25.0``)
- ``projection`` -- ``"ortho"`` (default) or ``"persp"``
- ``show_bounding_box`` -- draw a wireframe box around the full point cloud
  (default ``True``)
- ``show_axes`` -- show the XYZ orientation widget (default ``True``)

Custom Plotter
--------------

Pass an existing ``pv.Plotter`` to overlay multiple visualizations or
customize the scene. When a plotter is provided, ``space_profile`` adds meshes
to it but does **not** call ``show()``, so you retain full control:

.. code-block:: python

   import pyvista as pv

   pl = pv.Plotter()
   space_profile(d_grid, color_by="dz", plotter=pl)
   pl.show()

API Reference
-------------

.. autofunction:: ferrodispcalc.vis.space_plot.space_profile
