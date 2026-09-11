Parse CP2K Trajectory
=====================

This tutorial demonstrates how to read a CP2K molecular dynamics trajectory (XYZ format),
compute ionic displacements for A-site and B-site ions in a perovskite, and visualize the results
in 2D and 3D.

We use SrTiO\ :sub:`3` as an example: Sr sits on the A-site, Ti on the B-site, and O is the
shared neighbor species.

Complete Script
---------------

.. code-block:: python

   from ferrodispcalc.neighborlist import build_neighbor_list
   from ferrodispcalc.compute import calculate_displacement
   from ferrodispcalc.vis import grid_data, space_profile, plane_profile
   from ase.io import read
   from ase import Atoms

   # 1. Read CP2K trajectory
   traj: list[Atoms] = read("sto-pos-1.xyz", ":")

   # 2. Set cell and PBC
   cell = [
       [15.390867, 0.0, 0.0],
       [0.0, 15.390867, 0.0],
       [0.0, 0.0, 15.929633],
   ]
   for atoms in traj:
       atoms.set_cell(cell)
       atoms.set_pbc([True, True, True])

   # 3. Build neighbor lists
   nl_b = build_neighbor_list(traj[0], ["Ti"], ["O"], 4, 6)
   nl_a = build_neighbor_list(traj[0], ["Sr"], ["O"], 4, 12)

   # 4. Calculate displacements & grid
   disp_a = calculate_displacement(traj, nl_a, select=slice(0, None, 1))
   disp_b = calculate_displacement(traj, nl_b, select=slice(0, None, 1))

   import numpy as np
   # Print mean displacement of the last frame
   print("A-site mean displacement (last frame):", np.mean(disp_a[-1], axis=0))
   print("B-site mean displacement (last frame):", np.mean(disp_b[-1], axis=0))

   disp_a = grid_data(traj[0], disp_a, ["Sr"], target_size=[4, 4, 4])
   disp_b = grid_data(traj[0], disp_b, ["Ti"], target_size=[4, 4, 4])

   # 5. 2D visualization
   plane_profile(disp_a[-1], save_dir="d_a")
   plane_profile(disp_b[-1], save_dir="d_b")

   # 6. 3D visualization
   space_profile(disp_b[-1], color_by="dz", factor=6)

Step-by-Step Walkthrough
------------------------

Step 1: Read CP2K Trajectory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CP2K MD simulations produce trajectory files in ``*-pos-*.xyz`` format. Use ASE's ``read``
function to load them:

.. code-block:: python

   from ase.io import read
   from ase import Atoms

   traj: list[Atoms] = read("sto-pos-1.xyz", ":")

The ``":"`` index tells ASE to read all frames. The return value is a list of ``Atoms`` objects,
one per trajectory frame.

Step 2: Set Cell and PBC
^^^^^^^^^^^^^^^^^^^^^^^^^

CP2K's XYZ output typically does not contain cell information, so you need to set the cell
parameters and periodic boundary conditions (PBC) manually:

.. code-block:: python

   cell = [
       [15.390867, 0.0, 0.0],
       [0.0, 15.390867, 0.0],
       [0.0, 0.0, 15.929633],
   ]
   for atoms in traj:
       atoms.set_cell(cell)
       atoms.set_pbc([True, True, True])

The cell parameters should match the ``&CELL`` section in your CP2K input file. Setting PBC to
``True`` in all three directions is essential for the subsequent neighbor search and the minimum
image convention (MIC).

Step 3: Build Neighbor Lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Build separate neighbor lists for the A-site (Sr) and B-site (Ti) ions:

.. code-block:: python

   from ferrodispcalc.neighborlist import build_neighbor_list

   # B-site: Ti is coordinated by 6 O (octahedral)
   nl_b = build_neighbor_list(traj[0], ["Ti"], ["O"], cutoff=4, neighbor_num=6)

   # A-site: Sr is coordinated by 12 O
   nl_a = build_neighbor_list(traj[0], ["Sr"], ["O"], cutoff=4, neighbor_num=12)

Parameters:

- ``center_elements`` -- element symbols for the center atoms
- ``neighbor_elements`` -- element symbols for the neighbor atoms
- ``cutoff`` -- search cutoff radius (unit: Angstrom)
- ``neighbor_num`` -- number of neighbors to keep per center atom

Only the first frame (``traj[0]``) is needed to build the neighbor list; the same topology is
reused for all subsequent frames.

Step 4: Calculate Displacements & Grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Compute ionic displacements and map the results onto a regular grid:

.. code-block:: python

   from ferrodispcalc.compute import calculate_displacement
   from ferrodispcalc.vis import grid_data

   # Displacement = offset of each center atom from the centroid of its neighbors
   disp_a = calculate_displacement(traj, nl_a, select=slice(0, None, 1))
   disp_b = calculate_displacement(traj, nl_b, select=slice(0, None, 1))

``select=slice(0, None, 1)`` selects all frames (equivalent to ``[::1]``). If set to ``None``,
only the last 50% of frames are used by default.

The returned array has shape ``(n_frames, n_centers, 3)`` -- a 3D displacement vector for each
center atom in each frame.

You can print the spatially averaged displacement of the last frame to get a quick sanity check:

.. code-block:: python

   import numpy as np

   print("A-site mean displacement (last frame):", np.mean(disp_a[-1], axis=0))
   print("B-site mean displacement (last frame):", np.mean(disp_b[-1], axis=0))

This gives the mean displacement vector (dx, dy, dz) in Angstrom, averaged over all center atoms
in the last frame. A non-zero mean along a particular axis indicates a net polar displacement in
that direction.

Next, map the scattered data onto a regular grid:

.. code-block:: python

   disp_a = grid_data(traj[0], disp_a, ["Sr"], target_size=[4, 4, 4])
   disp_b = grid_data(traj[0], disp_b, ["Ti"], target_size=[4, 4, 4])

``target_size=[4, 4, 4]`` specifies the expected grid dimensions (i.e., 4 equivalent sites along
each direction in the supercell). The output shape is ``(n_frames, 4, 4, 4, 3)``.

Step 5: 2D Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``plane_profile`` to plot 2D vector fields on cross-sectional slices:

.. code-block:: python

   from ferrodispcalc.vis import plane_profile

   plane_profile(disp_a[-1], save_dir="d_a")
   plane_profile(disp_b[-1], save_dir="d_b")

``disp_a[-1]`` takes the last frame, with shape ``(4, 4, 4, 3)``. The function slices along the
x, y, and z directions layer by layer, projects the vectors onto the corresponding plane, and
saves PNG images to the ``save_dir`` directory.

You can use the ``select`` parameter to plot only specific layers, for example:

.. code-block:: python

   plane_profile(disp_b[-1], save_dir="d_b", select={"z": [0, 1]})

Step 6: 3D Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``space_profile`` to render a 3D vector field:

.. code-block:: python

   from ferrodispcalc.vis import space_profile

   space_profile(disp_b[-1], color_by="dz", factor=6)

Parameters:

- ``color_by`` -- coloring strategy: ``'dz'`` colors by the z-component; other options include ``'dx'``, ``'dy'``, ``'magnitude'``, ``'all'`` (RGB from components), etc.
- ``factor`` -- arrow scale factor; larger values produce longer arrows

This function is built on PyVista and opens an interactive 3D window.
