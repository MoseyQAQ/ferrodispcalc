Example: Wurtzite Ferroelectric (ScAlN)
=======================================

This example computes the out-of-plane cation displacement (D\ :sub:`z`) in a
wurtzite-type ferroelectric such as Sc\ :sub:`x`\ Al\ :sub:`1-x`\ N.

In wurtzite, each cation (Ga/Sc/Al) is tetrahedrally coordinated by 4 N atoms:
1 along the c-axis and 3 in the basal plane. The ferroelectric polarization is
governed by the cation's displacement relative to its N neighbors along z.
The key challenge is that we only want the 3 in-plane N neighbors (not the axial one)
to define the coordination center, so we need a custom neighbor list.
For core concepts, see :doc:`/tutorials/fdc`.

Complete Script
---------------

.. code-block:: python

   import numpy as np
   from ase.io import read
   from ferrodispcalc.neighborlist import build_neighbor_list
   from ferrodispcalc.compute import calculate_displacement

   atoms = read("avg.xsf")

   # --- Custom neighbor list for wurtzite ---
   # Start with all 4 N neighbors per cation
   nl_full = build_neighbor_list(atoms, ["Ga", "Sc"], ["N"], cutoff=4, neighbor_num=4)

   center_id = nl_full[:, 0]
   neighbor_id = nl_full[:, 1:]

   # Filter: keep only the 3 neighbors in the basal plane (small dz)
   box = atoms.get_cell().array
   box_inv = np.linalg.inv(box)
   center_pos = atoms.get_positions()[center_id - 1]
   nl = np.full((nl_full.shape[0], 4), -1, dtype=int)  # 1 center + 3 neighbors
   nl[:, 0] = center_id

   for i, neighbors in enumerate(neighbor_id):
       diff = atoms.get_positions()[neighbors - 1] - center_pos[i]
       diff_frac = diff @ box_inv
       diff_frac[diff_frac > 0.5] -= 1.0
       diff_frac[diff_frac < -0.5] += 1.0
       diff = diff_frac @ box
       in_plane = np.abs(diff[:, 2]) < 1.0  # small z-component = basal plane
       nl[i, 1:] = neighbors[in_plane]

   # --- Compute displacement ---
   disp = calculate_displacement(atoms, nl)
   mean_Dz = np.mean(disp[:, 2])
   print(f"Mean Dz: {mean_Dz:.4f} Angstrom")

Step-by-Step Breakdown
----------------------

Why a Custom Neighbor List?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``build_neighbor_list`` returns the 4 nearest N neighbors for each cation.
In wurtzite, these 4 neighbors split into two groups:

- 3 in the basal plane (roughly the same z as the cation)
- 1 along the c-axis (large dz)

For computing D\ :sub:`z`, we want the displacement of the cation relative to the
3 in-plane neighbors only. So we filter by ``|dz| < 1 Angstrom`` after applying
the minimum image convention:

.. code-block:: python

   nl_full = build_neighbor_list(atoms, ["Ga", "Sc"], ["N"], cutoff=4, neighbor_num=4)

   # For each center, compute MIC-corrected dz to each neighbor
   for i, neighbors in enumerate(neighbor_id):
       diff = atoms.get_positions()[neighbors - 1] - center_pos[i]
       diff_frac = diff @ box_inv
       diff_frac[diff_frac > 0.5] -= 1.0
       diff_frac[diff_frac < -0.5] += 1.0
       diff = diff_frac @ box
       in_plane = np.abs(diff[:, 2]) < 1.0
       nl[i, 1:] = neighbors[in_plane]

The resulting ``nl`` has shape ``(n_cations, 4)`` — column 0 is the center index,
columns 1-3 are the 3 in-plane N neighbors.

Compute D\ :sub:`z`
^^^^^^^^^^^^^^^^^^^^

With the filtered neighbor list, ``calculate_displacement`` gives the cation offset
from the center of its 3 basal-plane N neighbors. The z-component is D\ :sub:`z`:

.. code-block:: python

   disp = calculate_displacement(atoms, nl)  # shape (n_cations, 3)
   Dz = disp[:, 2]  # z-component only

See :func:`~ferrodispcalc.compute.calculate_displacement` and
:func:`~ferrodispcalc.neighborlist.build_neighbor_list` for API details.
