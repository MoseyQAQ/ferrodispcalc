Example: Dielectric Constant of PbTiO\ :sub:`3`
================================================

This example computes the dielectric tensor of PbTiO\ :sub:`3` directly from an
MD trajectory.
For background on local polarization, see :doc:`/tutorials/fdc`.

We start from a single trajectory file, ``movie.xyz``. The workflow is:

1. read the trajectory
2. build neighbor lists
3. calculate local polarization
4. calculate the averaged structure and its volume
5. calculate the dielectric tensor

Complete Script
---------------

.. code-block:: python

   from ase.io import read, write

   from ferrodispcalc.config import BEC
   from ferrodispcalc.compute import (
       calculate_averaged_structure,
       calculate_dielectric_constant,
       calculate_polarization,
   )
   from ferrodispcalc.neighborlist import build_neighbor_list


   traj = read("movie.xyz", index=":")
   atoms = traj[0]

   nl_bo = build_neighbor_list(atoms, ["Ti"], ["O"], cutoff=4, neighbor_num=6)
   nl_ba = build_neighbor_list(atoms, ["Ti"], ["Pb"], cutoff=4, neighbor_num=8)

   P = calculate_polarization(
       traj,
       nl_ba=nl_ba,
       nl_bx=nl_bo,
       born_effective_charge=BEC["PTO"],
       select=slice(None, None, 1),
   )

   avg_atoms = calculate_averaged_structure(traj, select=slice(None, None, 1))
   write("avg.vasp", avg_atoms)

   dielectric = calculate_dielectric_constant(
       P,
       volume=avg_atoms.cell.volume,
       temperature=300.0,
       atomic=False,
   )

   print(dielectric)

Step-by-Step Breakdown
----------------------

Read the Trajectory
^^^^^^^^^^^^^^^^^^^

Use ASE to read the MD trajectory.

.. code-block:: python

   from ase.io import read

   traj = read("movie.xyz", index=":")
   atoms = traj[0]

``traj`` is a list of ASE ``Atoms`` objects. We use the first frame to build
neighbor lists.

Build Neighbor Lists
^^^^^^^^^^^^^^^^^^^^

For PbTiO\ :sub:`3`, local polarization around each Ti site needs two neighbor
lists:

- Ti-O: 6 oxygen neighbors
- Ti-Pb: 8 A-site neighbors

.. code-block:: python

   from ferrodispcalc.neighborlist import build_neighbor_list

   nl_bo = build_neighbor_list(atoms, ["Ti"], ["O"], cutoff=4, neighbor_num=6)
   nl_ba = build_neighbor_list(atoms, ["Ti"], ["Pb"], cutoff=4, neighbor_num=8)

Calculate Local Polarization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now compute the local polarization trajectory using the Born effective charges
for PbTiO\ :sub:`3`.

.. code-block:: python

   from ferrodispcalc.config import BEC
   from ferrodispcalc.compute import calculate_polarization

   P = calculate_polarization(
       traj,
       nl_ba=nl_ba,
       nl_bx=nl_bo,
       born_effective_charge=BEC["PTO"],
       select=slice(None, None, 1),
   )

``P`` has shape ``(n_frames, n_cells, 3)``. The last dimension stores
``P_x``, ``P_y``, and ``P_z`` for each local unit cell.

Calculate the Averaged Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The dielectric calculation needs the system volume in Å\ :sup:`3`.
A convenient choice is to compute the time-averaged structure and use its cell
volume.

.. code-block:: python

   from ase.io import write
   from ferrodispcalc.compute import calculate_averaged_structure

   avg_atoms = calculate_averaged_structure(traj, select=slice(None, None, 1))
   write("avg.vasp", avg_atoms)

The averaged structure is also useful for later inspection or visualization.

Calculate the Dielectric Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, compute the dielectric tensor from the polarization fluctuation.

.. code-block:: python

   from ferrodispcalc.compute import calculate_dielectric_constant

   dielectric = calculate_dielectric_constant(
       P,
       volume=avg_atoms.cell.volume,
       temperature=300.0,
       atomic=False,
   )

With ``atomic=False``, the function first averages the local polarization over
all unit cells, then computes the dielectric tensor. The returned dictionary
contains six components:

- ``eps_xx``
- ``eps_yy``
- ``eps_zz``
- ``eps_xy``
- ``eps_xz``
- ``eps_yz``

Notes
-----

- This function computes the ionic fluctuation contribution to the dielectric
  response. The electronic contribution is not included.
- No error analysis is done internally. In production calculations, it is a
  good idea to estimate uncertainties with block averaging.
- In this example, all frames are used for both polarization and averaged
  structure calculations.
- If you want a local dielectric tensor for each unit cell, set ``atomic=True``.
  In that case, ``volume`` should be the volume of one unit cell, not the full
  supercell.

See :func:`~ferrodispcalc.compute.calculate_polarization`,
:func:`~ferrodispcalc.compute.calculate_averaged_structure`, and
:func:`~ferrodispcalc.compute.calculate_dielectric_constant` for API details.
