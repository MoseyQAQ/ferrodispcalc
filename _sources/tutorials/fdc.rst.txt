Introduction to ferrodispcalc
=============================

What is ferrodispcalc?
----------------------

ferrodispcalc is a Python library for computing order parameters in ferroelectric materials,
including ionic displacements, local polarization, and octahedral tilting.
It is designed for perovskite-type structures but can be extended to other systems
as long as the coordination environment is well-defined.

In addition, ferrodispcalc provides high-performance I/O utilities for LAMMPS,
including fast trajectory reading (``read_lammps_dump``) and log file parsing
(``read_lammps_log``). See the :doc:`LAMMPS output tutorial </tutorials/parse_lammps_output>`
and the :mod:`ferrodispcalc.io` API reference for details.

The Core Idea: Coordination and Neighbor Lists
-----------------------------------------------

.. note::

   This tutorial assumes basic familiarity with ASE's ``Atoms`` object.
   See :doc:`/tutorials/ase` if you are new to ASE.

All calculations in ferrodispcalc boil down to one thing: **finding the right neighbors**.

Take a perovskite ABO\ :sub:`3` as an example. The B-site cation sits at the center of an
oxygen octahedron, and the A-site cation occupies the corner of the unit cell.
If you can correctly identify which O atoms coordinate a given B atom (and vice versa),
you can compute:

- **Ionic displacement**: the offset of the center atom from the geometric center of its neighbors.
- **Local polarization**: displacement weighted by Born effective charges, converted to C/m².
- **Octahedral tilt**: the rotation angles of the BO\ :sub:`6` octahedron about the Cartesian axes.

ferrodispcalc uses ``build_neighbor_list`` to construct these coordination lists.
You specify the center element, the neighbor element, a cutoff distance, and the expected
coordination number. Everything else follows from there.

.. code-block:: python

   from ferrodispcalc.neighborlist import build_neighbor_list

   # B-O coordination (octahedron): 6 neighbors
   nl_bo = build_neighbor_list(atoms, ["Ti"], ["O"], cutoff=4, neighbor_num=6)

   # B-A coordination: 8 neighbors
   nl_ba = build_neighbor_list(atoms, ["Ti"], ["Pb"], cutoff=4, neighbor_num=8)

   # A-O coordination: 12 neighbors
   nl_ao = build_neighbor_list(atoms, ["Pb"], ["O"], cutoff=4, neighbor_num=12)

Example: PbTiO\ :sub:`3` Displacement and Polarization
-------------------------------------------------------

This example computes ionic displacements and local polarization for a PbTiO\ :sub:`3` (PTO) structure.

.. code-block:: python

   from ase.io import read
   from ferrodispcalc.neighborlist import build_neighbor_list
   from ferrodispcalc.compute import calculate_displacement, calculate_polarization
   from ferrodispcalc.config import BEC

   # 1. Read structure
   atoms = read("PTO.vasp")

   # 2. Build neighbor lists
   nl_bo = build_neighbor_list(atoms, ["Ti"], ["O"], cutoff=4, neighbor_num=6)
   nl_ba = build_neighbor_list(atoms, ["Ti"], ["Pb"], cutoff=4, neighbor_num=8)
   nl_ao = build_neighbor_list(atoms, ["Pb"], ["O"], cutoff=4, neighbor_num=12)

   # 3. Calculate displacements
   disp_bo = calculate_displacement(atoms, nl_bo)   # Ti off-center in O octahedron
   disp_ao = calculate_displacement(atoms, nl_ao)   # Pb off-center in O cage

   # 4. Calculate polarization
   bec = BEC.get("PTO")   # {'Pb': 3.44, 'Ti': 5.18, 'O': -2.87}
   P = calculate_polarization(atoms, nl_ba, nl_bo, bec)

``calculate_displacement`` returns an array of shape ``(n_centers, 3)``, where each row is
the displacement vector of a center atom relative to the geometric center of its neighbors.
For trajectories (``list[Atoms]``), the output shape is ``(n_frames, n_centers, 3)``.

``calculate_polarization`` converts these displacements into local polarization (in C/m²)
using Born effective charges. It requires two neighbor lists: B-A and B-X.

Beyond Perovskites
------------------

ferrodispcalc is not limited to simple ABO\ :sub:`3` perovskites. As long as you can define
the coordination environment, you can use the same workflow:

1. Identify center and neighbor elements.
2. Choose an appropriate cutoff and coordination number.
3. Build the neighbor list and compute.

This makes it applicable to:

- **Solid solutions** (e.g. PbTiO\ :sub:`3`-SrTiO\ :sub:`3`): use mixed element lists
  like ``["Ti"]`` for B-site and ``["Pb", "Sr"]`` for A-site, and provide averaged
  Born effective charges. See :doc:`/tutorials/example_perov_superlattice`.
- **Molecular ferroelectrics**: use bond vectors as dipole proxies.
  See :doc:`/tutorials/example_molecular` for a TMCM-CdCl\ :sub:`3` example.
- **Other coordination geometries**: any system where the order parameter can be defined
  as a displacement from the coordination center.
  See :doc:`/tutorials/example_aln` for a wurtzite example with a custom neighbor list.

.. code-block:: python

   # Solid solution example: PTO-STO superlattice
   nl_ba = build_neighbor_list(atoms, ["Ti"], ["Pb", "Sr"], cutoff=4, neighbor_num=8)
   nl_bo = build_neighbor_list(atoms, ["Ti"], ["O"], cutoff=4, neighbor_num=6)

   bec = {"Pb": 3.44, "Sr": 2.56, "Ti": 5.18, "O": -2.87}
   P = calculate_polarization(atoms, nl_ba, nl_bo, bec)
