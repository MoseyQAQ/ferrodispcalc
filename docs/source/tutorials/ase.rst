Introduction to ASE
===================

What is ASE?
------------

The Atomic Simulation Environment (`ASE <https://ase-lib.org/index.html>`_) is a Python library that provides
a flexible set of APIs for creating, manipulating, and analyzing atomic structures.
It supports a wide range of file formats and offers a consistent interface for working with
atoms regardless of the simulation code you use.

Why does ferrodispcalc use ASE?
-------------------------------

ferrodispcalc uses ASE's ``Atoms`` object as its core data structure.
Rather than reinventing the wheel, ferrodispcalc builds directly on ASE's I/O and structure
manipulation APIs, so you can seamlessly integrate ferrodispcalc into any ASE-based workflow.

Basic Operations
----------------

In ASE, everything revolves around two things:

- ``Atoms``: a single atomic structure (positions, cell, species, etc.)
- ``list[Atoms]``: a trajectory, i.e. a sequence of structures (e.g. from an MD simulation)

Once you are comfortable with these two, you can do almost anything.

Reading Files
^^^^^^^^^^^^^

Use ``ase.io.read`` to load structures from files. The ``format`` argument is usually
optional — ASE can infer it from the file extension — but being explicit never hurts.

**Single structure** (e.g. a POSCAR or LAMMPS data file):

.. code-block:: python

   from ase.io import read

   # VASP POSCAR
   atoms = read("POSCAR")

   # LAMMPS data file
   atoms = read("structure.data", format="lammps-data", style="atomic")

**Trajectory** (e.g. a multi-frame XYZ file):

.. code-block:: python

   # Read all frames — note the index=":" argument
   trajectory = read("trajectory.xyz", index=":")

   print(type(trajectory))        # <class 'list'>
   print(type(trajectory[0]))     # <class 'ase.atoms.Atoms'>
   print(len(trajectory))         # number of frames

Structure Manipulation
^^^^^^^^^^^^^^^^^^^^^^

All manipulations below operate on an ``Atoms`` object in-place or return a new one.

**Supercell expansion**

The simplest way is to multiply the ``Atoms`` object directly:

.. code-block:: python

   # 2x2x2 supercell
   supercell = atoms * (2, 2, 2)

**Substitution / doping**

Replace an atom by changing its symbol directly:

.. code-block:: python

   # Replace the first atom with Ba
   atoms.symbols[0] = "Ba"

   # Replace all Ti with Zr
   for i, symbol in enumerate(atoms.symbols):
       if symbol == "Ti":
           atoms.symbols[i] = "Zr"

**Coordinate operations**

.. code-block:: python

   # Get positions as a numpy array of shape (N, 3)
   pos = atoms.get_positions()

   # Set positions
   atoms.set_positions(pos)

   # Wrap atoms back into the unit cell (useful after displacement)
   atoms.wrap()

   # Translate all atoms by a vector
   atoms.translate([0.0, 0.0, 1.5])

**Applying strain**

To apply strain, scale the cell and (optionally) the atomic positions together:

.. code-block:: python

   import numpy as np

   cell = atoms.get_cell()

   # Apply 2% tensile strain along the x-axis
   strain = np.eye(3)
   strain[0, 0] = 1.02

   atoms.set_cell(cell @ strain.T, scale_atoms=True)

Setting ``scale_atoms=True`` ensures that fractional coordinates are preserved,
so the atoms move with the cell rather than staying at their Cartesian positions.

Writing Files
^^^^^^^^^^^^^

Use ``ase.io.write`` to save structures. The API mirrors ``read``:

.. code-block:: python

   from ase.io import write

   # Write a single structure
   write("POSCAR_out", atoms, format="vasp")

   # Write a trajectory to a multi-frame XYZ file
   write("trajectory_out.xyz", trajectory)

For more information, please refer to the `ASE documentation <https://ase-lib.org/examples_generated/tutorials/index.html>`_.

Once you are comfortable with ASE, head to :doc:`/tutorials/fdc` to learn how
ferrodispcalc builds on these concepts.