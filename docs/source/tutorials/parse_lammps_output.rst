Parse LAMMPS's output file
======================

Use :func:`ferrodispcalc.io.read_lammps_dump` to read LAMMPS dump files.

.. code-block:: python

    from ferrodispcalc.io import read_lammps_dump

    # type_map maps LAMMPS numeric types to element symbols
    type_map = ['Sr', 'Ti', 'O']

    # Read all frames (returns a list of ASE Atoms objects)
    trajectory = read_lammps_dump("dump.lammpstrj", type_map=type_map)

    # Access a single frame
    frame = trajectory[0]
    print(frame.info['timestep'])     # timestep number
    print(frame.get_positions())      # atomic positions, shape (N, 3)
    print(frame.get_cell())           # simulation box

Read a subset of frames using the ``select`` parameter:

.. code-block:: python

    # Read every other frame from index 0 to 100
    subset = read_lammps_dump("dump.lammpstrj", type_map=type_map, select=slice(0, 100, 2))

    # Read a single frame by index
    frame = read_lammps_dump("dump.lammpstrj", type_map=type_map, select=50)

.. note::

    An index file (e.g. ``dump.lammpstrj.idx``) is generated on first read to speed up subsequent reads.
