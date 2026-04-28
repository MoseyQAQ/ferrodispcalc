Parse LAMMPS Output Files
=========================

Read Trajectories
-----------------

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

Read Log Files
--------------

Use :func:`ferrodispcalc.io.read_lammps_log` to parse thermodynamic data from a LAMMPS log file.
It returns a dictionary mapping each thermo keyword to a NumPy array of values.

.. code-block:: python

    from ferrodispcalc.io import read_lammps_log

    log = read_lammps_log("log.lammps")

    print(log.keys())        # e.g. dict_keys(['Step', 'Temp', 'Press', 'TotEng', ...])
    print(log['nframes'])    # number of thermo output lines

    # Plot temperature vs. step
    import matplotlib.pyplot as plt

    plt.plot(log['Step'], log['Temp'])
    plt.xlabel('Step')
    plt.ylabel('Temperature (K)')
    plt.savefig('temp_vs_step.png')

See the :mod:`ferrodispcalc.io` API reference for full details.
