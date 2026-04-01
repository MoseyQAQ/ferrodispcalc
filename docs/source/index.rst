.. ferrodispcalc documentation master file

ferrodispcalc
=============

**ferrodispcalc** is a Python package for preprocessing and postprocessing
molecular dynamics simulations of ferroelectric materials, with a focus on
perovskite structures (e.g. PbTiO₃, BaTiO₃, PbTiO₃/SrTiO₃ superlattices).

It provides a high-level API for extracting physically meaningful descriptors
from LAMMPS trajectories: ionic displacements, local polarization, octahedral
tilt angles, and time-averaged structures.  A C++ backend (pybind11) handles
memory-mapped I/O for large trajectory files.

Features
--------

- Build neighbor lists with element-selective cutoffs
- Calculate ionic displacements from MD trajectories (minimum image convention)
- Compute local polarization per unit cell using Born effective charges
- Measure octahedral tilt angles about x, y, z axes
- Average atomic structures across MD frames
- Read LAMMPS dump, data, and log files via a high-performance C++ backend
- Visualize vector fields in 1-D profiles, 2-D planes, and 3-D space

Installation
------------

.. code-block:: bash

   git clone https://github.com/MoseyQAQ/ferrodispcalc.git
   cd ferrodispcalc
   pip install -e .

Quick Start
-----------

.. code-block:: python

   from ferrodispcalc.io import read_lammps_dump
   from ferrodispcalc.neighborlist import build_neighbor_list
   from ferrodispcalc.compute import calculate_displacement

   # Read LAMMPS trajectory
   traj = read_lammps_dump("dump.lammpstrj", type_map=["Pb", "Ti", "O"])

   # Build Ti–O neighbor list (6 nearest O neighbors per Ti)
   nl = build_neighbor_list(
       traj[0],
       center_elements=["Ti"],
       neighbor_elements=["O"],
       cutoff=3.0,
       neighbor_num=6,
   )

   # Calculate Ti displacements relative to O neighbors
   disp = calculate_displacement(traj, nl)   # shape: (n_frames, n_Ti, 3)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
