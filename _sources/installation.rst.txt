Installation
============

Python API
----------

Requires Python >= 3.9 and a C++17-capable compiler (GCC/Clang/MSVC).
Linux/Windows also require OpenMP; macOS builds without it.

.. code-block:: bash

   git clone https://github.com/MoseyQAQ/ferrodispcalc.git
   cd ferrodispcalc
   pip install -e .

This installs the core dependencies (``numpy``, ``matplotlib``, ``ase``,
``pymatgen``, ``dpdata``, ``scienceplots``) and compiles the C++ backend.

For the optional 3-D visualisation tools (pyvista):

.. code-block:: bash

   pip install -e ".[vis]"

LAMMPS API
----------

The LAMMPS API provides a file-based ``compute disp/atom`` runtime plugin.
It outputs displacement and displacement velocity from a user-provided neighbor list.

**Prerequisites**

A C++ compiler with MPI and OpenMP support, plus the LAMMPS source tree.

.. warning::

   The LAMMPS source version used to compile the plugin **should match**
   the LAMMPS binary you run simulations with.  Likewise, the compiler
   (and its version) must be the same in both cases — a mismatch will cause
   runtime errors when loading the plugin.

If you do not have the source locally:

.. code-block:: bash

   wget https://github.com/lammps/lammps/archive/stable_2Aug2023_update3.tar.gz
   tar -xzf stable_2Aug2023_update3.tar.gz

**Compile**

Load the required modules (example for a cluster with environment modules):

.. code-block:: bash

   module purge
   module load lammps/29aug2024-dpv3

Edit ``Makefile`` and set the two ``-I`` flags to your actual LAMMPS source paths:

.. code-block:: makefile

   CXX      = mpicxx
   CXXFLAGS = -I/path/to/lammps/src -Wall -Wextra -O3 -fPIC \
              -I/path/to/lammps/src/OPENMP -fopenmp
   LD       = $(CXX) -shared -rdynamic -fopenmp
   DSOEXT   = .so

   include Makefile.common

Then build:

.. code-block:: bash

   make

On success, ``dispplugin.so`` appears in the current directory.

**Verify**

In a LAMMPS input file, load the plugin and confirm it is registered:

.. code-block:: lammps

   plugin load /path/to/dispplugin.so
   plugin list

``plugin list`` should show only ``disp/atom`` from this plugin:

.. code-block:: bash

   Loading plugin: compute disp/atom by Denan Li (lidenan@westlake.edu.cn)
   Currently loaded plugins: 1
      1: compute style plugin disp/atom

A minimal input setup for displacement only is:

.. code-block:: lammps

   atom_modify map array
   plugin load /path/to/dispplugin.so
   compute d all disp/atom nnfile nn.dat

This returns three per-atom columns: ``c_d[1]``-``c_d[3]`` are
``r_center - mean(r_neighbors)``.

To also output the displacement velocity, enable ghost velocity communication
and pass ``vel yes``:

.. code-block:: lammps

   atom_modify map array
   comm_modify vel yes
   plugin load /path/to/dispplugin.so
   compute d all disp/atom nnfile nn.dat vel yes

With ``vel yes``, the compute returns six per-atom columns. ``c_d[4]``-``c_d[6]``
are ``v_center - mean(v_neighbors)``.
