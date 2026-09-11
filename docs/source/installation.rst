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

Set ``LMP_SOURCE_DIR`` to the matching LAMMPS source tree. 
The C++ standard must also match the runtime binary; 

.. code-block:: makefile

   LMP_SOURCE_DIR = /path/to/lammps/src
   CXX_STANDARD = c++20

Then build:

.. code-block:: bash

   make

On success, ``dispplugin.so`` appears in the current directory.

For plugin loading and compute usage, see :doc:`tutorials/lammps_plugin`.
