.. FerroDispCalc documentation master file, created by
   sphinx-quickstart on Tue Mar 18 15:15:44 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

FerroDispCalc documentation
===========================

FerroDispCalc (fdc) is a Python toolkit for pre- and post-processing of atomic simulations of ferroelectric materials, with special focus on analyzing ionic displacements and polarization calculations.

Features
---------

**Pre-processing**

- Build neighbor lists through intuitive APIs supporting various crystal systems (ABO<sub>3</sub> perovskites, HfO₂, In₂Se₃, etc.)
- Structure file format conversion (supports POSCAR, LAMMPS data, XYZ, etc.)

**Core Computation**

- *Python Interface*: Calculate ionic displacements, average structures, and perovskite polarization
- *C++ Interface*: High-performance processing of LAMMPS trajectory files for:

  - Average structure calculation
  - Ionic displacement
  - Local polarization (perovskite-specific)
  - Local lattice distortion
- *LAMMPS Plugin*: Real-time computation during MD simulations via ``compute`` commands:

  - On-the-fly displacement monitoring
  - Instant polarization access

**Post-processing**

- Automated visualization of vector fields (displacement/polarization)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting-started/index
   
   :caption: API Reference

   modules
