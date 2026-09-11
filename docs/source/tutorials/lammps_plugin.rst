LAMMPS plugin for real-time computation of polar displacement and local polarization
==================================================================================

Load the plugin
---------------

Build instructions: :doc:`/installation`.

.. code-block:: lammps

   plugin load /path/to/dispplugin.so  # Register the three compute styles

Alternatively, set the environment variable ``LAMMPS_PLUGIN_PATH`` to the plugin
directory for automatic loading at startup.

Compute commands
----------------

.. code-block:: lammps

   atom_modify map array  # Atom-ID lookup for file-based neighbors
   comm_modify vel yes   # Ghost velocities for vel yes

   group pb type 2       # A sites
   group ti type 3       # B sites
   group o type 4        # Oxygen sites

   # Ti displacement relative to neighbors listed in nn.dat
   compute d_file ti disp/atom nnfile nn.dat vel yes

   # Ti displacement relative to the nearest six O atoms
   compute d_auto ti disp/atom/auto o 6 vel yes

   # B-centered ABO3 polarization: A group/BEC, B group/BEC, O group/BEC
   compute p_auto all polar/abo3/auto pb 3.74 ti 6.17 o -3.303333 vel yes

**disp/atom:** ``nnfile`` rows contain the central atom ID followed by neighbor
IDs (1-based). It can be generated using python API. Columns 1-3 give the polar displacement;
``vel yes`` adds relative velocities in columns 4-6 (default: ``vel no``).

**disp/atom/auto:** same as ``disp/atom``. The only difference is that this compute doesn't need additional ``nnfile``. 
It uses the neighbor list from LAMMPS.

**polar/abo3/auto:** arguments are three ``group BEC`` pairs in A, B, O order.
Uses eight A and six O neighbors per B center. Columns 1-3 give polarization:
``16.02176634 * (Z_B*r_B + Z_A*sum(r_A)/8 + Z_O*sum(r_O)/2) / V_cell``
in C/m², where ``V_cell = box volume / global B count``; ``metal`` or
``real`` units are required.
 Currently, the velocity give ``(Z_B*v_B + Z_A*sum(v_A)/8 + Z_O*sum(v_O)/2)/5``, not ``dP/dt``.
