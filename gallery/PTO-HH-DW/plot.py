from ferrodispcalc.compute import calculate_polarization, calculate_displacement
from ferrodispcalc.neighborlist import build_neighbor_list
from ferrodispcalc.vis import grid_data
from ferrodispcalc.vis import line_profile
from ferrodispcalc.config import BEC
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read 


atoms = read("DW.vasp")

# 1. build neighbor list
nl_bo = build_neighbor_list(atoms, ['Ti'], ['O'], 4, 6)
nl_ba = build_neighbor_list(atoms, ['Ti'], ['Pb'], 4, 8)
nl_ao = build_neighbor_list(atoms, ['Pb'], ['O'], 4, 12)
bec = BEC.get('PTO')

# 2. calculate polarization and displacement
P = calculate_polarization(atoms, nl_ba, nl_bo, bec)
disp_ao = calculate_displacement(atoms, nl_ao)
disp_bo = calculate_displacement(atoms, nl_bo)
P_grid = grid_data(atoms, P, element=['Ti'])

# 3. grid data for visualization
disp_ao_grid = grid_data(atoms, disp_ao, element=['Pb'], target_size=(2,2,80))
disp_bo_grid = grid_data(atoms, disp_bo, element=['Ti'], target_size=(2,2,80))

# 4. plot
fig, axs = plt.subplots(1, 3, figsize=(9, 2))
line_profile(disp_ao_grid, ax=axs[0], along='z', field_prefix='d')
line_profile(disp_bo_grid, ax=axs[1], along='z', field_prefix='d')
line_profile(P_grid, ax=axs[2], along='z', field_prefix='P')
axs[0].set_title("Displacement for Pb")
axs[1].set_title("Displacement for Ti")
axs[2].set_title("Polarization")
plt.tight_layout()
plt.savefig("PTO-HH-DW-profile.png", dpi=300)