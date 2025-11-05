from ferrodispcalc.compute import calculate_polarization
from ferrodispcalc.neighborlist import build_neighbor_list
from ferrodispcalc.vis import grid_data
from ferrodispcalc.vis import plane_profile
from ase.io import read 

def calc_bec():
    z_pto_pb = 3.45
    z_pto_ti = 5.21
    z_sto_sr = 2.56
    z_sto_ti = 7.40
    z_a = 0.5 * (z_pto_pb + z_sto_sr)
    z_b = 0.5 * (z_pto_ti + z_sto_ti)
    z_o = - (z_a + z_b) / 3

    bec = {'Pb': z_a, 'Sr': z_a, 'Ti': z_b, 'O': z_o}
    return bec


atoms = read("stru.traj")

# 1. build neighbor list
nl_bo = build_neighbor_list(atoms, ['Ti'], ['O'], 4, 6)
nl_ba = build_neighbor_list(atoms, ['Ti'], ['Pb', 'Sr'], 4, 8)
bec = calc_bec()

# 2. calculate polarization and grid the data
P = calculate_polarization(atoms, nl_ba, nl_bo, bec)
P_grid = grid_data(atoms, P, element=['Ti'], target_size=(40, 20, 20))

# 3. plot the data
plane_profile(P_grid, save_dir='profile', select={'x': [0]})