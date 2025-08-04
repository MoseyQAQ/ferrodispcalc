from ase.io import read, write
from ferrodispcalc import NeighborList, Compute
import numpy as np

def get_disp(atoms, nl_ao, nl_bo):
    disp_ao = Compute([atoms]).get_displacement(nl_ao.copy(),slice(0,1,1))[0]
    disp_bo = Compute([atoms]).get_displacement(nl_bo.copy(),slice(0,1,1))[0]

    return disp_ao, disp_bo

atoms = read("PTO-T-444.xyz")
nl_bo = NeighborList(atoms).build(["Ti"], ["O"], 4, 6)
nl_ao = NeighborList(atoms).build(["Pb"], ["O"], 5, 12)
disp_ao, disp_bo = get_disp(atoms, nl_ao, nl_bo)

nimage = 15
ao_linespace = np.linspace(0, disp_ao, nimage)
bo_linespace = np.linspace(0, disp_bo, nimage)
for i in range(nimage):
    atoms_copy = atoms.copy()
    atoms_copy.positions[nl_ao[:,0]-1] -= ao_linespace[i]
    atoms_copy.positions[nl_bo[:,0]-1] -= bo_linespace[i]
    disp_ao_tmp, disp_bo_tmp = get_disp(atoms_copy, nl_ao, nl_bo)
    print(i, np.mean(disp_ao_tmp[:, -1]), np.mean(disp_bo_tmp[:, -1]))
    write(f"{i+1:02d}.xyz", atoms_copy)

