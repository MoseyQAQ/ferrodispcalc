from ferrodispcalc import NeighborList
from ase.io import read
import numpy as np

atoms = read("PTO-T.vasp")

nl_ao = NeighborList(atoms).build(['Pb', 'Sr'], ['O'], 4, 12, False)
np.savetxt('PTO_AO-nl.dat', nl_ao, fmt='%d')
nl_bo = NeighborList(atoms).build(['Ti'], ['O'], 4, 6, False)
np.savetxt('PTO_BO-nl.dat', nl_bo, fmt='%d')
nl_ba = NeighborList(atoms).build(['Ti'], ['Pb', 'Sr'], 4, 8, False)
np.savetxt('PTO_BA-nl.dat', nl_ba, fmt='%d')