from ferrodispcalc import NeighborList, Compute
import numpy as np
from ase.io import read
nl = NeighborList('avg.xsf').build(["Ti"], ["O"], 4, 6)
nl -= 1
np.savetxt("nl.dat", nl, fmt="%10d")
atoms = read("avg.xsf")
volume = atoms.get_volume()
volume_per_uc = volume / 4000
conversion_factor = 1.602176E-19 * 1.0E-10 * 1.0E30
disp = Compute([atoms]).get_displacement(nl=NeighborList('avg.xsf').build(["Ti"], ["O"], 4, 6))[0]
disp = disp * conversion_factor / volume_per_uc 
disp *= 6.5
print(disp.shape)
ap = np.loadtxt("ap")
print(ap.shape)

print(np.allclose(disp, ap))