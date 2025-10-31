from ase.io import read, write

atoms = read("sc.vasp")

atoms *= [5,5,5] # 10*10*10 supercell

# change Pb with Sr randomly 
import numpy as np
np.random.seed(0)
num_Pb = len([atom for atom in atoms if atom.symbol == "Pb"])
num_Sr = int(num_Pb * 0.2) # 20% Sr
Pb_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "Pb"]
Sr_indices = np.random.choice(Pb_indices, size=num_Sr, replace=False)
for i in Sr_indices:
    atoms[i].symbol = "Sr"
write("sc_20pct_Sr.vasp", atoms, vasp5=True, sort=True)