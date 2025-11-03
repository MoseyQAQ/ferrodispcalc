import pytest
import numpy as np
from pathlib import Path
from ferrodispcalc.neighborlist import build_neighbor_list
from ase.io import read 

DATA_DIR = Path(__file__).parent / "data"

def sort_neighbor_list(nl):
    '''Help function to sort neighbor lists for comparison.'''
    centers = nl[:, 0:1]
    neighbors = nl[:, 1:]
    neighbors_sorted = np.sort(neighbors, axis=1)
    return np.hstack((centers, neighbors_sorted))

def test_build_neighbor_list():
    # 1. Load atoms and reference
    atoms_pto = read(DATA_DIR / "PTO-T.vasp")
    atoms_psto = read(DATA_DIR / "PSTO.vasp")
    nl_pto = np.load(DATA_DIR / "PTO-T_nl.npz")
    nl_psto = np.load(DATA_DIR / "PSTO_nl.npz")

    # 2. Compare with reference
    for atoms, nl in zip([atoms_pto, atoms_psto], [nl_pto, nl_psto]):
        nl_ao = build_neighbor_list(
            atoms,
            center_elements=["Pb", "Sr"],
            neighbor_elements=["O"],
            cutoff=4,
            neighbor_num=12,
            defect=False
        )

        nl_bo = build_neighbor_list(
            atoms,
            center_elements=["Ti"],
            neighbor_elements=["O"],
            cutoff=4,
            neighbor_num=6,
            defect=False
        )
        
        nl_ba = build_neighbor_list(
            atoms,
            center_elements=["Ti"],
            neighbor_elements=["Pb", "Sr"],
            cutoff=4,
            neighbor_num=8,
            defect=False
        )
        
        nl_ao = sort_neighbor_list(nl_ao)
        nl_bo = sort_neighbor_list(nl_bo)
        nl_ba = sort_neighbor_list(nl_ba)
        nl_ao_ref = sort_neighbor_list(nl['ao'])
        nl_bo_ref = sort_neighbor_list(nl['bo'])
        nl_ba_ref = sort_neighbor_list(nl['ba'])

        assert np.allclose(nl_ao, nl_ao_ref)
        assert np.allclose(nl_bo, nl_bo_ref)
        assert np.allclose(nl_ba, nl_ba_ref)
