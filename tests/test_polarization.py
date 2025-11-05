import pytest
import numpy as np
from pathlib import Path
from ferrodispcalc.compute import calculate_polarization
from ferrodispcalc.neighborlist import build_neighbor_list
from ase.io import read 

DATA_DIR = Path(__file__).parent / "data"

def test_calculate_polarization():
    # 1. Load trajectory and reference
    atoms = read(DATA_DIR / "PTO-180DW.vasp")
    P_ref = np.load(DATA_DIR / 'PTO-180DW-P.npy')

    # 2. Build neighbor lists
    nl_bo = build_neighbor_list(
        atoms,
        ['Ti'],
        ['O'],
        4,
        6
    )

    nl_ba = build_neighbor_list(
        atoms,
        ['Ti'],
        ['Pb'],
        4,
        8
    )

    # 3. Calculate displacements
    bec = {'Pb': 3.44,'Ti': 5.18,'O': -(3.44+5.18)/3}
    P = calculate_polarization(atoms, nl_ba, nl_bo, born_effective_charge=bec)

    # 4. Compare with reference
    assert np.allclose(P, P_ref, atol=1e-5)