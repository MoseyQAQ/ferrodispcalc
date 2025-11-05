import pytest
import numpy as np
from pathlib import Path
from ferrodispcalc.neighborlist import build_neighbor_list
from ferrodispcalc.compute import calculate_octahedral_tilt
from ase.io import read 

DATA_DIR = Path(__file__).parent / "data"

def test_calculate_octahedral_tilt():
    # 1. Load trajectory and reference
    atoms = read(DATA_DIR / "STO-50K.vasp")
    tilt_ref = np.load(DATA_DIR / "STO-50K_octahedral_tilt.npy")

    # 2. Build neighbor lists
    nl_bo = build_neighbor_list(
        atoms,
        ['Ti'],
        ['O'],
        4,
        6
    )

    # 3. Calculate octahedral tilt
    tilt = calculate_octahedral_tilt(
        atoms,
        nl_bo
    )
    
    # 4. Compare with reference
    assert np.allclose(tilt, tilt_ref)