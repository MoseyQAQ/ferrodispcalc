import pytest
import numpy as np
from pathlib import Path
from ferrodispcalc.compute import compute_averaged_structure
from ase.io import read 

DATA_DIR = Path(__file__).parent / "data"

def test_averaged_structure():
    traj = read(DATA_DIR / "PTO-300K.traj", index=":")
    ref = read(DATA_DIR / "PTO-300K-avg.vasp")
    avg = compute_averaged_structure(traj, select=slice(0, 20, 1))

    cell = avg.get_cell().array
    ref_cell = ref.get_cell().array
    coord = avg.get_positions()
    ref_coord = ref.get_positions()

    np.allclose(cell, ref_cell, atol=1e-5)
    np.allclose(coord, ref_coord, atol=1e-5)



