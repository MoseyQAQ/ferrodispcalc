import pytest
import numpy as np
from pathlib import Path
from ferrodispcalc.neighborlist import build_neighbor_list
from ferrodispcalc.compute import calculate_displacement
from ase.io import read 

DATA_DIR = Path(__file__).parent / "data"

def test_calculate_displacement():
    # 1. Load trajectory and reference
    traj = read(DATA_DIR / "PTO-300K.traj", ":")
    disp_ref = np.load(DATA_DIR / "PTO-300K-disp.npz")

    # 2. Build neighbor lists
    nl_ao = build_neighbor_list(
        traj[0],
        ['Pb'],
        ['O'],
        4,
        12
    )

    nl_bo = build_neighbor_list(
        traj[0],
        ['Ti'],
        ['O'],
        4,
        6
    )

    # 3. Calculate displacements
    disp_ao = calculate_displacement(
        traj,
        nl_ao,
        select=slice(0,20,1)
    )
    disp_bo = calculate_displacement(
        traj,
        nl_bo,
        select=slice(0,20,1)
    )

    # 4. Compare with reference
    assert np.allclose(disp_ao, disp_ref['ao'])
    assert np.allclose(disp_bo, disp_ref['bo'])