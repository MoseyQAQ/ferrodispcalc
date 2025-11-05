#import pytest 
import numpy as np
from pathlib import Path
from ferrodispcalc.vis.grid import grid_data
from ase.io import read 
from ase import Atoms

DATA_DIR = Path(__file__).parent / "data"
traj: list[Atoms] = read(DATA_DIR / "PTO-300K.traj", index=":")
disp = np.load(DATA_DIR / "PTO-300K-disp.npz")
ref = np.load(DATA_DIR / "PTO-300K-disp-grid.npz")


def test_grid_data_single_frame():
    atoms = traj[0]

    grid_ao = grid_data(
        atoms=atoms,
        data=disp['ao'][0],
        tol=1,
        target_size=(5,5,5),
        element=['Pb']
    )
    grid_bo = grid_data(
        atoms=atoms,
        data=disp['bo'][0],
        tol=1,
        target_size=(5,5,5),
        element=['Ti']
    )

    ref_ao = ref['ao_single']
    ref_bo = ref['bo_single']

    assert np.allclose(grid_ao, ref_ao, equal_nan=True)
    assert np.allclose(grid_bo, ref_bo, equal_nan=True)


def test_grid_data_multi_frame():
    grid_ao = grid_data(
        atoms=traj[0],
        data=disp['ao'],
        tol=1,
        target_size=(5,5,5),
        element=['Pb']
    )

    grid_bo = grid_data(
        atoms=traj[0],
        data=disp['bo'],
        tol=1,
        target_size=(5,5,5),
        element=['Ti']
    )
    
    ref_ao = ref['ao_multi']
    ref_bo = ref['bo_multi']

    assert np.allclose(grid_ao, ref_ao, equal_nan=True)
    assert np.allclose(grid_bo, ref_bo, equal_nan=True)