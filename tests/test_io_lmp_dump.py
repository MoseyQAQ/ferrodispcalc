import pytest
import numpy as np
from pathlib import Path
from ferrodispcalc.io import read_lammps_dump
from ase.io import read

DATA_DIR = Path(__file__).parent / "data"
type_map = ['Sr', 'Pb', 'Ti', 'O']

def test_io_lmp_dump():
    # 1. Load reference
    ref_traj = read(DATA_DIR / 'test.traj', index=':')

    # 2. Load using ferrodispcalc
    traj = read_lammps_dump(DATA_DIR / "test.lammpstrj", type_map=type_map)

    # 3. Compare
    cells_ref = np.array([atoms.get_cell().array for atoms in ref_traj])
    cells = np.array([atoms.get_cell().array for atoms in traj])
    pos_ref = np.array([atoms.get_positions() for atoms in ref_traj])
    pos = np.array([atoms.get_positions() for atoms in traj])

    assert np.allclose(cells_ref, cells)
    assert np.allclose(pos_ref, pos)

def test_io_lmp_dump_select():
    # 1. Load reference
    ref_traj = read(DATA_DIR / 'test.traj', index=slice(2,5))

    # 2. Load using ferrodispcalc
    traj = read_lammps_dump(DATA_DIR / "test.lammpstrj", type_map=type_map, select=slice(2,5))

    # 3. Compare
    cells_ref = np.array([atoms.get_cell().array for atoms in ref_traj])
    cells = np.array([atoms.get_cell().array for atoms in traj])
    pos_ref = np.array([atoms.get_positions() for atoms in ref_traj])
    pos = np.array([atoms.get_positions() for atoms in traj])

    assert np.allclose(cells_ref, cells)
    assert np.allclose(pos_ref, pos)