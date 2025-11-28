import pytest
import numpy as np
from pathlib import Path
from ferrodispcalc.io import read_lammps_log
from ase.io import read

DATA_DIR = Path(__file__).parent / "data"

def test_io_lmp_log():
    # 1. Load reference
    ref_log = np.load(DATA_DIR / "ref_log.lammps.npz", allow_pickle=True)

    # 2. Load using ferrodispcalc
    log_data = read_lammps_log(DATA_DIR / "log.lammps")

    # 3. Compare keys
    assert set(log_data.keys()) == set(ref_log.keys()), "Keys do not match"

    # 4. Compare values
    for key in ref_log.keys():
        ref_value = ref_log[key]
        log_value = log_data[key]

        if isinstance(ref_value, np.ndarray):
            assert np.array_equal(ref_value, log_value), f"Values for key '{key}' do not match"
        else:
            assert ref_value == log_value, f"Values for key '{key}' do not match"