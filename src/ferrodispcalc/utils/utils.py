from ase import Atoms 
import numpy as np

def crystal_lattice_to_cartesian(atoms: Atoms, vector: np.ndarray) -> np.ndarray:
    '''
    Map the vector along the crystal lattice to the cartesian coordinates.
    
    Args:
    ---
    atoms: Atoms
        The Atoms object containing the crystal structure.
    vector: np.ndarray
        The vector in crystal lattice coordinates to be converted.
    '''
    cell = atoms.cell.array.copy()
    La, Lb, Lc, _, _, _ = atoms.cell.cellpar()
    cell[:, 0] /= La
    cell[:, 1] /= Lb
    cell[:, 2] /= Lc

    vector = cell.T @ vector
    return vector

def cartesian_to_crystal_lattice(atoms: Atoms, vector: np.ndarray) -> np.ndarray:
    '''
    Map the vector along the cartesian coordinates to the crystal lattice.
    
    Args:
    ---
    atoms: Atoms
        The Atoms object containing the crystal structure.
    vector: np.ndarray
        The vector in cartesian coordinates to be converted.
    '''
    cell = atoms.cell.array.copy()
    La, Lb, Lc, _, _, _ = atoms.cell.cellpar()
    cell[:, 0] *= La
    cell[:, 1] *= Lb
    cell[:, 2] *= Lc

    vector = np.linalg.inv(cell.T) @ vector
    return vector