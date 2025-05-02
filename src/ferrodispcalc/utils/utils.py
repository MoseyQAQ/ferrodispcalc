from ase import Atoms 
import numpy as np

def get_polarization_quanta(atoms: Atoms) -> np.ndarray:
    '''
    Calculate the polarization quanta along the crystal lattice vectors.

    Args:
    ---
    atoms: Atoms
        The Atoms object containing the crystal structure.

    Returns:
    ---
    mod: np.ndarray
        The polarization quanta ([modA, modB, modC]) along the crystal lattice vectors.  
    '''
    cell = atoms.cell.array.copy()
    La, Lb, Lc, _, _, _ = atoms.cell.cellpar()
    volume = atoms.get_volume()

    modA = La / volume * 1602
    modB = Lb / volume * 1602
    modC = Lc / volume * 1602
    mod = np.array([modA, modB, modC])
    return mod

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
    cell[0, :] /= La
    cell[1, :] /= Lb
    cell[2, :] /= Lc

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
    cell[0, :] /= La
    cell[1, :] /= Lb
    cell[2, :] /= Lc

    vector = np.linalg.inv(cell.T) @ vector
    return vector

def c2l(atoms: Atoms, vector: np.ndarray) -> np.ndarray:
    '''
    Map the vector along the cartesian coordinates to the crystal lattice.
    Alias for cartesian_to_crystal_lattice.
    
    Args:
    ---
    atoms: Atoms
        The Atoms object containing the crystal structure.
    vector: np.ndarray
        The vector in cartesian coordinates to be converted.
    '''
    return cartesian_to_crystal_lattice(atoms, vector)

def l2c(atoms: Atoms, vector: np.ndarray) -> np.ndarray:
    '''
    Map the vector along the crystal lattice to the cartesian coordinates.
    Alias for crystal_lattice_to_cartesian.
    
    Args:
    ---
    atoms: Atoms
        The Atoms object containing the crystal structure.
    vector: np.ndarray
        The vector in crystal lattice coordinates to be converted.
    '''
    return crystal_lattice_to_cartesian(atoms, vector)