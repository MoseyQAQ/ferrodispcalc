from ase import Atoms 
import numpy as np
from tqdm import tqdm
from ase.calculators.singlepoint import SinglePointCalculator

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

def match_atoms(ref_atoms: Atoms, atoms: Atoms) -> Atoms:
    ref_cell = ref_atoms.cell.array.copy()
    atoms_cell = atoms.cell.array.copy()

    assert np.allclose(ref_cell, atoms_cell), "Cells do not match, please check the input Atoms objects."
    assert len(ref_atoms) == len(atoms), "Number of atoms do not match, please check the input Atoms objects."
    
    inv_cell = np.linalg.inv(ref_cell)

    ref_coords = ref_atoms.get_positions()
    coords = atoms.get_positions()

    mapping_idx = np.zeros(len(ref_atoms), dtype=int) # map: ref_atoms -> atoms
    max_dist_sq = None
    for i in tqdm(range(len(ref_atoms)), desc="Matching atoms"):
        ref_coord = ref_coords[i]

        # Calculate the displacement in fractional coordinates
        displacement = coords - ref_coord
        displacement_frac = displacement @ inv_cell
        displacement_frac -= np.round(displacement_frac)
        displacement = displacement_frac @ ref_cell
        dist_sq = np.sum(displacement**2, axis=-1)

        # Find the index of the closest atom
        min_idx = np.argmin(dist_sq)
        if max_dist_sq is None or dist_sq[min_idx] > max_dist_sq:
            max_dist_sq = dist_sq[min_idx]
        mapping_idx[i] = min_idx

    assert len(set(mapping_idx)) == len(ref_atoms), "Some atoms in the reference structure do not have a unique match in the target structure."
    
    print(f"Minimum distance squared between matched atoms: {max_dist_sq:.3f} A^2")
    
    # return a new Atoms object with the matched atoms
    _atoms = atoms.copy()
    _atoms = _atoms[mapping_idx]

    # Attach the new calculator
    ## Note: only support forces and energy.
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    f_reordered = f[mapping_idx]
    sg = SinglePointCalculator(
        _atoms,
        energy=e,
        forces=f_reordered,
    )
    _atoms.set_calculator(sg)
    return _atoms