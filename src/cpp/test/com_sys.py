import numpy as np
from pathlib import Path
def parse_xsf_file(file_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    parse the xsf file and return the cell, coord, and type_index

    Args:
        file_name (str): the name of the xsf file
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: cell, coord, and type_index (in string)
    '''
    cell = np.zeros((3, 3))
    f = open(file_name, "r")
    f.readline()
    f.readline()

    for i in range(3):
        cell[i] = np.array(f.readline().split()).astype(float)
    
    f.readline()
    natoms = int(f.readline().split()[0])
    coord = np.zeros((natoms, 3))
    type_index = np.zeros(natoms, dtype=str)
    for i in range(natoms):
        line = f.readline().split()
        type_index[i] = line[0]
        coord[i] = np.array(line[1:]).astype(float)

    f.close()
    return cell, coord, type_index

def compare_stru(ref_file: str, test_file: str) -> tuple[bool, float]:
    '''
    compare the structure of two xsf files

    Args:
        ref_file (str): the reference xsf file
        test_file (str): the test xsf file
    
    Returns:
        tuple[bool, float]: whether the two structures are the same and the maximum difference between the two structures
    '''
    ref_cell, ref_coord, ref_type_index = parse_xsf_file(ref_file)
    test_cell, test_coord, test_type_index = parse_xsf_file(test_file)

    if np.allclose(ref_cell, test_cell) and np.allclose(ref_coord, test_coord) and np.all(ref_type_index == test_type_index):
        return True, np.max(np.linalg.norm(ref_coord - test_coord, axis=1))
    else:
        return False, np.max(np.linalg.norm(ref_coord - test_coord, axis=1))

def compare_disp(ref_file: str, test_file: str) -> tuple[bool, float]:
    '''
    compare the displacement in two files

    Args:
        ref_file (str): the reference file
        test_file (str): the test file
    
    Returns:
        tuple[bool, float]: whether the two displacements are the same and the maximum difference between the two displacements
    '''
    ref_disp = np.loadtxt(ref_file, skiprows=5, usecols=(0, 1, 2))
    test_disp = np.loadtxt(test_file, usecols=(3, 4, 5))

    if np.allclose(ref_disp, test_disp):
        return True, np.max(np.linalg.norm(ref_disp - test_disp, axis=1))
    else:
        return False, np.max(np.linalg.norm(ref_disp - test_disp, axis=1))
    
def compare_polar(ref_file: str, test_file: str) -> tuple[bool, float]:
    '''
    compare the polarization in two files

    Args:
        ref_file (str): the reference file
        test_file (str): the test file
    
    Returns:
        tuple[bool, float]: whether the two polarizations are the same and the maximum difference between the two polarizations
    '''
    ref_polar = np.loadtxt(ref_file, skiprows=1, usecols=(0, 1, 2))
    test_polar = np.loadtxt(test_file)

    if np.allclose(ref_polar, test_polar):
        return True, np.max(np.linalg.norm(ref_polar - test_polar, axis=1))
    else:
        return False, np.max(np.linalg.norm(ref_polar - test_polar, axis=1))