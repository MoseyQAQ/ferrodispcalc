import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms

def build_neighbor_list(atoms: Atoms,
                        center_elements: list[str],
                        neighbor_elements: list[str],
                        cutoff: float,
                        neighbor_num: int,
                        defect: bool=False) -> np.ndarray:
    """Build neighbor lists for selected atom types.

    Find the neighbor of selected center atoms within a cutoff radius. And return
    the 1-based neighbor list array.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object containing atomic positions, cell and element types.
    center_elements : list[str]
        Element symbols to treat as centers (e.g. ["Ti", "Ba"]).
    neighbor_elements : list[str]
        Element symbols considered as neighbors (e.g. ["O"]).
    cutoff : float
        Cutoff radius used to search for neighbors (Unit: Angstrom).
    neighbor_num : int
        Number of neighbors to keep per center. If fewer neighbors are found and
        ``defect`` is False a :class:`ValueError` is raised; if ``defect`` is
        True missing neighbor slots are filled with the center index.
    defect : bool, optional
        When True, allow centers with fewer than ``neighbor_num`` neighbors and
        fill missing entries with the center index (default: False).

    Returns
    -------
    np.ndarray
        Integer array of shape ``(n_centers, neighbor_num + 1)``. Each row
        contains the center atom index followed by the neighbor atom indices.
        Indices are 1-based.

    Raises
    ------
    ValueError
        If any center atom has fewer than ``neighbor_num`` neighbors and
        ``defect`` is False.

    Notes
    -----
    - Internally the function uses :meth:`pymatgen.core.Structure.get_neighbor_list`.
    - A small-cell check prints a warning if any cell vector length is less than
      4.0 Å.
    - Ensure that the provided ``atoms`` object has periodic boundary
      conditions (PBC) and a valid cell when you expect PBC behavior.

    Examples
    --------
        from ase.io import read
        from ferrodispcalc.neighborlist import build_neighbor_list

        atoms = read("POSCAR")
        nl = build_neighbor_list(
            atoms,
            center_elements=['Pb', 'Sr'],
            neighbor_elements=['O'],
            cutoff=4.0,
            neighbor_num=12,
            defect=False
        )
    """
    
    # check the dim of cell, the small cell may cause error in neighbor list
    # the cutoff for this check is 4 Ang.
    CUTOFF = 4.0
    cellpar = atoms.cell.cellpar()[[0,1,2]]
    if np.any(cellpar < CUTOFF):
        print("Warning: The cell length is smaller than 4 Angstrom, which may lead to unexpected error!")
        
    stru = AseAtomsAdaptor.get_structure(atoms)
    # initialize the index list
    center_elements_index = []
    neighbor_elements_index = []

    # use set to speed up the search
    center_elements_set = set(center_elements)
    neighbor_elements_set = set(neighbor_elements)

    # find the index of the center elements
    for idx in range(len(stru)):
        if str(stru[idx].specie) in center_elements_set:
            center_elements_index.append(idx)
        
    # build the neighbor list
    center_idx, point_idx, offset_vectors, distances = stru.get_neighbor_list(r=cutoff)

    # select the elements that are in the center_elements and neighbor_elements
    center_elements_mask = np.isin(center_idx, center_elements_index)
    neighbor_elements_mask = np.array([str(stru[idx].specie) in neighbor_elements_set for idx in point_idx])
    combined_mask = center_elements_mask & neighbor_elements_mask
    selected_center_elements_index = center_idx[combined_mask]
    selected_neighbor_elements_index = point_idx[combined_mask]
    selected_offset_vectors = offset_vectors[combined_mask]
    selected_distances = distances[combined_mask]

    # build the neighbor list in the format of {center: [neighbor1, neighbor2, ...]}
    result = {element_index: [] for element_index in center_elements_index}
    result_distance = {element_index: [] for element_index in center_elements_index}
    result_offset = {element_index: [] for element_index in center_elements_index}
    for center, point, offset, distance in zip(selected_center_elements_index, selected_neighbor_elements_index, selected_offset_vectors, selected_distances):
        result[center].append(point)
        result_distance[center].append(distance)
        result_offset[center].append(offset)
        
    # sort the neighbors by distance
    for center in center_elements_index:
        if len(result[center]) > 0:
            result[center] = np.array(result[center])
            result_distance[center] = np.array(result_distance[center])
            result_offset[center] = np.array(result_offset[center])
            result[center] = result[center][np.argsort(result_distance[center])]
        
    # check if the number of neighbors is correct
    # if defect is True, fill the missing neighbors with the center itself
    # if defect is False, raise an error
    # if the number of neighbors is more than neighbor_num, only keep the first neighbor_num neighbors
    for center in center_elements_index:
        if len(result[center]) < neighbor_num and not defect:
            raise ValueError(f"{center} {stru[center].specie} has {len(result[center])} neighbors, expected at least {neighbor_num}")
        elif len(result[center]) < neighbor_num and defect:
            print(f"Warning: {center} has {len(result[center])} neighbors, expected at least {neighbor_num}")
            neighbor_elements_index.append([center]*neighbor_num)
        elif len(result[center]) >= neighbor_num:
            neighbor_elements_index.append(result[center][:neighbor_num])
        
    center_elements_index = np.array(center_elements_index)
    neighbor_elements_index = np.array(neighbor_elements_index)
    nl = np.concatenate([center_elements_index[:,np.newaxis], neighbor_elements_index], axis=1)
    nl +=1 # convert the index to 1-based
    return nl

def save_neighbor_list(nl: np.ndarray, file_name: str, zero_based: bool=False) -> None:
    """Save a neighbor list array to a text file.

    Parameters
    ----------
    nl : np.ndarray
        Neighbor list array as returned by :func:`build_neighbor_list`. The
        function expects an integer array where each row contains a center
        index followed by neighbor indices.
    file_name : str
        Output file path. The neighbor list will be written as a text file
        with fixed-width integer columns.
    zero_based : bool, optional
        If True, convert indices to zero-based before saving. By default the
        neighbor list uses 1-based indices and ``zero_based`` is False.

    Examples
    --------
    >>> from ferrodispcalc.neighborlist import save_neighbor_list, build_neighbor_list
    >>> nl = build_neighbor_list(...)
    >>> save_neighbor_list(nl, 'nl.dat', zero_based=False)
    """

    nl_to_save = nl-1 if zero_based else nl
    np.savetxt(file_name, nl_to_save, fmt='%5d')
    print(f"Neighbor list saved to {file_name}")