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