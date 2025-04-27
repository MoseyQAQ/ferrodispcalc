'''
build_neighbor_list.py:
    This module is used to build the neighbor list for the ABO3 perovskite structure.
    The input structure file can be in the format of lammps dump file or other formats.

Example:
    from build_neighbor_list import NeighborListABO3
    nl = NeighborListABO3('structure.lmp', 'lmp-dump', type_map=['Pb', 'Sr', 'Ti', 'O'])
    center_elements = ['Pb', 'Sr']
    neighbor_elements = ['O']
    nl.build(center_elements, neighbor_elements, cutoff=4.0, neighbor_num=6)
    nl.write('neighbor_list.txt')

Author: Denan Li
Last modified: 2024-07-15
Email: lidenan@westlake.edu.cn   

Todo list:
    1. support HfO2 structure
'''
import numpy as np
from pymatgen.core import Structure, Lattice

class NeighborListABO3:
    def __init__(self, file_name: str, format: str,
                 type_map: list[str] = None):
        '''
        initialize the NeighborListABO3 object.

        Args:
            file_name (str): The file name of the structure file.
            format (str): The format of the structure file.
            type_map (list[str]): The list of element names in the structure file. Default is None.
        '''
        self.file_name = file_name
        self.format = format
        self.type_map = type_map
        
        # if the format is lmp-dump, read the structure from the lammps dump file
        if self.format == 'lmp-dump':
            f = open(self.file_name, 'r')
            self.natoms = self._get_natoms()
            cell, type_index, coord = self._read_lmp_traj(f)
            f.close()
            self.st = Structure(Lattice(cell), type_index, coord, coords_are_cartesian=True)
        
        # if other format, read the structure with pymatgen
        else:
            self.st = Structure.from_file(self.file_name)
    def build(self, center_elements: list[str], neighbor_elements: list[str],
              cutoff: float = 4.0, neighbor_num: int = 6,
              defect: bool = False) -> tuple[list, list]:
        '''
        Build the neighbor list for the center elements and neighbor elements.

        Args:
            center_elements (list[str]): The list of center elements. i.e. ['Pb', 'Sr']
            neighbor_elements (list[str]): The list of neighbor elements. i.e. ['O']
            cutoff (float): The cutoff radius for the neighbor search, unit is A. Default is 4.0.
            neighbor_num (int): The number of neighbors for each center element. Default is 6.
            defect (bool): If True, the number of neighbors can be less than neighbor_num. Default is False.
        
        Returns:
            tuple: The index list of the center elements and the neighbor elements.
        '''
        # initialize the index list
        center_elements_index = []
        neighbor_elements_index = []

        # use set to speed up the search
        center_elements_set = set(center_elements)
        neighbor_elements_set = set(neighbor_elements)

        # find the index of the center elements
        for idx in range(len(self.st)):
            if str(self.st[idx].specie) in center_elements_set:
                center_elements_index.append(idx)
        
        # build the neighbor list
        center_idx, point_idx, offset_vectors, distances = self.st.get_neighbor_list(r=cutoff)

        # select the elements that are in the center_elements and neighbor_elements
        center_elements_mask = np.isin(center_idx, center_elements_index)
        neighbor_elements_mask = np.array([str(self.st[idx].specie) in neighbor_elements_set for idx in point_idx])
        combined_mask = center_elements_mask & neighbor_elements_mask
        selected_center_elements_index = center_idx[combined_mask]
        selected_neighbor_elements_index = point_idx[combined_mask]

        # build the neighbor list in the format of {center: [neighbor1, neighbor2, ...]}
        result = {element_index: [] for element_index in center_elements_index}
        for center, point in zip(selected_center_elements_index, selected_neighbor_elements_index):
            result[center].append(point)
        
        # check if the number of neighbors is correct
        for center in center_elements_index:
            if len(result[center]) != neighbor_num and not defect:
                raise ValueError(f"{center}, {self.st[center].specie} has {len(result[center])} neighbors, expected {neighbor_num}")  
            elif len(result[center]) != neighbor_num and defect:
                print(f"Warning: {center} has {len(result[center])} neighbors, expected {neighbor_num}")
                neighbor_elements_index.append([center]*neighbor_num)   # fill the missing neighbors with the center itself
            else:
                neighbor_elements_index.append(result[center])
        
        self.center_elements_index = np.array(center_elements_index)
        self.neighbor_elements_index = np.array(neighbor_elements_index)

        return center_elements_index, neighbor_elements_index
    
    def write(self, file_name: str):
        '''
        write the neighbor list to a file.
        The first column is the index of the center atoms, 
        the rest columns are the index of the neighbor atoms.

        Args:
            file_name (str): The file name to write the neighbor list.
        '''
        result = np.concatenate([self.center_elements_index[:,np.newaxis], self.neighbor_elements_index], axis=1)
        np.savetxt(file_name, result, fmt='%10d')
    
    def _get_natoms(self) -> int:
        flag = False
        with open(self.file_name, 'r') as f:
            for line in f:
                if 'ITEM: NUMBER OF ATOMS' in line:
                    flag = True
                    continue
                if flag:
                    natoms = int(line)
                    break
        return natoms
    
    def _read_cell(self,f) -> np.ndarray:
        line = f.readline().split()
        line = [ float(x) for x in line ]
        xlo_bound = line[0]
        xhi_bound = line[1]
        xy = line[2]
        line = f.readline().split()
        line = [ float(x) for x in line ]
        ylo_bound = line[0]
        yhi_bound = line[1]
        xz = line[2]
        line = f.readline().split()
        line = [ float(x) for x in line ]
        zlo_bound = line[0]
        zhi_bound = line[1]
        yz = line[2]
        xlo = xlo_bound - min(0.0, xy, xz, xy+xz)
        xhi = xhi_bound - max(0.0, xy, xz, xy+xz)
        ylo = ylo_bound - min(0.0, yz)
        yhi = yhi_bound - max(0.0, yz)
        zlo = zlo_bound
        zhi = zhi_bound
        xx = xhi - xlo
        yy = yhi - ylo
        zz = zhi - zlo
        cell = np.zeros((3,3))
        cell[0,:] = [xx, 0, 0]
        cell[1,:] = [xy, yy, 0]
        cell[2,:] = [xz, yz, zz]
        return cell
    def _read_atoms(self,f) -> tuple:
        type_index = [None]*self.natoms
        coord = np.zeros((self.natoms,3))
        for i in range(self.natoms):
            line = f.readline().split()
            type_index[i] = self.type_map[int(line[1])-1]
            tmp = [ float(x) for x in line[2:] ]
            coord[i,0] = tmp[0]
            coord[i,1] = tmp[1]
            coord[i,2] = tmp[2]
        return type_index, coord
    def _skip_blank_line(self,f,n:int) -> None:
        for i in range(n):
            f.readline()
    def _read_lmp_traj(self,f) -> tuple:
        self._skip_blank_line(f,5)
        cell = self._read_cell(f)
        self._skip_blank_line(f,1)
        type_index, coord = self._read_atoms(f)
        return cell, type_index, coord