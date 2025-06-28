import numpy as np
from pymatgen.core import Structure, Lattice
from ase import Atoms
class LAMMPSdump:
    '''LAMMPSdump calss is used to read the lammps dump file.
        
        Attributes:
        ----------
        file_name: str
            The name of the lammps dump file.
        type_map: list[str]
            The list of atom types. The index of the list is the atom type in the lammps dump file.
        
        Example:
        --------
        >>> from ferrodispcalc.io import LAMMPSdump
        >>> from ferrodispcalc.type_map import UniPero
        >>> lmp = LAMMPSdump('dump.lammpstrj', UniPero)
        >>> stru = lmp.get_first_frame() # get the first frame, in pymatgen format.
        >>> nframes = lmp.get_nframes() # get the number of frames in the lammps dump file.

        Methods:
        -------
        get_first_frame():
            get the first frame, in pymatgen format.
        get_nframes():
            get the number of frames.
        get_natoms():
            get the number of atoms.
        '''
    def __init__(self, file_name: str, type_map: list[str] = None):
        '''
        Initialize the LAMMPSdump object.

        Parameters:
        ----------
        file_name: str
            The name of the lammps dump file.
        type_map: list[str]
            The list of atom types. The index of the list is the atom type in the lammps dump file.
        '''
        self.file_name = file_name
        self.type_map = type_map
        self.f = None # file object
        self.natoms = None # number of atoms in the lammps dump file

    def get_first_frame(self) -> Atoms:
        '''
        Get the first frame in the lammps dump file. Return the structure in ASE/atoms format.

        Parameters:
        ----------
        None

        Returns:
        -------
        Structure:
            The structure in pymatgen format.
        '''
        f = open(self.file_name, 'r')
        self.natoms = self.get_natoms()
        cell, type_index, coord = self._read_lmp_traj(f)
        f.close()
        stru = Atoms(symbols=type_index, positions=coord, cell=cell, pbc=True)
        return stru
    
    def get_next_frame(self) -> Atoms:
        '''
        Read the next frame in the lammps dump file iteratively.
        '''
        if self.f is None:
            self.f = open(self.file_name, 'r')
        if self.natoms is None:
            self.natoms = self.get_natoms()
        
        cell, type_index, coord = self._read_lmp_traj(self.f)
        stru = Atoms(symbols=type_index, positions=coord, cell=cell, pbc=True)

        return stru

    
    def get_nframes(self) -> int:
        '''
        Get the number of frames in the lammps dump file.

        Parameters:
        ----------
        None

        Returns:
        -------
        int: 
            The number of frames.
        '''

        f = open(self.file_name, 'r')
        nframes = 0
        for line in f:
            if 'ITEM: TIMESTEP' in line:
                nframes += 1
        f.close()
        return nframes
    
    def get_natoms(self) -> int:
        '''
        Get the number of atoms in the lammps dump file.

        Returns:
        -------
            int: The number of atoms.
        '''
        f = open(self.file_name, 'r')
        for line in f:
            if 'ITEM: NUMBER OF ATOMS' in line:
                natoms = int(f.readline().strip())
                break
        f.close()
        return natoms
    
    def __read_cell(self,f) -> np.ndarray:
        '''
        Read the cell information.

        Parameters:
        ----------
        f: file
            The file object of the lammps dump file.
        
        Returns:
        -------
        np.ndarray:
            The cell matrix.
        '''
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
    
    def __read_atoms(self,f) -> tuple:
        '''
        Read the atom type and coordinates.

        Parameters:
        ----------
        f: file
            The file object of the lammps dump file.
        
        Returns:
        -------
        tuple:
            type_index: list[str]
                The list of atom types.
            coord: np.ndarray
                The coordinates of atoms.
        '''
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
    
    def __skip_blank_line(self,f,n:int) -> None:
        '''
        Skip the blank lines.

        Parameters:
        ----------
        f: file
            The file object of the lammps dump file.
        n: int
            The number of lines to skip.
        
        Returns:
        -------
        None
        '''
        for i in range(n):
            f.readline()

    def _read_lmp_traj(self,f) -> tuple[np.ndarray, list[str], np.ndarray]:
        '''
        Read one frame

        Parameters:
        ----------
        f: file
            The file object of the lammps dump file.
        
        Returns:
        -------
        tuple:
            cell: np.ndarray
                The cell matrix.
            type_index: list[str]
                The list of atom types.
            coord: np.ndarray
                The coordinates of atoms
        '''
        self.__skip_blank_line(f,5)
        cell = self.__read_cell(f)
        self.__skip_blank_line(f,1)
        type_index, coord = self.__read_atoms(f)
        return cell, type_index, coord
    

# ------------------- LAMMPS data ------------------- #
class LAMMPSdata:
    def __init__():
        raise NotImplementedError('This class is not implemented yet')