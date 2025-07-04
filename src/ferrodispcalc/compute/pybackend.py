from ferrodispcalc.compute.backend import ComputeBackend
from ase import Atoms
import numpy as np
from tqdm import tqdm

class PyCompute(ComputeBackend):
    def get_averaged_structure(self, select: list[int]) -> Atoms:

        traj: list[Atoms] = self.traj

        # shape: (nframes, natoms, 3)
        coords = np.array([atoms.get_positions() for atoms in traj])
        cells = np.array([atoms.get_cell().array for atoms in traj])
        coords = coords[select]
        cells = cells[select]
        coords_frac = np.array([np.dot(coords[i], np.linalg.inv(cells[i])) for i in range(len(coords))])
        coords_frac_diff = coords_frac - coords_frac[0]
        coords_frac[coords_frac_diff > 0.5] -= 1
        coords_frac[coords_frac_diff < -0.5] += 1
        coords = np.array([np.dot(coords_frac[i], cells[i]) for i in range(len(coords))])
        avg_cell = np.mean(cells, axis=0)
        avg_coords = np.mean(coords, axis=0)
        symbols = [atom.symbol for atom in traj[0]]
        return Atoms(
            symbols=symbols,
            positions=avg_coords,
            cell=avg_cell,
            pbc=True
        )
        
    def get_polarization(self, select: np.ndarray, nl_ba: np.ndarray, nl_bx: np.ndarray, born_effective_charge: dict[str:list[float]]) -> np.ndarray:
        traj: list[Atoms] = self.traj
        coords = np.array([atoms.get_positions() for atoms in traj])
        cells = np.array([atoms.get_cell().array for atoms in traj])
        coords = coords[select]
        cells = cells[select]
        nl_ba -= 1
        nl_bx -= 1
        nframes = coords.shape[0]
        natoms = nl_ba.shape[0] # it's number of unit cells actually
        polarization = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, natoms, 3)
        bec = np.array([born_effective_charge[atom.symbol] for atom in traj[0]]) # convet born effective charge to list
        conversion_factor = 1.602176E-19 * 1.0E-10 * 1.0E30 # convert to C/m^2

        # walk through frames
        for i in range(nframes):
            cell = cells[i]
            volume = np.abs(np.linalg.det(cell))
            volume_per_uc = volume / natoms
            
            # walk through all unit cells
            for j in range(natoms):
                # 1. get id
                b_id = nl_ba[j, 0]
                a_id = nl_ba[j, 1:]
                x_id = nl_bx[j, 1:]

                # 2. get coordinates
                b_coords = coords[i, b_id]
                a_coords = coords[i, a_id]
                x_coords = coords[i, x_id]

                # 3. get frac coordinates
                b_coords_frac = np.dot(b_coords, np.linalg.inv(cell))
                a_coords_frac = np.dot(a_coords, np.linalg.inv(cell))
                x_coords_frac = np.dot(x_coords, np.linalg.inv(cell))

                # 4. apply mic to frac coordinates, and update coordinates of a and x
                a_coords_diff_frac = a_coords_frac - b_coords_frac
                x_coords_diff_frac = x_coords_frac - b_coords_frac
                a_coords_frac[a_coords_diff_frac > 0.5] -= 1
                a_coords_frac[a_coords_diff_frac < -0.5] += 1
                x_coords_frac[x_coords_diff_frac > 0.5] -= 1
                x_coords_frac[x_coords_diff_frac < -0.5] += 1
                a_coords = np.dot(a_coords_frac, cell)
                x_coords = np.dot(x_coords_frac, cell)

                # 5. calculate polarization
                polarization[i, j] = b_coords * bec[b_id] + np.sum(a_coords * bec[a_id][:, np.newaxis], axis=0) / 8 + np.sum(x_coords * bec[x_id][:, np.newaxis], axis=0) / 2
            
            # 6. convert to C/m^2
            polarization[i] = polarization[i] * conversion_factor / volume_per_uc
        
        return polarization

    def get_displacement(self, select: list[int], nl: np.ndarray) -> np.ndarray:

        traj: list[Atoms] = self.traj
        coords = np.array([atoms.get_positions() for atoms in traj])
        cells = np.array([atoms.get_cell().array for atoms in traj])
        coords = coords[select]
        cells = cells[select]
        nframes = coords.shape[0]
        natoms = nl.shape[0]
        nl -=1 # convert to 0-based index
        displacement = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, nneighbors, 3)
        
        # walk through frames
        for i in range(nframes):
            # select center atoms and their coordinates
            center_id = nl[:, 0]
            center_coords = coords[i, center_id]
            cell = cells[i]

            # walk through neighbors
            for j, neighbors in enumerate(nl):
                neighbors_id = neighbors[1:]
                neighbors_coords = coords[i,neighbors_id]
                neighbors_coords_diff = center_coords[j] - neighbors_coords
                neighbors_coords_diff_frac = np.dot(neighbors_coords_diff, np.linalg.inv(cell))
                neighbors_coords_diff_frac[neighbors_coords_diff_frac > 0.5] -= 1
                neighbors_coords_diff_frac[neighbors_coords_diff_frac < -0.5] += 1
                neighbors_coords_diff = np.dot(neighbors_coords_diff_frac, cell)
                displacement[i, j] = np.mean(neighbors_coords_diff, axis=0)
        
        return displacement
    
    def get_local_lattice(self, select: list[int], nl: np.ndarray) -> np.ndarray:
        """
        Calculates the local lattice vectors for each unit cell.

        This method determines the local lattice by classifying neighbor atoms
        based on their position relative to a central atom and averaging their
        displacement vectors.

        Args:
            select: A list of frame indices to include in the calculation.
            nl: The neighbor list array (n_unitcells, 9), with the first column
                being the central atom and the rest being its neighbors. Indices
                are 1-based.

        Returns:
            A NumPy array of shape (nframes, n_unitcells, 9) containing the
            flattened local lattice vectors (ax,ay,az,bx,by,bz,cx,cy,cz).
        """
        traj: list[Atoms] = self.traj
        coords = np.array([atoms.get_positions() for atoms in traj])
        cells = np.array([atoms.get_cell().array for atoms in traj])
        coords = coords[select]
        cells = cells[select]

        nframes = coords.shape[0]
        n_unitcells = nl.shape[0]
        nl_zero_based = nl - 1  # Convert to 0-based index

        all_local_lattices = np.full((nframes, n_unitcells, 9), np.nan)

        # walk through frames
        for i in tqdm(range(nframes), desc="Calculating local lattice vectors"):
            cell = cells[i]
            inv_cell = np.linalg.inv(cell)
            frame_coords = coords[i]

            # walk through all unit cells
            for j in range(n_unitcells):
                center_idx = nl_zero_based[j, 0]
                neighbor_indices = nl_zero_based[j, 1:]
                
                center_pos = frame_coords[center_idx]
                neighbor_pos = frame_coords[neighbor_indices]

                # Calculate vectors from center to neighbors, applying MIC
                vectors_cart = neighbor_pos - center_pos
                vectors_frac = np.dot(vectors_cart, inv_cell)
                vectors_frac[vectors_frac > 0.5] -= 1.0
                vectors_frac[vectors_frac < -0.5] += 1.0
                vectors = np.dot(vectors_frac, cell)

                # Classify vectors into alpha/beta groups based on cartesian components
                alpha_a = vectors[vectors[:, 0] < 0]
                beta_a = vectors[vectors[:, 0] >= 0]

                alpha_b = vectors[vectors[:, 1] < 0]
                beta_b = vectors[vectors[:, 1] >= 0]
                
                alpha_c = vectors[vectors[:, 2] < 0]
                beta_c = vectors[vectors[:, 2] >= 0]
                
                # Define calculation function
                def get_lattice_vector(beta, alpha):
                    # Check for valid groups to prevent errors with faulty neighbor lists
                    if alpha.shape[0] == 0 or beta.shape[0] == 0:
                        return np.array([np.nan, np.nan, np.nan])
                    return 0.25 * (np.sum(beta, axis=0) - np.sum(alpha, axis=0))

                # Calculate the three local lattice vectors
                lattice_vector_a = get_lattice_vector(beta_a, alpha_a)
                lattice_vector_b = get_lattice_vector(beta_b, alpha_b)
                lattice_vector_c = get_lattice_vector(beta_c, alpha_c)

                # Store the flattened 3x3 local lattice matrix
                all_local_lattices[i, j, :] = np.hstack(
                    (lattice_vector_a, lattice_vector_b, lattice_vector_c)
                )

        return all_local_lattices