import numpy as np
from ase import Atoms
np.set_printoptions(precision=2, suppress=True)

def __select_traj(traj: list[Atoms] | Atoms, select: list[int] | slice | None) -> list[Atoms]:
    nframe = len(traj)

    # 1. default: select last 50% frames
    if select is None:
        select = slice(nframe//2, nframe, 1)

    # 2. select frames
    if isinstance(traj, Atoms):
        selected_traj = [traj]
    else:
        selected_traj: list[Atoms] = traj[select]
    print(f"Number of Selected Frames: {len(selected_traj)}")
    return selected_traj

def calculate_displacement(traj: list[Atoms] | Atoms, nl: np.ndarray, select: list[int] | slice | None = None) -> np.ndarray:

    selected_traj: list[Atoms] = __select_traj(traj, select)

    # 2. get coord, cell arrays
    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])
    nframes = coords.shape[0]
    natoms = nl.shape[0]
    _nl = nl - 1 # convert to zero-based index
    displacement = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, nneighbors, 3)
        
    # 3. walk through frames
    for i in range(nframes):
        # 3.1 select center atoms and their coordinates
        center_id = _nl[:, 0]
        center_coords = coords[i, center_id]
        cell = cells[i]

        # 3.2 walk through neighbors
        for j, neighbors in enumerate(_nl):
            neighbors_id = neighbors[1:]
            neighbors_coords = coords[i,neighbors_id]
            neighbors_coords_diff = center_coords[j] - neighbors_coords
            neighbors_coords_diff_frac = np.dot(neighbors_coords_diff, np.linalg.inv(cell))
            neighbors_coords_diff_frac[neighbors_coords_diff_frac > 0.5] -= 1
            neighbors_coords_diff_frac[neighbors_coords_diff_frac < -0.5] += 1
            neighbors_coords_diff = np.dot(neighbors_coords_diff_frac, cell)
            displacement[i, j] = np.mean(neighbors_coords_diff, axis=0)
    
    if nframes == 1:
        displacement = displacement[0]

    return displacement

def calculate_polarization(traj, 
                           nl_ba:  np.ndarray, 
                           nl_bx: np.ndarray, 
                           born_effective_charge: dict, 
                           select: list[int] | slice | None = None) -> np.ndarray:

    # 1. determine frames to select
    selected_traj: list[Atoms] = __select_traj(traj, select)

    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])
    _nl_ba = nl_ba - 1
    _nl_bx = nl_bx - 1
    assert np.allclose(_nl_ba[:, 0], _nl_bx[:, 0]), "The center atom indices in nl_ba and nl_bx must be the same."

    nframes = coords.shape[0]
    natoms = _nl_ba.shape[0] # it's number of unit cells actually
    polarization = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, natoms, 3)
    bec = np.array([born_effective_charge[atom.symbol] for atom in selected_traj[0]]) # convet born effective charge to list
    # the sum of bec should be small
    if np.abs(np.sum(bec)) > 1.0E-5:
        print(f"Warning: The sum of Born charge is {np.sum(bec)}. May lead to unphysical results.")

    conversion_factor = 1.602176E-19 * 1.0E-10 * 1.0E30 # convert to C/m^2
    
    # walk through frames
    for i in range(nframes):
        cell = cells[i]
        cell_inv = np.linalg.inv(cell)
        volume = np.abs(np.linalg.det(cell))
        volume_per_uc = volume / natoms
            
        # walk through all unit cells
        for j in range(natoms):
            # 1. get id
            b_id = _nl_ba[j, 0]
            a_id = _nl_ba[j, 1:]
            x_id = _nl_bx[j, 1:]

            # 2. get coordinates
            b_coords = coords[i, b_id]
            a_coords = coords[i, a_id]
            x_coords = coords[i, x_id]

            # 3. get frac coordinates
            b_coords_frac = np.dot(b_coords, cell_inv)
            a_coords_frac = np.dot(a_coords, cell_inv)
            x_coords_frac = np.dot(x_coords, cell_inv)
            
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
    
    if nframes == 1:
        polarization = polarization[0]

    return polarization

def calculate_octahedral_tilt(traj: list[Atoms] | Atoms, nl_bo: np.ndarray, select: list[int] | slice | None = None) -> np.ndarray:
    selected_traj: list[Atoms] = __select_traj(traj, select)
    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])
    nframes = len(selected_traj)
    natoms = nl_bo.shape[0]
    _nl = nl_bo - 1 # convert to zero-based index
    assert _nl.shape[1] == 7, "Neighbor list for octahedral tilt calculation must have 6 neighbors."
    octahedral_tilt = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, natoms, 3)

    for i in range(nframes):
        center_id = _nl[:, 0]
        center_coords = coords[i, center_id]
        cell = cells[i]
        cell_inv = np.linalg.inv(cell)

        # 3.2 walk through neighbors
        for j, neighbors in enumerate(_nl):
            neighbors_id = neighbors[1:]
            neighbors_coords = coords[i,neighbors_id]
            neighbors_coords_frac = np.dot(neighbors_coords, cell_inv)
            neighbors_coords_diff = center_coords[j] - neighbors_coords
            neighbors_coords_diff_frac = np.dot(neighbors_coords_diff, cell_inv)
            neighbors_coords_frac[neighbors_coords_diff_frac > 0.5] -= 1
            neighbors_coords_frac[neighbors_coords_diff_frac < -0.5] += 1
            neighbors_coords = np.dot(neighbors_coords_frac, cell) # update neighbor coords after mic

            # sort the neighbors into +/-x, +/-y, +/-z based on their relative positions to center
            diffs = neighbors_coords - center_coords[j] # shape: (6, 3)
            
            # sort along the x direction
            x_pos_idx = np.argmax(diffs[:, 0])
            x_neg_idx = np.argmin(diffs[:, 0])
            # sort along the y direction
            y_pos_idx = np.argmax(diffs[:, 1])
            y_neg_idx = np.argmin(diffs[:, 1])
            # sort along the z direction
            z_pos_idx = np.argmax(diffs[:, 2])
            z_neg_idx = np.argmin(diffs[:, 2])

            x_vector = (neighbors_coords[x_pos_idx] - neighbors_coords[x_neg_idx])
            y_vector = (neighbors_coords[y_pos_idx] - neighbors_coords[y_neg_idx])
            z_vector = (neighbors_coords[z_pos_idx] - neighbors_coords[z_neg_idx])

            x_angle = np.arccos(np.dot(x_vector, np.array([1,0,0])) / np.linalg.norm(x_vector))
            y_angle = np.arccos(np.dot(y_vector, np.array([0,1,0])) / np.linalg.norm(y_vector))
            z_angle = np.arccos(np.dot(z_vector, np.array([0,0,1])) / np.linalg.norm(z_vector))
            octahedral_tilt[i, j] = np.rad2deg([x_angle, y_angle, z_angle])

    if nframes == 1:
        octahedral_tilt = octahedral_tilt[0]
        
    return octahedral_tilt

def calculate_averaged_structure(traj: list[Atoms], select: list[int] | slice | None = None) -> Atoms:
    
    selected_traj: list[Atoms] = __select_traj(traj, select)
    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])

    # 3. update coordinates to account for PBC
    coords_frac = np.array([np.dot(coords[i], np.linalg.inv(cells[i])) for i in range(len(coords))])
    coords_frac_diff = coords_frac - coords_frac[0]
    coords_frac[coords_frac_diff > 0.5] -= 1
    coords_frac[coords_frac_diff < -0.5] += 1
    coords = np.array([np.dot(coords_frac[i], cells[i]) for i in range(len(coords))])

    # 4. compute averaged structure
    avg_cell = np.mean(cells, axis=0)
    avg_coords = np.mean(coords, axis=0)
    symbols = [atom.symbol for atom in traj[0]]
    atoms = Atoms(symbols=symbols, positions=avg_coords, cell=avg_cell, pbc=True)
    return atoms