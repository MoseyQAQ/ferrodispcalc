import numpy as np
from ase import Atoms
from tqdm import tqdm
np.set_printoptions(precision=2, suppress=True)

def __select_traj(traj: list[Atoms] | Atoms, select: list[int] | slice | None = None) -> list[Atoms]:
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
    """Calculate ionic displacements from an MD trajectory.

    For each center atom in the neighbor list, the displacement is computed as
    the mean vector from its neighbors to itself. Periodic boundary conditions
    are handled via the minimum image convention (MIC).

    Parameters
    ----------
    traj : list[Atoms] | Atoms
        Full MD trajectory (list of ASE Atoms) or a single Atoms object.
    nl : np.ndarray
        Neighbor list array of shape ``(n_centers, n_neighbors + 1)`` with
        1-based indices as returned by
        :func:`~ferrodispcalc.neighborlist.build_neighbor_list`.
    select : list[int] | slice | None, optional
        Frame selection. ``None`` selects the last 50 % of frames.
        Defaults to ``None``.

    Returns
    -------
    np.ndarray
        Displacement vectors in Ångström. Shape ``(n_frames, n_centers, 3)``
        for multiple frames, or ``(n_centers, 3)`` for a single frame.
    """

    selected_traj: list[Atoms] = __select_traj(traj, select)

    # 2. get coord, cell arrays
    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])
    nframes = coords.shape[0]
    natoms = nl.shape[0]
    _nl = nl - 1 # convert to zero-based index
    displacement = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, nneighbors, 3)
    
    # 2. get indices
    center_id = _nl[:, 0]
    neighbors_id = _nl[:, 1:]

    # 3. walk through frames
    ranger = tqdm(range(nframes), desc="Calculating Displacement") if nframes > 10 else range(nframes)
    for i in ranger:
        # 3.1 select center atoms and their coordinates
        cell = cells[i]
        cell_inv = np.linalg.inv(cell)
        frame_coords = coords[i]
        center_coords = frame_coords[center_id] # shape: (natoms, 3)
        neighbors_coords = frame_coords[neighbors_id] # shape: (natoms, num_neighbor, 3)

        # 3.2 update coordinates to account for MIC
        center_coords_expanded = center_coords[:, np.newaxis, :]
        neighbors_coords_diff = center_coords_expanded - neighbors_coords # shape: (natoms, num_neighbor, 3)
        neighbors_coords_diff_frac = np.dot(neighbors_coords_diff, cell_inv)
        neighbors_coords_diff_frac[neighbors_coords_diff_frac > 0.5] -= 1
        neighbors_coords_diff_frac[neighbors_coords_diff_frac < -0.5] += 1

        # 3.3 calculate displacement
        neighbors_coords_diff = np.dot(neighbors_coords_diff_frac, cell)
        displacement[i] = np.mean(neighbors_coords_diff, axis=1)
    
    if nframes == 1:
        displacement = displacement[0]

    return displacement

def calculate_polarization(traj,
                           nl_ba:  np.ndarray,
                           nl_bx: np.ndarray,
                           born_effective_charge: dict,
                           select: list[int] | slice | None = None) -> np.ndarray:
    """Calculate local polarization for each unit cell in a perovskite.

    The local polarization at each B-site unit cell is computed from the
    positions of the B-site cation, the surrounding A-site cations (8
    neighbors), and the anions (6 neighbors), weighted by Born effective
    charges.  Results are returned in C/m².

    Parameters
    ----------
    traj : list[Atoms] | Atoms
        Full MD trajectory or a single ASE Atoms object.
    nl_ba : np.ndarray
        Neighbor list for B–A pairs (B-site center, 8 A-site neighbors),
        shape ``(n_cells, 9)``, 1-based indices.
    nl_bx : np.ndarray
        Neighbor list for B–X pairs (B-site center, 6 anion neighbors),
        shape ``(n_cells, 7)``, 1-based indices.
    born_effective_charge : dict
        Mapping of element symbol to scalar Born effective charge, e.g.
        ``{'Pb': 3.44, 'Ti': 5.18, 'O': -2.87}``.
    select : list[int] | slice | None, optional
        Frame selection. ``None`` selects the last 50 % of frames.
        Defaults to ``None``.

    Returns
    -------
    np.ndarray
        Local polarization in C/m². Shape ``(n_frames, n_cells, 3)`` for
        multiple frames, or ``(n_cells, 3)`` for a single frame.

    Raises
    ------
    AssertionError
        If the B-site center indices in ``nl_ba`` and ``nl_bx`` do not match,
        or if the neighbor counts are not 8 (A-site) and 6 (X-site).
    """

    # 1. determine frames to select
    selected_traj: list[Atoms] = __select_traj(traj, select)

    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])
    _nl_ba = nl_ba - 1
    _nl_bx = nl_bx - 1
    assert np.allclose(_nl_ba[:, 0], _nl_bx[:, 0]), "The center atom indices in nl_ba and nl_bx must be the same."
    assert _nl_ba.shape[1] == 9 and _nl_bx.shape[1] == 7, "Neighbor list for polarization calculation must have 8 A-site neighbors and 6 X-site neighbors."

    nframes = coords.shape[0]
    natoms = _nl_ba.shape[0] # it's number of unit cells actually
    polarization = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, natoms, 3)
    bec = np.array([born_effective_charge[atom.symbol] for atom in selected_traj[0]]) # convet born effective charge to list
    # the sum of bec should be small
    if np.abs(np.sum(bec)) > 1.0E-5:
        print(f"Warning: The sum of Born charge is {np.sum(bec)}. May lead to unphysical results.")

    conversion_factor = 1.602176E-19 * 1.0E-10 * 1.0E30 # convert to C/m^2

    # get indices and bec
    b_id = _nl_ba[:, 0] # shape: (natoms,)
    a_id = _nl_ba[:, 1:] # shape: (natoms, 8)
    x_id = _nl_bx[:, 1:] # shape: (natoms, 6)
    bec_b = bec[b_id] # shape: (natoms,)
    bec_a = bec[a_id] # shape: (natoms, 8)
    bec_x = bec[x_id] # shape: (natoms, 6)

    # walk through frames
    ranger = tqdm(range(nframes), desc="Calculating Polarization") if nframes > 10 else range(nframes)
    for i in ranger:
        cell = cells[i]
        cell_inv = np.linalg.inv(cell)
        volume = np.abs(np.linalg.det(cell))
        volume_per_uc = volume / natoms

        # get the coordinates
        frame_coords = coords[i]
        b_coords = frame_coords[b_id] # shape: (natoms, 3)
        a_coords = frame_coords[a_id] # shape: (natoms, 8, 3)
        x_coords = frame_coords[x_id] # shape: (natoms, 6, 3)
        b_coords_frac = np.dot(b_coords, cell_inv) # shape: (natoms, 3)
        a_coords_frac = np.dot(a_coords, cell_inv) # shape: (natoms, 8, 3)
        x_coords_frac = np.dot(x_coords, cell_inv) # shape: (natoms, 6, 3)

        # update coordinates to account for MIC
        b_frac_expanded = b_coords_frac[:, np.newaxis, :] # shape: (natoms, 1, 3)
        a_coords_diff_frac = a_coords_frac - b_frac_expanded # shape: (natoms, 8, 3)
        x_coords_diff_frac = x_coords_frac - b_frac_expanded # shape: (natoms, 6, 3)
        a_coords_frac[a_coords_diff_frac > 0.5] -= 1
        a_coords_frac[a_coords_diff_frac < -0.5] += 1
        x_coords_frac[x_coords_diff_frac > 0.5] -= 1
        x_coords_frac[x_coords_diff_frac < -0.5] += 1
        a_coords = np.dot(a_coords_frac, cell) # shape: (natoms, 8, 3)
        x_coords = np.dot(x_coords_frac, cell) # shape: (natoms, 6, 3)

        # calculate polarization
        pol_b = b_coords * bec_b[:, np.newaxis] # shape: (natoms, 3)
        pol_a = np.sum(a_coords * bec_a[:, :, np.newaxis], axis=1) / 8 # shape: (natoms, 3)
        pol_x = np.sum(x_coords * bec_x[:, :, np.newaxis], axis=1) / 2 # shape: (natoms, 3)
        polarization[i] = (pol_b + pol_a + pol_x) * conversion_factor / volume_per_uc
    
    if nframes == 1:
        polarization = polarization[0]

    return polarization

def calculate_octahedral_tilt(traj: list[Atoms] | Atoms, nl_bo: np.ndarray, select: list[int] | slice | None = None) -> np.ndarray:
    """Calculate octahedral tilt angles in a perovskite structure.

    For each octahedral center, the six neighboring anions are sorted into
    ±x, ±y, ±z pairs. The tilt angle about each Cartesian axis is the angle
    between the anion–anion bond vector and the corresponding unit vector.

    Parameters
    ----------
    traj : list[Atoms] | Atoms
        Full MD trajectory or a single ASE Atoms object.
    nl_bo : np.ndarray
        Neighbor list for B–O pairs, shape ``(n_centers, 7)`` (center + 6
        anion neighbors), 1-based indices.
    select : list[int] | slice | None, optional
        Frame selection. ``None`` selects the last 50 % of frames.
        Defaults to ``None``.

    Returns
    -------
    np.ndarray
        Tilt angles in degrees. Shape ``(n_frames, n_centers, 3)`` for
        multiple frames (columns: x-tilt, y-tilt, z-tilt), or
        ``(n_centers, 3)`` for a single frame.

    Raises
    ------
    AssertionError
        If ``nl_bo`` does not have exactly 6 neighbors per center.
    """
    selected_traj: list[Atoms] = __select_traj(traj, select)
    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])
    nframes = len(selected_traj)
    natoms = nl_bo.shape[0]
    _nl = nl_bo - 1 # convert to zero-based index
    assert _nl.shape[1] == 7, "Neighbor list for octahedral tilt calculation must have 6 neighbors."
    octahedral_tilt = np.full((nframes, natoms, 3), np.nan) # shape: (nframes, natoms, 3)

    # 3. walk through frames
    ranger = tqdm(range(nframes), desc="Calculating Octahedral Tilt") if nframes > 10 else range(nframes)
    for i in ranger:
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


def calculate_dielectric_constant(polarization: np.ndarray,
                                  volume: float,
                                  temperature: float = 300.0,
                                  atomic: bool = False) -> dict[str, float | np.ndarray]:
    """Calculate dielectric tensor components from polarization fluctuations.

    The dielectric tensor is computed from the fluctuation formula
    ``eps_ij = V / (eps0 * kB * T) * (<Pi Pj> - <Pi><Pj>)``, where the input
    polarization is assumed to be in C/m². The input volume must be given in
    Å³ and is converted internally to m³. When ``atomic=False``, the
    polarization is first averaged over all unit cells. When ``atomic=True``,
    the dielectric tensor is computed independently for each unit cell without
    spatial averaging, and ``volume`` is interpreted as the volume per unit
    cell.

    Parameters
    ----------
    polarization : np.ndarray
        Polarization time series. Shape ``(n_frames, n_cells, 3)`` for local
        polarization from :func:`calculate_polarization`, or ``(n_frames, 3)``
        for an already averaged polarization trajectory.
    volume : float
        Volume in Å³. For ``atomic=False``, this is the total volume of the
        system corresponding to the polarization trajectory. For
        ``atomic=True``, this must be the volume of a single unit cell.
    temperature : float, optional
        Temperature in K. Defaults to ``300.0``.
    atomic : bool, optional
        Whether to compute local dielectric tensor components for each unit
        cell. Defaults to ``False``.

    Returns
    -------
    dict[str, float | np.ndarray]
        Dielectric tensor components ``eps_xx``, ``eps_yy``, ``eps_zz``,
        ``eps_xy``, ``eps_xz``, and ``eps_yz``. Each value is a scalar when
        ``atomic=False`` and an array of shape ``(n_cells,)`` when
        ``atomic=True``.

    Notes
    -----
    This function computes only the ionic fluctuation contribution to the
    dielectric response and neglects the electronic contribution.

    No statistical error analysis is performed internally. Users should apply
    their own block averaging or related analysis to estimate uncertainties.
    By default, all provided frames are used in the calculation.

    Raises
    ------
    ValueError
        If the input shape is invalid, if ``atomic=True`` but the input is not
        local polarization, or if volume or temperature is not positive.
    """
    if volume <= 0:
        raise ValueError("volume must be positive.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if polarization.ndim not in (2, 3):
        raise ValueError("polarization must have shape (n_frames, 3) or (n_frames, n_cells, 3).")
    if polarization.shape[-1] != 3:
        raise ValueError("The last dimension of polarization must be 3.")
    if polarization.shape[0] < 1:
        raise ValueError("polarization must contain at least one frame.")
    if atomic and polarization.ndim != 3:
        raise ValueError("atomic=True requires polarization with shape (n_frames, n_cells, 3).")

    if atomic:
        pol = polarization
    elif polarization.ndim == 3:
        pol = np.mean(polarization, axis=1)
    else:
        pol = polarization

    eps0 = 8.854187817e-12
    kB = 1.380649e-23
    volume_m3 = volume * 1.0E-30
    factor = volume_m3 / (eps0 * kB * temperature)

    components = {
        "xx": (0, 0),
        "yy": (1, 1),
        "zz": (2, 2),
        "xy": (0, 1),
        "xz": (0, 2),
        "yz": (1, 2),
    }
    dielectric = {}
    for label, (i, j) in components.items():
        mean_pi = np.mean(pol[..., i], axis=0)
        mean_pj = np.mean(pol[..., j], axis=0)
        mean_pipj = np.mean(pol[..., i] * pol[..., j], axis=0)
        dielectric[f"eps_{label}"] = factor * (mean_pipj - mean_pi * mean_pj)

    return dielectric


def calculate_averaged_structure(traj: list[Atoms], select: list[int] | slice | None = None) -> Atoms:
    """Compute the time-averaged atomic structure from an MD trajectory.

    Atomic coordinates are unwrapped with respect to the first selected frame
    before averaging to avoid artefacts from periodic boundary crossings.

    Parameters
    ----------
    traj : list[Atoms]
        Full MD trajectory as a list of ASE Atoms objects.
    select : list[int] | slice | None, optional
        Frame selection. ``None`` selects the last 50 % of frames.
        Defaults to ``None``.

    Returns
    -------
    Atoms
        ASE Atoms object with averaged positions and cell. Element symbols and
        PBC flags are taken from the first frame of the trajectory.
    """

    selected_traj: list[Atoms] = __select_traj(traj, select)
    coords = np.array([atoms.get_positions() for atoms in selected_traj])
    cells = np.array([atoms.get_cell().array for atoms in selected_traj])

    cells_inv = np.linalg.inv(cells)
    coords_frac = np.matmul(coords, cells_inv)
    coords_frac_diff = coords_frac - coords_frac[0]
    coords_frac[coords_frac_diff > 0.5] -= 1
    coords_frac[coords_frac_diff < -0.5] += 1
    coords_unwrapped = np.matmul(coords_frac, cells)

    # 3. update coordinates to account for PBC
    #coords_frac = np.array([np.dot(coords[i], np.linalg.inv(cells[i])) for i in range(len(coords))])
    #coords_frac_diff = coords_frac - coords_frac[0]
    #coords_frac[coords_frac_diff > 0.5] -= 1
    #coords_frac[coords_frac_diff < -0.5] += 1
    #coords = np.array([np.dot(coords_frac[i], cells[i]) for i in range(len(coords))])

    # 4. compute averaged structure
    avg_cell = np.mean(cells, axis=0)
    avg_coords = np.mean(coords_unwrapped, axis=0)
    symbols = [atom.symbol for atom in traj[0]]
    atoms = Atoms(symbols=symbols, positions=avg_coords, cell=avg_cell, pbc=True)
    return atoms