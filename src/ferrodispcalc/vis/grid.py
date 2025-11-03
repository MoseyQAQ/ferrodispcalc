import numpy as np
from ase.geometry import get_layers
from ase import Atoms

def grid_data(atoms: Atoms,
              data: np.ndarray,
              element: list[str],
              tol: float = 1.0,
              axis: tuple[tuple, tuple, tuple] = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
              target_size: tuple[int, int, int] | None = None):

    # 1. get the atoms of interest
    element_index = [idx for idx, i in enumerate(atoms) if i.symbol in element]
    clean_atoms = atoms[element_index]

    # 2. get the layer tags and size
    def __get_layers(atoms, axis, tol):
        tag_x, _ = get_layers(clean_atoms, axis[0], tolerance=tol)
        tag_y, _ = get_layers(clean_atoms, axis[1], tolerance=tol)
        tag_z, _ = get_layers(clean_atoms, axis[2], tolerance=tol)
        tag = np.concatenate((tag_x[:, np.newaxis], tag_y[:, np.newaxis], tag_z[:, np.newaxis]), axis=1)
        size = [len(set(tag_x)), len(set(tag_y)), len(set(tag_z))]
        return tag, size

    tag, size = __get_layers(clean_atoms, axis, tol)

    # 3. check target size; if not matched, try to avoid boundary issues
    if target_size is not None and not np.allclose(size, target_size):
        clean_atoms.positions += 1 # avoid boundary issues
        tag, size = __get_layers(clean_atoms, axis, tol)
        
        if not np.allclose(size, target_size):
            raise ValueError(f"Target size {target_size} not matched with actual size {size}.")
    
    # 4. grid the data
    # datashape: [n_atoms, n_features]
    # or [nframe, n_atoms, n_features]
    # namely, we accept the multi-frame data
    if data.ndim == 2:
        results = np.full((size[0], size[1], size[2], data.shape[1]), np.nan)
        for i in range(len(data)):
            results[tag[i,0], tag[i,1], tag[i,2], :] = data[i, :]
    elif data.ndim == 3:
        results = np.full((data.shape[0], size[0], size[1], size[2], data.shape[2]), np.nan)
        for frame in range(data.shape[0]):
            for i in range(len(data[frame])):
                results[frame, tag[i,0], tag[i,1], tag[i,2], :] = data[frame, i, :]
    else:
        raise ValueError(f"We only accept data with shape [n_atoms, n_features] or [nframe, n_atoms, n_features]. Your data shape is {data.shape}.")
    
    return results