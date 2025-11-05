import matplotlib.pyplot as plt
import numpy as np

def line_profile(data: np.ndarray,
                 ax: plt.Axes = None,
                 along: str = 'x',
                 savepath: str = None,
                 field_prefix: str = None,
                 call_back: callable = None):

    # 1. perfrom some checks on the input
    if data.ndim != 4 or data.shape[3] != 3:
        raise ValueError(f"Your data shape: {data.shape}. Expected shape: (nx, ny, nz, 3).")
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if along not in ['x', 'y', 'z']:
        raise ValueError(f"Invalid 'along' parameter: {along}. Expected 'x', 'y', or 'z'.")
    
    if field_prefix is None:
        field_prefix = 'v'

    # 2. compute the mean profile along the specified axis
    mean_data = None
    x_label = ''
    if along == 'x':    
        mean_data = np.mean(data, axis=(1,2))  # Average over y and z
        x_label = 'X-axis'
    elif along == 'y':
        mean_data = np.mean(data, axis=(0,2))  # Average over x and z
        x_label = 'Y-axis'
    elif along == 'z':
        mean_data = np.mean(data, axis=(0,1))  # Average over x and y
        x_label = 'Z-axis'

    # 3. plot the mean profiles
    ax.plot(mean_data[:,0], label=f'${field_prefix}_x$', marker='o')
    ax.plot(mean_data[:,1], label=f'${field_prefix}_y$', marker='o')
    ax.plot(mean_data[:,2], label=f'${field_prefix}_z$', marker='o')
    ax.set_xlabel(x_label)
    ax.set_ylabel(f'${field_prefix}$')
    ax.legend()

    # 4. execute the callback function if provided
    if call_back is not None:
        call_back(ax)
    
    # 5. save the figure if savepath is provided
    if savepath is not None:
        plt.savefig(savepath, dpi=600) # a hardcoded dpi value, consider remove later