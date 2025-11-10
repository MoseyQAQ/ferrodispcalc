import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
from matplotlib import animation


def __cal_angle(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """
    calculate the angle of the vector
    
    Parameters:
    -----------
    dx: np.ndarray
        The x component of the vector
    dy: np.ndarray
        The y component of the vector
        
    Returns:
    --------
    angle: np.ndarray
        The angle of the vector
    """
    pp = np.sqrt(dx * dx + dy * dy)
    
    # 避免除零错误 (当 dx 和 dy 均为 0 时)
    # 创建一个副本以避免修改原始pp（尽管在这里可能不是必需的，但更安全）
    pp_safe = pp.copy()
    pp_safe[pp_safe == 0] = 1e-9  # 用一个极小的数替换0
    
    # 裁剪 dx / pp_safe 的值到 [-1.0, 1.0] 范围内，防止 arccos 的数值错误
    dx_norm = np.clip(dx / pp_safe, -1.0, 1.0)
    
    angle = np.arccos(dx_norm) / np.pi * 180.0
    
    # 将 dy < 0 的角度翻转到 180-360 度范围
    index = np.where(dy < 0.0)
    angle[index] = 360.0 - angle[index]
    
    # 将原始大小为 0 的向量的角度设为 0
    angle[pp == 0] = 0.0
    
    return angle

def __plot_plane(dx: np.ndarray,
                 dy: np.ndarray,
                 plane_name: str,
                 index: int,
                 fig_size: Optional[Tuple[float, float]],
                 quiver_kwargs: dict,
                 hot: bool,
                 xlabel: str,
                 ylabel: str,
                 save_dir: Path) -> None:
    """
    Internal helper function to plot a single 2D plane.
    
    Parameters:
    -----------
    dx: np.ndarray
        The 2D array (already transposed) for the x-component of the quiver.
    dy: np.ndarray
        The 2D array (already transposed) for the y-component of the quiver.
    plane_name: str
        Name of the plane (e.g., "XY", "XZ").
    index: int
        The layer index being plotted.
    fig_size: tuple | None
        The figure size tuple, or None for default.
    quiver_kwargs: dict
        Arguments for ax.quiver.
    hot: bool
        Whether to plot the angle colormap.
    xlabel: str
        Label for the x-axis.
    ylabel: str
        Label for the y-axis.
    save_dir: Path
        The directory to save the plot.
    """
    
    print(f"Plotting {plane_name} plane, {index}th layer")

    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig, ax = plt.subplots()

    angle = __cal_angle(dx, dy)
    
    ax.quiver(dx, dy, **quiver_kwargs)
    if hot:
        # 使用 imshow 绘制角度的热图
        # origin='lower' 使 (0,0) 在左下角，与 quiver 默认对齐
        # aspect=1.0 保持像素为正方形
        sc = ax.imshow(angle, cmap='hsv', vmax=360, vmin=0, aspect=1.0, 
                       origin='lower', interpolation='none')
    
    ax.set_title(f"{plane_name} plane, {index}th layer")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    save_path = save_dir / f"{plane_name}_{index}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)  # 关闭图像以释放内存

def plane_profile(data: np.ndarray,
                save_dir: Union[str, Path] = 'plane_plots',
                relative: bool = True,
                select: Optional[Dict[str, List[int]]] = None,
                hot: bool = True,
                fig_size: Union[str, Tuple[float, float]] = 'auto') -> None:
    """
    Plot 2D cross-sections of 3D vector data.

    Parameters:
    -----------
    data: np.ndarray
        The 4D vector data in (nx, ny, nz, 3) format.
    save_dir: str | Path
        The directory to save the plots.
    relative: bool
        Whether to plot the vector in the relative scale (default True).
    select: dict, optional
        The selected layers to plot. Keys should be 'x', 'y', 'z' 
        (for YZ, XZ, XY planes respectively).
        e.g., {'z': [0, 1], 'y': [0]}
        If None, all layers are plotted.
    hot: bool
        Whether to plot the angle in the hot colormap (default True).
    fig_size: str | tuple
        The figure size: 'auto', 'default', or a (width, height) tuple.
    """
    
    # 1. 执行输入检查
    if data.ndim != 4 or data.shape[3] != 3:
        raise ValueError(f"Your data shape: {data.shape}. Expected shape: (nx, ny, nz, 3).")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    quiver_kwargs = {}
    if relative == False:
        quiver_kwargs['scale'] = 1
        quiver_kwargs['scale_units'] = 'xy'

    size = data.shape[:3]  # (nx, ny, nz)

    # 2. 解析 fig_size
    fig_sizes: Dict[str, Optional[Tuple[float, float]]] = {}
    if fig_size == 'auto':
        fig_sizes['XY'] = (1.0 * size[0], 1.0 * size[1])  # (nx, ny)
        fig_sizes['XZ'] = (1.0 * size[0], 1.0 * size[2])  # (nx, nz)
        fig_sizes['YZ'] = (1.0 * size[1], 1.0 * size[2])  # (ny, nz)
    elif fig_size == 'default':
        fig_sizes['XY'] = None
        fig_sizes['XZ'] = None
        fig_sizes['YZ'] = None
    elif isinstance(fig_size, tuple):
        fig_sizes['XY'] = fig_size
        fig_sizes['XZ'] = fig_size
        fig_sizes['YZ'] = fig_size
    else:
        raise ValueError(f"Invalid figure size: {fig_size}")

    # 3. 解析 select 字典
    if select is None:
        select = {'x': list(range(size[0])),
                  'y': list(range(size[1])),
                  'z': list(range(size[2]))}
    
    # 'z' 键用于选择 XY 平面 (沿 z 轴切片)
    # 'y' 键用于选择 XZ 平面 (沿 y 轴切片)
    # 'x' 键用于选择 YZ 平面 (沿 x 轴切片)
    indices_z = select.get('z', [])
    indices_y = select.get('y', [])
    indices_x = select.get('x', [])

    # --- 4. 循环绘图 ---

    # 绘制 XY 平面 (沿 Z 轴切片)
    for z_index in indices_z:
        if not (0 <= z_index < size[2]):
            print(f"Warning: Skipping invalid z-index {z_index} for XY plot.")
            continue
        
        # 原始数据是 (nx, ny)，绘图需要 (ny, nx)
        dx = data[:, :, z_index, 0].T 
        dy = data[:, :, z_index, 1].T
        
        __plot_plane(dx, dy, 'XY', z_index, fig_sizes['XY'], 
                     quiver_kwargs, hot, 
                     xlabel="[100] (X-axis)", 
                     ylabel="[010] (Y-axis)", 
                     save_dir=save_dir)

    # 绘制 XZ 平面 (沿 Y 轴切片)
    for y_index in indices_y:
        if not (0 <= y_index < size[1]):
            print(f"Warning: Skipping invalid y-index {y_index} for XZ plot.")
            continue

        # 原始数据是 (nx, nz)，绘图需要 (nz, nx)
        dx = data[:, y_index, :, 0].T
        dz = data[:, y_index, :, 2].T

        __plot_plane(dx, dz, 'XZ', y_index, fig_sizes['XZ'],
                     quiver_kwargs, hot,
                     xlabel="[100] (X-axis)",
                     ylabel="[001] (Z-axis)",
                     save_dir=save_dir)

    # 绘制 YZ 平面 (沿 X 轴切片)
    for x_index in indices_x:
        if not (0 <= x_index < size[0]):
            print(f"Warning: Skipping invalid x-index {x_index} for YZ plot.")
            continue
            
        # 原始数据是 (ny, nz)，绘图需要 (nz, ny)
        dy = data[x_index, :, :, 1].T
        dz = data[x_index, :, :, 2].T
        
        __plot_plane(dy, dz, 'YZ', x_index, fig_sizes['YZ'],
                     quiver_kwargs, hot,
                     xlabel="[010] (Y-axis)",
                     ylabel="[001] (Z-axis)",
                     save_dir=save_dir)
        
def plane_profile_multi(data: np.ndarray,
                          select: List[Union[str, int, Union[slice, List[int]]]],
                          save_name: Union[str, Path] = 'plane_plots.gif',
                          relative: bool = True,
                          hot: bool = True,
                          fig_size: Union[str, Tuple[float, float]] = 'auto',
                          fps: int = 10):
    """
    Plot 2D cross-sections of 5D vector data (nframe, nx, ny, nz, 3)
    and save as a GIF using matplotlib.animation.
    (使用 matplotlib.animation 绘制 5D 矢量数据的 2D 横截面并另存为 GIF)

    Parameters:
    -----------
    data: np.ndarray
        The 5D vector data in (nframe, nx, ny, nz, 3) format.
    select: list
        Defines the plot. Format: [plane_name, layer_index, frame_spec]
        e.g., ['xz', 0, slice(0, None, 2)] -> XZ plane, y_index=0, every 2nd frame.
    save_name: str | Path
        The output filename (e.g., 'animation.gif').
    relative: bool
        Whether to plot the vector in the relative scale.
    hot: bool
        Whether to plot the angle in the hot colormap.
    fig_size: str | tuple
        The figure size: 'auto', 'default', or a (width, height) tuple.
    fps: int
        Frames per second for the output GIF.
    """
    
    # 1. 解析 select 输入
    plane_name = select[0].lower() # e.g., 'xz'
    layer_index = select[1]      # e.g., 0
    frame_spec = select[2]       # e.g., slice(0, None, 1)

    # 2. 获取帧索引
    nframe = data.shape[0]
    if isinstance(frame_spec, slice):
        frame_indices = list(range(nframe))[frame_spec]
    else:
        frame_indices = frame_spec
        
    if not frame_indices:
        print("No frames selected to animate.")
        return

    # 3. 准备 quiver (矢量) 参数
    quiver_kwargs = {}
    if not relative:
        quiver_kwargs['scale'] = 1
        quiver_kwargs['scale_units'] = 'xy'
            
    # 4. 准备 fig_size (图像大小)
    size = data.shape[1:4] # (nx, ny, nz)
    fig_size_tuple: Optional[Tuple[float, float]] = None
    
    if fig_size == 'auto':
        if plane_name == 'xy':
            fig_size_tuple = (1.0 * size[0], 1.0 * size[1]) # (nx, ny)
        elif plane_name == 'xz':
            fig_size_tuple = (1.0 * size[0], 1.0 * size[2]) # (nx, nz)
        elif plane_name == 'yz':
            fig_size_tuple = (1.0 * size[1], 1.0 * size[2]) # (ny, nz)
    elif fig_size == 'default':
        fig_size_tuple = None
    elif isinstance(fig_size, tuple):
        fig_size_tuple = fig_size

    # 5. 定义轴标签
    labels = {
        'xy': ("[100] (X-axis)", "[010] (Y-axis)"),
        'xz': ("[100] (X-axis)", "[001] (Z-axis)"),
        'yz': ("[010] (Y-axis)", "[001] (Z-axis)")
    }
    xlabel, ylabel = labels[plane_name]

    # --- 6. 设置 Animation ---

    # 6.1. 创建 Figure 和 Artists (仅一次)
    if fig_size_tuple is not None:
        fig, ax = plt.subplots(figsize=fig_size_tuple)
    else:
        fig, ax = plt.subplots()

    # 6.2. 绘制第一帧 (frame_indices[0]) 作为初始状态
    first_frame_index = frame_indices[0]
    if plane_name == 'xy':
        dx = data[first_frame_index, :, :, layer_index, 0].T 
        dy = data[first_frame_index, :, :, layer_index, 1].T
    elif plane_name == 'xz':
        dx = data[first_frame_index, :, layer_index, :, 0].T
        dy = data[first_frame_index, :, layer_index, :, 2].T
    elif plane_name == 'yz':
        dx = data[first_frame_index, layer_index, :, :, 1].T
        dy = data[first_frame_index, layer_index, :, :, 2].T
    
    angle = __cal_angle(dx, dy)
    
    # 存储对 artists 的引用
    qv = ax.quiver(dx, dy, **quiver_kwargs)
    sc = None
    if hot:
        sc = ax.imshow(angle, cmap='hsv', vmax=360, vmin=0, aspect=1.0, 
                      origin='lower', interpolation='none')
    
    title_obj = ax.set_title(f"{plane_name.upper()} plane, layer {layer_index}, frame {first_frame_index}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 6.3. 定义 update 函数
    def update(frame_index: int):
        """
        Animation update function.
        (动画更新功能)
        """
        # (切片逻辑)
        if plane_name == 'xy':
            dx = data[frame_index, :, :, layer_index, 0].T 
            dy = data[frame_index, :, :, layer_index, 1].T
        elif plane_name == 'xz':
            dx = data[frame_index, :, layer_index, :, 0].T
            dy = data[frame_index, :, layer_index, :, 2].T
        elif plane_name == 'yz':
            dx = data[frame_index, layer_index, :, :, 1].T
            dy = data[frame_index, layer_index, :, :, 2].T

        # 更新 Artists 数据
        qv.set_UVC(dx, dy)
        
        artists_to_return = [qv]
        
        if hot and sc is not None:
            angle = __cal_angle(dx, dy)
            sc.set_data(angle)
            artists_to_return.append(sc)
        
        title_obj.set_text(f"{plane_name.upper()} plane, layer {layer_index}, frame {frame_index}")
        artists_to_return.append(title_obj)
        return tuple(artists_to_return)

    # 6.4. 创建并保存动画
    print(f"Creating animation ({len(frame_indices)} frames)...")
    
    # interval = 1000ms / fps
    interval = 1000.0 / fps
    
    # blit=True 极大提高性能，它只重绘已更改的 artists
    anim = animation.FuncAnimation(fig, update, frames=frame_indices, 
                                   interval=interval, blit=True)
    
    # 7. 保存动画为 GIF
    try:
        anim.save(save_name, writer='pillow', fps=fps)
        print(f"Successfully saved animation to {save_name}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Make sure 'pillow' (PIL) library is installed: pip install pillow")
    
    # 清理
    plt.close(fig)