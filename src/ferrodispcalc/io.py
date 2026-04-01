from ase import Atoms 
import dpdata
from typing import List, Union, Dict
import numpy as np
import pathlib
try:
    import ferrodispcalc._cpp_bindings as _cpp
except ImportError:
    print("Warning: Could not import ferrodispcalc C++ bindings. Some IO functions may not work.")
    
def read_xyz(filename: str, select_frames: list = None, cache: bool = True) -> list[Atoms]:
    '''
    Read an XYZ trajectory file and return a list of ASE Atoms objects.

    Parameters:
    filename (str): Path to the XYZ file.
    select_frames (list, optional): List of frame indices to read. If None, read all frames. Defaults to None.
    cache (bool, optional): Whether to cache the read frames in memory. Defaults to True.
    '''
    pass

def _convert_frame_to_atoms(cpp_frame: _cpp.Frame, type_map: List[str]) -> Atoms:
    """
    (私有) 助手函数：将 C++ Frame 对象转换为 ase.Atoms 对象。
    
    C++ Frame 已经通过 pybind11 将其数据暴露为 NumPy 数组 (零拷贝)。
    """

    symbols = [type_map[t - 1] for t in cpp_frame.types]
    atoms = Atoms(
        symbols=symbols,
        positions=cpp_frame.positions,
        cell=cpp_frame.cell,
        pbc=True  # LAMMPS dump 几乎总是有 PBC
    )
    
    # 附加 timestep
    atoms.info['timestep'] = cpp_frame.timestep
    
    # 附加所有其他 per-atom 属性 (例如 'fx', 'fy', 'fz', 'q')
    # cpp_frame.arrays 是一个 dict[str, np.ndarray(N,)]
    for key, array in cpp_frame.arrays.items():
        atoms.set_array(key, array)
        
    return atoms


def read_lammps_dump(
    filename: str,
    type_map: List[str],
    select: Union[int, slice, List[int], None] = None,
    cache: bool = True
) -> Union[Atoms, List[Atoms], None]:
    """Read a LAMMPS dump trajectory file using the high-performance C++ backend.

    The reader memory-maps the file and optionally caches the frame index to a
    sidecar file (``.{filename}.idx``) for fast subsequent reads.

    .. note::
       Atom coordinates are read as stored in the dump file and are **not**
       shifted back to the primary unit cell.  A constant offset may therefore
       exist compared to tools such as dpdata, but this does not affect any
       relative-displacement calculations.

    Parameters
    ----------
    filename : str
        Path to the LAMMPS dump file.
    type_map : list[str]
        Ordered list of element symbols corresponding to LAMMPS atom types,
        e.g. ``['Pb', 'Ti', 'O']``.
    select : int | slice | list[int] | None, optional
        Frames to read.

        * ``None`` – read all frames (default).
        * ``int`` – read a single frame by index (negative indices supported);
          returns a single :class:`~ase.Atoms` object instead of a list.
        * ``slice`` – read a contiguous or strided range of frames.
        * ``list[int]`` – read an arbitrary set of frame indices.
    cache : bool, optional
        If ``True`` (default), write/read a sidecar index file so that
        re-reading the same dump file is faster.

    Returns
    -------
    Atoms | list[Atoms] | None
        A single :class:`~ase.Atoms` when *select* is an integer, otherwise a
        list of :class:`~ase.Atoms`.  Returns ``None`` (integer select) or
        ``[]`` (other select) if the file is empty.

    Raises
    ------
    FileNotFoundError
        If *filename* does not exist.
    IOError
        If the C++ backend fails to open or memory-map the file.
    IndexError
        If an integer *select* is out of range.
    """
    
    # --- 1. (1a) (1c) 缓存管理 ---
    
    # 确定原始文件和缓存文件的路径
    try:
        f_path = pathlib.Path(filename).resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"LAMMPS dump file not found: {filename}")

    filename_str = str(f_path)
    
    # 缓存文件名: ".dump.traj.idx" (Linux 隐藏)
    # (根据你的要求 '." + filename'，我们使用 '.{name}.idx' 格式)
    cache_path = f_path.parent / f".{f_path.name}.idx"
    cache_filename_str = str(cache_path)

    # --- 2. (1b) 实例化 C++ Reader ---
    # 构造函数执行 mmap。如果失败，会抛出异常。
    try:
        reader = _cpp.LAMMPSReader(filename_str)
    except RuntimeError as e:
        raise IOError(f"C++ 内核 mmap 文件失败: {filename_str}. Error: {e}")

    use_cache = cache
    is_cache_valid = False

    if use_cache:
        if cache_path.exists():
            try:
                # 检查缓存是否过期
                orig_mtime = f_path.stat().st_mtime
                cache_mtime = cache_path.stat().st_mtime
                if cache_mtime >= orig_mtime:
                    is_cache_valid = True
                else:
                    # 缓存已过期 (原始文件已被修改)
                    pass 
            except OSError:
                # 无法获取 stat，视为无效
                pass

    # --- 3. 调用 C++ 索引/加载 ---
    try:
        if is_cache_valid:
            # (1d) 缓存有效：调用 C++ load_index
            reader.load_index(cache_filename_str)
        else:
            # (1c) 缓存无效或被禁用：调用 C++ index
            reader.index()
            if use_cache:
                # (新增) 保存新索引到文件
                reader.save_index(cache_filename_str)

    except RuntimeError as e:
        # 处理 C++ 端的索引/加载/保存错误
        raise RuntimeError(f"C++ 内核索引/缓存操作失败: {e}")

    # --- 4. (1a) 解析 'select' 参数 ---
    total_frames = reader.n_frames # 使用绑定的 .n_frames 属性
    
    if total_frames == 0:
        return [] if not isinstance(select, int) else None

    if select is None:
        # 读取所有
        indices = list(range(total_frames))
        return_single = False
    elif isinstance(select, int):
        # 归一化负索引
        norm_idx = select if select >= 0 else select + total_frames
        if 0 <= norm_idx < total_frames:
            indices = [norm_idx]
            return_single = True
        else:
            raise IndexError(f"Frame index {select} out of range for file with {total_frames} frames.")
    elif isinstance(select, slice):
        indices = list(range(*select.indices(total_frames)))
        return_single = False
    elif isinstance(select, list):
        # 归一化负索引
        indices = [i % total_frames for i in select]
        return_single = False
    else:
        raise TypeError(f"不支持的 'select' 类型: {type(select)}")

    if not indices:
        return [] if not return_single else None

    # --- 5. (1d) 调用 C++ 核心读取 ---
    try:
        # C++ 端返回一个 _cpp.Frame 列表
        raw_frames = reader.read_frames(indices)
    except (RuntimeError, IndexError, std.out_of_range) as e:
        raise IOError(f"C++ 内核读取帧失败: {e}")

    # --- 6. (1e) 将 C++ 结构体包装为 Python Atoms 对象 ---
    atoms_list = [_convert_frame_to_atoms(frame, type_map) for frame in raw_frames]

    # --- 7. (1a) 根据 'select' 类型返回结果 ---
    return atoms_list[0] if return_single else atoms_list

def read_lammps_data(filename: str, type_map: list[str]) -> Atoms:
    """Read a LAMMPS data file and return the structure as an ASE Atoms object.

    Parameters
    ----------
    filename : str
        Path to the LAMMPS data file.
    type_map : list[str]
        Ordered list of element symbols corresponding to LAMMPS atom types,
        e.g. ``['Pb', 'Ti', 'O']``.

    Returns
    -------
    Atoms
        The structure described in the data file.
    """
    sys = dpdata.System(filename, fmt='lmp', type_map=type_map)
    atoms =  sys.to_ase_structure()[0]
    return atoms

def read_lammps_log(file_name: str) -> Dict:
    """Parse thermodynamic data from the last run in a LAMMPS log file.

    The function locates the final ``Per MPI rank memory`` header and reads
    all thermo lines up to the matching ``Loop time of`` footer.

    Parameters
    ----------
    file_name : str
        Path to the LAMMPS log file.

    Returns
    -------
    dict
        Dictionary mapping each thermo keyword (e.g. ``'Step'``, ``'Temp'``,
        ``'Press'``) to a NumPy array of values.  An extra key ``'nframes'``
        holds the number of thermo records read.

    Raises
    ------
    AssertionError
        If the number of header columns does not match the data columns.
    """
    with open(file_name, 'r') as f:
        line_idx = []
        end_line_idx = []
        for i, line in enumerate(f):
            if line.startswith('Per MPI rank memory'):
                line_idx.append(i+1)
            if 'Loop time of' in line:
                end_line_idx.append(i)
                break
    start_line_idx = line_idx[-1]
    end_line_idx = end_line_idx[-1]
    nframes = end_line_idx - start_line_idx - 1
    keys = np.genfromtxt(file_name, skip_header=start_line_idx, max_rows=1, dtype=str)
    data=np.genfromtxt(file_name, skip_header=start_line_idx+1, max_rows=nframes)
    
    assert len(keys) == data.shape[1], "Keys and data columns do not match."

    data = {key: data[:, i] for i, key in enumerate(keys)}
    data['nframes'] = nframes
    print(f"Read {nframes} frames from {file_name}")
    print(f"Found Keys: {keys}")

    return data