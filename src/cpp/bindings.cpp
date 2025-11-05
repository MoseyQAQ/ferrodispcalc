#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // 用于 std::vector, std::map 的自动转换
#include <pybind11/numpy.h>     // 用于 NumPy 数组
#include <cstddef>
#include <cstring>
#include "reader.hpp"           // 包含我们的 LAMMPSReader 和 Frame 声明

namespace py = pybind11;

#ifdef _WIN32
// Windows doesn't have POSIX ssize_t; use ptrdiff_t which is a signed
// integer type able to represent pointer differences. This mirrors common
// definitions used when building on Windows.
using ssize_t = std::ptrdiff_t;
#endif

// 帮助函数：将 std::vector<double> (扁平) 转换为 (N, 3) NumPy 数组 (零拷贝，只读)
// 注意：这只在 Frame 对象的生命周期内有效
py::array_t<double> make_positions_array(const Frame& f) {
    return py::array_t<double>(
        {(ssize_t)f.n_atoms, (ssize_t)3},     // 形状 (N, 3)
        {sizeof(double) * 3, sizeof(double)}, // 步长 (strides)
        f.positions.data(),                   // 数据指针
        py::cast(f)                           // 基础对象 (用于管理生命周期)
    );
}

// 帮助函数：将 std::vector<int> 转换为 (N,) NumPy 数组 (零拷贝，只读)
py::array_t<int> make_types_array(const Frame& f) {
    return py::array_t<int>(
        {(ssize_t)f.n_atoms}, // 形状 (N,)
        f.types.data(),       // 数据指针
        py::cast(f)           // 基础对象
    );
}

// 帮助函数：将 std::map<string, vector<double>> 转换为 dict[str, np.ndarray(N,)]
py::dict make_arrays_dict(const Frame& f) {
    py::dict d;
    for (const auto& pair : f.arrays) {
        // 创建一个 NumPy 数组 (零拷贝)
        py::array_t<double> arr(
            {(ssize_t)f.n_atoms}, // 形状 (N,)
            pair.second.data(),   // 数据指针
            py::cast(f)           // 基础对象
        );
        d[py::str(pair.first)] = arr;
    }
    return d;
}


PYBIND11_MODULE(_cpp_bindings, m) {
    m.doc() = "C++11 bindings for ferrodispcalc high-performance mmap IO";

    // -----------------------------------------------------------------------
    // 绑定 Frame 结构 (你重命名的 FrameData)
    // -----------------------------------------------------------------------
    // 我们将 C++ struct 绑定为 Python class, 
    // 并使用 .def_property 将 C++ 成员作为 Python 属性暴露。
    py::class_<Frame>(m, "Frame")
        .def(py::init<>()) // 绑定默认构造函数
        .def_readonly("n_atoms", &Frame::n_atoms,
            "Number of atoms in this frame.")
        .def_readonly("timestep", &Frame::timestep,
            "Timestep of this frame.")
        
        // 绑定 cell (Lattice)
        // 返回一个 (3, 3) NumPy 数组 (拷贝)
        .def_property("cell",
            [](const Frame &f) -> py::array_t<double> {
                // 创建一个 (3, 3) 数组的拷贝
                py::array_t<double> arr({3, 3});
                double* ptr = static_cast<double*>(arr.mutable_data());
                // std::array<std::array...>> 在内存中是连续的
                std::memcpy(ptr, f.cell.data(), 9 * sizeof(double));
                return arr;
            },
            nullptr, // 只读属性
            "Lattice vectors as a (3, 3) NumPy array.")
        
        // 绑定 positions
        // 返回一个 (N, 3) NumPy 数组 (零拷贝视图)
        .def_property_readonly("positions",
            &make_positions_array,
            "Atom positions as an (N, 3) NumPy array (read-only view).")

        // 绑定 types
        // 返回一个 (N,) NumPy 数组 (零拷贝视图)
        .def_property_readonly("types",
            &make_types_array,
            "Atom types as an (N,) NumPy array (read-only view).")
        
        // 绑定 arrays (其他属性)
        // 返回一个 dict[str, np.ndarray(N,)]
        .def_property_readonly("arrays",
            &make_arrays_dict,
            "Other per-atom properties as a dict of (N,) NumPy arrays (read-only views).");

    // -----------------------------------------------------------------------
    // 绑定 LAMMPSReader 类
    // -----------------------------------------------------------------------
    py::class_<LAMMPSReader>(m, "LAMMPSReader")
        // 绑定构造函数：
        // def __init__(self, filename: str)
        .def(py::init<const std::string&>(), 
             py::arg("filename"),
             "Constructor: takes the path to the LAMMPS dump file.\n"
             "This call performs the mmap operation.")

        // 绑定方法 (全部释放 GIL)
        
        // def index(self)
        .def("index", &LAMMPSReader::index, 
             py::call_guard<py::gil_scoped_release>(),
             "Scans the entire mmap'd file to build the frame index.")

        // def save_index(self, cache_filename: str)
        .def("save_index", &LAMMPSReader::save_index,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("cache_filename"),
             "Saves the built index to a binary cache file.")
        
        // def load_index(self, cache_filename: str)
        .def("load_index", &LAMMPSReader::load_index,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("cache_filename"),
             "Loads the index from a binary cache file, skipping the scan.")
        
        // def read_frames(self, indices: List[int]) -> List[Frame]
        .def("read_frames", &LAMMPSReader::read_frames,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("indices"),
             "Reads and parses specific frames by their zero-based indices.\n"
             "This is the main data retrieval function and is parallelized.")
        
        // 绑定 Getters 作为只读属性

        // def n_atoms(self) -> int
        .def_property_readonly("n_atoms", &LAMMPSReader::get_num_atoms,
             "Get the number of atoms (read after indexing).")
        
        // def n_frames(self) -> int
        .def_property_readonly("n_frames", &LAMMPSReader::get_num_frames,
             "Get the total number of frames found in the index.");
}