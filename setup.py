# setup.py
import sys
from setuptools import setup, Extension
import pybind11

# --- C++ 扩展定义 ---
# (C++ Extension Definition)

# 定义编译器特定的标志
# (Define compiler-specific flags)
if sys.platform == 'win32':
    # MSVC 标志
    cpp_args = [
        '/std:c++17',  # 需要 C++17
        '/openmp',     # 启用 OpenMP
        '/D_WIN32'
    ]
    link_args = []
elif sys.platform == 'darwin':
    # For now, openMP is disabled in MacOS due to complexity in setup.
    cpp_args = [
        '-std=c++17'
    ]
    link_args = []
else:
    # GCC/Clang 标志
    cpp_args = [
        '-std=c++17',  # 需要 C++17
        '-fopenmp',    # 启用 OpenMP
        '-fvisibility=hidden' # 良好的实践
    ]
    link_args = ['-fopenmp'] # 链接 OpenMP

# 定义 pybind11 扩展模块
# (Define the pybind11 extension module)
ext_modules = [
    Extension(
        # 模块的 Python 完整名称
        # (The full Python name of the module)
        'ferrodispcalc._cpp_bindings', 
        
        # C++ 源文件列表
        # (List of C++ source files)
        [
            'src/cpp/bindings.cpp',
            'src/cpp/reader.cpp'
        ],
        
        # 包含目录
        # (Include directories)
        include_dirs=[
            # pybind11 的 include 路径
            pybind11.get_include(),
            
            # C++ 头文件 (reader.hpp) 的路径
            'src/cpp', 
        ],
        
        # 语言
        language='c++',
        
        # 传递给 C++ 编译器的额外参数
        # (Extra arguments passed to the C++ compiler)
        extra_compile_args=cpp_args,
        
        # 传递给链接器的额外参数
        # (Extra arguments passed to the linker)
        extra_link_args=link_args,
    ),
]

# --- Setup 函数 ---
# (Setup Function)
setup(
    # 这是构建 C++ 扩展所必需的最小配置。
    # 所有其他元数据 (如 'name', 'version', 'install_requires') 
    # 都应放在 pyproject.toml 中。
    # (This is the minimal config needed to build the C++ extension.)
    # (All other metadata (name, version, install_requires)
    #  should live in pyproject.toml.)
    ext_modules=ext_modules,
)