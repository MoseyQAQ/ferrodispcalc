LMP_SOURCE_DIR ?= /opt/apps/lammps/lammps-4Jul2026/src/
CXX=mpicxx
# Match the C++ standard used by the LAMMPS binary (including its format ABI).
CXX_STANDARD ?= c++20
CXXFLAGS=-std=$(CXX_STANDARD) -I${LMP_SOURCE_DIR} -Wall -Wextra -O3 -fPIC -I${LMP_SOURCE_DIR}/OPENMP -fopenmp
LD=$(CXX) -shared -rdynamic -fopenmp
DSOEXT=.so

include Makefile.common
