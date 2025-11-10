LMP_SOURCE_DIR=/shared_storage/home/lidenan/lmp_plugin_test/lammps-stable_2Aug2023_update3/src
CXX=mpicxx
CXXFLAGS=-I${LMP_SOURCE_DIR} -Wall -Wextra -O3 -fPIC -I${LMP_SOURCE_DIR}/OPENMP -fopenmp
LD=$(CXX) -shared -rdynamic -fopenmp
DSOEXT=.so

include Makefile.common
