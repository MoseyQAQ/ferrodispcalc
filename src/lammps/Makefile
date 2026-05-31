LMP_SOURCE_DIR=/opt/apps/lammps/lammps-29Aug2024_update4/src/
CXX=mpicxx
CXXFLAGS=-I${LMP_SOURCE_DIR} -Wall -Wextra -O3 -fPIC -I${LMP_SOURCE_DIR}/OPENMP -fopenmp
LD=$(CXX) -shared -rdynamic -fopenmp
DSOEXT=.so

include Makefile.common
