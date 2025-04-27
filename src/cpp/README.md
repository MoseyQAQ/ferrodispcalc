# FerroDispCalc
FerroDispCalc (FDC) is a Calculator of polarization displacements and polarization of ferroelectrics in molecular dynamics simulations. 

This code is designed for: 
* build neighbor list for ferrorelectrics.
* calculate the polarization, displacement, and local lattice from a LAMMPS dump file. 

For better performence, the main part of code is written in C++.

## Code Structure

```
.
├── example  //example of how to use code, in todo list
├── ferrodispcalc // python library
│   ├── build_neighbor_list.py  //Build neighbor list for ABO3 perovskite
│   ├── PolarPolter.py
│   └── type_map.py  //store the common type map
├── install.sh //installation script
├── LICENSE
├── README.md
├── src  //c++ source code
│   ├── basic.hpp //This file contains basic functions to read lammps dump files.
│   ├── get_averaged_structure.cpp //Calculate the average structure from LAMMPS dump file
│   ├── get_local_lattice.cpp // cauculate the local lattice vector for ABO3 perovskite
│   ├── get_polarization.cpp //Calculate the polarization of each perovskite unit call using LAMMPS dump file
│   └── get_polarization_displacement.cpp //Calculate the polarization displacement from LAMMPS dump file
└── test //files related to testing
    ├── BaTiO3
    ├── compare.py
    ├── com_sys.py
    ├── PIN-PMN-PT
    ├── PIN-PMN-PT_multi_domain
    ├── PSZO
    ├── PZO
    ├── README.md
    └── run_test.sh 
```

## Installation
1. Get source code:
```Bash
git clone https://github.com/MoseyQAQ/FerroDispCalc.git
```
2. Prerequisites:
    * C++ compiler
    * Eigen (For matrix operation)
    * numpy, pymatgen (can be easily installed through: <kbd>pip3 install pymatgen</kbd>)

3. Edit the following variables in <kbd>install.sh</kbd>:
```Bash
####### general seeting #######
EIGEN=/home/liulab/eigen-3.4.0  # path to your eigen header
CXX=g++                         # your c++ compiler
CXXFLAGS="-O3" # or "-O3 -march=native" for example
INSTALL_DIR=$(pwd)
###############################
```

4. Run the <kbd>install.sh</kbd>:
```Bash
chmod +x install.sh
./install.sh make    # compile the source code
./install.sh clean   # clean the compiled code
./install.sh test    # test the executable
./install.sh install # installation
```
## Usage
1. <kbd>get_averaged_structure</kbd> / <kbd>get_a</kbd> : calculate averaged structure from a LAMMPS dump file. \
Options:
    * <kbd>input_file</kbd>: LAMMPS dump file
    * <kbd>output_file</kbd>: output file in .xsf format
    * <kbd>type_map_file</kbd>: a file containing the atom types.
    * <kbd>ratio</kbd>: If < 1, the last "ratio"% of frame will be read. If >= 1, the last "ratio" frames will be read. 

2. <kbd>get_polarization_displacement</kbd> / <kbd>get_d</kbd> : calculate displacement from: <kbd>LAMMPS dump</kbd> / <kbd>.xsf</kbd> file. \
Options:
    * <kbd>traj_file</kbd>: <kbd>LAMMPS dump</kbd> or <kbd>.xsf</kbd>  file
    * <kbd>output_file</kbd>: output file, each line contains the original coordinates and displacements of a cation.
    * <kbd>nl_file</kbd>: neighbor list file, it can be generated using <kbd>build_neighbor_list.py</kbd>
    * <kbd>ratio/last_frame</kbd>: If < 1, the last "ratio"% of frame will be read. If >= 1, the last "ratio" frames will be read. (only work when reading <kbd>LAMMPS dump</kbd>)

3. <kbd>get_polarization</kbd> / <kbd>get_p</kbd> : calculate polarization from: <kbd>LAMMPS dump</kbd> / <kbd>.xsf</kbd> file. \
Options:
    * <kbd>traj_file</kbd>: <kbd>LAMMPS dump</kbd> or <kbd>.xsf</kbd>  file
    * <kbd>output_file</kbd>: output file, each line contains the original coordinates and displacements of a cation.
    * <kbd>a_nl_file</kbd>: neighbor list file for A site cations
    * <kbd>x_nl_file</kbd>: neighbor list file for X site anions
    * <kbd>type_map_file</kbd>: file contains the type map of atom types
    * <kbd>bec_file</kbd>: file contains the Born effective charge of each atom type
    * <kbd>ratio/last_frame</kbd>: If < 1, the last "ratio"% of frame will be read. If >= 1, the last "ratio" frames will be read. (only work when reading <kbd>LAMMPS dump</kbd>)

4. <kbd>get_local_lattice</kbd> / <kbd>get_l</kbd>: calculate the local lattice vector for each perovskite unit cell in the large simulation cell. \
Options:
    * <kbd>input_file</kbd>: <kbd>.xsf</kbd>  file
    * <kbd>output_file</kbd>: the output file to store the local lattice vectors
    * <kbd>nl_file</kbd>: the neighbor list file
    * <kbd>angle_x</kbd>: the rotation angles about x axis (in degree), default is 0
    * <kbd>angle_y</kbd>: the rotation angles about y axis (in degree), default is 0
    * <kbd>angle_z</kbd>: the rotation angles about z axis (in degree), default is 0

## Example
The example is still under developed. You can find examples in "test" folder.

## Todo list
1. interface with Python
2. CL tool, and preset workflow
3. can: pip install!
4. more testing, example, readme, and doc
5. lammps plugin