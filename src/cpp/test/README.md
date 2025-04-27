# How to contribute test cases

1. make a new folder: 
```Bash
mkdir YOUR_FOLDER_NAME 
```
2. prepare the necessary input files, the following as an example in <kbd>PIN-PMN-PT</kbd> folder:
```Bash
PIN-PMN-PT
├── bec_file             # input for get_polarization
├── build_neighbor.py    # for build neighbor list
├── ref.disp             # the reference displacement
├── ref.polar            # the reference polarization
├── test.sh              # specify how to run the test
├── traj.lammpstrj       # the input dump file 
└── type_map_file        # input for get_polarization
```
3. prepare the <kbd>test.sh</kbd>, here is an example:
```Bash
##################### Test for get_d #####################
# build_neighbor
python3 -B build_neighbor.py   # use -B to avoid the generation of "__pycache__" folder

# test get_d
$1/get_d traj.lammpstrj B.dat test_B.disp  0 1 1 > /dev/null   # $1 is the path of the compiled codes
$1/get_d traj.lammpstrj A.dat test_A.disp  0 1 1 > /dev/null
cat test_A.disp test_B.disp > test.disp

# compare the output with the reference
python3 -B ../compare.py ref.disp test.disp disp   # call compare.py to compare the two files

# clean up
rm -f test.disp test_A.disp test_B.disp A.dat      # clean the generated files
##################### End Test get_d #####################
```