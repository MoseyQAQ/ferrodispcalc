#!/bin/bash

# build the neighbor list
python3 -B build_neighbor.py

# convert the dump file into xsf
$1/get_a traj.lammpstrj test.xsf type_map_file 1 > /dev/null

# test get_d (dump)
$1/get_d traj.lammpstrj test_B.disp B.dat 1 > /dev/null
$1/get_d traj.lammpstrj test_A.disp A.dat 1 > /dev/null
cat test_A.disp test_B.disp > test.disp
python3 -B ../compare.py ref.disp test.disp disp

# test get_d (xsf)
$1/get_d test.xsf test_B_xsf.disp B.dat > /dev/null
$1/get_d test.xsf test_A_xsf.disp A.dat > /dev/null
cat test_A_xsf.disp test_B_xsf.disp > test_xsf.disp
python3 -B ../compare.py ref.disp test_xsf.disp disp

# test get_p (dump)
$1/get_p traj.lammpstrj test.polar BA.dat B.dat type_map_file bec_file 1 > /dev/null
python3 -B ../compare.py ref.polar test.polar polar

# test get_p (xsf)
$1/get_p test.xsf test_xsf.polar BA.dat B.dat type_map_file bec_file > /dev/null
python3 -B ../compare.py ref.polar test_xsf.polar polar

# clean up
rm test*.disp test*.xsf *.dat test*.polar