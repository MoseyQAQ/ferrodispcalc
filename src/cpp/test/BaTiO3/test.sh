#!/bin/bash

# Test get_a
$1/get_a traj.lammpstrj test_1.xsf type_map_file 10 > /dev/null
$1/get_a traj.lammpstrj test_2.xsf type_map_file 1.0 > /dev/null

# compare the output with the reference
python3 -B ../compare.py ref.xsf test_1.xsf stru
python3 -B ../compare.py ref.xsf test_2.xsf stru

# clean up
rm -f test_1.xsf test_2.xsf