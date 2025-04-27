#!/usr/bin/python
# This script calculates the average structure for a lammps dump file
# Usage: python x.py input.dump output.xsf

import sys
import subprocess
import numpy as np
from numpy import linalg as la

# Input and output
inputfile = sys.argv[1]
outputfile = sys.argv[2]

# Set atom type
atoms_type = ["Ba", "Ti", "O"]

# Define readcell function
def readcell(f):
    line = f.readline().split()
    line = [ float(x) for x in line ]
    xlo_bound = line[0]
    xhi_bound = line[1]
    xy = line[2]
    line = f.readline().split()
    line = [ float(x) for x in line ]
    ylo_bound = line[0]
    yhi_bound = line[1]
    xz = line[2]
    line = f.readline().split()
    line = [ float(x) for x in line ]
    zlo_bound = line[0]
    zhi_bound = line[1]
    yz = line[2]
    xlo = xlo_bound - min(0.0, xy, xz, xy+xz)
    xhi = xhi_bound - max(0.0, xy, xz, xy+xz)
    ylo = ylo_bound - min(0.0, yz)
    yhi = yhi_bound - max(0.0, yz)
    zlo = zlo_bound
    zhi = zhi_bound
    xx = xhi - xlo
    yy = yhi - ylo
    zz = zhi - zlo
    cell = np.zeros((3,3))
    cell[0,:] = [xx, 0, 0]
    cell[1,:] = [xy, yy, 0]
    cell[2,:] = [xz, yz, zz]
    return cell

# Define readatoms function
def readatoms(f, natoms):
    type_index = np.zeros(natoms)
    coord = np.zeros((natoms,3))
    for i in range(natoms):
        line = f.readline().split()
        if float(line[1])==1:
            type_index[i]=1
        elif float(line[1]) == 9:
            type_index[i]=2
        elif float(line[1]) == 15:
            type_index[i]=3
        tmp = [ float(x) for x in line[2:] ]
        coord[i,0] = tmp[0]
        coord[i,1] = tmp[1]
        coord[i,2] = tmp[2]
    return type_index, coord

# Read number of structures and atoms
nstructures = subprocess.check_output("grep TIMESTEP %s | wc -l "%inputfile, shell=True)
nstructures = int(nstructures.decode().strip("\n"))
natoms = subprocess.check_output("grep -A1 \"NUMBER OF ATOMS\" %s | tail -1 "%inputfile, shell=True)
natoms = int(natoms.decode().strip("\n"))
print ("number of structures:", nstructures)
print ("number of atoms per structure:", natoms)

# Take average of the structures
infile = open(inputfile,'r')
cell_avg = np.zeros((3,3))
cell_tmp = np.zeros((3,3))
type_index = np.zeros(natoms)
coord_avg = np.zeros((natoms,3))
coord_tmp = np.zeros((natoms,3))
coord_ini = np.zeros((natoms,3))

a = []
b = []
c = []
for i in range(nstructures):
    for j in range(5):
        infile.readline()
    cell_tmp = readcell(infile)
    a.append(cell_tmp[0,0]/10.0)
    b.append(cell_tmp[1,1]/10.0)
    c.append(cell_tmp[2,2]/10.0)
    cell_tmp_inv = la.inv(cell_tmp)
    cell_avg += cell_tmp
    infile.readline()
    if i == 0:
        type_index, coord_tmp = readatoms(infile, natoms)
        coord_ini = coord_tmp
        coord_avg += coord_tmp
    else:
        type_index, coord_tmp = readatoms(infile, natoms)
        coord_diff = coord_tmp - coord_ini
        coord_tmp_frac = np.dot(coord_tmp, cell_tmp_inv)  # fractional coordinates
        coord_diff_frac = np.dot(coord_diff, cell_tmp_inv)
        for k in range(3):
            index1 = np.where(coord_diff_frac[:,k] > 0.5)  # periodic boundary condition
            coord_tmp_frac[index1,k] -= 1.0
            index2 = np.where(coord_diff_frac[:,k] < -0.5)
            coord_tmp_frac[index2,k] += 1.0
        coord_tmp = np.dot(coord_tmp_frac, cell_tmp)
        coord_avg += coord_tmp
cell_avg = cell_avg/nstructures
coord_avg = coord_avg/nstructures
infile.close()

# Print the average structure
outfile = open(outputfile,'w')
print ("CRYSTAL", file=outfile)
print ("PRIMVEC", file=outfile)
for i in range(3):
    print ("%25.16f %22.16f %22.16f" %(cell_avg[i,0], cell_avg[i,1], cell_avg[i,2]), file=outfile)
print ("PRIMCOORD", file=outfile)
print (natoms, file=outfile)
for i in range(natoms):
    print ("%-2s %22.16f %22.16f %22.16f" %(atoms_type[int(type_index[i])-1], coord_avg[i,0], coord_avg[i,1], coord_avg[i,2]), file=outfile)
outfile.close()

