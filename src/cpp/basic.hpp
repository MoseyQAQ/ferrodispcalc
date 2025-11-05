#pragma once

#include <vector>
#include <string>
#include <array>
#include <array>
#include <map>

using Lattice = std::array<std::array<double, 3>, 3>;

struct Frame
{
    Lattice cell;
    // Atom types/numbers stored as integers (matches reader.cpp which
    // parses 'type' as an integer). This matches the Python side which
    // expects `Frame.types` to be an array of atomic numbers/numbers.
    std::vector<int> types;
    std::vector<double> positions; // coord in flat array [x1,y1,z1,x2,y2,z2,...]
    int n_atoms = 0;
    int timestep = 0;

    // Other properties (per-atom double arrays)
    std::map<std::string, std::vector<double>> arrays;
};
