/*
basic.hpp: 
    This file contains basic functions to read lammps dump files.
    
Author: Denan Li
Last modified: 2024-07-14
Email: lidenan@westlake.edu.cn

structs:
    Frame: store cell and coordinates of a frame
    Traj: store all frames and atom types

functions:
    skip_lines: skip n lines in the file
    read_cell: read cell matrix from lammps dump file
    read_coords: read coordinates from lammps dump file
    get_nframes: get number of frames
    get_frame_positions: get position of each frame
    get_natoms: get number of atoms
    read_single_frame: read single frame
    read_atom_types: read atom types (integer, not string)
    read_all_frames: read all frames
    read_selected_frames: read selected frames
    read_atom_types_xsf: read atom types in xsf (integer, not string)
    read_xsf: read xsf file
    parse_neighbor_list_file: parse neighbor list file
    get_type_map: get type map
    apply_pbc: calculate the neighbor coord after applying the PBC

Todo list:
    1. support IO of .xsf format file
    2. .npy file support
*/
#ifndef BASIC_HPP
#define BASIC_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <iterator>
#include <unordered_map>

//structs Frame: store cell and coordinates of a frame
struct Frame {
    Eigen::Matrix3d cell;               // Cell matrix
    Eigen::MatrixXd coords;             // Atomic coordinates
    std::vector<int> ids;               // Atomic ordering (IDs) in LAMMPS
    Eigen::MatrixXd properties;         // Per-atom properties
};

// struct Traj: store all frames and atom types
struct Traj {
    std::vector<Frame> frames;
    std::vector<int> atom_types;
};

/* skip n lines in the file */
void skip_lines(std::ifstream &file, int n) {
    std::string line;
    for (int i = 0; i < n; ++i) {
        std::getline(file, line);
    }
}


/* read cell matrix from lammps dump file */
Eigen::Matrix3d read_cell(std::ifstream &file) {
    Eigen::Matrix3d cell;
    std::string line;
    std::vector<double> bounds;

    // Read three lines for x, y, z dimensions
    for (int i = 0; i < 3; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        std::vector<double> lineData((std::istream_iterator<double>(iss)), std::istream_iterator<double>());

        // Handle cases where only two columns (lo, hi) are provided
        if (lineData.size() == 2) {
            lineData.push_back(0.0); // Add a default tilt factor of 0.0
        }

        bounds.insert(bounds.end(), lineData.begin(), lineData.end());
    }

    // Extract bounds and tilt factors from the read data
    double xlo_bound = bounds[0], xhi_bound = bounds[1], xy = bounds[2];
    double ylo_bound = bounds[3], yhi_bound = bounds[4], xz = bounds[5];
    double zlo_bound = bounds[6], zhi_bound = bounds[7], yz = bounds[8];

    // Calculate lo and hi values adjusted for tilt (triclinic corrections)
    double xlo = xlo_bound - std::min({0.0, xy, xz, xy + xz});
    double xhi = xhi_bound - std::max({0.0, xy, xz, xy + xz});
    double ylo = ylo_bound - std::min(0.0, yz);
    double yhi = yhi_bound - std::max(0.0, yz);

    // Set the matrix values
    cell(0, 0) = xhi - xlo;
    cell(0, 1) = 0;
    cell(0, 2) = 0;
    cell(1, 0) = xy;
    cell(1, 1) = yhi - ylo;
    cell(1, 2) = 0;
    cell(2, 0) = xz;
    cell(2, 1) = yz;
    cell(2, 2) = zhi_bound - zlo_bound;

    return cell;
}

/* Parse the number of atomic properties and their names from a LAMMPS dump file */
int parse_properties(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // Skip the first 8 lines
    skip_lines(file, 8);

    // Read the line containing "ITEM: ATOMS ..."
    std::string line;
    std::getline(file, line);

    // Parse the line to extract column names
    std::istringstream iss(line);
    std::vector<std::string> tokens((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());

    // Find the position of "z"
    auto it_z = std::find(tokens.begin(), tokens.end(), "z");
    if (it_z == tokens.end()) {
        throw std::runtime_error("Error: Could not find 'z' column in the ATOMS line");
    }

    // Collect property names after "z"
    std::vector<std::string> property_names(it_z + 1, tokens.end());
    int num_properties = property_names.size();

    // Print the number of properties and their names
    std::cout << "Number of atomic properties: " << num_properties << std::endl;
    std::cout << "Atomic properties: ";
    for (const auto &name : property_names) {
        std::cout << name << " ";
    }
    std::cout << std::endl;

    return num_properties;
}

/* Read coordinates from lammps dump file */
/*
Eigen::MatrixXd read_coords(std::ifstream &file, int n_atoms) {
    Eigen::MatrixXd coords(n_atoms, 3); // 3 columns for x, y, z
    std::string line;
    for (int i = 0; i < n_atoms; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        std::vector<double> lineData((std::istream_iterator<double>(iss)), std::istream_iterator<double>());
        coords(i, 0) = lineData[2];
        coords(i, 1) = lineData[3];
        coords(i, 2) = lineData[4];
    }
    return coords;
}
*/
/* Read coordinates and properties from LAMMPS dump file */
void read_coords(std::ifstream &file, Frame &frame, int n_atoms, int num_properties) {
    // Initialize coordinates and properties matrices
    frame.coords = Eigen::MatrixXd(n_atoms, 3); // 3 columns for x, y, z
    frame.properties = Eigen::MatrixXd(n_atoms, num_properties); // num_properties columns for properties
    frame.ids.resize(n_atoms); // Resize the ids vector to store atomic IDs

    std::string line;
    for (int i = 0; i < n_atoms; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        // std::vector<double> lineData((std::istream_iterator<double>(iss)), std::istream_iterator<double>());
        
        // Tokenize the line into individual strings
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        // Convert tokens to doubles, handling errors
        std::vector<double> lineData;
        for (const auto &t : tokens) {
            try {
                double value = std::stod(t);
                lineData.push_back(value);
            } catch (const std::invalid_argument &e) {
                std::cerr << "Error: Invalid number format: " << t << " (" << e.what() << ")" << std::endl;
                lineData.push_back(0.0); // Replace invalid numbers with 0
            } catch (const std::out_of_range &e) {
                // std::cerr << "Warning: Number out of range: " << t << " (" << e.what() << ")" << std::endl;
                lineData.push_back(0.0); // Replace out-of-range numbers with 0
            }
        }

        // Check if the line has enough data
        if (lineData.size() < 5 + num_properties) {
            std::cerr << "Error: Incomplete data in line " << i + 1 << std::endl;
            std::cerr << "Raw line content: " << line << std::endl;
            std::cerr << "Line data: ";
            for (const auto &data : lineData) {
                std::cerr << data << " ";
            }
            std::cerr << std::endl;
            for (char c : line) {
                std::cerr << "[" << c << "] (" << static_cast<int>(c) << ") ";
            }
            std::cerr << std::endl;
            std::cerr << "Expected at least " << (5 + num_properties) << " values, but got " << lineData.size() << std::endl;
            exit(1);
        }

        // Read ID
        frame.ids[i] = static_cast<int>(lineData[0]);

        // Read coordinates (x, y, z)
        frame.coords(i, 0) = lineData[2];
        frame.coords(i, 1) = lineData[3];
        frame.coords(i, 2) = lineData[4];

        // Read properties (starting from index 5)
        for (int j = 0; j < num_properties; ++j) {
            frame.properties(i, j) = lineData[5 + j];
        }
    }
}

/* get number of frames */
int get_nframes(std::string filename) {
    std::ifstream file(filename);
    std::string line;
    int nframes = 0;
    while (std::getline(file, line)) {
        if (line.find("ITEM: TIMESTEP") != std::string::npos) {
            nframes++;
        }
    }
    return nframes;
}

/* get position of each frame */
std::vector<std::streampos> get_frame_positions(std::string filename) {
    // initialize file stream
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    // define variables
    std::string line;
    std::vector<std::streampos> frame_positions;

    // write positions of each frame to a vector
    // BE CAREFUL: THE "ITEM: TIMESTEP" LINE HAS BEEN SKIPPED
    while (std::getline(file, line)) {
        if (line.find("ITEM: TIMESTEP") != std::string::npos) {
            frame_positions.push_back(file.tellg());
        }
    }

    return frame_positions;
}

/* get number of atoms */
int get_natoms(std::string filename) {
    std::ifstream file(filename);
    std::string line;
    skip_lines(file, 3);
    std::getline(file, line);
    int n_atoms = std::stoi(line);

    return n_atoms;
}

/* read single frame */
Frame read_single_frame(std::ifstream &file, int n_atoms, int num_properties = 0) {
    Frame frame;
    
    // Since we are already at the beginning of the frame, 
    // we only need to skip 4 lines to read the cell and coordinates
    skip_lines(file, 4);
    frame.cell = read_cell(file);
    skip_lines(file, 1);
    read_coords(file, frame, n_atoms, num_properties);
    
    return frame;
}

/* read atom types in intger */
std::vector<int> read_atom_types(std::string filename, int n_atoms) {
    // The atom types here is integer, not string
    std::ifstream file(filename);
    std::string line;
    std::vector<std::pair<int, int>> id_type_pairs; // Pair of (ID, type)
    skip_lines(file, 9);

    // Read the atom types and IDs
    for (int i = 0; i < n_atoms; ++i) {
        std::getline(file, line);
        std::istringstream iss(line);
        std::vector<int> lineData((std::istream_iterator<int>(iss)), std::istream_iterator<int>());

        if (lineData.size() < 2) {
            std::cerr << "Error: Incomplete data in line " << i + 1 << std::endl;
            exit(1);
        }

        id_type_pairs.emplace_back(lineData[0], lineData[1]); // Store (ID, type)
    }

    // Sort by atom IDs
    std::sort(id_type_pairs.begin(), id_type_pairs.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
        return a.first < b.first;
    });

    // Extract sorted atom types
    std::vector<int> atom_types(n_atoms);
    for (int i = 0; i < n_atoms; ++i) {
        atom_types[i] = id_type_pairs[i].second;
    }

    return atom_types;
}

/* read all frames */
Traj read_all_frames(std::string filename, int n_atoms, int n_frames, int num_properties = 0) {
    std::ifstream file(filename);
    std::vector<Frame> frames;
    Traj traj;
    traj.atom_types = read_atom_types(filename, n_atoms);
    for (int i = 0; i < n_frames; ++i) {
        Frame frame = read_single_frame(file, n_atoms, num_properties);
        frames.push_back(frame);
    }
    traj.frames = frames;
    return traj;
}

/* read selected frames */
std::vector<Frame> read_selected_frames(std::string filename, int n_atoms, std::vector<std::streampos> frame_positions, 
                                        int start, int end, int step,
                                        int num_properties = 0) {
    // check if the start and end are within the range
    if (start < 0 || end > frame_positions.size() || start > end) {
        std::cerr << "Error: Invalid start or end frame" << std::endl;
        exit(1);
    }

    // initialize file stream and vector of frames
    std::ifstream file(filename);
    std::vector<Frame> frames;

    // read frames from start to end with step
    for (int i = start; i < end; i += step) {
        file.seekg(frame_positions[i]);
        Frame frame = read_single_frame(file, n_atoms, num_properties);
        frames.push_back(frame);
    }

    return frames;
}

/* read atom type in xsf
    It convert the atom type from string to integer, according to the type_map
*/
std::vector<int> read_atom_types_xsf(std::string filename, std::vector<std::string> type_map) {
    std::ifstream file(filename);
    std::string line;
    skip_lines(file, 6);
    std::getline(file, line);
    int n_atoms = std::stoi(line.substr(0, line.find(" ")));

    // initialize variables
    std::vector<int> atom_types(n_atoms);
    std::unordered_map<std::string, int> type_index_map;
    for (size_t j = 0; j < type_map.size(); ++j) {
        type_index_map[type_map[j]] = j+1;
    }

    // read atom types
    for (int i = 0; i < n_atoms; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        std::string atom_type;
        iss >> atom_type;
        auto it = type_index_map.find(atom_type);
        atom_types[i] = it->second;
    }

    return atom_types;
}

/* read xsf file */
Frame read_xsf(std::string filename) {
    std::ifstream file(filename);
    std::string line;

    // skip the first 2 lines
    skip_lines(file, 2);

    // read cell matrix
    Eigen::Matrix3d cell;
    for (int i = 0; i < 3; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        std::vector<double> lineData((std::istream_iterator<double>(iss)), std::istream_iterator<double>());
        cell(i, 0) = lineData[0];
        cell(i, 1) = lineData[1];
        cell(i, 2) = lineData[2];
    }

    // skip 1 line
    skip_lines(file, 1);

    // read number of atoms, which is the first number in the line
    std::getline(file, line);
    int n_atoms = std::stoi(line.substr(0, line.find(" ")));

    // read coordinates
    Eigen::MatrixXd coords(n_atoms, 3);
    std::string atom_type;
    double x, y, z;
    for (int i = 0; i < n_atoms; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        iss >> atom_type;
        iss >> x >> y >> z;
        coords(i, 0) = x;
        coords(i, 1) = y;
        coords(i, 2) = z;
    }

    Frame frame;
    frame.cell = cell;
    frame.coords = coords;
    return frame;
}

std::vector<std::vector<int>> parse_neighbor_list_file(std::string nl_file) {
    // initialize the neighbor list
    std::ifstream file(nl_file);
    std::string line;
    std::vector<std::vector<int>> neighbor_list;

    // read the neighbor list
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> lineData((std::istream_iterator<int>(iss)), std::istream_iterator<int>());
        neighbor_list.push_back(lineData);
    }

    return neighbor_list;
}

/* Read Type map from a give file,

example_type_map_file:
    Ba,Pb,Ca,Sr,Bi,K,Na,Hf,Ti,Zr,Nb,Mg,In,Zn,O
*/
std::vector<std::string> get_type_map(std::string filename) {
    // initialize file stream and variables
    std::ifstream file(filename);
    std::string line;
    std::vector<std::string> type_map;

    // parse the file
    std::getline(file, line);
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        type_map.push_back(token);
    }

    return type_map;
}

/* calculate the neighbor coord after applying the PBC */
Eigen::RowVector3d apply_pbc(Eigen::RowVector3d neighbor, Eigen::RowVector3d center, Eigen::Matrix3d cell) {
    Eigen::RowVector3d diff = neighbor - center;
    Eigen::RowVector3d diff_frac = diff * cell.inverse();
    Eigen::RowVector3d neighbor_frac = neighbor * cell.inverse();

    for (int i = 0; i < 3; i++) {
        if (diff_frac(i) > 0.5) {
            neighbor_frac(i) -= 1;
        } else if (diff_frac(i) < -0.5) {
            neighbor_frac(i) += 1;
        }
    }

    return neighbor_frac * cell;
}
#endif 