/*
get_averaged_structure.cpp: 
    This program read a LAMMPS dump file and output the averaged structure in .xsf format.
    Eigen library is required for matrix operations.

Compile:
    g++ get_averaged_structure.cpp -O3 -o get_averaged_structure -I /path/to/eigen

Usage:
    ./get_averaged_structure input_file output_file type_map_file ratio/last_frame
    OPTIONS:
        input_file: LAMMPS dump file
        output_file: output file in .xsf format
        type_map_file: a file containing the atom types 
        ratio/last_frame: If the number < 1, it is the ratio of frames to be read. i.e. 0.5 means last 50% of frames will be read to calculate the average structure.
                          If the number >= 1, it is the last frame to be read. i.e. 2500 means the last 2500 frames will be read to calculate the average structure.
        
Author: Denan Li
Email: lidenan@westlake.edu.cn
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <Eigen/Dense>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include "basic.hpp"

// define functions
Eigen::MatrixXd get_avg_properties(const std::vector<Frame> &frames, int natoms, int num_properties);
Eigen::Matrix3d get_avg_cell(std::vector<Frame> frames);
Eigen::MatrixXd get_avg_coords(std::vector<Frame> frames, int natoms);
Frame sort_frame_by_id(const Frame &frame);
void write_atomic_properties(const std::string &filename, const Eigen::MatrixXd &avg_properties);
void write_xsf(std::string filename, Eigen::Matrix3d cell, Eigen::MatrixXd coords, std::vector<int> atom_types, std::vector<std::string> type_map);

/* Main function */
int main(int argc, char** argv) {
    // input parameters
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string type_map_file = argv[3];
    double ratio = std::stod(argv[4]);

    // check whether the output file exists, if so, exit
    std::ifstream check_file(output_file);
    if (check_file.good() == true) {
        std::cerr << "File already exists: " << output_file << std::endl;
        exit(1);
    }

    // read input file
    std::cout << "Reading file: " << input_file << std::endl;
    int natoms = get_natoms(input_file);
    std::vector<int> atom_types = read_atom_types(input_file, natoms);
    std::vector<std::streampos> frame_pos = get_frame_positions(input_file);
    std::vector<std::string> type_map = get_type_map(type_map_file);

    // calculate the frame index to read in
    int end_frame = frame_pos.size();
    int step = 1;
    int start_frame;
    if (ratio > 1) {
        start_frame = end_frame - ratio;
    } else if (ratio <= 1 && ratio > 0) {
        start_frame = frame_pos.size() * (1 - ratio);
    } else {
        std::cerr << "Invalid ratio: " << ratio << std::endl;
        exit(1);
    }

    // print information
    std::cout << "Number of atoms: " << natoms << std::endl;
    std::cout << "Number of frames: " << frame_pos.size() << std::endl;
    std::cout << "Type map: "; 
    for (int i = 0; i < type_map.size(); ++i) {
        std::cout << type_map[i] << " ";
    }
    std::cout << std::endl;

    // parse atomic properties
    int num_properties = parse_properties(input_file);

    // read selected frames
    std::vector<Frame> frames = read_selected_frames(input_file, natoms, frame_pos, start_frame, end_frame, step, num_properties);
    std::cout << "Frames read: " << frames.size() << std::endl;

    // Sort the first frame by IDs if necessary
    frames[0] = sort_frame_by_id(frames[0]);

    // Ensure all frames follow the same order as the first frame
    for (int i = 1; i < frames.size(); ++i) {
        frames[i] = sort_frame_by_id(frames[i]);
    }

    // Calculate the average cell, coordinates, and properties
    Eigen::Matrix3d avg_cell = get_avg_cell(frames);
    Eigen::MatrixXd avg_coords = get_avg_coords(frames, natoms);
    Eigen::MatrixXd avg_properties;
    if (num_properties > 0) {
        avg_properties = get_avg_properties(frames, natoms, num_properties);
    }

    // write output file
    std::cout << "Writing file: " << output_file << std::endl;
    write_xsf(output_file, avg_cell, avg_coords, atom_types, type_map);

    // Write averaged atomic properties to a file if num_properties > 0
    if (num_properties > 0) {
        std::string properties_file = "properties_" + output_file;
        std::cout << "Writing atomic properties to file: " << properties_file << std::endl;
        write_atomic_properties(properties_file, avg_properties);
    }

    return 0;
}

/* calculate the averaged cell */
Eigen::Matrix3d get_avg_cell(std::vector<Frame> frames) {
    Eigen::Matrix3d avg_cell = Eigen::Matrix3d::Zero();
    for (int i = 0; i < frames.size(); ++i) {
        avg_cell += frames[i].cell;
    }
    avg_cell /= frames.size();

    return avg_cell;
}

/* calculate the averaged coordinates */
Eigen::MatrixXd get_avg_coords(std::vector<Frame> frames, int natoms) {
    // initialize the averaged coordinates
    Eigen::MatrixXd avg_coords = frames[0].coords;
    Eigen::MatrixXd coord_init = frames[0].coords;

    for (int i = 1; i < frames.size(); ++i) {
        
        // check the periodic boundary conditions
        Eigen::MatrixXd coord_current = frames[i].coords;
        Eigen::MatrixXd coord_current_frac = coord_current * frames[i].cell.inverse();
        Eigen::MatrixXd coord_diff = coord_current - coord_init;
        Eigen::MatrixXd coord_diff_frac = coord_diff * frames[i].cell.inverse();

        for (int j = 0; j < natoms; ++j) {
            for (int k = 0; k < 3; ++k) {
                if (coord_diff_frac(j, k) > 0.5) {
                    coord_current_frac(j, k) -= 1;
                } else if (coord_diff_frac(j, k) < -0.5) {
                    coord_current_frac(j, k) += 1;
                }
            }
        }

        coord_current = coord_current_frac * frames[i].cell;
        avg_coords += coord_current;
    }

    avg_coords /= frames.size();
    return avg_coords;
}

/* write the .xsf file */
void write_xsf(std::string filename, Eigen::Matrix3d cell, Eigen::MatrixXd coords, std::vector<int> atom_types, std::vector<std::string> type_map) {

    // write the file
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(16); // set 16 decimal places
    file << "CRYSTAL\n";
    file << "PRIMVEC\n";
    file << cell.row(0) << std::endl;
    file << cell.row(1) << std::endl;
    file << cell.row(2) << std::endl;
    file << "PRIMCOORD\n";
    file << coords.rows() << " 1\n";
    for (int i = 0; i < coords.rows(); ++i) {
        file << type_map[atom_types[i]-1] << " " << coords.row(i) << std::endl;
    }
}

/* Check and sort the first frame by IDs */
Frame sort_frame_by_id(const Frame &frame) {
    // Create a vector of indices for sorting
    std::vector<int> indices(frame.ids.size());
    std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1

    // Check if IDs are already sorted
    if (std::is_sorted(frame.ids.begin(), frame.ids.end())) {
        return frame; // Return the original frame if IDs are sorted
    }

    // Sort indices based on the IDs
    std::sort(indices.begin(), indices.end(), [&frame](int i, int j) {
        return frame.ids[i] < frame.ids[j];
    });

    // Create a new frame with sorted data
    Frame sorted_frame;
    sorted_frame.ids.resize(frame.ids.size());
    sorted_frame.coords = Eigen::MatrixXd(frame.coords.rows(), frame.coords.cols());
    sorted_frame.properties = Eigen::MatrixXd(frame.properties.rows(), frame.properties.cols());
    sorted_frame.cell = frame.cell; // Cell isn't sorted, just copied

    for (int i = 0; i < indices.size(); ++i) {
        sorted_frame.ids[i] = frame.ids[indices[i]];
        sorted_frame.coords.row(i) = frame.coords.row(indices[i]);
        if (frame.properties.rows() > 0) { // Check if properties exist
            sorted_frame.properties.row(i) = frame.properties.row(indices[i]);
        }
    }

    return sorted_frame;
}

/* Calculate the averaged atomic properties */
Eigen::MatrixXd get_avg_properties(const std::vector<Frame> &frames, int natoms, int num_properties) {
    // Initialize the averaged properties matrix
    Eigen::MatrixXd avg_properties = Eigen::MatrixXd::Zero(natoms, num_properties);

    // Accumulate properties from all frames
    for (const auto &frame : frames) {
        avg_properties += frame.properties;
    }

    // Divide by the number of frames to get the average
    avg_properties /= frames.size();

    return avg_properties;
}

/* Write averaged atomic properties to a file */
void write_atomic_properties(const std::string &filename, const Eigen::MatrixXd &avg_properties) {
    // Open the file for writing
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        exit(1);
    }

    // Set precision to 4 decimal places
    file << std::fixed << std::setprecision(4);

    // Write the properties matrix
    for (int i = 0; i < avg_properties.rows(); ++i) {
        for (int j = 0; j < avg_properties.cols(); ++j) {
            file << avg_properties(i, j);
            if (j < avg_properties.cols() - 1) {
                file << " "; // Add space between columns
            }
        }
        file << "\n"; // Newline after each row
    }

    file.close();
}