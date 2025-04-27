/*
get_polarization_displacement.cpp:
    This program reads a LAMMPS dump file and a neighbor list file, and calculates the polarization displacement of cations.
    The neighbor list file contains the indices of the neighbors of each cation, starting from 0. It can be generated using 
    "build_neighbor_list.py"
    Eigen library is required for matrix operations.

Compile:
    g++ get_polarization_displacement.cpp -O3 -o get_polarization_displacement -I /path/to/eigen

Usage:
    ./get_polarization_displacement traj_file output_file nl_file ratio/last_frame
    OPTIONS:
        traj_file: LAMMPS dump file or xsf file
        output_file: output file, each line contains the original coordinates and displacements of a cation.
        nl_file: neighbor list file, it can be generated using "build_neighbor_list.py"
        ratio/last_frame: If the number < 1, it is the ratio of frames to be read. i.e. 0.5 means last 50% of frames will be read
                          If the number >= 1, it is the last frame to be read. i.e. 2500 means the last 2500 frames will be read

Author: Denan Li
Email: lidenan@westlake.edu.cn

Todo list:
    3. support output an averaged polarization displacement.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <Eigen/Dense>
#include <iomanip>
#include "basic.hpp"

/* define struct for polarization data */
struct polarization_data
{
    Eigen::MatrixXd original_coords; // original coordinates of cations
    Eigen::MatrixXd displacements; // displacements of cations
};

// define functions
std::vector<std::vector<int>> parse_nl_file(std::string nl_file);
polarization_data get_polarization_displacement_in_one_frame(Frame frame, std::vector<std::vector<int>> neighbor_list);

/* Main function */
int main(int argc, char** argv) {
    // setup input parameters
    std::string traj_file = argv[1];
    std::string output_file = argv[2];
    std::string nl_file = argv[3];

    // initialize neighbor list and frames
    std::vector<std::vector<int>> neighbor_list = parse_nl_file(nl_file);
    std::vector<Frame> frames;

    // check whether the output file exists, if so, exit
    std::ifstream check_file(output_file);
    if (check_file.good() == true) {
        std::cerr << "File already exists: " << output_file << std::endl;
        exit(1);
    }

    // detect the traj_file format
    // If the traj_file is in xsf format, read it using read_xsf function
    std::string traj_file_format = traj_file.substr(traj_file.find_last_of(".") + 1);
    if (traj_file_format == "xsf") {
        // read xsf file
        frames.push_back(read_xsf(traj_file));
        std::cout << "Frames read: " << frames.size() << std::endl;
    } else {
        // else: the traj_file is in lammps dump format
        // get the number of atoms and frame positions
        int natoms = get_natoms(traj_file);
        std::vector<std::streampos> frame_pos = get_frame_positions(traj_file);

        // print information
        std::cout << "Number of atoms: " << natoms << std::endl;
        std::cout << "Number of frames: " << frame_pos.size() << std::endl;

        // calculate the frame index to read in
        int end_frame = frame_pos.size();
        int step = 1;
        double ratio = std::stod(argv[4]);
        int start_frame;
        if (ratio > 1) {
            start_frame = end_frame - ratio;
        } else if (ratio <= 1 && ratio > 0) {
            start_frame = frame_pos.size() * (1 - ratio);
        } else {
            std::cerr << "Invalid ratio: " << ratio << std::endl;
            exit(1);
        }

        // read selected frames
        frames = read_selected_frames(traj_file, natoms, frame_pos, start_frame, end_frame, step);
        std::cout << "Frames read: " << frames.size() << std::endl;
    }

    // get polarization displacement data in all frames
    std::vector<polarization_data> data(frames.size());
    for (int i = 0; i < frames.size(); i++) {
        data[i] = get_polarization_displacement_in_one_frame(frames[i], neighbor_list);
    }

    // write output file
    std::ofstream file(output_file);
    file << std::fixed << std::setprecision(16);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].original_coords.rows(); j++) {
            file << data[i].original_coords.row(j) << " " << data[i].displacements.row(j) << std::endl;
        }
    }

    return 0;
}

/* parse the neighbor list file */
std::vector<std::vector<int>> parse_nl_file(std::string nl_file) {
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

/* calculate the polarization displacement in one frame */
polarization_data get_polarization_displacement_in_one_frame(Frame frame, std::vector<std::vector<int>> neighbor_list) {
    // initialize the original coordinates and displacements
    int natoms = neighbor_list.size();
    Eigen::MatrixXd original_coords = Eigen::MatrixXd::Zero(natoms, 3);
    Eigen::MatrixXd displacements = Eigen::MatrixXd::Zero(natoms, 3);

    // loop over all center atoms
    for (int i = 0; i < neighbor_list.size(); i++) {
        Eigen::RowVector3d center = frame.coords.row(neighbor_list[i][0]);
        Eigen::RowVector3d neighbor = Eigen::RowVector3d::Zero();

        // loop over all neighbors of the center atom
        for (int j = 1; j < neighbor_list[i].size(); j++) {
            Eigen::RowVector3d neighbor_coord = apply_pbc(frame.coords.row(neighbor_list[i][j]), center, frame.cell);
            neighbor += neighbor_coord;
        }

        // calculate the average position of the neighbors
        neighbor /= neighbor_list[i].size() - 1;
        Eigen::RowVector3d displacement = center - neighbor;
        original_coords.row(i) = center;
        displacements.row(i) = displacement;
    }

    polarization_data data;
    data.original_coords = original_coords;
    data.displacements = displacements;
    return data;
}