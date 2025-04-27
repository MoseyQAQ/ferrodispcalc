/*
get_atomic_polarization.cpp:
    This program reads a LAMMPS dump file and a neighbor list file, and calculates the polarization displacement of cations.
    The neighbor list file contains the indices of the neighbors of each cation, starting from 0. It can be generated using 
    "build_neighbor_list.py"
    Eigen library is required for matrix operations.

Compile:
    g++ get_atomic_polarization.cpp -O3 -o get_atomic_polarization -I /path/to/eigen

Usage:
    ./get_polarization traj_file output_file nl_file bec ratio/last_frame
    OPTIONS:
        traj_file: LAMMPS dump file or xsf file
        output_file: output file, each line contains the original coordinates and displacements of a cation.
        nl_file: neighbor list file, it can be generated using "build_neighbor_list.py"
        ratio/last_frame: If the number < 1, it is the ratio of frames to be read. i.e. 0.5 means last 50% of frames will be read
                          If the number >= 1, it is the last frame to be read. i.e. 2500 means the last 2500 frames will be read

Author: Denan Li
Email: lidenan@westlake.edu.cn
*/

#include "basic.hpp"
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <Eigen/Dense>

Eigen::MatrixXd get_atomic_polarization_in_one_frame(Frame frame, 
    std::vector<std::vector<int>> nl,
    double atomic_bec);

/* Main function */
int main(int argc, char** argv) {
    // setup input parameters
    std::string traj_file = argv[1];
    std::string output_file = argv[2];
    std::string nl_file = argv[3];
    double bec = std::stod(argv[4]);

    // initialize neighbor list and frames
    std::vector<std::vector<int>> neighbor_list = parse_neighbor_list_file(nl_file);
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
        double ratio = std::stod(argv[5]);
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
    std::vector<Eigen::MatrixXd> data(frames.size());
    for (int i = 0; i < frames.size(); i++) {
        data[i] = get_atomic_polarization_in_one_frame(frames[i], neighbor_list, bec);
    }

    // write output file
    std::ofstream file(output_file);
    file << std::fixed << std::setprecision(16);
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[i].rows(); j++) {
            file << data[i].row(j) << std::endl;
        }
    }

    return 0;
}

/* calculate the atomic polarization in one frame */
Eigen::MatrixXd get_atomic_polarization_in_one_frame(Frame frame, 
    std::vector<std::vector<int>> nl,
    double atomic_bec)
{
// initialize the original coordinates and displacements
int natoms = nl.size();
Eigen::MatrixXd displacements = Eigen::MatrixXd::Zero(natoms, 3);

// loop over all center atoms
for (int i = 0; i < nl.size(); i++) {
Eigen::RowVector3d center = frame.coords.row(nl[i][0]);
Eigen::RowVector3d neighbor = Eigen::RowVector3d::Zero();

// loop over all neighbors of the center atom
for (int j = 1; j < nl[i].size(); j++) {
Eigen::RowVector3d neighbor_coord = apply_pbc(frame.coords.row(nl[i][j]), center, frame.cell);
neighbor += neighbor_coord;
}

// calculate the average position of the neighbors
neighbor /= nl[i].size() - 1;
Eigen::RowVector3d displacement = center - neighbor;
displacements.row(i) = displacement;
}

double volume = std::abs(frame.cell.determinant());
double conversion_factor = 1.602176E-19 * 1.0E-10 * 1.0E30; // convert to C/m^2
double volume_per_unit_cell = volume / natoms;
displacements = atomic_bec * displacements * conversion_factor / volume_per_unit_cell; // here, we assume the volume is the same for all unit cells.

return displacements;
}