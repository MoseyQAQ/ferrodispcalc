/*
get_polarization.cpp:
    This program read a LAMMPS dump file and two neighbor list files, and calculate the polarization of each perovskite unit call.
    The polarization of each unit cell, P, is calculated by the formula (could be found at: PHYSICAL REVIEWB95,094102 (2017)):
        P = (bec * B + bec * 0.5X +  bec * 0.125A)
    where B, X, and A are the coordinates of B site cations, X site anions, and A site cations, respectively.
    The bec of each atom type is read from a bec file.

Compile:
    g++ get_polarization.cpp -o get_polarization -I /path/to/eigen

Usage:
    ./get_polarization input_file output_file a_nl_file x_nl_file type_map_file bec_file ratio/last_frame
    OPTIONS:
        input_file: LAMMPS dump file or xsf file
        output_file: output file, each line contains the polarization of a unit cell.
        a_nl_file: neighbor list file for A site cations
        x_nl_file: neighbor list file for X site anions
        type_map_file: file contains the type map of atom types
        bec_file: file contains the Born effective charge of each atom type
        ratio/last_frame: If the number < 1, it is the ratio of frames to be read. i.e. 0.5 means last 50% of frames will be read
                          If the number >= 1, it is the last frame to be read. i.e. 2500 means the last 2500 frames will be read

Author: Denan Li
Email: lidenan@westlake.edu.cn

Todo list:
    3. support output an averaged polarization.
    4. flexible IO
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iterator>
#include <Eigen/Dense>
#include <iomanip>
#include "basic.hpp"

std::vector<double> get_bec(std::string bec_file);
Eigen::MatrixXd get_polarization_in_one_frame(Frame frame, 
                                              std::vector<std::vector<int>> a_neighbor_list, 
                                              std::vector<std::vector<int>> x_neighbor_list,
                                              std::vector<double> bec_of_atoms);

int main(int argc, char** argv) {
    // initialize input parameters
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string a_nl_file = argv[3];
    std::string x_nl_file = argv[4];
    std::string type_map_file = argv[5];
    std::string bec_file = argv[6];

    // initialize neighbor list, bec, and frames
    std::vector<Frame> frames;
    std::vector<double> bec = get_bec(bec_file);
    std::vector<double> bec_of_atoms;
    std::vector<std::vector<int>> a_neighbor_list = parse_neighbor_list_file(a_nl_file);
    std::vector<std::vector<int>> x_neighbor_list = parse_neighbor_list_file(x_nl_file);
    std::vector<std::string> type_map = get_type_map(type_map_file);
    
    // detect the traj_file format
    // If the traj_file is in xsf format, read it using read_xsf function
    std::string traj_file_format = input_file.substr(input_file.find_last_of(".") + 1);
    if (traj_file_format == "xsf") {
        frames.push_back(read_xsf(input_file)); // read xsf file
        
        //build bec
        std::vector<int> atom_types = read_atom_types_xsf(input_file, type_map);
        for (int i = 0; i < atom_types.size(); i++) {
            bec_of_atoms.push_back(bec[atom_types[i]-1]);
        }
    } else {
        // else: the traj_file is in lammps dump format
        // get the number of atoms and frame positions
        int natoms = get_natoms(input_file);
        std::vector<std::streampos> frame_pos = get_frame_positions(input_file);

        // set start_frame, end_frame, and step
        int end_frame = frame_pos.size();
        int step = 1;
        double ratio = std::stod(argv[7]);
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
        frames = read_selected_frames(input_file, natoms, frame_pos, start_frame, end_frame, step);

        // build bec
        std::vector<int> atom_types = read_atom_types(input_file, natoms);
        for (int i = 0; i < natoms; i++) {
            bec_of_atoms.push_back(bec[atom_types[i]-1]);
        }
    }

    // get polarization data in all frames
    std::vector<Eigen::MatrixXd> polarization(frames.size());
    for (int i = 0; i < frames.size(); i++) {
        polarization[i] = get_polarization_in_one_frame(frames[i], a_neighbor_list, x_neighbor_list, bec_of_atoms);
    }

    // write output file
    std::ofstream file(output_file);
    file << std::fixed << std::setprecision(16);
    for (int i = 0; i < polarization.size(); i++) {
        for (int j = 0; j < polarization[i].rows(); j++) {
            file << polarization[i].row(j) << std::endl;
        }
    
    }

    return 0;
}

Eigen::MatrixXd get_polarization_in_one_frame(Frame frame, 
                                              std::vector<std::vector<int>> a_neighbor_list, 
                                              std::vector<std::vector<int>> x_neighbor_list,
                                              std::vector<double> bec_of_atoms) {
    // check if the number of atoms in a_neighbor_list and x_neighbor_list are the same
    if (a_neighbor_list.size() != x_neighbor_list.size()) {
        std::cerr << "The number of atoms in a_neighbor_list and x_neighbor_list are different" << std::endl;
        exit(1);
    }

    int natoms = a_neighbor_list.size();
    
    Eigen::MatrixXd polarization(natoms, 3);

    // loop over all B site cations
    for (int i = 0; i < a_neighbor_list.size(); i++) {
        Eigen::RowVector3d term_a = Eigen::RowVector3d::Zero();
        Eigen::RowVector3d term_b = Eigen::RowVector3d::Zero();
        Eigen::RowVector3d term_x = Eigen::RowVector3d::Zero();
        Eigen::RowVector3d center = frame.coords.row(a_neighbor_list[i][0]);

        // loop over all A site cations
        for (int j = 1; j < a_neighbor_list[i].size(); j++) {

            // check pbc
            Eigen::RowVector3d neighbor_coord = apply_pbc(frame.coords.row(a_neighbor_list[i][j]), center, frame.cell);
            term_a += bec_of_atoms[a_neighbor_list[i][j]] * neighbor_coord;
        }

        // loop over all X site anions
        for (int j = 1; j < x_neighbor_list[i].size(); j++) {
            Eigen::RowVector3d neighbor_coord = apply_pbc(frame.coords.row(x_neighbor_list[i][j]), center, frame.cell);
            term_x += bec_of_atoms[x_neighbor_list[i][j]] * neighbor_coord;
        }

        // calculate the polarization
        term_b = bec_of_atoms[a_neighbor_list[i][0]] * center;
        polarization.row(i) = term_b + term_x * 0.5 + term_a * 0.125;
    }

    double volume = std::abs(frame.cell.determinant());
    double conversion_factor = 1.602176E-19 * 1.0E-10 * 1.0E30;
    polarization = a_neighbor_list.size() * polarization * conversion_factor / volume;

    return polarization;
}

/* parse the bec file */
std::vector<double> get_bec(std::string bec_file) {
    // initialize file stream and variables
    std::ifstream file(bec_file);
    std::string line;
    std::vector<double> bec;

    // parse the file
    std::getline(file, line);
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, ',')) {
        bec.push_back(std::stod(token));
    }

    return bec;
}