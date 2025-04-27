/*
get_local_lattice.cpp:
    This program read xsf file for ABO3 perovskite structure and calculate the local lattice vectors for each unit cell.
    Ref: Phys. Rev. Lett. 119, 177602 (2017)

Compile:
    g++ -o get_local_lattice get_local_lattice.cpp -O3 -I /path/to/eigen -std=c++11

Usage:
    ./get_local_lattice input_file output_file nl_file [angle_x angle_y angle_z]
    OPTIONS:
        input_file: the input xsf file
        output_file: the output file to store the local lattice vectors
        nl_file: the neighbor list file
        angle_x, angle_y, angle_z: the rotation angles about x, y, z axis in degree (default: 0 0 0)

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
#include <cmath>
#include "basic.hpp"

/* defined functions */
Eigen::MatrixXd get_local_lattice_vectors(Frame frame, std::vector<std::vector<int>> neighbor_list, Eigen::Matrix3d tm_x, Eigen::Matrix3d tm_y, Eigen::Matrix3d tm_z);
std::vector<Eigen::Matrix3d> construct_rotation_matrix(double angle_x, double angle_y, double angle_z);
Eigen::RowVector3d apply_rotation_matrix(Eigen::RowVector3d coord, Eigen::Matrix3d tm_x, Eigen::Matrix3d tm_y, Eigen::Matrix3d tm_z);
Eigen::RowVector3d get_lattice(std::vector<Eigen::RowVector3d> beta, std::vector<Eigen::RowVector3d> alpha);

int main(int argc, char** argv) {
    if (argc != 4 && argc != 7) {
        std::cerr << "Usage: ./get_local_lattice input_file output_file nl_file [angle_x angle_y angle_z]" << std::endl;
        return 1;
    }
    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string nl_file = argv[3];

    // Initialize angles to 0 by default
    double angle_about_x = 0.0, angle_about_y = 0.0, angle_about_z = 0.0;

    // Set angles if provided
    if (argc == 7) {
        angle_about_x = std::stod(argv[4]) * M_PI / 180;
        angle_about_y = std::stod(argv[5]) * M_PI / 180;
        angle_about_z = std::stod(argv[6]) * M_PI / 180;
    }

    // construct transformation matrix, read xsf file and neighbor list file
    std::vector<Eigen::Matrix3d> rotation_matrix = construct_rotation_matrix(angle_about_x, angle_about_y, angle_about_z);
    std::vector<std::vector<int>> neighbor_list = parse_neighbor_list_file(nl_file);
    Frame frame = read_xsf(input_file);
    Eigen::MatrixXd local_lattice = get_local_lattice_vectors(frame, neighbor_list, rotation_matrix[0], rotation_matrix[1], rotation_matrix[2]);

    // write the local lattice vectors to output file
    std::ofstream out(output_file);
    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < local_lattice.rows(); i++) {
        out << local_lattice.row(i) << std::endl;
    }
}

/* construct the rotation matrix */
std::vector<Eigen::Matrix3d> construct_rotation_matrix(double angle_x, double angle_y, double angle_z) {
    Eigen::Matrix3d tm_x, tm_y, tm_z;
    tm_x << 
        1, 0, 0,
        0, cos(angle_x), -sin(angle_x),
        0, sin(angle_x), cos(angle_x);
    tm_y <<
        cos(angle_y), 0, sin(angle_y),
        0, 1, 0,
        -sin(angle_y), 0, cos(angle_y);
    tm_z <<
        cos(angle_z), -sin(angle_z), 0,
        sin(angle_z), cos(angle_z), 0,
        0, 0, 1;
    return {tm_x, tm_y, tm_z};
}

/* For a given coord, apply the rotation matrix */
Eigen::RowVector3d apply_rotation_matrix(Eigen::RowVector3d coord, Eigen::Matrix3d tm_x, Eigen::Matrix3d tm_y, Eigen::Matrix3d tm_z) {
    Eigen::Vector3d new_coord = tm_x * tm_y * tm_z * coord.transpose();
    return new_coord.transpose();
}

/* calcuate local lattice for a give unit cell */
Eigen::RowVector3d get_lattice(std::vector<Eigen::RowVector3d> beta, std::vector<Eigen::RowVector3d> alpha) {
    Eigen::RowVector3d alpha_sum = Eigen::RowVector3d::Zero();
    Eigen::RowVector3d beta_sum = Eigen::RowVector3d::Zero();
    for (int i = 0; i < alpha.size(); i++) {
        alpha_sum += alpha[i];
    }
    for (int i = 0; i < beta.size(); i++) {
        beta_sum += beta[i];
    }
    return 0.25 * (beta_sum - alpha_sum);
}

/* Main function: calculate local lattice vector for all unit cells */
Eigen::MatrixXd get_local_lattice_vectors(Frame frame, std::vector<std::vector<int>> neighbor_list,
                                          Eigen::Matrix3d tm_x, Eigen::Matrix3d tm_y, Eigen::Matrix3d tm_z) {
    int n_unitcells = neighbor_list.size();
    Eigen::MatrixXd lattice_vector_a = Eigen::MatrixXd::Zero(n_unitcells, 3);
    Eigen::MatrixXd lattice_vector_b = Eigen::MatrixXd::Zero(n_unitcells, 3);
    Eigen::MatrixXd lattice_vector_c = Eigen::MatrixXd::Zero(n_unitcells, 3);

    // loop over all unit cells
    for (int i = 0; i < n_unitcells; i++) {
        Eigen::RowVector3d center = frame.coords.row(neighbor_list[i][0]);
        Eigen::RowVector3d center_transformed = apply_rotation_matrix(center, tm_x, tm_y, tm_z);
        std::vector<Eigen::RowVector3d> alpha_a, alpha_b, alpha_c;
        std::vector<Eigen::RowVector3d> beta_a, beta_b, beta_c;

        // loop over all neighbors
        for (int j = 1; j < neighbor_list[i].size(); j++) {
            Eigen::RowVector3d neighbor_coord = apply_pbc(frame.coords.row(neighbor_list[i][j]), center, frame.cell);
            Eigen::RowVector3d neighbor_coord_transformed = apply_rotation_matrix(neighbor_coord, tm_x, tm_y, tm_z);
            Eigen::RowVector3d vector = neighbor_coord - center;
            Eigen::RowVector3d vector_transformed = neighbor_coord_transformed - center_transformed;
            
            if (vector_transformed(0) < 0) {
                alpha_a.push_back(vector);
            } else {
                beta_a.push_back(vector);
            }

            if (vector_transformed(1) < 0) {
                alpha_b.push_back(vector);
            } else {
                beta_b.push_back(vector);
            }

            if (vector_transformed(2) < 0) {
                alpha_c.push_back(vector);
            } else {
                beta_c.push_back(vector);
            }
        }

        // check the size of alpha and beta layer, should be 4
        if (alpha_a.size() != 4 || beta_a.size() != 4 || alpha_b.size() != 4 || beta_b.size() != 4 || alpha_c.size() != 4 || beta_c.size() != 4) {
            std::cerr << "Error: the number of neighbors is not 8" << std::endl;
            exit(1);
        }

        // calculate the local lattice vectors of this unit cell
        lattice_vector_a.row(i) = get_lattice(beta_a, alpha_a);
        lattice_vector_b.row(i) = get_lattice(beta_b, alpha_b);
        lattice_vector_c.row(i) = get_lattice(beta_c, alpha_c);
    }

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_unitcells, 9);
    out << lattice_vector_a, lattice_vector_b, lattice_vector_c;
    return out;
}