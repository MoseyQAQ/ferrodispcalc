/*

LAMMPS plugin for calculating the displacement of atoms relative to the
centroid of user-provided neighbors, and the matching displacement velocity.

The neighbor list file is provided by the user.
The first column is the central atom ID, and the rest of the columns are the neighbor IDs.

Usage:
atom_modify map array
compute compute-ID all disp/atom nnfile file_name [vel yes|no]
nnfile = neighbor list file name
vel = output displacement velocity; default is no

Output columns:
1-3: displacement = average(r_center - r_neighbor)
4-6: displacement velocity = average(v_center - v_neighbor), only with vel yes

*/

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(disp/atom,ComputeDispCustomNN);
// clang-format on
#else

#ifndef COMPUTE_DISP_CUSTOM_NN_H
#define COMPUTE_DISP_CUSTOM_NN_H

#include "compute.h"
#include <vector>

namespace LAMMPS_NS {

    class ComputeDispCustomNN : public Compute {
        public:
            ComputeDispCustomNN(class LAMMPS *, int, char **);
            ~ComputeDispCustomNN() override;
            void compute_peratom() override;
            void init() override;
        
        private:
            char *nnfile; // neighbor list file name
            void read_file();
            std::vector<int> central_id; // central atom ID
            std::vector<std::vector<int>> neighbor_id; // neighbor atom ID
            int nmax;
            int velocity_flag;
    };

}   // namespace LAMMPS_NS

#endif
#endif
