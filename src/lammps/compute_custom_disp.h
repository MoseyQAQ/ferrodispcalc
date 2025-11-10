/*

LAMMPS plugin for calculate the displacement of atoms relative to the centroid of their neighbors.

The neighbor list file is provided by the user.
The first column is the central atom ID, and the rest of the columns are the neighbor IDs.

Usage:
compute compute-ID all customdisp nnfile file_name
nnfile = neighbor list file name

*/

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(disp/atom,ComputeCustomDisp);
// clang-format on
#else

#ifndef COMPUTE_CUSTOM_DISP_H
#define COMPUTE_CUSTOM_DISP_H

#include "compute.h"
#include <vector>

namespace LAMMPS_NS {

    class ComputeCustomDisp : public Compute {
        public:
            ComputeCustomDisp(class LAMMPS *, int, char **);
            ~ComputeCustomDisp() override;
            void compute_peratom() override;
            void init() override;
        
        private:
            char *nnfile; // neighbor list file name
            void read_file();
            std::vector<int> central_id; // central atom ID
            std::vector<std::vector<int>> neighbor_id; // neighbor atom ID
            int nmax;
    };

}   // namespace LAMMPS_NS

#endif
#endif