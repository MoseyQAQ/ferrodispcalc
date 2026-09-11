/*

LAMMPS plugin for calculating the displacement of atoms relative to the
centroid of the nearest N atoms in a neighbor group, and the matching
displacement velocity. Neighbors are selected anew on each invocation.

Usage:
compute compute-ID central_group disp/atom/auto neighbor_group N [vel yes|no]
N = positive number of nearest neighbors within the pair cutoff
vel = output displacement velocity; default is no

Output columns:
1-3: displacement = average(r_center - r_neighbor)
4-6: displacement velocity = average(v_center - v_neighbor), only with vel yes

Requires a pair style and a full LAMMPS neighbor list. Neighbor exclusions
also apply here. Periodic ghost coordinates provide the neighbor images.
Atoms with fewer than N neighbors output zero and generate a summary warning.
With vel yes, use comm_modify vel yes. If neighbors change, the reported
relative velocity does not include the jump caused by that change.

*/

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(disp/atom/auto,ComputeDispLmpNN);
// clang-format on
#else

#ifndef COMPUTE_DISP_LMP_NN_H
#define COMPUTE_DISP_LMP_NN_H

#include "compute.h"
#include <utility>
#include <vector>

namespace LAMMPS_NS {

    class ComputeDispLmpNN : public Compute {
        public:
            ComputeDispLmpNN(class LAMMPS *, int, char **);
            ~ComputeDispLmpNN() override;
            void compute_peratom() override;
            void init() override;
            void init_list(int, class NeighList *) override;

        private:
            int neighbor_groupbit;
            int neighbor_number;
            int nmax;
            int velocity_flag;
            class NeighList *list;
            std::vector<std::pair<double, int>> nearest; // squared distance, local index
    };

}   // namespace LAMMPS_NS

#endif
#endif
