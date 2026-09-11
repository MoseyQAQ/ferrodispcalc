/*

B-centered ABO3 local polarization using the nearest 8 A and 6 X images.

Usage:
compute ID group polar/abo3/auto A_group BEC_A B_group BEC_B X_group BEC_X [vel yes|no]

Columns 1-3: P = (BEC_B*r_B + BEC_A*sum(r_A)/8 + BEC_X*sum(r_X)/2)
                * 16.02176634 / V_cell, in C/m^2.
V_cell is the current box volume divided by the global number of B atoms,
including B atoms outside the compute group. Positions are in Angstrom and
BECs in elementary-charge units (units metal or real). BEC_A+BEC_B+3*BEC_X
must vanish for origin-independent polarization; supplied BECs are not adjusted.

Columns 4-6 with vel yes: (BEC_B*v_B + BEC_A*sum(v_A)/8
                          + BEC_X*sum(v_X)/2) / 5.
These are BEC-weighted unit-cell average velocities, without volume or SI
conversion (e*Angstrom/ps for metal; e*Angstrom/fs for real). They are not dP/dt.

Only B atoms in the compute group have nonzero output. Neighbors are selected
on every invocation within the pair cutoff, respecting neighbor exclusions.
Periodic ghost images, including repeated IDs in small cells, are retained.
Insufficient coordination gives zero output and a summary warning.
vel defaults to no; vel yes requires comm_modify vel yes.

*/

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(polar/abo3/auto,ComputePolarABO3LmpNN);
// clang-format on
#else

#ifndef COMPUTE_POLAR_ABO3_LMP_NN_H
#define COMPUTE_POLAR_ABO3_LMP_NN_H

#include "compute.h"
#include <utility>
#include <vector>

namespace LAMMPS_NS {

    class ComputePolarABO3LmpNN : public Compute {
        public:
            ComputePolarABO3LmpNN(class LAMMPS *, int, char **);
            ~ComputePolarABO3LmpNN() override;
            void compute_peratom() override;
            void init() override;
            void init_list(int, class NeighList *) override;

        private:
            int Agroupbit, Bgroupbit, Xgroupbit;
            double bec_A, bec_B, bec_X;
            int nmax;
            int velocity_flag;
            class NeighList *list;
            std::vector<std::pair<double, int>> nearest_A, nearest_X;
    };

}   // namespace LAMMPS_NS

#endif
#endif
