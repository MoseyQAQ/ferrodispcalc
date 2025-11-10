/*

LAMMPS plugin for calculate the local polarization of Perovskite ABX3.

Usage:
compute compute-ID all polar/atom GROUP-A BEC-A GROUP-B BEC-B GROUP-X BEC-X every N

BEC-B: born effective charge of B cation
BEC-A: born effective charge of A cation
BEC-X: born effective charge of X anion
every N: calculate every N steps, default is 1
*/

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(polar/atom,ComputePolarAtom);
// clang-format on
#else

#ifndef COMPUTE_POLAR_ATOM_H
#define COMPUTE_POLAR_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

    class ComputePolarAtom : public Compute {
        public:
            ComputePolarAtom(class LAMMPS *, int, char **);
            ~ComputePolarAtom() override;
            void init() override;
            void init_list(int, class NeighList *) override;
            void compute_peratom() override;
        
        private:
            double *distsq_A, *distsq_X;
            int *nearest_A, *nearest_X;
            class NeighList *list;
            int nmax, maxneigh;
            int nevery;
            int Agroup, Agroupit;
            int Bgroup, Bgroupit;
            int Xgroup, Xgroupit;
            double bec_B, bec_A, bec_X;
            int num_B;
            
        
        void select2(int, int, double *, int *);
    };

}   // namespace LAMMPS_NS

#endif
#endif