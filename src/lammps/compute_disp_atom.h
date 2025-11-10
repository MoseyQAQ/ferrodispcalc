/*

LAMMPS plugin for calculate the displacement of atoms relative to the centroid of their neighbors.

Usage:
compute compute-ID group-ID disp/atom neighbor_group neighbor_number every N

*/

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(disp/atom,ComputeDispAtom);
// clang-format on
#else

#ifndef COMPUTE_DISP_ATOM_H
#define COMPUTE_DISP_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

    class ComputeDispAtom : public Compute {
        public:
            ComputeDispAtom(class LAMMPS *, int, char **);
            ~ComputeDispAtom() override;
            void init() override;
            void init_list(int, class NeighList *) override;
            void compute_peratom() override;
        
        private:
            double *distsq;
            int *nearest;
            class NeighList *list;
            int neighbor_number, nmax, maxneigh;
            int nevery;
            char *neighbor_group;
            int jgroup, jgroupit;
        
        void select2(int, int, double *, int *);
    };

}   // namespace LAMMPS_NS

#endif
#endif