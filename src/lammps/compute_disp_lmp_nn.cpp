/* ----------------------------------------------------------------------
    Contributors: Denan LI
----------------------------------------------------------------------- */

#include "compute_disp_lmp_nn.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <algorithm>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeDispLmpNN::ComputeDispLmpNN(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), list(nullptr)
{
    if (narg < 5) error->all(FLERR, "Illegal compute disp/atom/auto command");

    // read the neighbor group and number
    int jgroup = group->find(arg[3]);
    if (jgroup == -1)
        error->all(FLERR, "Compute disp/atom/auto: neighbor group ID does not exist");
    neighbor_groupbit = group->bitmask[jgroup];
    neighbor_number = utils::inumeric(FLERR, arg[4], false, lmp);
    if (neighbor_number <= 0)
        error->all(FLERR, "Compute disp/atom/auto: neighbor number must be positive");
    velocity_flag = 0;

    // read the optional parameters
    int iarg = 5;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "vel") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal compute disp/atom/auto command: vel");
            if (strcmp(arg[iarg + 1], "yes") == 0) {
                velocity_flag = 1;
            } else if (strcmp(arg[iarg + 1], "no") == 0) {
                velocity_flag = 0;
            } else {
                error->all(FLERR, "Illegal compute disp/atom/auto command: vel must be yes or no");
            }
            iarg += 2;
        } else {
            error->all(FLERR, "Illegal compute disp/atom/auto command");
        }
    }

    peratom_flag = 1;
    size_peratom_cols = velocity_flag ? 6 : 3;
    nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeDispLmpNN::~ComputeDispLmpNN()
{
    memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeDispLmpNN::init()
{
    if (force->pair == nullptr)
        error->all(FLERR, "Compute disp/atom/auto requires a pair style be defined");

    if (velocity_flag && !comm->ghost_velocity)
        error->all(FLERR, "Compute disp/atom/auto with vel yes requires ghost velocities. Use comm_modify vel yes");

    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void ComputeDispLmpNN::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeDispLmpNN::compute_peratom()
{
    invoked_peratom = update->ntimestep;

    if (atom->nmax > nmax) {
        memory->destroy(array_atom);
        nmax = atom->nmax;
        memory->create(array_atom, nmax, size_peratom_cols, "disp/atom/auto:array_atom");
    }

    for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
        for (int k = 0; k < size_peratom_cols; k++) {
            array_atom[i][k] = 0.0;
        }
    }

    neighbor->build_one(list);

    double **x = atom->x;
    double **v = velocity_flag ? atom->v : nullptr;
    int *mask = atom->mask;
    double cutsq = force->pair->cutforce * force->pair->cutforce;
    bigint insufficient = 0;

    for (int ii = 0; ii < list->inum; ii++) {
        int i = list->ilist[ii];
        if (!(mask[i] & groupbit)) continue;

        // select candidate neighbors using their periodic ghost coordinates
        nearest.clear();
        int *jlist = list->firstneigh[i];
        for (int jj = 0; jj < list->numneigh[i]; jj++) {
            int j = jlist[jj] & NEIGHMASK;
            if (!(mask[j] & neighbor_groupbit)) continue;

            double dx = x[i][0] - x[j][0];
            double dy = x[i][1] - x[j][1];
            double dz = x[i][2] - x[j][2];
            double rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < cutsq) nearest.emplace_back(rsq, j);
        }

        if (nearest.size() < static_cast<size_t>(neighbor_number)) {
            insufficient++;
            continue;
        }

        // use atom IDs to break distance ties consistently across MPI ranks
        auto closer = [this](const std::pair<double, int> &a,
                             const std::pair<double, int> &b) {
            if (a.first != b.first) return a.first < b.first;
            return atom->tag[a.second] < atom->tag[b.second];
        };
        std::partial_sort(nearest.begin(), nearest.begin() + neighbor_number,
                          nearest.end(), closer);

        double dx=0, dy=0, dz=0;
        double dvx=0, dvy=0, dvz=0;
        for (int k = 0; k < neighbor_number; k++) {
            int j = nearest[k].second;
            dx += x[i][0] - x[j][0];
            dy += x[i][1] - x[j][1];
            dz += x[i][2] - x[j][2];
            if (velocity_flag) {
                dvx += v[i][0] - v[j][0];
                dvy += v[i][1] - v[j][1];
                dvz += v[i][2] - v[j][2];
            }
        }
        array_atom[i][0] = dx / neighbor_number;
        array_atom[i][1] = dy / neighbor_number;
        array_atom[i][2] = dz / neighbor_number;
        if (velocity_flag) {
            array_atom[i][3] = dvx / neighbor_number;
            array_atom[i][4] = dvy / neighbor_number;
            array_atom[i][5] = dvz / neighbor_number;
        }
    }

    bigint insufficient_all;
    MPI_Allreduce(&insufficient, &insufficient_all, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    if (insufficient_all && comm->me == 0)
        error->warning(FLERR, "Compute disp/atom/auto: {} atoms have fewer than {} neighbors; output set to zero",
                       insufficient_all, neighbor_number);
}
