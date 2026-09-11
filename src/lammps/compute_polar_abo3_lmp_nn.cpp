/* ----------------------------------------------------------------------
    Contributors: Denan LI
----------------------------------------------------------------------- */

#include "compute_polar_abo3_lmp_nn.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
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

ComputePolarABO3LmpNN::ComputePolarABO3LmpNN(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), list(nullptr)
{
    if (narg < 9) error->all(FLERR, "Illegal compute polar/abo3/auto command");

    // read A, B, X groups and scalar Born effective charges
    int Agroup = group->find(arg[3]);
    int Bgroup = group->find(arg[5]);
    int Xgroup = group->find(arg[7]);
    if (Agroup == -1 || Bgroup == -1 || Xgroup == -1)
        error->all(FLERR, "Compute polar/abo3/auto: A, B or X group ID does not exist");
    Agroupbit = group->bitmask[Agroup];
    Bgroupbit = group->bitmask[Bgroup];
    Xgroupbit = group->bitmask[Xgroup];
    bec_A = utils::numeric(FLERR, arg[4], false, lmp);
    bec_B = utils::numeric(FLERR, arg[6], false, lmp);
    bec_X = utils::numeric(FLERR, arg[8], false, lmp);
    if (comm->me == 0)
        utils::logmesg(lmp, "Compute polar/abo3/auto: BEC residual (Z_A + Z_B + 3*Z_X) = {:.12g} e\n",
                      bec_A + bec_B + 3*bec_X);
    velocity_flag = 0;

    int iarg = 9;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "vel") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal compute polar/abo3/auto command: vel");
            if (strcmp(arg[iarg + 1], "yes") == 0) {
                velocity_flag = 1;
            } else if (strcmp(arg[iarg + 1], "no") == 0) {
                velocity_flag = 0;
            } else {
                error->all(FLERR, "Illegal compute polar/abo3/auto command: vel must be yes or no");
            }
            iarg += 2;
        } else {
            error->all(FLERR, "Illegal compute polar/abo3/auto command");
        }
    }

    peratom_flag = 1;
    size_peratom_cols = velocity_flag ? 6 : 3;
    nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputePolarABO3LmpNN::~ComputePolarABO3LmpNN()
{
    memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */

void ComputePolarABO3LmpNN::init()
{
    if (force->pair == nullptr)
        error->all(FLERR, "Compute polar/abo3/auto requires a pair style be defined");
    if (strcmp(update->unit_style, "metal") != 0 && strcmp(update->unit_style, "real") != 0)
        error->all(FLERR, "Compute polar/abo3/auto requires units metal or real");
    if (domain->dimension != 3)
        error->all(FLERR, "Compute polar/abo3/auto requires a three-dimensional box");
    if (velocity_flag && !comm->ghost_velocity)
        error->all(FLERR, "Compute polar/abo3/auto with vel yes requires ghost velocities. Use comm_modify vel yes");

    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void ComputePolarABO3LmpNN::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePolarABO3LmpNN::compute_peratom()
{
    invoked_peratom = update->ntimestep;

    if (atom->nmax > nmax) {
        memory->destroy(array_atom);
        nmax = atom->nmax;
        memory->create(array_atom, nmax, size_peratom_cols, "polar/abo3/auto:array_atom");
    }
    for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
        for (int k = 0; k < size_peratom_cols; k++) array_atom[i][k] = 0.0;
    }

    // count owned B atoms only, then reduce across MPI ranks
    int *mask = atom->mask;
    bigint num_B_local = 0, num_B;
    for (int i = 0; i < atom->nlocal; i++) {
        if (mask[i] & Bgroupbit) num_B_local++;
    }
    MPI_Allreduce(&num_B_local, &num_B, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    if (num_B == 0) error->all(FLERR, "Compute polar/abo3/auto: B group is empty");
    double mean_vol = domain->xprd * domain->yprd * domain->zprd / num_B;
    double polar_scale = 16.02176634 / mean_vol; // e/Angstrom^2 -> C/m^2

    neighbor->build_one(list);
    double **x = atom->x;
    double **v = velocity_flag ? atom->v : nullptr;
    double cutsq = force->pair->cutforce * force->pair->cutforce;
    bigint insufficient = 0;

    for (int ii = 0; ii < list->inum; ii++) {
        int i = list->ilist[ii];
        if (!(mask[i] & groupbit) || !(mask[i] & Bgroupbit)) continue;

        nearest_A.clear();
        nearest_X.clear();
        int *jlist = list->firstneigh[i];
        for (int jj = 0; jj < list->numneigh[i]; jj++) {
            int j = jlist[jj] & NEIGHMASK;
            if (!(mask[j] & (Agroupbit | Xgroupbit))) continue;
            double dx = x[i][0] - x[j][0];
            double dy = x[i][1] - x[j][1];
            double dz = x[i][2] - x[j][2];
            double rsq = dx*dx + dy*dy + dz*dz;
            if (rsq >= cutsq) continue;
            if (mask[j] & Agroupbit) nearest_A.emplace_back(rsq, j);
            if (mask[j] & Xgroupbit) nearest_X.emplace_back(rsq, j);
        }

        if (nearest_A.size() < 8 || nearest_X.size() < 6) {
            insufficient++;
            continue;
        }

        // ties between images of one atom are ordered by image coordinates
        auto closer = [this, x](const std::pair<double, int> &a,
                               const std::pair<double, int> &b) {
            if (a.first != b.first) return a.first < b.first;
            if (atom->tag[a.second] != atom->tag[b.second])
                return atom->tag[a.second] < atom->tag[b.second];
            for (int k = 0; k < 3; k++) {
                if (x[a.second][k] != x[b.second][k]) return x[a.second][k] < x[b.second][k];
            }
            return false;
        };
        std::partial_sort(nearest_A.begin(), nearest_A.begin() + 8, nearest_A.end(), closer);
        std::partial_sort(nearest_X.begin(), nearest_X.begin() + 6, nearest_X.end(), closer);

        for (int k = 0; k < 3; k++) {
            // Relative coordinates reduce cancellation; retain the supplied net BEC.
            double polar = (bec_A + bec_B + 3*bec_X) * x[i][k];
            for (int n = 0; n < 8; n++)
                polar += bec_A * (x[nearest_A[n].second][k] - x[i][k]) / 8;
            for (int n = 0; n < 6; n++)
                polar += bec_X * (x[nearest_X[n].second][k] - x[i][k]) / 2;
            array_atom[i][k] = polar * polar_scale;
            if (velocity_flag) {
                double velocity = bec_B * v[i][k];
                for (int n = 0; n < 8; n++)
                    velocity += bec_A * v[nearest_A[n].second][k] / 8;
                for (int n = 0; n < 6; n++)
                    velocity += bec_X * v[nearest_X[n].second][k] / 2;
                array_atom[i][k + 3] = velocity / 5;
            }
        }
    }

    bigint insufficient_all;
    MPI_Allreduce(&insufficient, &insufficient_all, 1, MPI_LMP_BIGINT, MPI_SUM, world);
    if (insufficient_all && comm->me == 0)
        error->warning(FLERR, "Compute polar/abo3/auto: {} B atoms have fewer than 8 A or 6 X neighbors; output set to zero",
                       insufficient_all);
}
