/* ----------------------------------------------------------------------
    Contributors: Denan LI
----------------------------------------------------------------------- */

#include "compute_polar_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "group.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"
#include "domain.h"
#include <cstring>
#include <utility>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputePolarAtom::ComputePolarAtom(LAMMPS *lmp, int narg, char **arg):
    Compute(lmp, narg, arg), distsq_A(nullptr), distsq_X(nullptr), nearest_A(nullptr), nearest_X(nullptr)
{   
    // check the number of arguments
    if (narg < 9) error->all(FLERR, "Illegal compute polar/atom command");

    // set parameters for A, B, X cations and anions
    char *group_str = utils::strdup(arg[3]);
    Agroup = group->find(group_str);
    if (Agroup == -1)
        error->all(FLERR, "Compute polar/atom: A group ID does not exist");
    Agroupit = group->bitmask[Agroup];
    bec_A = utils::numeric(FLERR, arg[4], false, lmp);

    group_str = utils::strdup(arg[5]);
    Bgroup = group->find(group_str);
    if (Bgroup == -1)
        error->all(FLERR, "Compute polar/atom: B group ID does not exist");
    Bgroupit = group->bitmask[Bgroup];
    bec_B = utils::numeric(FLERR, arg[6], false, lmp);

    group_str = utils::strdup(arg[7]);
    Xgroup = group->find(group_str);
    if (Xgroup == -1)
        error->all(FLERR, "Compute polar/atom: X group ID does not exist");
    Xgroupit = group->bitmask[Xgroup];
    bec_X = utils::numeric(FLERR, arg[8], false, lmp);
    
    // set nevery
    nevery = 1;
    int iarg = 9;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "every") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal compute polar/atom command");
            nevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else {
            error->all(FLERR, "Illegal compute polar/atom command");
        }
    }

    // set the flag
    peratom_flag = 1;
    size_peratom_cols = 3;
    nmax = 0;
    maxneigh = 0;
}

/* ---------------------------------------------------------------------- */

void ComputePolarAtom::init()
{
    if (force->pair == nullptr)
        error->all(FLERR, "compute disp/atom requires a pair style be defined");

    // Request a full, occasional neighbor list
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);

    // get the number of B group atoms
    int i;
    num_B = 0;
    for (i = 0; i < atom->natoms; i++) {
        if (atom->mask[i] & Bgroupit) {
            num_B++;
        }
    }
}

/* ---------------------------------------------------------------------- */

ComputePolarAtom::~ComputePolarAtom()
{
    memory->destroy(distsq_A);
    memory->destroy(nearest_A);
    memory->destroy(distsq_X);
    memory->destroy(nearest_X);
}

/* ---------------------------------------------------------------------- */

void ComputePolarAtom::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePolarAtom::compute_peratom()
{   
    int i, j, ii, jj;
    double b_x, b_y, b_z;
    double delx, dely, delz, rsq;
    int inum, jnum, *ilist, *jlist, *numneigh, **firstneigh; // for neighbor list

    // allocate memory for array_atom
    if (atom->nmax > nmax) {
        nmax = atom->nmax;
        memory->create(array_atom, nmax, size_peratom_cols, "polar/atom:polarization");
    }

    // if not every, set the array to 0, return
    if (update->ntimestep % nevery) {
        for (i = 0; i < atom->natoms; i++) {
            array_atom[i][0] = 0.0;
            array_atom[i][1] = 0.0;
            array_atom[i][2] = 0.0;
        }
        return;
    }

    // calculate the mean volume of each u.c., which is total volume divided by the number of u.c.
    // the number of u.c. is the number of B group atoms
    double total_vol = domain->xprd * domain->yprd * domain->zprd;
    double mean_vol = total_vol / num_B;


    // invoke full neighbor list
    neighbor->build_one(list);
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // assign the coord and type
    double **x = atom->x;
    int *mask = atom->mask;
    double cutsq = force->pair->cutforce * force->pair->cutforce;

    // loop over all atoms
    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii]; 
        if (mask[i] & Bgroupit) { // only handle the B site atom
            b_x = x[i][0];
            b_y = x[i][1];
            b_z = x[i][2];
            jlist = firstneigh[i]; // get neighbor atoms' local index
            jnum = numneigh[i]; // get the number of neighbors

            // ensure the array is big enough
            if (jnum > maxneigh) {
                    memory->destroy(distsq_A);
                    memory->destroy(nearest_A);
                    memory->destroy(distsq_X);
                    memory->destroy(nearest_X);
                    maxneigh = jnum;
                    memory->create(distsq_A,maxneigh,"disp/atom:distsq_A");
                    memory->create(nearest_A,maxneigh,"disp/atom:nearest_A");
                    memory->create(distsq_X,maxneigh,"disp/atom:distsq_X");
                    memory->create(nearest_X,maxneigh,"disp/atom:nearest_X");
                }

            // loop over neighbor atoms
            int n_a = 0, n_x = 0;
            for (jj = 0; jj < jnum; jj++) {
                j = jlist[jj];
                j &= NEIGHMASK;

                // If the neighbor is in A group
                if (mask[j] & Agroupit) {
                    delx = b_x - x[j][0];
                    dely = b_y - x[j][1];
                    delz = b_z - x[j][2];
                    rsq = delx*delx + dely*dely + delz*delz;
                    if (rsq < cutsq) {
                        distsq_A[n_a] = rsq;
                        nearest_A[n_a++] = j;
                    }
                }

                // If the neighbor is in X group
                if (mask[j] & Xgroupit) {
                    delx = b_x - x[j][0];
                    dely = b_y - x[j][1];
                    delz = b_z - x[j][2];
                    rsq = delx*delx + dely*dely + delz*delz;
                    if (rsq < cutsq) {
                        distsq_X[n_x] = rsq;
                        nearest_X[n_x++] = j;
                    }
                }
            }

            // check the number of neighbor
            // for A, it should be 8;
            // for X, it should be 6;
            if (n_a < 8 || n_x < 6) {
                error->warning(FLERR, "Insufficient neighbors for compute polar/atom");
                array_atom[i][0] = 0.0;
                array_atom[i][1] = 0.0;
                array_atom[i][2] = 0.0;
                continue;
            }

            // store 8 nearest neighbors for A, 6 nearest neighbors for X
            select2(8, n_a, distsq_A, nearest_A);
            select2(6, n_x, distsq_X, nearest_X);

            // calculate the local polarization
            // fomula: P = BEC_B * R_B + (BEC_A * R_A) / 8 + BEC_X * R_X / 2
            double px=0, py=0, pz=0;
            px += bec_B * b_x;
            py += bec_B * b_y;
            pz += bec_B * b_z;

            double pa_x = 0, pa_y = 0, pa_z = 0;
            for (j = 0; j < 8; j++) {
                jj = nearest_A[j];
                pa_x += bec_A * x[jj][0];
                pa_y += bec_A * x[jj][1];
                pa_z += bec_A * x[jj][2];
            }
            px += pa_x / 8;
            py += pa_y / 8;
            pz += pa_z / 8;

            double px_x = 0, px_y = 0, px_z = 0;
            for (j = 0; j < 6; j++) {
                jj = nearest_X[j];
                px_x += bec_X * x[jj][0];
                px_y += bec_X * x[jj][1];
                px_z += bec_X * x[jj][2];
            }
            px += px_x / 2;
            py += px_y / 2;
            pz += px_z / 2;

            px /= mean_vol;
            py /= mean_vol;
            pz /= mean_vol;

            // unit conversion, from e/Å^3 to C/m^2
            px *= 1.602176E-19 * 1.0E-10 * 1.0E30;
            py *= 1.602176E-19 * 1.0E-10 * 1.0E30;
            pz *= 1.602176E-19 * 1.0E-10 * 1.0E30;

            array_atom[i][0] = px;
            array_atom[i][1] = py;
            array_atom[i][2] = pz;
            } else {
                array_atom[i][0] = 0.0;
                array_atom[i][1] = 0.0;
                array_atom[i][2] = 0.0;
            }
        
    } 
}

/* ---------------------------------------------------------------------- */

void ComputePolarAtom::select2(int k, int n, double *arr, int *iarr)
{
  int i, ir, j, l, mid, ia;
  double a;

  arr--;
  iarr--;
  l = 1;
  ir = n;
  while (true) {
    if (ir <= l + 1) {
      if (ir == l + 1 && arr[ir] < arr[l]) {
        std::swap(arr[l], arr[ir]);
        std::swap(iarr[l], iarr[ir]);
      }
      return;
    } else {
      mid = (l + ir) >> 1;
      std::swap(arr[mid], arr[l + 1]);
      std::swap(iarr[mid], iarr[l + 1]);
      if (arr[l] > arr[ir]) {
        std::swap(arr[l], arr[ir]);
        std::swap(iarr[l], iarr[ir]);
      }
      if (arr[l + 1] > arr[ir]) {
        std::swap(arr[l + 1], arr[ir]);
        std::swap(iarr[l + 1], iarr[ir]);
      }
      if (arr[l] > arr[l + 1]) {
        std::swap(arr[l], arr[l + 1]);
        std::swap(iarr[l], iarr[l + 1]);
      }
      i = l + 1;
      j = ir;
      a = arr[l + 1];
      ia = iarr[l + 1];
      while (true) {
        do i++;
        while (arr[i] < a);
        do j--;
        while (arr[j] > a);
        if (j < i) break;
        std::swap(arr[i], arr[j]);
        std::swap(iarr[i], iarr[j]);
      }
      arr[l + 1] = arr[j];
      arr[j] = a;
      iarr[l + 1] = iarr[j];
      iarr[j] = ia;
      if (j >= k) ir = j - 1;
      if (j <= k) l = i;
    }
  }
}