/* ----------------------------------------------------------------------
    Contributors: Denan LI
----------------------------------------------------------------------- */

#include "compute_disp_atom.h"

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
#include <cstring>
#include <utility>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeDispAtom::ComputeDispAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), distsq(nullptr), nearest(nullptr)
{
    if (narg < 5) error->all(FLERR, "Illegal compute disp/atom command");

    // set neighbor group mask
    neighbor_group = utils::strdup(arg[3]);
    jgroup = group->find(neighbor_group);
    if (jgroup == -1)
        error->all(FLERR, "Compute disp/atom: neighbor group ID does not exist");
    jgroupit = group->bitmask[jgroup];

    // set neighbor number and every
    neighbor_number = utils::inumeric(FLERR, arg[4], false, lmp);
    nevery = 1;

    int iarg = 5;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "every") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal compute disp/atom command");
            nevery = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else 
        error->all(FLERR, "Illegal compute disp/atom command");
    }

    peratom_flag = 1;
    size_peratom_cols = 3;
    nmax = 0;
    maxneigh = 0;
}

/* ---------------------------------------------------------------------- */

ComputeDispAtom::~ComputeDispAtom()
{
    memory->destroy(distsq);
    memory->destroy(nearest);
}

/* ---------------------------------------------------------------------- */

void ComputeDispAtom::init()
{
    if (force->pair == nullptr)
        error->all(FLERR, "compute disp/atom requires a pair style be defined");

    // Request a full, occasional neighbor list
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void ComputeDispAtom::init_list(int /*id*/, NeighList *ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeDispAtom::compute_peratom()
{
    int i, j, ii, jj, n, inum, jnum;
    double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
    int *ilist, *jlist, *numneigh, **firstneigh;

    if (atom->nmax > nmax) {
        memory->destroy(array_atom);
        nmax = atom->nmax;
        memory->create(array_atom, nmax, size_peratom_cols, "disp/atom:centro");
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
        i = ilist[ii];  // store the local indice of ii atom
        if (mask[i] & groupbit) {  // only handle the atom in group 
            xtmp = x[i][0];
            ytmp = x[i][1];
            ztmp = x[i][2];
            jlist = firstneigh[i]; // get neighbor atoms' local index
            jnum = numneigh[i];    // get number of neighbors

            //ensure distsq and nearst arrays are long enough
            if (jnum > maxneigh) {
                memory->destroy(distsq);
                memory->destroy(nearest);
                maxneigh = jnum;
                memory->create(distsq,maxneigh,"disp/atom:distsq");
                memory->create(nearest,maxneigh,"disp/atom:nearest");
            }


            // loop over neighbor atoms
            n = 0;
            for(jj = 0; jj < jnum; jj++) {
                j = jlist[jj]; // get neighbor inddex
                j &= NEIGHMASK; 

                if (!(mask[j] & jgroupit)) continue; // skip if the neighbor isn't in neighbor_group

                delx = xtmp - x[j][0];
                dely = ytmp - x[j][1];
                delz = ztmp - x[j][2];
                rsq = delx*delx + dely*dely + delz*delz;
                if (rsq < cutsq) {
                    distsq[n] = rsq;
                    nearest[n++] = j;
                }

            }

            // check the number of neighbor 
            if (n < neighbor_number) {
                error->warning(FLERR, "Insufficient neighbors for comput disp/atom");
                array_atom[i][0] = 0.0;
                array_atom[i][1] = 0.0;
                array_atom[i][2] = 0.0;
                continue;
            }

            // store neighbor_number nearest neighbors 
            select2(neighbor_number, n, distsq, nearest);

            // calculate the displacement vector
            double dx=0, dy=0, dz=0;
            for (j = 0; j < neighbor_number; j++) {
              jj = nearest[j];
              delx = xtmp - x[jj][0];
              dely = ytmp - x[jj][1];
              delz = ztmp - x[jj][2];
              dx += delx;
              dy += dely;
              dz += delz;
            }
            array_atom[i][0] = dx / neighbor_number;
            array_atom[i][1] = dy / neighbor_number;
            array_atom[i][2] = dz / neighbor_number;
        } else {
          array_atom[i][0]=0.0; 
          array_atom[i][1]=0.0;
          array_atom[i][2]=0.0;
        }
    }

}

/* ---------------------------------------------------------------------- */

void ComputeDispAtom::select2(int k, int n, double *arr, int *iarr)
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