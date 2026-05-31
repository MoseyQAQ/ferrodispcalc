/* ----------------------------------------------------------------------
    Contributors: Denan LI
----------------------------------------------------------------------- */

#include "compute_custom_disp.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "memory.h"
#include "update.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

namespace {

template <typename DomainType>
auto minimum_image_compat(DomainType *domain, double &dx, double &dy, double &dz, int)
    -> decltype(domain->minimum_image(FLERR, dx, dy, dz), void())
{
    domain->minimum_image(FLERR, dx, dy, dz);
}

template <typename DomainType>
void minimum_image_compat(DomainType *domain, double &dx, double &dy, double &dz, long)
{
    domain->minimum_image(dx, dy, dz);
}

} // namespace

/* ---------------------------------------------------------------------- */

ComputeCustomDisp::ComputeCustomDisp(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
    if (narg < 3) error->all(FLERR, "Illegal compute disp/atom command");

    // initialize the parameters
    nnfile = utils::strdup("nn.dat");
    velocity_flag = 0;

    // read the parameters
    int iarg = 3;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "nnfile") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal compute disp/atom command: nnfile");
            delete[] nnfile;
            nnfile = utils::strdup(arg[iarg + 1]);
            iarg += 2;
        } else if (strcmp(arg[iarg], "vel") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal compute disp/atom command: vel");
            if (strcmp(arg[iarg + 1], "yes") == 0) {
                velocity_flag = 1;
            } else if (strcmp(arg[iarg + 1], "no") == 0) {
                velocity_flag = 0;
            } else {
                error->all(FLERR, "Illegal compute disp/atom command: vel must be yes or no");
            }
            iarg += 2;
        } else {
            error->all(FLERR, "Illegal compute disp/atom command");
        }
    }

    // read the neighbor list file
    peratom_flag = 1;
    size_peratom_cols = velocity_flag ? 6 : 3;
    nmax = 0;

    read_file();
}
/* ---------------------------------------------------------------------- */

void ComputeCustomDisp::init()
{
    if (atom->map_style == Atom::MAP_NONE)
        error->all(FLERR, "Compute disp/atom requires an atom map. Use atom_modify map array");

    if (velocity_flag && !comm->ghost_velocity)
        error->all(FLERR, "Compute disp/atom with vel yes requires ghost velocities. Use comm_modify vel yes");
}

/* ---------------------------------------------------------------------- */

ComputeCustomDisp::~ComputeCustomDisp()
{
    memory->destroy(array_atom);
    delete[] nnfile;

    // free the memory of central_id and neighbor_id
    central_id.clear();
    neighbor_id.clear();
}

/* ---------------------------------------------------------------------- */

void ComputeCustomDisp::compute_peratom()
{
    invoked_peratom = update->ntimestep;

    // check number of atoms
    if (atom->nmax > nmax) {
        memory->destroy(array_atom);
        nmax = atom->nmax;
        memory->create(array_atom, nmax, size_peratom_cols, "disp/atom:array_atom");
    }

    // reset the array_atom to 0
    for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
        for (int k = 0; k < size_peratom_cols; k++) {
            array_atom[i][k] = 0.0;
        }
    }

    // assign coordinates and velocities
    double **x = atom->x;
    double **v = velocity_flag ? atom->v : nullptr;
    int *mask = atom->mask;

    // loop over all central atoms
    size_t i,j;
    for(i=0; i < central_id.size(); i++) {
        int central_global_id = central_id[i];
        int central_local_id = atom->map(central_global_id);

        if (central_local_id < 0 || central_local_id >= atom->nlocal) continue;
        if (!(mask[central_local_id] & groupbit)) continue;

        // loop over all neighbors of the central atom
        double dx=0, dy=0, dz=0;
        double dvx=0, dvy=0, dvz=0;
        double tmpx, tmpy, tmpz;
        int neighbor_count = 0;
        for(j = 0; j < neighbor_id[i].size(); j++) {
            int neighbor_global_id = neighbor_id[i][j];
            if (neighbor_global_id == -1) {
                continue;
            }
            int neighbor_local_id = atom->map(neighbor_global_id);

            if (neighbor_local_id < 0) {
                continue;
            }

            tmpx = x[central_local_id][0] - x[neighbor_local_id][0];
            tmpy = x[central_local_id][1] - x[neighbor_local_id][1];
            tmpz = x[central_local_id][2] - x[neighbor_local_id][2];
            minimum_image_compat(domain, tmpx, tmpy, tmpz, 0);
            dx += tmpx;
            dy += tmpy;
            dz += tmpz;
            if (velocity_flag) {
                dvx += v[central_local_id][0] - v[neighbor_local_id][0];
                dvy += v[central_local_id][1] - v[neighbor_local_id][1];
                dvz += v[central_local_id][2] - v[neighbor_local_id][2];
            }
            neighbor_count++;
        }
        if (neighbor_count == 0) continue;

        dx /= neighbor_count;
        dy /= neighbor_count;
        dz /= neighbor_count;
        array_atom[central_local_id][0] = dx;
        array_atom[central_local_id][1] = dy;
        array_atom[central_local_id][2] = dz;
        if (velocity_flag) {
            dvx /= neighbor_count;
            dvy /= neighbor_count;
            dvz /= neighbor_count;
            array_atom[central_local_id][3] = dvx;
            array_atom[central_local_id][4] = dvy;
            array_atom[central_local_id][5] = dvz;
        }
    }
}

/* ---------------------------------------------------------------------- */

void ComputeCustomDisp::read_file()
{
    // Open the neighbor list file, and issue an error if it cannot be opened
    std::ifstream file(nnfile);
    if (!file.is_open()) {
        error->all(FLERR, "compute disp/atom: cannot open neighbor list file '{}'", nnfile);
    }

    //
    // The ID provided by user should be 1-based. LAMMPS atom IDs are also 1-based.
    //

    // Read the file line by line
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines
        if (line.empty()) continue;

        // Create a string stream from the line
        std::istringstream iss(line);

        // Read the central atom ID
        int centralAtomID;
        if (!(iss >> centralAtomID)) {
            error->all(FLERR, "Error reading central atom ID in neighbor list file");
        }
        central_id.push_back(centralAtomID);

        // Read the neighbor atom IDs
        std::vector<int> neighborIDs;
        int neighborID;
        while (iss >> neighborID) {
            neighborIDs.push_back(neighborID);
        }

        // Check if neighbor IDs were read; if not, issue an error
        if (neighborIDs.empty()) {
            error->all(FLERR, "No neighbor IDs found for a central atom in neighbor list file");
        }

        neighbor_id.push_back(neighborIDs);
    }

    file.close();
}
