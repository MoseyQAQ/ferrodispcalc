/* ----------------------------------------------------------------------
    Contributors: Denan LI
----------------------------------------------------------------------- */

#include "compute_custom_disp.h"

#include "atom.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "domain.h"
#include <cstring>
#include <utility>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeCustomDisp::ComputeCustomDisp(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
    if (narg < 3) error->all(FLERR, "Illegal compute custom_disp command");

    // initialize the parameters
    nnfile = utils::strdup("nn.dat");

    // read the parameters
    int iarg = 3;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "nnfile") == 0) {
            if (iarg + 2 > narg) error->all(FLERR, "Illegal compute custom_disp command: nnfile");
            strcpy(nnfile, arg[iarg + 1]);
            iarg += 2;
        } else {
            error->all(FLERR, "Illegal compute custom_disp command");
        }
    }

    // read the neighbor list file
    peratom_flag = 1;
    size_peratom_cols = 3;
    nmax = 0;

    // read the neighbor list file, and initialize the atom map
    read_file();
    atom->map_init();
}
/* ---------------------------------------------------------------------- */

void ComputeCustomDisp::init()
{
    // to be implemented
}

/* ---------------------------------------------------------------------- */

ComputeCustomDisp::~ComputeCustomDisp()
{
    std::cout << "Destroying the ComputeCustomDisp" << std::endl;
    memory->destroy(array_atom);
    atom->map_delete();

    // free the memory of central_id and neighbor_id
    central_id.clear();
    neighbor_id.clear();
}

/* ---------------------------------------------------------------------- */

void ComputeCustomDisp::compute_peratom() 
{
    // update the atom map
    atom->map_set();

    // check number of atoms
    if (atom->nmax > nmax) {
        memory->destroy(array_atom);
        nmax = atom->nmax;
        memory->create(array_atom, nmax, size_peratom_cols, "custom_disp:array_atom");
    }

    // reset the array_atom to 0
    for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
        array_atom[i][0] = 0.0;
        array_atom[i][1] = 0.0;
        array_atom[i][2] = 0.0;
    }

    // assign the coord
    double **x = atom->x;
    
    // loop over all central atoms
    size_t i,j;
    for(i=0; i < central_id.size(); i++) {
        int central_global_id = central_id[i];
        int central_local_id = atom->map(central_global_id);

        if (central_local_id < 0 || central_local_id > atom->nlocal) continue; // skip if the central atom isn't in the processor

        // loop over all neighbors of the central atom
        double dx=0, dy=0, dz=0;
        double tmpx, tmpy, tmpz;
        int neighbor_count = 0;
        for(j = 0; j < neighbor_id[i].size(); j++) {
            int neighbor_global_id = neighbor_id[i][j];
            int neighbor_local_id = atom->map(neighbor_global_id);

            if (neighbor_local_id < 0) {
                error->warning(FLERR, "Cannot find the local index of the neighbor atom");
                continue;
            }

            tmpx = x[central_local_id][0] - x[neighbor_local_id][0];
            tmpy = x[central_local_id][1] - x[neighbor_local_id][1];
            tmpz = x[central_local_id][2] - x[neighbor_local_id][2];
            domain->minimum_image(tmpx, tmpy, tmpz);
            dx += tmpx;
            dy += tmpy;
            dz += tmpz;
            neighbor_count++;
        }
        dx /= neighbor_count;
        dy /= neighbor_count;
        dz /= neighbor_count;

        array_atom[central_local_id][0] = dx;
        array_atom[central_local_id][1] = dy;
        array_atom[central_local_id][2] = dz;
    }
}

/* ---------------------------------------------------------------------- */

void ComputeCustomDisp::read_file()
{
    // Open the neighbor list file, and issue an error if it cannot be opened
    std::ifstream file(nnfile);
    if (!file.is_open()) {
        error->all(FLERR, "Cannot open the neighbor list file");
    }

    //
    //The ID provided by user should be 1-based. The LAMMPS is also 1-based.
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