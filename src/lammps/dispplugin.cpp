#include "lammpsplugin.h"
#include "version.h"
#include "compute_disp_atom.h"
#include "compute_polar_atom.h"
#include "compute_custom_disp.h"

using namespace  LAMMPS_NS;

static Compute *computedispatom(LAMMPS *lmp, int narg, char **arg) {
    return new ComputeDispAtom(lmp, narg, arg);
}

static Compute *computepolaratom(LAMMPS *lmp, int narg, char **arg) {
    return new ComputePolarAtom(lmp, narg, arg);
}

static Compute *computecustomdisp(LAMMPS *lmp, int narg, char **arg) {
    return new ComputeCustomDisp(lmp, narg, arg);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
    lammpsplugin_t plugin;
    lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

    plugin.version = LAMMPS_VERSION;
    plugin.author = "Denan Li (lidenan@westlake.edu.cn)";

    plugin.style = "compute";
    plugin.name = "disp/atom";
    plugin.info = "compute disp/atom";
    plugin.creator.v2 = (lammpsplugin_factory2 *) &computedispatom;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);

    plugin.style = "compute";
    plugin.name = "polar/atom";
    plugin.info = "compute polar/atom";
    plugin.creator.v2 = (lammpsplugin_factory2 *) &computepolaratom;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);

    plugin.style = "compute";
    plugin.name = "customdisp";
    plugin.info = "compute customdisp";
    plugin.creator.v2 = (lammpsplugin_factory2 *) &computecustomdisp;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);
}