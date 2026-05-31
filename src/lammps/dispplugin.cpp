#include "lammpsplugin.h"
#include "version.h"
#include "compute_custom_disp.h"

using namespace LAMMPS_NS;

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
    plugin.info = "compute disp/atom - file-based displacement and displacement velocity";
    plugin.creator.v2 = (lammpsplugin_factory2 *) &computecustomdisp;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);
}
