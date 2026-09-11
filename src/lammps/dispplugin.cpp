#include "lammpsplugin.h"
#include "version.h"
#include "compute_disp_custom_nn.h"
#include "compute_disp_lmp_nn.h"
#include "compute_polar_abo3_lmp_nn.h"

using namespace LAMMPS_NS;

static Compute *computedispcustomnn(LAMMPS *lmp, int narg, char **arg) {
    return new ComputeDispCustomNN(lmp, narg, arg);
}

static Compute *computedisplmpnn(LAMMPS *lmp, int narg, char **arg) {
    return new ComputeDispLmpNN(lmp, narg, arg);
}

static Compute *computepolarabo3lmpnn(LAMMPS *lmp, int narg, char **arg) {
    return new ComputePolarABO3LmpNN(lmp, narg, arg);
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
    plugin.creator.v2 = (lammpsplugin_factory2 *) &computedispcustomnn;
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);

    plugin.name = "disp/atom/auto";
    plugin.info = "compute disp/atom/auto - nearest-neighbor displacement and displacement velocity";
    plugin.creator.v2 = (lammpsplugin_factory2 *) &computedisplmpnn;
    (*register_plugin)(&plugin, lmp);

    plugin.name = "polar/abo3/auto";
    plugin.info = "compute polar/abo3/auto - B-centered ABO3 polarization and BEC-weighted velocity";
    plugin.creator.v2 = (lammpsplugin_factory2 *) &computepolarabo3lmpnn;
    (*register_plugin)(&plugin, lmp);
}
