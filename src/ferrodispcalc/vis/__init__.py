import matplotlib.pyplot as plt
from .line_plot import line_profile
from .plane_plot import plane_profile
from .space_plot import space_profile
from .grid import grid_data

__all__ = ['line_profile', 'plane_profile', 'space_profile', 'grid_data', 'use_science_style']

_science_style_enabled = False


def use_science_style(enable: bool = True) -> None:
    """Enable or disable the SciencePlots style globally.

    Parameters
    ----------
    enable : bool
        *True* to activate ``['science', 'no-latex']``,
        *False* to revert to matplotlib defaults.
    """
    global _science_style_enabled
    if enable:
        import scienceplots  # noqa: F401
        plt.style.use(['science', 'no-latex'])
    else:
        plt.style.use('default')
    _science_style_enabled = enable