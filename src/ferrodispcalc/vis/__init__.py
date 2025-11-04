import scienceplots
import matplotlib.pyplot as plt
from .line_plot import line_profile
from .plane_plot import plane_profile
from .space_plot import space_profile
plt.style.use(['science', 'no-latex'])

__all__ = ['line_profile', 'plane_profile', 'space_profile']