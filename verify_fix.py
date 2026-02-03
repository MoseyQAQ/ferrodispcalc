import sys
import os
from unittest.mock import MagicMock
import builtins

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Simulate pytest environment
sys.modules['pytest'] = MagicMock()
sys.modules['scienceplots'] = MagicMock()
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# Mock import to force ImportError for vispy
real_import = builtins.__import__
def mock_import(name, *args, **kwargs):
    if name.startswith('vispy'):
        raise ImportError(f"No module named '{name}' (simulated)")
    return real_import(name, *args, **kwargs)

builtins.__import__ = mock_import

print("Attempting to import ferrodispcalc.vis.space_plot with mocked pytest and missing vispy...")
try:
    import ferrodispcalc.vis.space_plot as sp
    print("Import successful!")
    print(f"scene type: {type(sp.scene)}")
    print(f"SpaceProfileCanvas bases: {sp.SpaceProfileCanvas.__bases__}")
except ImportError as e:
    print(f"Import FAILED: {e}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Other error: {e}")
