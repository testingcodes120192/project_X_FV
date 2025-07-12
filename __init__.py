# __init__.py (in main directory)
"""
FV Heat Diffusion Simulator

A finite volume solver for 2D heat diffusion with optional chemical reactions
and adaptive mesh refinement.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make imports cleaner
from .mesh import FVMesh
from .solver import FVHeatSolver
from .postprocessor import FVPostProcessor
from .weno import WENOReconstructor
from .initial_conditions import get_fv_initial_condition
from .simple_amr import SimpleAMR