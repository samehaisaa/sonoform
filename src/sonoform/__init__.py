"""sonoform: every shape has a sound, and sand will show you."""

from sonoform.geometry import FourierShape, polygon_mesh, star_mesh
from sonoform.spectrum import Spectrum, solve_spectrum

__version__ = "0.1.0"
__all__ = [
    "FourierShape",
    "Spectrum",
    "polygon_mesh",
    "solve_spectrum",
    "star_mesh",
]
