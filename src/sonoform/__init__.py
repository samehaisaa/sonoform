"""sonoform: every shape has a sound, and sand will show you."""

from sonoform.geometry import FourierShape, polygon_mesh, star_mesh
from sonoform.inverse import (
    CHORDS,
    PPW_FREQUENCY_BOUND,
    Feasibility,
    InverseResult,
    check_feasibility,
    solve_inverse,
)
from sonoform.plate import BRASS, PlateSpectrum, solve_plate
from sonoform.spectrum import Spectrum, solve_spectrum

__version__ = "0.1.0"
__all__ = [
    "CHORDS",
    "PPW_FREQUENCY_BOUND",
    "Feasibility",
    "FourierShape",
    "InverseResult",
    "BRASS",
    "PlateSpectrum",
    "Spectrum",
    "check_feasibility",
    "polygon_mesh",
    "solve_inverse",
    "solve_plate",
    "solve_spectrum",
    "star_mesh",
]
