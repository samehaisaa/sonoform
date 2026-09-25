"""The sheets the page can switch between.

Stiffness, density and Poisson's ratio are handbook values for the common
grades. Loss factors sit inside the ranges quoted in L. Cremer,
M. Heckl and B. A. T. Petersson, *Structure-Borne Sound*, 3rd ed., Springer
(2005), and they matter: the loss factor alone sets how long a note rings
after the bow comes off, ``exp(-π η f t)``, so steel sings on long after glass
has gone quiet.

Chladni worked with brass and with glass. The others are what a teaching lab
is more likely to have.
"""

from __future__ import annotations

from sonoform.plate import BRASS, PlateMaterial

__all__ = ["MATERIALS"]

MATERIALS: dict[str, dict] = {
    "brass": {
        "label": "brass",
        "material": BRASS,
        "note": "What Chladni used, and the reference for everything here.",
    },
    "copper": {
        "label": "copper",
        "material": PlateMaterial(
            young=117.0e9, poisson=0.34, density=8960.0, thickness=0.002,
            loss_factor=2.0e-3,
        ),
        "note": "Stiffer and heavier than brass, which nearly cancel: a little "
        "higher, and it damps twice as fast.",
    },
    "aluminium": {
        "label": "aluminium",
        "material": PlateMaterial(
            young=69.0e9, poisson=0.33, density=2700.0, thickness=0.002,
            loss_factor=1.0e-4,
        ),
        "note": "A third of the density for two thirds of the stiffness, so "
        "every note sits well above brass, and rings several times longer.",
    },
    "steel": {
        "label": "steel",
        "material": PlateMaterial(
            young=200.0e9, poisson=0.29, density=7850.0, thickness=0.002,
            loss_factor=2.0e-4,
        ),
        "note": "Twice as stiff as brass. Its lower Poisson ratio also moves "
        "the notes against each other, which no rescaling can do.",
    },
    "glass": {
        "label": "glass",
        "material": PlateMaterial(
            young=72.0e9, poisson=0.22, density=2500.0, thickness=0.002,
            loss_factor=1.0e-3,
        ),
        "note": "Chladni's other plate. The lowest Poisson ratio here, so its "
        "figures differ most from the metals.",
    },
}
