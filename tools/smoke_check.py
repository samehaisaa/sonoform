"""Check what the container actually served.

Run by the image workflow against plate.json, which is the square's payload
fetched from a running container.

The first partial is asserted as a band rather than a value. The exact decimal
moves with the BLAS the base image happens to ship, and 135.4 against 135.5 is
0.07 percent, far below anything that matters. A pin to one decimal tests the
library rather than the physics, and fails the day the base image updates.
"""

import json
import pathlib
import sys

REFERENCE_HZ = 135.4
BAND_HZ = 2.0


def main() -> int:
    plate = json.loads(pathlib.Path("plate.json").read_text(encoding="utf-8"))
    hz = [mode["hz"] for mode in plate["modes"]]
    print("partials:", [round(value, 1) for value in hz])

    if not plate.get("valid"):
        print("solver reported the plate as invalid")
        return 1
    if len(hz) != 6:
        print(f"expected 6 partials, got {len(hz)}")
        return 1
    if hz != sorted(hz):
        print("partials are not ascending")
        return 1
    if abs(hz[0] - REFERENCE_HZ) > BAND_HZ:
        print(f"first partial {hz[0]:.1f} Hz, want {REFERENCE_HZ} +/- {BAND_HZ}")
        return 1

    print(f"ok: first partial {hz[0]:.1f} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
