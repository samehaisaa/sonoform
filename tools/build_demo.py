"""Build the static demo.

The published demo has no solver behind it, so every preset is solved here
ahead of time and written out as the exact payload the local server would have
returned. The page is the same index.html the app ships: it picks up
window.SONOFORM_STATIC and reads from these files instead of posting to a
server. Drawing is unavailable, because there is nothing to solve with.

    python tools/build_demo.py site/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

from sonoform.plate import SPAN
from sonoform.presets import PRESETS, preset_request
from sonoform.server import _plate_for, _plate_payload

WEB = pathlib.Path(__file__).resolve().parent.parent / "src" / "sonoform" / "web"

SHIM = """// Static backend for the published demo.
//
// The presets below were solved by tools/build_demo.py with the same code the
// local app calls, so what you see here is the real solution and not a
// simplified stand-in. Drawing your own outline needs the solver, which is a
// Python package, so that button is disabled and points at the repository.
window.SONOFORM_STATIC = {
  presets: %(presets)s,

  async get(path) {
    if (path === '/presets') return this.presets;
    throw new Error('not available in the demo: ' + path);
  },

  async post(path, body) {
    if (path === '/plate' && body && body.preset) {
      const res = await fetch('plates/' + body.preset + '.json');
      if (!res.ok) throw new Error('could not load ' + body.preset);
      return res.json();
    }
    if (path === '/plate') {
      return { valid: false, error: 'drawing needs the solver, see the repo' };
    }
    throw new Error('not available in the demo: ' + path);
  },
};

// Tell the reader what this is, once the app has settled.
addEventListener('load', () => {
  const draw = document.getElementById('draw');
  if (draw) {
    draw.disabled = true;
    draw.title = 'drawing needs the local solver: '
      + 'pip install -e . and run sonoform play';
  }
  const hint = document.getElementById('hint');
  if (hint) hint.textContent = 'pick a plate and press bow';
});
"""


def build(out: pathlib.Path) -> None:
    plates = out / "plates"
    plates.mkdir(parents=True, exist_ok=True)

    total = 0
    for key in PRESETS:
        request = preset_request(key)
        spec, outline = _plate_for(request)
        payload = _plate_payload(spec, outline)
        path = plates / f"{key}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        size = path.stat().st_size
        total += size
        hz = [round(m["hz"], 1) for m in payload["modes"]]
        print(f"  {key:9s} {size / 1024:7.0f} KB  {hz}")

    shutil.copy(WEB / "index.html", out / "index.html")
    labels = {key: {"label": plate["label"]} for key, plate in PRESETS.items()}
    (out / "static.js").write_text(
        SHIM % {"presets": json.dumps(labels)}, encoding="utf-8"
    )

    # the page has to load the shim before its own script runs
    page = (out / "index.html").read_text(encoding="utf-8")
    marker = "<script>\nconst cv ="
    if marker not in page:
        raise SystemExit("could not find the script tag to inject before")
    page = page.replace(marker, '<script src="static.js"></script>\n' + marker, 1)
    (out / "index.html").write_text(page, encoding="utf-8")

    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"\n  {total / 1024:.0f} KB of plates, span {SPAN} m")
    print(f"  wrote {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default="site", type=pathlib.Path)
    args = parser.parse_args(argv)
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
