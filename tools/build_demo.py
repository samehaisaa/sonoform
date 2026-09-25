"""Build the published site.

The site is the page ``sonoform play`` serves, plus every data file that page
would ask the local server for, computed ahead of time by the same functions.
Nothing is simplified for the static copy. Every plate on it was solved by
this checkout's solver, and the benchmark numbers its physics panel shows are
the ones this build got.

    python tools/build_demo.py site/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

from sonoform import export
from sonoform.presets import PRESETS

WEB = pathlib.Path(__file__).resolve().parent.parent / "src" / "sonoform" / "web"


def _write(path: pathlib.Path, payload) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, separators=(",", ":"))
    path.write_text(body, encoding="utf-8")
    return len(body)


def build(out: pathlib.Path) -> None:
    started = time.perf_counter()
    shutil.copytree(WEB, out, dirs_exist_ok=True)
    data = out / "data"

    _write(data / "manifest.json", export.manifest())
    _write(data / "verification.json", export.verification())

    total = 0
    for key in PRESETS:
        sizes = []
        for poisson in export.poisson_ratios():
            payload = export.plate_payload(key, poisson)
            size = _write(data / export.plate_file(key, poisson), payload)
            sizes.append(size)
            total += size
        notes = len(payload["clusters"])
        print(f"  {key:9s} {sum(sizes) / 1024:6.0f} KB  {notes:2d} notes")

    # the card link previews show, rendered by the page itself
    card = WEB.parent.parent.parent / "docs" / "og.jpg"
    if card.exists():
        shutil.copy(card, out / "og.jpg")

    (out / ".nojekyll").write_text("", encoding="utf-8")
    elapsed = time.perf_counter() - started
    print(f"\n  {total / 1024 / 1024:.1f} MB of plates in {elapsed:.0f} s")
    print(f"  wrote {out}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default="site", type=pathlib.Path)
    args = parser.parse_args(argv)
    build(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
