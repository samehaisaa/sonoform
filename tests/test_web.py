"""The page's JavaScript, held to the package.

The page evaluates plates, walks sand and synthesises sound on its own, so
it is checked against the code it mirrors: tools/export_fixtures.py writes the
plates the app has always shown next to what scikit-fem and sonoform.audio
say about them, and tests/js runs the page's modules against that.

Skipped where Node is not installed. GitHub's runners have it.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_the_page_agrees_with_the_package(tmp_path):
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "export_fixtures.py"), str(tmp_path)],
        check=True,
        capture_output=True,
    )
    files = sorted(str(p) for p in (ROOT / "tests" / "js").glob("*.test.mjs"))
    assert files, "no JavaScript tests found"
    result = subprocess.run(
        ["node", "--test", *files],
        env={**os.environ, "SONOFORM_FIXTURES": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_every_page_module_parses(tmp_path):
    """The modules the tests above cannot import still have to be valid.

    main.js and its companions need a browser to run, so they are only
    parsed. Each is copied to .mjs, which every Node release reads as a module.
    """
    for source in sorted((ROOT / "src" / "sonoform" / "web" / "js").glob("*.js")):
        copy = tmp_path / f"{source.stem}.mjs"
        copy.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(copy)], capture_output=True, text=True
        )
        assert result.returncode == 0, f"{source.name}: {result.stderr}"
