// The page's plate against scikit-fem, on the plates the app has always shown.
//
// Fixtures come from tools/export_fixtures.py. tests/test_web.py writes them
// to a temporary directory and points SONOFORM_FIXTURES at it.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { EXPONENTS, Plate } from '../../src/sonoform/web/js/plate.js';

const DIR = process.env.SONOFORM_FIXTURES ?? new URL('./fixtures', import.meta.url).pathname;
const load = (key) => JSON.parse(readFileSync(join(DIR, `${key}.json`), 'utf8'));
const KEYS = ['square', 'circle', 'triangle'];

test('the monomials come in the order the package fits them', () => {
  assert.deepEqual(EXPONENTS, load('square').exponents);
});

for (const key of KEYS) {
  test(`${key}: the quintics reproduce scikit-fem at points it chose`, () => {
    const f = load(key);
    const plate = new Plate(f.payload);
    let worst = 0;
    f.probe.points.forEach(([x, y], i) => {
      const e = plate.locate(x, y);
      assert.ok(e >= 0, `probe ${i} fell outside the plate`);
      for (let m = 0; m < plate.modes; m++) {
        const got = plate.valueIn(plate.mode(m), e, x, y);
        worst = Math.max(worst, Math.abs(got - f.probe.values[i][m]));
      }
    });
    // values peak at one; float32 coefficients leave a few parts in 1e7
    assert.ok(worst < 2e-5, `worst disagreement ${worst.toExponential(2)}`);
  });

  test(`${key}: mass norms are exact`, () => {
    const f = load(key);
    const plate = new Plate(f.payload);
    for (let m = 0; m < plate.modes; m++) {
      const rel = Math.abs(plate.norms[m] - f.norms[m]) / f.norms[m];
      assert.ok(rel < 1e-5, `mode ${m}: ${rel.toExponential(2)}`);
    }
  });

  test(`${key}: the grid covers the plate and nothing else`, () => {
    const plate = new Plate(load(key).payload);
    const grid = plate.makeGrid(320);
    let inside = 0;
    for (let k = 0; k < grid.sdf.length; k++) if (grid.sdf[k] < 0) inside++;
    const area = inside * grid.h * grid.h;
    assert.ok(Math.abs(area - plate.area) / plate.area < 0.01, `${area} against ${plate.area}`);
  });
}

test('the gradient is the derivative of the value', () => {
  const plate = new Plate(load('circle').payload);
  const field = plate.mode(5);
  const g = [0, 0];
  const e = plate.locate(0.11, -0.07);
  const v = plate.valueIn(field, e, 0.11, -0.07, g);
  const d = 1e-5;
  const gx = (plate.valueIn(field, e, 0.11 + d, -0.07) - plate.valueIn(field, e, 0.11 - d, -0.07)) / (2 * d);
  const gy = (plate.valueIn(field, e, 0.11, -0.07 + d) - plate.valueIn(field, e, 0.11, -0.07 - d)) / (2 * d);
  assert.ok(Math.abs(g[0] - gx) < 1e-3 * (Math.abs(gx) + 1));
  assert.ok(Math.abs(g[1] - gy) < 1e-3 * (Math.abs(gy) + 1));
  assert.ok(Number.isFinite(v));
});

test('the field is continuous across element edges', () => {
  // Argyris is C1, so two neighbouring quintics agree on their shared edge.
  const plate = new Plate(load('triangle').payload);
  const t = plate.triangles, p = plate.points;
  let worst = 0;
  const field = plate.mode(7);
  const seen = new Map();
  for (let e = 0; e < plate.nt; e++) {
    for (let k = 0; k < 3; k++) {
      const a = t[3 * e + k], b = t[3 * e + ((k + 1) % 3)];
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      if (!seen.has(key)) {
        seen.set(key, e);
        continue;
      }
      const other = seen.get(key);
      for (const s of [0.25, 0.5, 0.75]) {
        const x = (1 - s) * p[2 * a] + s * p[2 * b];
        const y = (1 - s) * p[2 * a + 1] + s * p[2 * b + 1];
        worst = Math.max(worst, Math.abs(plate.valueIn(field, e, x, y) - plate.valueIn(field, other, x, y)));
      }
    }
  }
  assert.ok(worst < 2e-5, `jump across an edge ${worst.toExponential(2)}`);
});

test('degenerate pairs arrive as one note', () => {
  const circle = new Plate(load('circle').payload);
  assert.deepEqual(circle.clusters[0], [0, 1]);
  const square = new Plate(load('square').payload);
  assert.ok(square.clusters.some((c) => c.length === 2 && c[0] === 3 && c[1] === 4));
});

test('rim points know where on the rim they are', () => {
  // The bow is kept as a rim point and put back on the rim by its index when
  // the sheet changes under it, so the index has to travel with the point.
  const plate = new Plate(load('circle').payload);
  const n = plate.outline.length / 2;
  const p = plate.rimPoint(17);
  assert.equal(p.index, 17);
  assert.ok(p.e >= 0 && plate._contains(p.e, p.x, p.y, 1e-9));
  assert.equal(plate.rimPoint(-1).index, n - 1);
  assert.equal(plate.rimPoint(n + 3).index, 3);
  assert.equal(plate.rimPoint(plate.rimPoint(5).index).index, 5);
});

test('the outline runs counter-clockwise', () => {
  const plate = new Plate(load('triangle').payload);
  const o = plate.outline;
  let twice = 0;
  for (let i = 0, n = o.length / 2; i < n; i++) {
    const j = (i + 1) % n;
    twice += o[2 * i] * o[2 * j + 1] - o[2 * j] * o[2 * i + 1];
  }
  assert.ok(Math.abs(0.5 * twice - plate.area) < 1e-6 * plate.area);
});
