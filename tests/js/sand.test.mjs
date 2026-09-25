// The transport law, as tests/test_sand_transport.py states it, for the
// page's walk. No fixture needed: the sand does not care what shape made
// the field, only what the field is.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { Sand } from '../../src/sonoform/web/js/sand.js';

// A unit square with reflecting walls, as a grid the walk accepts.
function box(n = 128) {
  const h = 1 / (n - 8);
  const x0 = -4 * h, y0 = -4 * h;
  const sdf = new Float32Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = x0 + (i + 0.5) * h, y = y0 + (j + 0.5) * h;
      const dx = Math.max(-x, x - 1), dy = Math.max(-y, y - 1);
      sdf[j * n + i] = dx > 0 || dy > 0 ? Math.hypot(Math.max(dx, 0), Math.max(dy, 0)) : Math.max(dx, dy);
    }
  }
  return { n, h, x0, y0, sdf, inside: (n - 8) * (n - 8) };
}

function field(grid, f) {
  const out = new Float32Array(grid.n * grid.n);
  for (let j = 0; j < grid.n; j++) {
    for (let i = 0; i < grid.n; i++) out[j * grid.n + i] = f(grid.x0 + (i + 0.5) * grid.h);
  }
  return out;
}

function histogram(sand, bins) {
  const counts = new Float64Array(bins);
  for (let i = 0; i < sand.count; i++) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(sand.x[i] * bins)));
    counts[b]++;
  }
  const mean = sand.count / bins;
  return counts.map((c) => c / mean);
}

function correlation(a, b) {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n, mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0, saa = 0, sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return sab / Math.sqrt(saa * sbb);
}

test('the Ito walk piles grains up where D is small, as 1 / D', () => {
  const grid = box();
  const floor = 0.05;
  const A = (x) => Math.sin(2 * Math.PI * x);
  const bow = field(grid, A);
  const sand = new Sand(40000, 7);
  sand.pour(grid, { now: false });
  const plate = { bow, level: 1, diffusion: 1, floor };
  for (let s = 0; s < 1500; s++) sand.step(4e-4, grid, plate);

  const bins = 40;
  const rho = histogram(sand, bins);
  const predicted = Array.from({ length: bins }, (_, i) => 1 / (floor + Math.abs(A((i + 0.5) / bins))));
  const mean = predicted.reduce((s, v) => s + v, 0) / bins;
  const r = correlation(Array.from(rho), predicted.map((v) => v / mean));
  assert.ok(r > 0.97, `rho against 1/D correlated at ${r.toFixed(4)}`);
  assert.ok(Math.max(...rho) / Math.min(...rho) > 3, 'the grains did not concentrate');
});

test('uniform D leaves the sand uniform: there is no drift', () => {
  const grid = box();
  const bow = field(grid, () => 0.6);
  const sand = new Sand(40000, 11);
  sand.pour(grid, { now: false });
  for (let s = 0; s < 800; s++) sand.step(4e-4, grid, { bow, level: 1, diffusion: 1, floor: 0.05 });
  const rho = histogram(sand, 20);
  for (const v of rho) assert.ok(Math.abs(v - 1) < 0.08, `density ${v.toFixed(3)} is not flat`);
});

test('a still plate moves nothing', () => {
  const grid = box();
  const bow = field(grid, (x) => Math.sin(6 * x));
  const sand = new Sand(2000, 3);
  sand.pour(grid, { now: false });
  const before = Float32Array.from(sand.x);
  for (let s = 0; s < 50; s++) sand.step(0.016, grid, { bow, level: 0 });
  assert.deepEqual(sand.x, before);
});

test('reflection keeps every grain on the plate', () => {
  const grid = box();
  const bow = field(grid, () => 1);
  const sand = new Sand(5000, 5);
  sand.pour(grid, { now: false });
  for (let s = 0; s < 200; s++) sand.step(0.02, grid, { bow, level: 1, diffusion: 0.05 });
  for (let i = 0; i < sand.count; i++) {
    assert.ok(sand.x[i] >= -1e-3 && sand.x[i] <= 1 + 1e-3, `x ${sand.x[i]}`);
    assert.ok(sand.y[i] >= -1e-3 && sand.y[i] <= 1 + 1e-3, `y ${sand.y[i]}`);
  }
});

test('a pour lands every grain', () => {
  const grid = box();
  const sand = new Sand(3000, 9);
  sand.pour(grid);
  assert.ok(sand.pouring);
  for (let s = 0; s < 400 && sand.pouring; s++) sand.step(1 / 60, grid, { bow: null, level: 0 });
  assert.equal(sand.pouring, 0);
  assert.ok(sand.lift.every((z) => z === 0));
});
