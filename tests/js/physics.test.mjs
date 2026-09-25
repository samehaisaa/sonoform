// What the page plays, against sonoform.audio on the same plate.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { test } from 'node:test';

import { Acoustics, hertz, malletSpectrum, note } from '../../src/sonoform/web/js/physics.js';
import { Plate } from '../../src/sonoform/web/js/plate.js';

const DIR = process.env.SONOFORM_FIXTURES ?? new URL('./fixtures', import.meta.url).pathname;
const load = (key) => JSON.parse(readFileSync(join(DIR, `${key}.json`), 'utf8'));

// Against the package mode by mode, so each mode keeps its own frequency.
// The page shares one frequency across a degenerate cluster, which the
// package does not, and that is tested separately below.
function acoustics(f, shareClusters = false) {
  const plate = new Plate(f.payload);
  return new Acoustics(plate, f.material, { thickness: f.thickness, span: f.span, shareClusters });
}

const close = (got, want, rtol, what) => {
  const err = Math.abs(got - want) / Math.max(Math.abs(want), 1e-300);
  assert.ok(err < rtol, `${what}: ${got} against ${want} (${err.toExponential(2)})`);
};

test('hertz from Ω is the frequency the package solved for', () => {
  for (const key of ['square', 'circle', 'triangle']) {
    const f = load(key);
    f.payload.omega.forEach((omega, m) => {
      close(hertz(omega, f.material, f.thickness, f.span), f.hz[m], 1e-6, `${key} ${m}`);
    });
  }
});

test('the square sings C sharp, 39 cents flat', () => {
  const n = note(135.54);
  assert.equal(n.label, 'C♯3');
  assert.equal(n.cents, -39);
  assert.equal(note(440).label, 'A4');
  assert.equal(note(440).cents, 0);
  assert.equal(note(261.6256).label, 'C4');
});

test('thickness scales the pitch linearly and size by its inverse square', () => {
  const f = load('square');
  const base = hertz(13.1, f.material, 0.002, 0.18);
  close(hertz(13.1, f.material, 0.004, 0.18), 2 * base, 1e-12, 'thickness');
  close(hertz(13.1, f.material, 0.002, 0.36), base / 4, 1e-12, 'span');
});

test('a mallet leaves low partials alone and rolls off high ones', () => {
  assert.equal(malletSpectrum(1000, 0), 1);
  close(malletSpectrum(2 * Math.PI * 50, 6e-4), 1, 1e-3, 'low');
  assert.ok(malletSpectrum(2 * Math.PI * 2000, 6e-4) < 0.35);
  close(malletSpectrum(Math.PI / 6e-4, 6e-4), Math.PI / 4, 1e-9, 'at the pole');
});

for (const key of ['square', 'circle', 'triangle']) {
  test(`${key}: radiation damping matches radiation_decay`, () => {
    const f = load(key);
    const a = acoustics(f);
    f.radiation.forEach((want, m) => close(a.radiation[m], want, 2e-4, `mode ${m}`));
  });

  test(`${key}: the Rayleigh integral matches rayleigh_integral`, () => {
    // Measured against the size of the integrand, ∫ |φ| / R dS, not of the
    // integral. Most modes of a small plate radiate almost nothing: their
    // positive and negative regions cancel to a part in 1e8 toward the
    // listener, and relative to a result that small any two quadratures
    // disagree completely while both being right.
    const f = load(key);
    const a = acoustics(f);
    const q = a.plate.coarse, L = f.span;
    const [x, y, z] = f.listener;
    f.rayleigh.forEach(([re, im], m) => {
      const [gr, gi] = a.rayleigh(m, x, y, z);
      let size = 0;
      for (let i = 0; i < q.count; i++) {
        const r = Math.hypot(q.x[i] * L - x, q.y[i] * L - y, z);
        size += (Math.abs(q.values[m][i]) * q.weights[i]) / r;
      }
      size *= a.unit[m] * L * L;
      const err = Math.hypot(gr - re, gi - im) / size;
      assert.ok(err < 2e-5, `mode ${m}: ${err.toExponential(2)}`);
    });
  });

  test(`${key}: a strike gives render_plate's velocities, decays and pressures`, () => {
    const f = load(key);
    const a = acoustics(f);
    const [sx, sy] = f.strike.point;
    const point = { x: sx, y: sy, e: a.plate.locate(sx, sy) };
    const out = a.strike(point, [f.strike.listener], { duration: 0.05, contact: 0 });
    const loudest = Math.max(...f.strike.modes.map((m) => Math.hypot(...m.pressure)));
    f.strike.modes.forEach((want, m) => {
      const got = out.modes[m];
      close(got.hz, want.hz, 1e-6, `hz ${m}`);
      close(got.gamma, want.gamma, 2e-4, `gamma ${m}`);
      close(got.velocity, want.velocity, 2e-5, `velocity ${m}`);
      const [pr, pi] = got.pressure[0];
      const err = Math.hypot(pr - want.pressure[0], pi - want.pressure[1]) / loudest;
      assert.ok(err < 2e-3, `pressure ${m}: ${err.toExponential(2)}`);
    });
  });
}

test('a degenerate cluster sounds at one frequency', () => {
  const f = load('triangle');
  const a = acoustics(f, true);
  for (const members of a.plate.clusters) {
    for (const m of members) assert.equal(a.hz[m], a.hz[members[0]]);
  }
  // and the shared frequency is the members' mean, a part in 1e3 from each
  const split = acoustics(f, false);
  for (const members of a.plate.clusters) {
    for (const m of members) close(a.hz[m], split.hz[m], 2e-3, `mode ${m}`);
  }
});

test('a strike is two finite channels that start and end at rest', () => {
  const f = load('square');
  const a = acoustics(f);
  const point = { x: 0.2, y: 0.1, e: a.plate.locate(0.2, 0.1) };
  const ears = [[-0.08, -0.3, 0.3], [0.08, -0.3, 0.3]];
  const { channels } = a.strike(point, ears, { duration: 1 });
  assert.equal(channels.length, 2);
  for (const c of channels) {
    assert.equal(c.length, 44100);
    assert.ok(c.every(Number.isFinite));
    assert.equal(c[0], 0);
    assert.equal(c[c.length - 1], 0);
  }
  // two ears in different places hear different things
  let diff = 0;
  for (let i = 0; i < 44100; i++) diff += Math.abs(channels[0][i] - channels[1][i]);
  assert.ok(diff > 0);
});
