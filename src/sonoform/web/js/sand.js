// Grains that find the still lines.
//
// This is sonoform.sand.sand_step, and tests/js holds it to the same law.
// Grains are not pushed toward the nodes, and there is no average force on
// them at all. Where the plate moves they are thrown and land somewhere
// random; where it is quiet they stay put. That is a random walk whose
// diffusivity follows the local amplitude,
//
//     dx = sqrt(2 D(x) dt) ξ,     D(x) = D0 (floor + |A(x)|)
//
// with no drift, and its steady state is ρ ∝ 1 / D, which is the nodal set.
//
// D is sampled where the grain starts. Sampling it anywhere else, or adding
// the ∇D "correction", turns this into the anti-Itô walk whose steady state
// is uniform: a blank plate, and no error to say so.
//
// A strike rings every mode at once. Modes at different frequencies do not
// interfere on average, so the amplitude a grain feels is the root sum of
// squares of what each mode is doing where it sits.

import { Normal, seeded } from './util.js';

// Diffusivity on a perfect node, as a fraction of an antinode's. The steady
// density goes as 1/D, so this alone sets the contrast: 0.004 puts a node at
// 126 times the density of a half-amplitude point, the regime real Chladni
// photographs are in. Lowering it is what driving the plate harder does.
export const FLOOR = 0.004;

// Plate units squared per second at full drive: 1.8e-3 m²/s on an 18 cm
// plate, sized so a figure forms in a few seconds of steady bowing, which is
// about what a real plate takes.
export const DIFFUSION = 1.8e-3 / (0.18 * 0.18);

const GRAVITY = 2.4; // plate units per second squared, for the pour only

export class Sand {
  constructor(count, seed = 1787) {
    this.count = count;
    this.random = seeded(seed);
    this.normal = new Normal(this.random);
    this.x = new Float32Array(count);
    this.y = new Float32Array(count);
    this.amp = new Float32Array(count); // signed bowed field under the grain
    this.act = new Float32Array(count); // how hard the plate moves under it
    this.pile = new Float32Array(count);
    this.lift = new Float32Array(count); // height during a pour
    this.fall = new Float32Array(count);
    this.delay = new Float32Array(count);
    // phase of the hop, size, and tone of each grain, fixed for its life
    this.traits = new Float32Array(3 * count);
    for (let i = 0; i < count; i++) {
      this.traits[3 * i] = this.random();
      this.traits[3 * i + 1] = 0.7 + 0.6 * this.random() ** 1.6;
      this.traits[3 * i + 2] = this.random();
    }
    this.pouring = 0;
    this.packed = new Float32Array(5 * count);
    this._density = null;
  }

  // Scatter the sand uniformly over the plate, dropped from a height.
  pour(grid, { from = 0.22, to = 0.65, over = 0.9, now = true } = {}) {
    const { n, h, x0, y0 } = grid;
    const margin = 0.6 * h;
    for (let i = 0; i < this.count; i++) {
      let x, y;
      do {
        x = x0 + this.random() * n * h;
        y = y0 + this.random() * n * h;
      } while (!(this._sdf(grid, x, y) < -margin));
      this.x[i] = x;
      this.y[i] = y;
      this.lift[i] = now ? from + (to - from) * this.random() : 0;
      this.fall[i] = 0;
      this.delay[i] = now ? over * this.random() ** 1.5 : 0;
    }
    this.pouring = now ? 1 : 0;
    this.pourTime = 0;
  }

  // Keep grains on the plate after its outline changes under them.
  settle(grid) {
    for (let i = 0; i < this.count; i++) {
      if (this._sdf(grid, this.x[i], this.y[i]) < 0) continue;
      let x, y;
      do {
        x = grid.x0 + this.random() * grid.n * grid.h;
        y = grid.y0 + this.random() * grid.n * grid.h;
      } while (!(this._sdf(grid, x, y) < -0.6 * grid.h));
      this.x[i] = x;
      this.y[i] = y;
    }
  }

  // Advance the pour. Grains fall and settle; nothing walks until they land.
  _pourStep(dt) {
    this.pourTime += dt;
    let airborne = 0;
    for (let i = 0; i < this.count; i++) {
      if (this.lift[i] <= 0 && this.fall[i] === 0) continue;
      if (this.pourTime < this.delay[i]) {
        airborne++;
        continue;
      }
      this.fall[i] += GRAVITY * dt;
      this.lift[i] -= this.fall[i] * dt;
      if (this.lift[i] <= 0) {
        this.lift[i] = 0;
        // a small bounce, then rest
        this.fall[i] = this.fall[i] > 0.35 ? -0.22 * this.fall[i] : 0;
      }
      if (this.lift[i] > 0 || this.fall[i] !== 0) airborne++;
    }
    if (airborne === 0) this.pouring = 0;
  }

  // One step of the walk. `field` holds what the plate is doing:
  //   voices    [{ values, level }]: each sounding note's field sampled on
  //             `grid`, peak one, and how hard it is ringing, zero to one.
  //             A bowed note builds while the last one rings down, so there
  //             can be more than one. The first is the one on screen.
  //   bow, level  the same for a single voice
  //   strike    optional { grid, values, level }: root-sum-square amplitude
  //             of a ringing strike, on its own grid
  //   diffusion D0, and floor
  step(dt, grid, field) {
    if (this.pouring) this._pourStep(dt);
    const { n, h, x0, y0 } = grid;
    const voices = field.voices || (field.bow ? [{ values: field.bow, level: field.level || 0 }] : []);
    const strike = field.strike || null;
    let activity = strike ? strike.level : 0;
    for (const v of voices) activity = Math.max(activity, v.level);
    const D0 = field.diffusion ?? DIFFUSION;
    const floor = field.floor ?? FLOOR;
    const walking = activity > 1e-4 && dt > 0 && !this.pouring;
    const normal = this.normal;
    const x = this.x, y = this.y;
    const lim = n - 2;
    const first = voices.length ? voices[0] : null;
    const rest = voices.slice(1);

    for (let i = 0; i < this.count; i++) {
      const px = x[i], py = y[i];
      let gx = (px - x0) / h - 0.5, gy = (py - y0) / h - 0.5;
      let ix = Math.floor(gx), iy = Math.floor(gy);
      if (ix < 0) ix = 0; else if (ix > lim) ix = lim;
      if (iy < 0) iy = 0; else if (iy > lim) iy = lim;
      const fx = gx - ix, fy = gy - iy;
      const k = iy * n + ix;
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
      let e2 = 0;
      let a = 0;
      if (first) {
        const v = first.values;
        a = w00 * v[k] + w10 * v[k + 1] + w01 * v[k + n] + w11 * v[k + n + 1];
        e2 = first.level * first.level * a * a;
      }
      for (const voice of rest) {
        const v = voice.values;
        const b = w00 * v[k] + w10 * v[k + 1] + w01 * v[k + n] + w11 * v[k + n + 1];
        e2 += voice.level * voice.level * b * b;
      }
      this.amp[i] = a;
      if (strike) e2 += this._strikeAt(strike, px, py) ** 2;
      const e = Math.sqrt(e2);
      this.act[i] = e;
      if (!walking) continue;

      // D where the grain is now, before it moves. See the header.
      const D = D0 * (activity * floor + e);
      const sigma = Math.sqrt(2 * D * dt);
      let nx = px + sigma * normal.next();
      let ny = py + sigma * normal.next();
      if (this._sdf(grid, nx, ny) > 0) [nx, ny] = this._reflect(grid, px, py, nx, ny);
      x[i] = nx;
      y[i] = ny;
    }
  }

  // Bounce a step that leaves the plate back in along the same line, as
  // sonoform.sand._reflect does, against the smooth outline of the grid's
  // signed distance.
  _reflect(grid, px, py, nx, ny) {
    if (this._sdf(grid, px, py) > 0) return [px, py];
    const sx = nx - px, sy = ny - py;
    let lo = 0, hi = 1;
    for (let k = 0; k < 14; k++) {
      const mid = 0.5 * (lo + hi);
      if (this._sdf(grid, px + mid * sx, py + mid * sy) <= 0) lo = mid;
      else hi = mid;
    }
    const hx = px + lo * sx, hy = py + lo * sy;
    const bx = hx - (1 - lo) * sx, by = hy - (1 - lo) * sy;
    return this._sdf(grid, bx, by) <= 0 ? [bx, by] : [hx, hy];
  }

  _sdf(grid, x, y) {
    const { n, h, x0, y0, sdf } = grid;
    const gx = (x - x0) / h - 0.5, gy = (y - y0) / h - 0.5;
    const ix = Math.floor(gx), iy = Math.floor(gy);
    if (ix < 0 || iy < 0 || ix > n - 2 || iy > n - 2) return 1;
    const fx = gx - ix, fy = gy - iy, k = iy * n + ix;
    return (1 - fx) * (1 - fy) * sdf[k] + fx * (1 - fy) * sdf[k + 1] +
      (1 - fx) * fy * sdf[k + n] + fx * fy * sdf[k + n + 1];
  }

  _strikeAt(strike, x, y) {
    const { n, h, x0, y0 } = strike.grid;
    const v = strike.values;
    const gx = (x - x0) / h - 0.5, gy = (y - y0) / h - 0.5;
    let ix = Math.floor(gx), iy = Math.floor(gy);
    if (ix < 0) ix = 0; else if (ix > n - 2) ix = n - 2;
    if (iy < 0) iy = 0; else if (iy > n - 2) iy = n - 2;
    const fx = gx - ix, fy = gy - iy, k = iy * n + ix;
    return (1 - fx) * (1 - fy) * v[k] + fx * (1 - fy) * v[k + 1] +
      (1 - fx) * fy * v[k + n] + fx * fy * v[k + n + 1];
  }

  // How high the sand stands where it has gathered. Visual only: a histogram
  // of the grains, blurred, lifts each grain in proportion to the square root
  // of the excess density under it, so the lines read as ridges.
  piles(grid, cells = 112, height = 0.0011) {
    const { n, h, x0, y0 } = grid;
    const size = (n * h) / cells;
    if (!this._density || this._density.length !== cells * cells) {
      this._density = new Float32Array(cells * cells);
      this._scratch = new Float32Array(cells * cells);
    }
    const d = this._density, tmp = this._scratch;
    d.fill(0);
    for (let i = 0; i < this.count; i++) {
      const cx = Math.floor((this.x[i] - x0) / size), cy = Math.floor((this.y[i] - y0) / size);
      if (cx >= 0 && cy >= 0 && cx < cells && cy < cells) d[cy * cells + cx]++;
    }
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < cells; j++) {
        for (let i = 0; i < cells; i++) {
          let s = 0, w = 0;
          for (let dj = -1; dj <= 1; dj++) {
            const jj = j + dj;
            if (jj < 0 || jj >= cells) continue;
            for (let di = -1; di <= 1; di++) {
              const ii = i + di;
              if (ii < 0 || ii >= cells) continue;
              s += d[jj * cells + ii];
              w++;
            }
          }
          tmp[j * cells + i] = s / w;
        }
      }
      d.set(tmp);
    }
    const mean = (this.count * size * size) / Math.max(grid.inside * h * h, 1e-12);
    for (let i = 0; i < this.count; i++) {
      const cx = Math.floor((this.x[i] - x0) / size), cy = Math.floor((this.y[i] - y0) / size);
      const rho = cx >= 0 && cy >= 0 && cx < cells && cy < cells ? d[cy * cells + cx] / mean : 0;
      const excess = Math.sqrt(Math.max(rho - 1.2, 0));
      this.pile[i] = height * Math.min(excess, 3.2) * (0.25 + 0.75 * this.traits[3 * i]);
    }
    this.densityCells = cells;
    this.densitySize = size;
    this.densityMean = mean;
  }

  // Interleaved for the GPU: x, y, height above the plate, bowed field,
  // activity.
  pack() {
    const out = this.packed;
    for (let i = 0; i < this.count; i++) {
      const o = 5 * i;
      out[o] = this.x[i];
      out[o + 1] = this.y[i];
      out[o + 2] = this.lift[i] + this.pile[i];
      out[o + 3] = this.amp[i];
      out[o + 4] = this.act[i];
    }
    return out;
  }
}
