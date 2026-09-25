// One solved plate, evaluated exactly.
//
// The payload carries every mode as a quintic polynomial on each element, in
// a frame centred on that element (sonoform/export.py). A value anywhere on
// the plate is one element lookup and 21 multiply-adds, which is what lets
// the page draw the true solution rather than an interpolation of samples of
// it. Coordinates are plate units: the physical span divided out, so the
// outline's longest side is one.

import { float32From, indexFrom } from './util.js';

export const TERMS = 21;

// x^a y^b of total degree at most five, grouped by degree. The same order as
// sonoform.export.EXPONENTS, and tests/js pins the two against each other.
export const EXPONENTS = [
  [0, 0],
  [1, 0], [0, 1],
  [2, 0], [1, 1], [0, 2],
  [3, 0], [2, 1], [1, 2], [0, 3],
  [4, 0], [3, 1], [2, 2], [1, 3], [0, 4],
  [5, 0], [4, 1], [3, 2], [2, 3], [1, 4], [0, 5],
];

// Value of the quintic with coefficients c[o .. o+20] at local (x, y), and,
// when `out` is given, its gradient with respect to the local coordinates.
export function quintic(c, o, x, y, out) {
  const x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
  const y2 = y * y, y3 = y2 * y, y4 = y3 * y, y5 = y4 * y;
  const value =
    c[o] + c[o + 1] * x + c[o + 2] * y +
    c[o + 3] * x2 + c[o + 4] * x * y + c[o + 5] * y2 +
    c[o + 6] * x3 + c[o + 7] * x2 * y + c[o + 8] * x * y2 + c[o + 9] * y3 +
    c[o + 10] * x4 + c[o + 11] * x3 * y + c[o + 12] * x2 * y2 +
    c[o + 13] * x * y3 + c[o + 14] * y4 +
    c[o + 15] * x5 + c[o + 16] * x4 * y + c[o + 17] * x3 * y2 +
    c[o + 18] * x2 * y3 + c[o + 19] * x * y4 + c[o + 20] * y5;
  if (out) {
    out[0] =
      c[o + 1] + 2 * c[o + 3] * x + c[o + 4] * y +
      3 * c[o + 6] * x2 + 2 * c[o + 7] * x * y + c[o + 8] * y2 +
      4 * c[o + 10] * x3 + 3 * c[o + 11] * x2 * y + 2 * c[o + 12] * x * y2 +
      c[o + 13] * y3 +
      5 * c[o + 15] * x4 + 4 * c[o + 16] * x3 * y + 3 * c[o + 17] * x2 * y2 +
      2 * c[o + 18] * x * y3 + c[o + 19] * y4;
    out[1] =
      c[o + 2] + c[o + 4] * x + 2 * c[o + 5] * y +
      c[o + 7] * x2 + 2 * c[o + 8] * x * y + 3 * c[o + 9] * y2 +
      c[o + 11] * x3 + 2 * c[o + 12] * x2 * y + 3 * c[o + 13] * x * y2 +
      4 * c[o + 14] * y3 +
      c[o + 16] * x4 + 2 * c[o + 17] * x3 * y + 3 * c[o + 18] * x2 * y2 +
      4 * c[o + 19] * x * y3 + 5 * c[o + 20] * y4;
  }
  return value;
}

// Gauss-Legendre on [0, 1], six points: exact to degree eleven.
const GAUSS6 = (() => {
  const t = [0.2386191860831969, 0.6612093864662645, 0.9324695142031521];
  const w = [0.467913934572691, 0.3607615730481386, 0.1713244923791704];
  const nodes = [], weights = [];
  for (let i = 0; i < 3; i++) {
    nodes.push(0.5 * (1 - t[i]), 0.5 * (1 + t[i]));
    weights.push(0.5 * w[i], 0.5 * w[i]);
  }
  return { nodes, weights };
})();

// Gauss-Legendre on [0, 1], three points: exact to degree five.
const GAUSS3 = {
  nodes: [0.5 - 0.5 * Math.sqrt(0.6), 0.5, 0.5 + 0.5 * Math.sqrt(0.6)],
  weights: [5 / 18, 8 / 18, 5 / 18],
};

export class Plate {
  constructor(payload) {
    if (payload.format !== 1) throw new Error('unknown plate format');
    this.key = payload.key;
    this.poisson = payload.poisson;
    this.omega = Float64Array.from(payload.omega);
    this.clusters = payload.clusters.map((c) => c.slice());
    this.kernel = payload.kernel;
    this.dofs = payload.dofs;
    this.points = float32From(payload.points);
    this.triangles = indexFrom(payload.triangles, payload.index);
    this.boundary = indexFrom(payload.boundary, payload.index);
    this.frames = float32From(payload.frames);
    this.coefficients = float32From(payload.coefficients);
    this.nv = this.points.length / 2;
    this.nt = this.triangles.length / 3;
    this.modes = this.omega.length;
    if (this.coefficients.length !== this.modes * this.nt * TERMS) {
      throw new Error('coefficients do not match the mesh');
    }
    this._geometry();
    this._buckets();
    // Exact for φ², degree ten, so the mass norms are exact.
    this.fine = this._quadrature(GAUSS6);
    // Degree five, plenty for φ times a smooth Green's function.
    this.coarse = this._quadrature(GAUSS3);
    this.norms = this._norms();
  }

  mode(m) {
    const size = this.nt * TERMS;
    return this.coefficients.subarray(m * size, (m + 1) * size);
  }

  // Σ weights[i] · mode(members[i]), as a field of its own.
  combine(members, weights) {
    const size = this.nt * TERMS;
    const out = new Float32Array(size);
    for (let k = 0; k < members.length; k++) {
      const src = this.mode(members[k]);
      const w = weights[k];
      for (let i = 0; i < size; i++) out[i] += w * src[i];
    }
    return out;
  }

  // Value of a field at a point already known to lie in element e.
  valueIn(field, e, x, y, grad) {
    const f = this.frames;
    const s = f[3 * e + 2];
    const v = quintic(field, e * TERMS, (x - f[3 * e]) / s, (y - f[3 * e + 1]) / s, grad);
    if (grad) {
      grad[0] /= s;
      grad[1] /= s;
    }
    return v;
  }

  value(field, x, y, grad) {
    const e = this.locate(x, y);
    return e < 0 ? 0 : this.valueIn(field, e, x, y, grad);
  }

  // The element containing (x, y), or -1.
  locate(x, y) {
    const b = this._bucket;
    const i = Math.floor((x - b.x0) / b.w);
    const j = Math.floor((y - b.y0) / b.h);
    if (i < 0 || j < 0 || i >= b.n || j >= b.n) return -1;
    const list = b.cells[j * b.n + i];
    for (let k = 0; k < list.length; k++) {
      if (this._contains(list[k], x, y, 1e-9)) return list[k];
    }
    return -1;
  }

  // Nearest point of the plate to (x, y), pulled a hair inside, with the
  // element it lies in. Used to put a bow on the rim.
  nearestInside(x, y, inset = 0.004) {
    const e = this.locate(x, y);
    if (e >= 0 && this.distanceToEdge(x, y) >= inset) return { x, y, e };
    const edge = this.nearestEdge(x, y);
    const px = edge.x - inset * edge.nx;
    const py = edge.y - inset * edge.ny;
    let el = this.locate(px, py);
    if (el < 0) el = this.edgeElement[edge.index];
    return { x: px, y: py, e: el };
  }

  distanceToEdge(x, y) {
    return this.nearestEdge(x, y).distance;
  }

  // Closest point on the outline, with the outward normal of that edge.
  nearestEdge(x, y) {
    const o = this.outline;
    const n = o.length / 2;
    let best = { distance: Infinity };
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const ax = o[2 * i], ay = o[2 * i + 1];
      const dx = o[2 * j] - ax, dy = o[2 * j + 1] - ay;
      const len2 = dx * dx + dy * dy;
      let t = ((x - ax) * dx + (y - ay) * dy) / len2;
      t = Math.min(1, Math.max(0, t));
      const qx = ax + t * dx, qy = ay + t * dy;
      const d = Math.hypot(x - qx, y - qy);
      if (d < best.distance) {
        const len = Math.sqrt(len2);
        best = { distance: d, x: qx, y: qy, nx: dy / len, ny: -dx / len, index: i, t };
      }
    }
    return best;
  }

  // Point i of the outline, a hair inside, as { x, y, e, index }.
  rimPoint(i, inset = 0.004) {
    const o = this.outline;
    const n = o.length / 2;
    const k = Number.isInteger(i) ? ((i % n) + n) % n : 0;
    const nx = this.rimNormals[2 * k], ny = this.rimNormals[2 * k + 1];
    const x = o[2 * k] - inset * nx, y = o[2 * k + 1] - inset * ny;
    let e = this.locate(x, y);
    if (e < 0) e = this.edgeElement[k];
    return { x, y, e, index: k };
  }

  // A regular grid over the plate: which element owns each cell centre, its
  // local coordinates there, and a signed distance to the outline.
  //
  // Cells just outside the outline are owned too, by the element of the
  // nearest boundary edge, whose quintic is carried a fraction of a cell past
  // its edge. Without that, bilinear lookups near the rim would mix in zeros
  // and report a free edge, which swings hardest, as nearly still.
  makeGrid(n, band = 3) {
    const pad = band + 1;
    const extent = Math.max(this.xmax - this.xmin, this.ymax - this.ymin);
    const h = extent / (n - 2 * pad);
    const x0 = 0.5 * (this.xmin + this.xmax) - 0.5 * n * h;
    const y0 = 0.5 * (this.ymin + this.ymax) - 0.5 * n * h;
    const owner = new Int32Array(n * n).fill(-1);
    const xi = new Float32Array(n * n);
    const eta = new Float32Array(n * n);
    const sdf = new Float32Array(n * n);
    const f = this.frames, p = this.points, t = this.triangles;

    let inside = 0;
    for (let e = 0; e < this.nt; e++) {
      const a = t[3 * e], b = t[3 * e + 1], c = t[3 * e + 2];
      const xlo = Math.min(p[2 * a], p[2 * b], p[2 * c]);
      const xhi = Math.max(p[2 * a], p[2 * b], p[2 * c]);
      const ylo = Math.min(p[2 * a + 1], p[2 * b + 1], p[2 * c + 1]);
      const yhi = Math.max(p[2 * a + 1], p[2 * b + 1], p[2 * c + 1]);
      const i0 = Math.max(0, Math.ceil((xlo - x0) / h - 0.5));
      const i1 = Math.min(n - 1, Math.floor((xhi - x0) / h - 0.5));
      const j0 = Math.max(0, Math.ceil((ylo - y0) / h - 0.5));
      const j1 = Math.min(n - 1, Math.floor((yhi - y0) / h - 0.5));
      const cx = f[3 * e], cy = f[3 * e + 1], s = f[3 * e + 2];
      for (let j = j0; j <= j1; j++) {
        const y = y0 + (j + 0.5) * h;
        for (let i = i0; i <= i1; i++) {
          const k = j * n + i;
          if (owner[k] >= 0) continue;
          const x = x0 + (i + 0.5) * h;
          if (!this._contains(e, x, y, 1e-12)) continue;
          owner[k] = e;
          xi[k] = (x - cx) / s;
          eta[k] = (y - cy) / s;
          inside++;
        }
      }
    }

    // Signed distance in a band around the outline, one boundary edge at a
    // time, so the cost is the perimeter rather than the area.
    const far = band * h;
    for (let k = 0; k < n * n; k++) sdf[k] = owner[k] >= 0 ? -far : far;
    const nearest = new Int32Array(n * n).fill(-1);
    const best = new Float32Array(n * n).fill(far);
    const o = this.outline;
    const nb = o.length / 2;
    for (let s = 0; s < nb; s++) {
      const u = (s + 1) % nb;
      const ax = o[2 * s], ay = o[2 * s + 1], bx = o[2 * u], by = o[2 * u + 1];
      const i0 = Math.max(0, Math.floor((Math.min(ax, bx) - x0) / h - 0.5) - band);
      const i1 = Math.min(n - 1, Math.ceil((Math.max(ax, bx) - x0) / h - 0.5) + band);
      const j0 = Math.max(0, Math.floor((Math.min(ay, by) - y0) / h - 0.5) - band);
      const j1 = Math.min(n - 1, Math.ceil((Math.max(ay, by) - y0) / h - 0.5) + band);
      const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
      for (let j = j0; j <= j1; j++) {
        const y = y0 + (j + 0.5) * h;
        for (let i = i0; i <= i1; i++) {
          const x = x0 + (i + 0.5) * h;
          let tt = ((x - ax) * dx + (y - ay) * dy) / len2;
          tt = tt < 0 ? 0 : tt > 1 ? 1 : tt;
          const d = Math.hypot(x - ax - tt * dx, y - ay - tt * dy);
          const k = j * n + i;
          if (d < best[k]) {
            best[k] = d;
            nearest[k] = s;
          }
        }
      }
    }
    for (let k = 0; k < n * n; k++) {
      if (nearest[k] < 0) continue;
      sdf[k] = owner[k] >= 0 ? -best[k] : best[k];
      if (owner[k] < 0) {
        const e = this.edgeElement[nearest[k]];
        const i = k % n, j = (k - i) / n;
        owner[k] = e;
        xi[k] = (x0 + (i + 0.5) * h - f[3 * e]) / f[3 * e + 2];
        eta[k] = (y0 + (j + 0.5) * h - f[3 * e + 1]) / f[3 * e + 2];
      }
    }
    return { n, h, x0, y0, owner, xi, eta, sdf, inside };
  }

  // A field sampled on a grid from makeGrid. Unowned cells read zero.
  sampleGrid(grid, field, out = new Float32Array(grid.n * grid.n)) {
    const { owner, xi, eta } = grid;
    for (let k = 0; k < out.length; k++) {
      const e = owner[k];
      out[k] = e < 0 ? 0 : quintic(field, e * TERMS, xi[k], eta[k]);
    }
    return out;
  }

  // ∫ φ_m² dA for every mode, in plate units. The page's modes are scaled to
  // a peak of one, so these are what turns them back into orthonormal ones.
  _norms() {
    const q = this.fine;
    const out = new Float64Array(this.modes);
    for (let m = 0; m < this.modes; m++) {
      const v = q.values[m];
      let sum = 0;
      for (let i = 0; i < q.count; i++) sum += q.weights[i] * v[i] * v[i];
      out[m] = sum;
    }
    return out;
  }

  // Collapsed Gauss rule on every element, with every mode evaluated at
  // every point. Duffy's map takes the unit square onto the triangle with a
  // Jacobian of (1 - u), which the tensor rule then integrates exactly.
  _quadrature(rule) {
    const { nodes, weights } = rule;
    const per = nodes.length * nodes.length;
    const count = this.nt * per;
    const x = new Float64Array(count), y = new Float64Array(count);
    const w = new Float64Array(count), elem = new Int32Array(count);
    const p = this.points, t = this.triangles;
    let q = 0;
    for (let e = 0; e < this.nt; e++) {
      const a = t[3 * e], b = t[3 * e + 1], c = t[3 * e + 2];
      const ax = p[2 * a], ay = p[2 * a + 1];
      const ux = p[2 * b] - ax, uy = p[2 * b + 1] - ay;
      const vx = p[2 * c] - ax, vy = p[2 * c + 1] - ay;
      const jac = Math.abs(ux * vy - uy * vx);
      for (let i = 0; i < nodes.length; i++) {
        for (let j = 0; j < nodes.length; j++) {
          const u = nodes[i], v = nodes[j] * (1 - nodes[i]);
          x[q] = ax + u * ux + v * vx;
          y[q] = ay + u * uy + v * vy;
          w[q] = jac * weights[i] * weights[j] * (1 - nodes[i]);
          elem[q] = e;
          q++;
        }
      }
    }
    const values = [];
    for (let m = 0; m < this.modes; m++) {
      const field = this.mode(m);
      const v = new Float64Array(count);
      for (let i = 0; i < count; i++) v[i] = this.valueIn(field, elem[i], x[i], y[i]);
      values.push(v);
    }
    return { count, x, y, weights: w, elem, values };
  }

  _contains(e, x, y, tol) {
    const p = this.points, t = this.triangles;
    const a = t[3 * e], b = t[3 * e + 1], c = t[3 * e + 2];
    const ax = p[2 * a], ay = p[2 * a + 1];
    const bx = p[2 * b], by = p[2 * b + 1];
    const cx = p[2 * c], cy = p[2 * c + 1];
    const d = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    const w0 = ((bx - x) * (cy - y) - (cx - x) * (by - y)) / d;
    const w1 = ((cx - x) * (ay - y) - (ax - x) * (cy - y)) / d;
    const w2 = 1 - w0 - w1;
    return w0 >= -tol && w1 >= -tol && w2 >= -tol;
  }

  _geometry() {
    const p = this.points, t = this.triangles;
    let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
    for (let i = 0; i < this.nv; i++) {
      xmin = Math.min(xmin, p[2 * i]); xmax = Math.max(xmax, p[2 * i]);
      ymin = Math.min(ymin, p[2 * i + 1]); ymax = Math.max(ymax, p[2 * i + 1]);
    }
    Object.assign(this, { xmin, xmax, ymin, ymax });

    let area = 0;
    this.areas = new Float64Array(this.nt);
    for (let e = 0; e < this.nt; e++) {
      const a = t[3 * e], b = t[3 * e + 1], c = t[3 * e + 2];
      const s = 0.5 * Math.abs(
        (p[2 * b] - p[2 * a]) * (p[2 * c + 1] - p[2 * a + 1]) -
        (p[2 * c] - p[2 * a]) * (p[2 * b + 1] - p[2 * a + 1]));
      this.areas[e] = s;
      area += s;
    }
    this.area = area;

    const nb = this.boundary.length;
    this.outline = new Float64Array(2 * nb);
    for (let i = 0; i < nb; i++) {
      this.outline[2 * i] = p[2 * this.boundary[i]];
      this.outline[2 * i + 1] = p[2 * this.boundary[i] + 1];
    }

    // The element each boundary edge belongs to. Exactly one, on a mesh
    // whose boundary is a single loop.
    const owner = new Map();
    const key = (a, b) => (a < b ? a * 1048576 + b : b * 1048576 + a);
    for (let e = 0; e < this.nt; e++) {
      const v = [t[3 * e], t[3 * e + 1], t[3 * e + 2]];
      for (let k = 0; k < 3; k++) owner.set(key(v[k], v[(k + 1) % 3]), e);
    }
    this.edgeElement = new Int32Array(nb);
    for (let i = 0; i < nb; i++) {
      this.edgeElement[i] = owner.get(key(this.boundary[i], this.boundary[(i + 1) % nb]));
    }

    // Outward normals at the outline's vertices, averaged from the two edges
    // that meet there. The loop runs counter-clockwise, so outward is to the
    // right of the direction of travel.
    this.rimNormals = new Float64Array(2 * nb);
    let perimeter = 0;
    for (let i = 0; i < nb; i++) {
      const j = (i + 1) % nb;
      const dx = this.outline[2 * j] - this.outline[2 * i];
      const dy = this.outline[2 * j + 1] - this.outline[2 * i + 1];
      const len = Math.hypot(dx, dy);
      perimeter += len;
      for (const k of [i, j]) {
        this.rimNormals[2 * k] += dy / len;
        this.rimNormals[2 * k + 1] -= dx / len;
      }
    }
    for (let i = 0; i < nb; i++) {
      const len = Math.hypot(this.rimNormals[2 * i], this.rimNormals[2 * i + 1]) || 1;
      this.rimNormals[2 * i] /= len;
      this.rimNormals[2 * i + 1] /= len;
    }
    this.perimeter = perimeter;
  }

  _buckets() {
    const p = this.points, t = this.triangles;
    const n = Math.max(8, Math.min(128, Math.round(1.5 * Math.sqrt(this.nt))));
    const pad = 1e-6;
    const x0 = this.xmin - pad, y0 = this.ymin - pad;
    const w = (this.xmax - this.xmin + 2 * pad) / n;
    const h = (this.ymax - this.ymin + 2 * pad) / n;
    const cells = Array.from({ length: n * n }, () => []);
    for (let e = 0; e < this.nt; e++) {
      const xs = [p[2 * t[3 * e]], p[2 * t[3 * e + 1]], p[2 * t[3 * e + 2]]];
      const ys = [p[2 * t[3 * e] + 1], p[2 * t[3 * e + 1] + 1], p[2 * t[3 * e + 2] + 1]];
      const i0 = Math.max(0, Math.floor((Math.min(...xs) - x0) / w));
      const i1 = Math.min(n - 1, Math.floor((Math.max(...xs) - x0) / w));
      const j0 = Math.max(0, Math.floor((Math.min(...ys) - y0) / h));
      const j1 = Math.min(n - 1, Math.floor((Math.max(...ys) - y0) / h));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) cells[j * n + i].push(e);
    }
    this._bucket = { n, x0, y0, w, h, cells };
  }
}
