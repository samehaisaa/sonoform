// What the plate sounds like, and for how long.
//
// This mirrors sonoform/audio.py, and tests/js checks it against that file
// on the same plate. A point strike gives each mass-normalised mode a
// velocity equal to its value at the strike. Each mode radiates into the half
// space above the plate through the Rayleigh integral, and decays at the
// structural rate π η f plus whatever its own radiation carries away.
//
// Two things are added. The listener is a pair of ears wherever the camera
// is, each with its own Rayleigh integral, so turning the plate changes what
// you hear, as walking round a real one does. And the strike is a mallet
// rather than a delta: a half-sine of force lasting a fraction of a
// millisecond, whose spectrum rolls the highest partials off the way felt
// does.

export const AIR_DENSITY = 1.204;
export const AIR_SPEED = 343.0;

export function rigidity(material, thickness) {
  const nu = material.poisson;
  return (material.young * thickness ** 3) / (12 * (1 - nu * nu));
}

// f = Ω √(D / ρh) / (2π L²). Exact: Kirchhoff's equation has no other length
// or stiffness in it, so this is all a change of sheet or size can do.
export function hertz(omega, material, thickness, span) {
  const stiff = Math.sqrt(rigidity(material, thickness) / (material.density * thickness));
  return (omega * stiff) / (2 * Math.PI * span * span);
}

const NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];

// Nearest equal-tempered note, A4 = 440 Hz, and how far off it is in cents.
export function note(hz, a4 = 440) {
  const midi = 69 + 12 * Math.log2(hz / a4);
  const nearest = Math.round(midi);
  const cents = Math.round((midi - nearest) * 100);
  const name = NAMES[((nearest % 12) + 12) % 12];
  const octave = Math.floor(nearest / 12) - 1;
  const sign = cents > 0 ? '+' : cents < 0 ? '−' : '±';
  return {
    midi,
    name,
    octave,
    cents,
    label: `${name}${octave}`,
    offset: `${sign}${Math.abs(cents)}¢`,
  };
}

// |F(ω)| / F(0) for a half-sine force pulse of duration τ. The first zero is
// at 1.5 / τ, so 0.6 ms leaves the low partials alone and softens everything
// above a couple of kilohertz, which is what a felt mallet does.
export function malletSpectrum(omega, tau) {
  if (!(tau > 0)) return 1;
  const x = (omega * tau) / Math.PI;
  if (Math.abs(1 - x) < 1e-6) return Math.PI / 4;
  return Math.abs(Math.cos(0.5 * omega * tau) / (1 - x * x));
}

export class Acoustics {
  constructor(plate, material, { thickness, span, air = {}, shareClusters = true }) {
    this.plate = plate;
    this.material = material;
    this.thickness = thickness;
    this.span = span;
    this.airDensity = air.density ?? AIR_DENSITY;
    this.airSpeed = air.speed ?? AIR_SPEED;
    this.rhoH = material.density * thickness;
    const scale = Math.sqrt(rigidity(material, thickness) / this.rhoH) / (span * span);

    // A cluster is one note, so its members sound at one frequency. The
    // split between them is the mesh's, not the plate's, and left in it would
    // beat at a fraction of a hertz.
    const n = plate.modes;
    this.omega = new Float64Array(n);
    for (const members of plate.clusters) {
      let mean = 0;
      for (const m of members) mean += plate.omega[m];
      mean /= members.length;
      for (const m of members) this.omega[m] = (shareClusters ? mean : plate.omega[m]) * scale;
    }
    this.hz = this.omega.map((w) => w / (2 * Math.PI));
    this.wave = this.omega.map((w) => w / this.airSpeed);

    // φ on the page peaks at one. Mass-normalised, ∫ ρh φ² dS = 1 over the
    // physical plate, it is the page's φ times this.
    this.unit = new Float64Array(n);
    for (let m = 0; m < n; m++) {
      this.unit[m] = 1 / Math.sqrt(this.rhoH * span * span * plate.norms[m]);
    }
    this.structural = this.hz.map((f) => Math.PI * material.lossFactor * f);
    this._radiation = null;
  }

  // Amplitude decay rate from the power each mode radiates, by the self-power
  // of the Rayleigh integral over element centroids, as radiation_decay does.
  get radiation() {
    if (this._radiation) return this._radiation;
    const p = this.plate;
    const nt = p.nt;
    const L = this.span;
    const out = new Float64Array(p.modes);
    const cx = new Float64Array(nt), cy = new Float64Array(nt);
    for (let e = 0; e < nt; e++) {
      cx[e] = p.frames[3 * e] * L;
      cy[e] = p.frames[3 * e + 1] * L;
    }
    const u = new Float64Array(nt);
    for (let m = 0; m < p.modes; m++) {
      const field = p.mode(m);
      const k = this.wave[m];
      // The frame of every element is centred on its centroid, so the value
      // there is the constant coefficient.
      for (let e = 0; e < nt; e++) u[e] = field[21 * e] * this.unit[m] * p.areas[e] * L * L;
      let sum = 0;
      for (let a = 0; a < nt; a++) {
        sum += u[a] * u[a] * k;
        for (let b = a + 1; b < nt; b++) {
          const d = Math.hypot(cx[a] - cx[b], cy[a] - cy[b]);
          sum += 2 * u[a] * u[b] * (d < 1e-12 ? k : Math.sin(k * d) / d);
        }
      }
      out[m] = ((this.airDensity * this.omega[m]) / (4 * Math.PI)) * Math.max(sum, 0);
    }
    this._radiation = out;
    return out;
  }

  gamma(m) {
    return this.structural[m] + this.radiation[m];
  }

  // ∫ φ exp(-ikR) / R dS from the mass-normalised mode to a listener at
  // (x, y, z) metres, the plate lying in z = 0 and centred on the origin.
  rayleigh(m, x, y, z) {
    const q = this.plate.coarse;
    const v = q.values[m];
    const L = this.span, k = this.wave[m];
    let re = 0, im = 0;
    for (let i = 0; i < q.count; i++) {
      const dx = q.x[i] * L - x, dy = q.y[i] * L - y;
      const r = Math.sqrt(dx * dx + dy * dy + z * z);
      const w = (v[i] * q.weights[i]) / r;
      re += w * Math.cos(k * r);
      im -= w * Math.sin(k * r);
    }
    const s = this.unit[m] * L * L;
    return [re * s, im * s];
  }

  // Pressure phasor at an ear for modal velocity amplitude `velocity`.
  // The Rayleigh kernel carries iωρ/2π.
  pressure(m, velocity, ear) {
    const [re, im] = this.rayleigh(m, ear[0], ear[1], ear[2]);
    const c = (this.omega[m] * this.airDensity * velocity) / (2 * Math.PI);
    return [-c * im, c * re];
  }

  // Modal velocities from an impulse at `point` ({x, y, e} in plate units).
  strikeVelocities(point, impulse = 1, contact = 0) {
    const p = this.plate;
    const out = new Float64Array(p.modes);
    for (let m = 0; m < p.modes; m++) {
      const phi = p.valueIn(p.mode(m), point.e, point.x, point.y) * this.unit[m];
      out[m] = impulse * phi * malletSpectrum(this.omega[m], contact);
    }
    return out;
  }

  // Pressure at each ear after a strike, as two sample arrays. Each mode is a
  // decaying phasor advanced by one complex multiplication per sample, which
  // is exact for exp((iω - γ) t) and far cheaper than a sine per sample.
  strike(point, ears, { duration = 4, rate = 44100, impulse = 1, contact = 6e-4 } = {}) {
    const n = Math.round(duration * rate);
    const channels = ears.map(() => new Float32Array(n));
    const velocities = this.strikeVelocities(point, impulse, contact);
    const modes = [];
    for (let m = 0; m < this.plate.modes; m++) {
      if (this.hz[m] >= 0.5 * rate) continue;
      const g = this.gamma(m);
      const rr = Math.exp(-g / rate) * Math.cos(this.omega[m] / rate);
      const ri = Math.exp(-g / rate) * Math.sin(this.omega[m] / rate);
      // Past -100 dB there is nothing left to hear.
      const audible = Math.min(n, Math.ceil((11.5 / Math.max(g, 1e-6)) * rate));
      const phasors = [];
      ears.forEach((ear, c) => {
        const [pr, pi] = this.pressure(m, velocities[m], ear);
        phasors.push([pr, pi]);
        const out = channels[c];
        let zr = pr, zi = pi;
        for (let i = 0; i < audible; i++) {
          out[i] += zr;
          const t = zr * rr - zi * ri;
          zi = zr * ri + zi * rr;
          zr = t;
        }
      });
      modes.push({ index: m, hz: this.hz[m], gamma: g, velocity: velocities[m], pressure: phasors });
    }
    // Four milliseconds of fade at each end, so the clip does not click.
    const edge = Math.min(Math.floor(0.004 * rate), n >> 1);
    for (const out of channels) {
      for (let i = 0; i < edge; i++) {
        const w = i / Math.max(edge - 1, 1);
        out[i] *= w;
        out[n - 1 - i] *= w;
      }
    }
    return { channels, modes };
  }
}
