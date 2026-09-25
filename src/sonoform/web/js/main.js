// sonoform: the page.
//
// Everything on screen follows from three things solved ahead of time and
// shipped as data: the shape of each mode, its frequency parameter, and the
// outline. The rest happens here, every frame: how hard each note is ringing,
// where the sand goes, what the plate radiates toward the camera, and what
// that looks like.

import { AudioEngine } from './audio.js';
import { Atlas } from './atlas.js';
import { Camera } from './camera.js';
import { Acoustics } from './physics.js';
import { Plate } from './plate.js';
import { Renderer } from './renderer.js';
import { DIFFUSION, FLOOR, Sand } from './sand.js';
import { copyLink, readHash, saveImage, writeHash } from './share.js';
import { FINISHES, finish } from './stone.js';
import { Tour } from './tour.js';
import { UI } from './ui.js';
import { clamp } from './util.js';

const norm3 = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / l, v[1] / l, v[2] / l];
};
const KEY_DIR = norm3([-0.42, 0.36, 0.83]);
const KEY_COLOR = [3.1, 2.9, 2.6];
const FILL_DIR = norm3([0.62, -0.52, 0.46]);
const FILL_COLOR = [0.42, 0.5, 0.64];

const STROBE_HZ = 0.5; // how fast the stroboscope makes the sheet seem to move
const STROBE_DEPTH = 0.055; // its exaggerated bend, in plate units
const HOLO_AMPLITUDE = 1.5e-6; // metres at the antinode, bowed as hard as it goes
const LASER = 632.8e-9; // helium-neon
// How fast a bow builds a note. The bow sets this, not the plate, so it is a
// choice. The ring-down after the bow lifts is not a choice: it is the mode's
// own decay rate, structural plus radiated.
const BOW_RISE = 0.45;
const LISTEN_AT = 0.55; // metres from the plate centre, toward the camera
const EAR_GAP = 0.175; // metres between the ears
const BOW_GAIN = 0.2;
const STRIKE_GAIN = 0.34;
const STRIKE_DRIVE = 0.75; // how hard a tap shakes the sand, against a full bow

const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

class App {
  constructor() {
    this.canvas = document.getElementById('gl');
    this.state = { plate: 'square', metal: 'brass', thickness: 0.002, span: 0.18, note: 0, view: 'sand', nodes: false };
    this.payloads = new Map();
    this.voices = [];
    this.strikes = [];
    this.ripples = [];
    this.bowing = false;
    this.latched = false;
    this.time = 0;
    this.frame = 0;
    this.zen = null;
    this.figure = 0;
    this.started = false;
  }

  async start() {
    try {
      this.renderer = new Renderer(this.canvas);
    } catch (err) {
      document.getElementById('fallbackText').textContent = err.message;
      document.getElementById('fallback').hidden = false;
      document.getElementById('ui').hidden = true;
      return;
    }
    document.body.classList.add('intro-on');
    this.camera = new Camera();
    if (reduced) this.camera.drift = 0;
    this.audio = new AudioEngine();
    this.manifest = await getJSON('data/manifest.json');
    Object.assign(this.state, readHash(this.manifest, this.state));
    this.deepLink = !!location.hash;

    const small = Math.min(screen.width, screen.height) < 700;
    this.gridSize = small ? 256 : 384;
    this.sand = new Sand(small ? 16000 : 30000);
    this.renderer.setSand(this.sand);

    this.ui = new UI(this, this.manifest, FINISHES);
    this.atlas = new Atlas(this);
    this.tour = new Tour(this);
    this._input();
    this._keys();

    await this.load(this.state.plate, this.state.metal, { pour: false });
    this.ui.ready();
    requestAnimationFrame((t) => this._loop(t));
  }

  // Called by the intro's button: the gesture that browsers need for sound.
  begin(tour) {
    if (this.started) return;
    this.started = true;
    this.audio.unlock();
    document.body.classList.remove('intro-on');
    this.pour();
    this.camera.fly({ az: -Math.PI / 2 - 0.32, el: 0.7, glide: 1.4 });
    this.ui.layout();
    if (tour && !this.deepLink) this.tour.start();
    else this.ui.hint();
  }

  // --- plates, sheets, notes -------------------------------------------

  material(key = this.state.metal) {
    return this.manifest.materials.find((m) => m.key === key);
  }

  async payload(key, metal) {
    const entry = this.manifest.plates.find((p) => p.key === key);
    const file = entry.files[this.material(metal).poisson.toFixed(2)];
    if (!this.payloads.has(file)) this.payloads.set(file, getJSON(`data/${file}`));
    return this.payloads.get(file);
  }

  async load(key, metal, { pour = true } = {}) {
    // what was asked for last, so a quick plate-then-sheet does not reload
    // the old plate in the new sheet
    this.wanted = { plate: key, metal };
    const token = (this._loading = {});
    this.ui.busy(true);
    let data;
    try {
      data = await this.payload(key, metal);
    } catch (err) {
      this.ui.toast(`Could not load that plate: ${err.message}`);
      this.ui.busy(false);
      return;
    }
    if (token !== this._loading) return;
    const sameOutline = this.plate && this.plate.key === key;
    const plate = new Plate(data);
    this.plate = plate;
    this.state.plate = key;
    this.state.metal = metal;
    this.grid = plate.makeGrid(this.gridSize);
    this.strikeGrid = plate.makeGrid(144);
    this.modeGrids = new Map();
    this.strikeModes = null;
    this.voices = [];
    this.strikes = [];
    this.radius = 0;
    for (let i = 0; i < plate.outline.length; i += 2) {
      this.radius = Math.max(this.radius, Math.hypot(plate.outline[i], plate.outline[i + 1]));
    }
    this.renderer.setPlate(plate);
    this.renderer.setGround(plate);
    this._acoustics();
    this.state.note = clamp(this.state.note, 0, plate.clusters.length - 1);
    if (!sameOutline || !this.bowAt) this.bowAt = this._bestRim(0);
    else this.bowAt = plate.rimPoint(this.bowAt.index);
    this._refreshField();
    if (sameOutline) this.sand.settle(this.grid);
    else if (pour) this.pour();
    else this.sand.pour(this.grid, { now: false });
    this.ui.plate();
    this.ui.busy(false);
    writeHash(this.state);
    if (!sameOutline) this.ui.layout({ now: !this.started });
  }

  _acoustics() {
    const m = this.material();
    this.acoustics = new Acoustics(this.plate, m, {
      thickness: this.state.thickness,
      span: this.state.span,
      air: this.manifest.air,
    });
    this.notes = this.plate.clusters.map((members, i) => ({
      index: i,
      members,
      hz: this.acoustics.hz[members[0]],
    }));
    this._reference = null;
  }

  // Thickness or span changed: every note rescales, nothing else does.
  setDimensions(thickness, span) {
    this.state.thickness = thickness;
    this.state.span = span;
    this._acoustics();
    for (const v of this.voices) v.gamma = this._gamma(v);
    this.ui.notes();
    writeHash(this.state);
  }

  setMetal(key) {
    const want = this.wanted || this.state;
    if (key === want.metal) return;
    this.load(want.plate, key);
  }

  setPlate(key) {
    const want = this.wanted || this.state;
    if (key === want.plate) return;
    this.state.note = 0;
    this.load(key, want.metal);
  }

  pour() {
    this.sand.pour(this.grid, { from: 0.1, to: 0.38, over: 1.2 });
    this.voices.forEach((v) => (v.level *= 0.2));
  }

  selectNote(i, { quiet = false } = {}) {
    i = clamp(i, 0, this.notes.length - 1);
    if (i === this.state.note && this.voices.length) return;
    this.state.note = i;
    this._refreshField();
    this.ui.selectNote(i);
    if (!quiet) this.ui.explain();
    writeHash(this.state);
    this.emit('note', i);
  }

  // --- the field of the selected note ------------------------------------

  modeGrid(m) {
    if (!this.modeGrids.has(m)) this.modeGrids.set(m, this.plate.sampleGrid(this.grid, this.plate.mode(m)));
    return this.modeGrids.get(m);
  }

  // The field a bow at `at` drives for note i. A single mode is itself. A
  // degenerate cluster answers as the one combination with the most motion
  // under the bow: the projection of a point force onto the eigenspace,
  // Σ φ_m(p) φ_m over orthonormal members. Everything orthogonal to it has a
  // node exactly where the bow is, so the bow cannot reach it.
  noteField(i, at) {
    const p = this.plate;
    const members = this.notes[i].members;
    if (members.length === 1) {
      const m = members[0];
      return {
        members, weights: [1], coef: p.mode(m), grid: this.modeGrid(m),
        drive: Math.abs(p.valueIn(p.mode(m), at.e, at.x, at.y)),
      };
    }
    let weights = members.map((m) => p.valueIn(p.mode(m), at.e, at.x, at.y) / p.norms[m]);
    if (weights.every((w) => Math.abs(w) < 1e-9)) weights = members.map((_, k) => (k === 0 ? 1 : 0));
    const grids = members.map((m) => this.modeGrid(m));
    const grid = new Float32Array(grids[0].length);
    let peak = 0;
    for (let k = 0; k < grid.length; k++) {
      let s = 0;
      for (let j = 0; j < members.length; j++) s += weights[j] * grids[j][k];
      grid[k] = s;
      if (Math.abs(s) > peak) peak = Math.abs(s);
    }
    const inv = peak > 0 ? 1 / peak : 1;
    for (let k = 0; k < grid.length; k++) grid[k] *= inv;
    weights = weights.map((w) => w * inv);
    const coef = p.combine(members, weights);
    const drive = Math.abs(p.valueIn(coef, at.e, at.x, at.y));
    return { members, weights, coef, grid, drive };
  }

  // How well a bow on the rim at each point could drive note i, zero to one.
  // For a cluster this is the envelope of the whole eigenspace, which does
  // not depend on which combination ends up sounding.
  rimReach(i) {
    const p = this.plate;
    const members = this.notes[i].members;
    const n = p.outline.length / 2;
    const out = new Float32Array(n);
    let peak = 0;
    for (let r = 0; r < n; r++) {
      const at = p.rimPoint(r);
      let s = 0;
      for (const m of members) s += p.valueIn(p.mode(m), at.e, at.x, at.y) ** 2 / p.norms[m];
      out[r] = Math.sqrt(s);
    }
    // the envelope's largest value anywhere, from the grids already sampled
    for (let k = 0; k < this.grid.n * this.grid.n; k += 3) {
      if (this.grid.sdf[k] > 0) continue;
      let s = 0;
      for (const m of members) s += this.modeGrid(m)[k] ** 2 / p.norms[m];
      if (s > peak) peak = s;
    }
    peak = Math.sqrt(peak) || 1;
    for (let r = 0; r < n; r++) out[r] = Math.min(1, out[r] / peak);
    return out;
  }

  // Where the bow drives note i hardest. A symmetric plate has several such
  // places; of those, the one on the far side, so the bow stands behind the
  // plate rather than between it and the viewer.
  _bestRim(i) {
    const reach = this.rimReach(i);
    let top = 0;
    for (let r = 0; r < reach.length; r++) top = Math.max(top, reach[r]);
    const az = this.camera ? this.camera.goal.az : 0;
    const ex = Math.cos(az), ey = Math.sin(az), o = this.plate.outline;
    let best = 0, far = Infinity;
    for (let r = 0; r < reach.length; r++) {
      if (reach[r] < 0.97 * top) continue;
      const toward = o[2 * r] * ex + o[2 * r + 1] * ey;
      if (toward < far) {
        far = toward;
        best = r;
      }
    }
    return this.plate.rimPoint(best);
  }

  moveBowToBest() {
    this.bowAt = this._bestRim(this.state.note);
    this._refreshField();
  }

  // Recompute what the selected note looks like with the bow where it is.
  _refreshField() {
    const i = this.state.note;
    const f = this.noteField(i, this.bowAt);
    this.field = f;
    this.reach = this.rimReach(i);
    this.renderer.setField(f.coef);
    const key = `${this.plate.key}:${this.state.metal}:${i}`;
    let v = this.voices.find((x) => x.key === key);
    if (!v) {
      v = { key, note: i, level: 0 };
      this.voices.unshift(v);
    } else {
      this.voices.splice(this.voices.indexOf(v), 1);
      this.voices.unshift(v);
    }
    Object.assign(v, { grid: f.grid, members: f.members, weights: f.weights, drive: f.drive });
    v.gamma = this._gamma(v);
    // keep one previous note ringing down, as a real plate would
    this.voices = this.voices.slice(0, 3);
    this.emit('field');
  }

  // Orthonormal weights of a voice's combination, for anything physical.
  _unitWeights(v) {
    const a = this.acoustics;
    const c = v.members.map((m, j) => v.weights[j] / a.unit[m]);
    const l = Math.hypot(...c) || 1;
    return c.map((x) => x / l);
  }

  _gamma(v) {
    const b = this._unitWeights(v);
    return v.members.reduce((s, m, j) => s + b[j] * b[j] * this.acoustics.gamma(m), 0);
  }

  setBow(on, { latch = false } = {}) {
    if (!this.plate) return;
    if (on) this.audio.unlock();
    this.bowing = on;
    this.latched = on && latch;
    this.ui.bowState();
    this.emit(on ? 'bow' : 'unbow');
  }

  toggleLatch() {
    if (this.bowing && this.latched) this.setBow(false);
    else this.setBow(true, { latch: true });
  }

  placeBow(x, y) {
    const edge = this.plate.nearestEdge(x, y);
    const n = this.plate.outline.length / 2;
    const index = edge.t < 0.5 ? edge.index : (edge.index + 1) % n;
    if (this.bowAt && this.bowAt.index === index) return;
    this.bowAt = this.plate.rimPoint(index);
    this._refreshField();
    this.ui.explain();
  }

  // --- a tap ---------------------------------------------------------------

  strike(x, y) {
    const p = this.plate;
    const e = p.locate(x, y);
    if (e < 0) return;
    this.audio.unlock();
    const a = this.acoustics;
    const point = { x, y, e };
    const velocity = a.strikeVelocities(point, 1, 6e-4);
    // modal displacement, in the page's units, for the sand
    const c = new Float64Array(p.modes);
    for (let m = 0; m < p.modes; m++) c[m] = (velocity[m] / a.omega[m]) / a.unit[m];
    if (!this.strikeModes) this.strikeModes = Array.from({ length: p.modes }, (_, m) => p.sampleGrid(this.strikeGrid, p.mode(m)));
    let peak = 0;
    for (let k = 0; k < this.strikeModes[0].length; k++) {
      let s = 0;
      for (let m = 0; m < p.modes; m++) s += (c[m] * this.strikeModes[m][k]) ** 2;
      if (s > peak) peak = s;
    }
    const scale = peak > 0 ? STRIKE_DRIVE / Math.sqrt(peak) : 0;
    const gammas = Float64Array.from({ length: p.modes }, (_, m) => a.gamma(m));
    this.strikes = [{ t: 0, c: c.map((v) => v * scale), gammas, values: new Float32Array(this.strikeModes[0].length), level: STRIKE_DRIVE, next: 0 }];
    this.ripples.push({ x, y, t: 0 });
    this.flash = 0.06;

    const ears = this._ears();
    const slowest = Math.min(...gammas);
    const duration = clamp(6.9 / slowest, 1.5, 7);
    const rate = this.audio.ctx ? this.audio.ctx.sampleRate : 44100;
    const { channels } = a.strike(point, ears, { duration, rate, contact: 6e-4 });
    const ref = this._strikeReference();
    this.audio.play(channels, rate, STRIKE_GAIN / ref);
    this.emit('strike');
  }

  // A tap at a random point well inside the plate, for the keyboard.
  tapSomewhere() {
    const g = this.grid;
    for (let tries = 0; tries < 200; tries++) {
      const x = g.x0 + Math.random() * g.n * g.h, y = g.y0 + Math.random() * g.n * g.h;
      if (this.sand._sdf(g, x, y) < -0.08 && this.plate.locate(x, y) >= 0) {
        this.strike(x, y);
        return;
      }
    }
  }

  // A fixed yardstick for loudness on this plate, so a tap on a quiet spot
  // sounds quieter rather than being normalised back up.
  _strikeReference() {
    if (this._strikeRef) return this._strikeRef;
    const ears = this._ears(this._defaultListener());
    const at = this._bestRim(0);
    const { channels } = this.acoustics.strike(at, ears, { duration: 0.4, rate: 22050, contact: 6e-4 });
    let peak = 0;
    for (const ch of channels) for (let i = 0; i < ch.length; i++) peak = Math.max(peak, Math.abs(ch[i]));
    this._strikeRef = peak || 1;
    return this._strikeRef;
  }

  // --- sound ---------------------------------------------------------------

  _defaultListener() {
    const az = -Math.PI / 2 - 0.32, el = 0.7;
    return { eye: [Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)], right: [-Math.sin(az), Math.cos(az), 0] };
  }

  // Two ears on the camera's line of sight, LISTEN_AT from the plate.
  _ears(view = this.camera.last) {
    const e = view.eye, l = Math.hypot(e[0], e[1], e[2]);
    const c = [(e[0] / l) * LISTEN_AT, (e[1] / l) * LISTEN_AT, Math.max((e[2] / l) * LISTEN_AT, 0.05)];
    const r = view.right, g = EAR_GAP / 2;
    return [[c[0] - r[0] * g, c[1] - r[1] * g, c[2] - r[2] * g], [c[0] + r[0] * g, c[1] + r[1] * g, c[2] + r[2] * g]];
  }

  // Pressure at each ear from a voice at unit modal velocity.
  _voicePressure(v, ears) {
    const b = this._unitWeights(v);
    return ears.map((ear) => {
      let re = 0, im = 0;
      v.members.forEach((m, j) => {
        const [pr, pi] = this.acoustics.pressure(m, b[j], ear);
        re += pr;
        im += pi;
      });
      return Math.hypot(re, im);
    });
  }

  // The median note's pressure, as the yardstick bowed levels are read on.
  _bowReference() {
    if (this._reference) return this._reference;
    const ears = this._ears(this._defaultListener());
    const levels = this.notes.map((n) => {
      const v = { members: [n.members[0]], weights: [1] };
      const [l, r] = this._voicePressure(v, ears);
      return Math.hypot(l, r);
    }).filter((x) => x > 0).sort((a, b) => a - b);
    this._reference = levels.length ? levels[levels.length >> 1] : 1;
    return this._reference;
  }

  _sound() {
    if (!this.audio.ctx) return;
    const ears = this._ears();
    const keep = new Set();
    const ref = this._bowReference();
    for (const v of this.voices) {
      if (v.level < 1e-3) continue;
      keep.add(v.key);
      const [pl, pr] = this._voicePressure(v, ears);
      // levels follow what reaches each ear, read on a square-root scale so
      // the quietest radiators can still be heard
      const gl = BOW_GAIN * v.level * Math.min(1.8, Math.sqrt(pl / ref));
      const gr = BOW_GAIN * v.level * Math.min(1.8, Math.sqrt(pr / ref));
      const dl = Math.hypot(...ears[0]) / 343, dr = Math.hypot(...ears[1]) / 343;
      const d0 = Math.min(dl, dr);
      this.audio.sing(v.key, this.notes[v.note] ? this.notes[v.note].hz : 220, gl, gr, dl - d0, dr - d0);
    }
    this.audio.release(keep);
  }

  // --- the frame -------------------------------------------------------------

  _loop(now) {
    requestAnimationFrame((t) => this._loop(t));
    const dt = Math.min(0.05, Math.max(0, (now - (this._last || now)) / 1000));
    this._last = now;
    this.time += dt;
    this.frame++;
    if (!this.plate) return;

    // how hard each note is ringing
    const current = this.voices[0];
    for (const v of this.voices) {
      if (v === current && this.bowing) v.level += (v.drive - v.level) * (1 - Math.exp(-dt / BOW_RISE));
      else v.level *= Math.exp(-v.gamma * dt);
    }
    this.voices = this.voices.filter((v) => v === current || v.level > 1e-3);

    // a ringing strike, as a root-sum-square field on its own grid
    let strikeField = null;
    for (const s of this.strikes) {
      s.t += dt;
      if (s.t >= s.next) {
        s.next = s.t + 0.07;
        const w = s.c.map((c, m) => (c * Math.exp(-s.gammas[m] * s.t)) ** 2);
        let peak = 0;
        const out = s.values, modes = this.strikeModes;
        for (let k = 0; k < out.length; k++) {
          let q = 0;
          for (let m = 0; m < w.length; m++) q += w[m] * modes[m][k] * modes[m][k];
          out[k] = Math.sqrt(q);
          if (out[k] > peak) peak = out[k];
        }
        s.level = peak;
      }
      strikeField = { grid: this.strikeGrid, values: s.values, level: s.level };
    }
    this.strikes = this.strikes.filter((s) => s.level > 0.004);

    this.sand.step(dt, this.grid, {
      voices: this.voices.filter((v) => v.level > 1e-4 || v === current).map((v) => ({ values: v.grid, level: v.level })),
      strike: strikeField,
      diffusion: DIFFUSION,
      floor: FLOOR,
    });
    if (this.frame % 3 === 0) this.sand.piles(this.grid);
    if (this.frame % 12 === 0) this._measureFigure();
    // the plate waits bare behind the title, and Begin pours the sand
    this.renderer.updateSand(this.sand.pack(), this.started ? this.sand.count : 0);

    const still = !this.pointer && !this.bowing;
    this.camera.update(dt, still);
    const aspect = this.canvas.clientWidth / Math.max(this.canvas.clientHeight, 1);
    const cam = this.camera.matrices(aspect);
    if (this.frame % 3 === 0) this._sound();
    if (this.zen) this._zen(dt);
    this.tour.update(dt);
    this.ui.frame(dt);

    this.renderer.render(this._scene(cam, dt));
  }

  _scene(cam, dt) {
    const level = this.voices[0] ? this.voices[0].level : 0;
    const view = this.state.view;
    const wave = Math.cos(2 * Math.PI * STROBE_HZ * this.time);
    const keyView = [
      cam.right[0] * KEY_DIR[0] + cam.right[1] * KEY_DIR[1] + cam.right[2] * KEY_DIR[2],
      cam.up[0] * KEY_DIR[0] + cam.up[1] * KEY_DIR[1] + cam.up[2] * KEY_DIR[2],
      cam.back[0] * KEY_DIR[0] + cam.back[1] * KEY_DIR[1] + cam.back[2] * KEY_DIR[2],
    ];
    this.flash = Math.max(0, (this.flash || 0) - dt * 0.4);

    // glows and rings on the plate
    const sprites = [];
    const at = this.bowAt;
    const shown = this.started;
    const idle = this.bowing ? 0 : 1;
    const pulse = 0.55 + 0.45 * Math.sin(this.time * 2.4);
    // where the bow touches: a small warm glow that breathes while idle, to
    // say "here", and steadies while bowing
    sprites.push([at.x, at.y, 0.032, 0, 0.95, 0.92, 0.86, this.bowing ? 0.12 + 0.16 * level : 0.07 + 0.08 * pulse]);
    if (idle && this.reach && view !== 'hologram') {
      const o = this.plate.outline, n = o.length / 2, step = Math.max(1, Math.round(n / 70));
      for (let r = 0; r < n; r += step) {
        const q = this.reach[r];
        if (q < 0.55) continue;
        sprites.push([o[2 * r], o[2 * r + 1], 0.011 + 0.008 * q, 2, 0.95, 0.92, 0.86, 0.45 * (q - 0.5) * this.ui.rimHint]);
      }
    }
    for (const r of this.ripples) {
      r.t += dt;
      const k = r.t / 0.9;
      if (k < 1) sprites.push([r.x, r.y, 0.03 + 0.28 * k, 1, 0.95, 0.93, 0.88, 1.1 * (1 - k) * (1 - k)]);
    }
    this.ripples = this.ripples.filter((r) => r.t < 0.9);

    // the bow, standing on the rim with its hair against the edge
    const nx = this.plate.rimNormals[2 * at.index], ny = this.plate.rimNormals[2 * at.index + 1];
    const stroke = this.bowing ? 0.12 * Math.sin((this.time * 2 * Math.PI) / 3.4) : 0.1;
    const ex = this.plate.outline[2 * at.index] + nx * 0.004, ey = this.plate.outline[2 * at.index + 1] + ny * 0.004;
    // held from above, leaning out from the plate and a little across it
    const lean = 0.2, cross = 0.12;
    const tx = -ny, ty = nx;
    const up = norm3([nx * lean + tx * cross, ny * lean + ty * cross, 1]);
    const out = norm3([nx - up[0] * (nx * up[0] + ny * up[1]), ny - up[1] * (nx * up[0] + ny * up[1]), -up[2] * (nx * up[0] + ny * up[1])]);
    const side = [out[1] * up[2] - out[2] * up[1], out[2] * up[0] - out[0] * up[2], out[0] * up[1] - out[1] * up[0]];
    const model = new Float32Array([
      side[0], side[1], side[2], 0,
      out[0], out[1], out[2], 0,
      up[0], up[1], up[2], 0,
      ex + up[0] * stroke, ey + up[1] * stroke, 0.02 + up[2] * stroke, 1,
    ]);
    const look = finish(this.state.metal);

    return {
      viewProj: cam.viewProj,
      eye: cam.eye,
      keyView,
      keyDir: KEY_DIR,
      keyColor: KEY_COLOR,
      fillDir: FILL_DIR,
      fillColor: FILL_COLOR,
      envTurn: 0,
      material: look,
      thickness: this.state.thickness / this.state.span,
      time: this.time,
      view: view === 'strobe' ? 1 : view === 'hologram' ? 2 : 0,
      disp: view === 'strobe' ? STROBE_DEPTH * level * wave : 0,
      signed: level * wave,
      holoK: view === 'hologram' ? ((4 * Math.PI * HOLO_AMPLITUDE) / LASER) * level : 0,
      nodes: this.state.nodes ? 1 : this.ui.nodeHint,
      field: 1,
      hop: reduced ? 0 : 0.011,
      grain: 0.0027,
      pixel: this.canvas.clientHeight / (2 * Math.tan(this.camera.fov / 2)),
      sprites: shown ? sprites : [],
      bow: { model, fade: view === 'hologram' || !this.started ? 0 : this.bowing ? 1 : 0.7 },
      exposure: view === 'hologram' ? 1.05 : 1.0,
      flash: this.flash,
      sandVisible: view !== 'hologram',
    };
  }

  // How much of the sand sits on the still lines of the sounding note.
  _measureFigure() {
    const s = this.sand;
    let near = 0;
    const step = Math.max(1, Math.floor(s.count / 4000));
    let total = 0;
    for (let i = 0; i < s.count; i += step) {
      total++;
      if (Math.abs(s.amp[i]) < 0.07) near++;
    }
    this.figure = near / Math.max(total, 1);
  }

  // --- zen: the page plays itself ---------------------------------------

  toggleZen(on = !this.zen) {
    if (on) {
      this.audio.unlock();
      this.tour.stop();
      this.zen = { t: 0, phase: 'rest', wait: 1.5, played: [] };
      document.body.classList.add('zen');
      this.ui.toast('zen · any key to come back');
    } else if (this.zen) {
      this.zen = null;
      document.body.classList.remove('zen');
      if (!this.latched) this.setBow(false);
    }
  }

  _zen(dt) {
    const z = this.zen;
    z.t += dt;
    if (z.t < z.wait) return;
    z.t = 0;
    if (z.phase === 'rest') {
      // pick a note not played lately, and put the bow where it sings
      const options = this.notes.map((n) => n.index).filter((i) => !z.played.includes(i));
      const i = options.length ? options[Math.floor(Math.random() * options.length)] : 0;
      z.played = [...z.played.slice(-5), i];
      this.selectNote(i, { quiet: true });
      this.moveBowToBest();
      this.setBow(true);
      z.phase = 'bow';
      z.wait = 9 + 3 * Math.random();
    } else {
      this.setBow(false);
      z.phase = 'rest';
      z.wait = 3.5 + 2 * Math.random();
    }
  }

  // --- input -------------------------------------------------------------

  _input() {
    const c = this.canvas;
    const pointers = new Map();
    let mode = null, pinch = 0, down = null;

    const local = (e) => {
      const r = c.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    };
    const onPlate = (px, py) => this.camera.pick(px, py, c.clientWidth, c.clientHeight, 0);

    c.addEventListener('pointerdown', (e) => {
      if (this.zen) this.toggleZen(false);
      this.audio.unlock();
      c.setPointerCapture(e.pointerId);
      const [px, py] = local(e);
      pointers.set(e.pointerId, [px, py]);
      this.pointer = true;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinch = Math.hypot(a[0] - b[0], a[1] - b[1]);
        mode = 'pinch';
        if (this.bowing && !this.latched) this.setBow(false);
        return;
      }
      down = { x: px, y: py, moved: 0 };
      const hit = onPlate(px, py);
      mode = 'tap';
      if (hit && this.plate) {
        const edge = this.plate.nearestEdge(hit[0], hit[1]);
        const toBow = Math.hypot(hit[0] - this.bowAt.x, hit[1] - this.bowAt.y);
        if (toBow < 0.075 || edge.distance < 0.045) {
          mode = 'bow';
          this.placeBow(hit[0], hit[1]);
          this.setBow(true);
        }
      }
      down.hit = hit;
    });

    c.addEventListener('pointermove', (e) => {
      const [px, py] = local(e);
      if (!pointers.has(e.pointerId)) {
        this._hover(px, py);
        return;
      }
      const prev = pointers.get(e.pointerId);
      pointers.set(e.pointerId, [px, py]);
      if (mode === 'pinch' && pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (pinch > 0) this.camera.zoom(pinch / d);
        pinch = d;
        return;
      }
      const dx = px - prev[0], dy = py - prev[1];
      if (down) down.moved += Math.hypot(dx, dy);
      if (mode === 'bow') {
        const hit = onPlate(px, py);
        if (hit) this.placeBow(hit[0], hit[1]);
        return;
      }
      if (mode === 'tap' && down && down.moved > 6) {
        mode = 'orbit';
        c.classList.add('grabbing');
      }
      if (mode === 'orbit') this.camera.orbit(dx, dy);
    });

    const end = (e) => {
      if (!pointers.has(e.pointerId)) return;
      pointers.delete(e.pointerId);
      c.classList.remove('grabbing');
      if (mode === 'bow' && !this.latched) this.setBow(false);
      if (mode === 'tap' && down && down.moved <= 6 && down.hit && this.plate.locate(down.hit[0], down.hit[1]) >= 0) {
        this.strike(down.hit[0], down.hit[1]);
      }
      if (pointers.size === 0) {
        mode = null;
        down = null;
        this.pointer = false;
      }
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.camera.zoom(Math.exp(e.deltaY * 0.0012));
    }, { passive: false });
    window.addEventListener('resize', () => this.ui.resize());
  }

  _hover(px, py) {
    const c = this.canvas;
    const hit = this.camera.pick(px, py, c.clientWidth, c.clientHeight, 0);
    c.classList.remove('over-rim', 'over-plate');
    if (!hit || !this.plate) return;
    const edge = this.plate.nearestEdge(hit[0], hit[1]);
    const toBow = Math.hypot(hit[0] - this.bowAt.x, hit[1] - this.bowAt.y);
    if (toBow < 0.075 || edge.distance < 0.045) {
      c.classList.add('over-rim');
      this.ui.rimHover(true);
    } else if (this.plate.locate(hit[0], hit[1]) >= 0) {
      c.classList.add('over-plate');
      this.ui.rimHover(false);
    } else {
      this.ui.rimHover(false);
    }
  }

  _keys() {
    const held = new Set();
    window.addEventListener('keydown', (e) => {
      if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
      if (this.zen && e.key !== 'z' && e.key !== 'Z') {
        this.toggleZen(false);
        return;
      }
      if (document.body.classList.contains('intro-on')) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          this.ui.clickBegin();
        }
        return;
      }
      const k = e.key.toLowerCase();
      if (e.key === ' ') {
        e.preventDefault();
        if (held.has(' ')) return;
        held.add(' ');
        this.setBow(true);
        this._spaceAt = performance.now();
        return;
      }
      if (e.key === 'ArrowRight') this.selectNote(this.state.note + 1);
      else if (e.key === 'ArrowLeft') this.selectNote(this.state.note - 1);
      else if (/^[1-9]$/.test(e.key)) this.selectNote(Number(e.key) - 1);
      else if (k === 'b') this.toggleLatch();
      else if (k === 't') this.tapSomewhere();
      else if (k === 'v') this.ui.cycleView();
      else if (k === 'n') this.ui.toggleNodes();
      else if (k === 'p') this.pour();
      else if (k === 'a') this.atlas.toggle();
      else if (k === 'z') this.toggleZen();
      else if (k === 'm') this.ui.toggleMute();
      else if (k === 'i') this.ui.toggleSheet();
      else if (k === 's') this.ui.toggleTray();
      else if (e.key === 'Escape') this.ui.closeAll();
    });
    window.addEventListener('keyup', (e) => {
      if (e.key !== ' ') return;
      held.delete(' ');
      // a short press keeps bowing; a held one stops on release
      const quick = performance.now() - (this._spaceAt || 0) < 260;
      if (quick) this.setBow(true, { latch: true });
      else if (!this.latched) this.setBow(false);
    });
  }

  // --- small event bus for the tour and the interface -------------------

  on(name, fn) {
    (this._events ||= {})[name] ||= [];
    this._events[name].push(fn);
  }

  emit(name, ...args) {
    for (const fn of (this._events && this._events[name]) || []) fn(...args);
  }

  shareLink() {
    copyLink().then((ok) => this.ui.toast(ok ? 'link copied' : location.href));
  }

  saveImage() {
    const note = this.notes[this.state.note];
    saveImage(this, `${this.plate.key}-${Math.round(note.hz)}hz`);
  }
}

const app = new App();
window.sonoform = app;
app.start().catch((err) => {
  console.error(err);
  const text = document.getElementById('fallbackText');
  if (text) text.textContent = `Something went wrong while starting: ${err.message}`;
  document.getElementById('fallback').hidden = false;
});
