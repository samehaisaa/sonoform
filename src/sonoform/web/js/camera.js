// An orbit around the plate, damped so it glides rather than jumps.

import { mat4 } from './gl.js';
import { clamp } from './util.js';

export class Camera {
  constructor() {
    this.fov = 0.56;
    this.az = -Math.PI / 2 - 0.55;
    this.el = 0.52;
    this.dist = 2.3;
    this.goal = { az: this.az, el: this.el, dist: this.dist };
    this.minEl = 0.14;
    this.maxEl = 1.48;
    this.minDist = 0.85;
    this.maxDist = 5.5;
    // Where the plate should sit on screen, in normalised device units, so
    // it centres in the space the interface leaves free.
    this.offsetX = 0;
    this.offsetY = 0;
    this.room = { hw: 0.8, hh: 0.8 };
    this.drift = 0.035; // radians per second while nobody is touching it
    this.idle = 0;
    this.glide = 0.16;
  }

  orbit(dx, dy) {
    this.goal.az -= dx * 0.0065;
    this.goal.el = clamp(this.goal.el + dy * 0.005, this.minEl, this.maxEl);
    this.idle = 0;
  }

  zoom(factor) {
    this.goal.dist = clamp(this.goal.dist * factor, this.minDist, this.maxDist);
    this.idle = 0;
  }

  fly({ az = this.goal.az, el = this.goal.el, dist = this.goal.dist, glide = 0.9 } = {}) {
    if (!Number.isFinite(dist)) dist = this.goal.dist;
    this.goal = { az, el: clamp(el, this.minEl, this.maxEl), dist: clamp(dist, this.minDist, this.maxDist) };
    this.glide = glide;
  }

  // The free part of the screen, from CSS-pixel insets for the interface.
  frame(width, height, { left = 0, right = 0, top = 0, bottom = 0 }, now = false) {
    // a window mid-resize can report no size at all, and one NaN here would
    // stick in every smoothed value after it
    if (!(width > 1 && height > 1)) return;
    const cx = (left + (width - right)) / 2, cy = (top + (height - bottom)) / 2;
    this.aim = [(cx / width) * 2 - 1, 1 - (cy / height) * 2];
    if (now) [this.offsetX, this.offsetY] = this.aim;
    this.room = { hw: Math.max(0.2, (width - left - right) / width), hh: Math.max(0.2, (height - top - bottom) / height) };
  }

  // Far enough back that a plate of this radius fits the free area.
  fit(radius, aspect) {
    const f = 1 / Math.tan(this.fov / 2);
    const vertical = (radius * f) / (0.9 * this.room.hh);
    const horizontal = (radius * f) / (0.9 * this.room.hw * aspect);
    return clamp(Math.max(vertical * 0.82, horizontal), this.minDist, this.maxDist);
  }

  update(dt, still) {
    if (!Number.isFinite(dt)) dt = 0;
    for (const key of ['az', 'el', 'dist']) if (!Number.isFinite(this[key])) this[key] = this.goal[key];
    if (!Number.isFinite(this.offsetX) || !Number.isFinite(this.offsetY)) [this.offsetX, this.offsetY] = this.aim || [0, 0];
    this.idle += dt;
    if (still && this.idle > 6) this.goal.az += this.drift * dt * Math.min(1, (this.idle - 6) / 4);
    const k = 1 - Math.exp(-dt / this.glide);
    this.az += (this.goal.az - this.az) * k;
    this.el += (this.goal.el - this.el) * k;
    this.dist += (this.goal.dist - this.dist) * k;
    if (this.aim) {
      const j = 1 - Math.exp(-dt / 0.5);
      this.offsetX += (this.aim[0] - this.offsetX) * j;
      this.offsetY += (this.aim[1] - this.offsetY) * j;
    }
    if (Math.abs(this.goal.az - this.az) < 1e-3 && this.glide > 0.16) this.glide = Math.max(0.16, this.glide * 0.9);
  }

  matrices(aspect) {
    const ce = Math.cos(this.el), se = Math.sin(this.el);
    const eye = [this.dist * ce * Math.cos(this.az), this.dist * ce * Math.sin(this.az), this.dist * se];
    const look = mat4.lookAt(eye, [0, 0, -0.02], [0, 0, 1]);
    const proj = mat4.perspective(this.fov, aspect, 0.02, 40, this.offsetY, this.offsetX);
    this.last = { eye, ...look, proj, aspect, viewProj: mat4.multiply(proj, look.m) };
    return this.last;
  }

  // Where a screen point lands on the plane z = height, in plate units.
  pick(px, py, width, height, z = 0) {
    const m = this.last;
    if (!m) return null;
    const f = 1 / Math.tan(this.fov / 2);
    const nx = (px / width) * 2 - 1;
    const ny = 1 - (py / height) * 2;
    const vx = ((nx - this.offsetX) * m.aspect) / f;
    const vy = (ny - this.offsetY) / f;
    const dir = [
      m.right[0] * vx + m.up[0] * vy - m.back[0],
      m.right[1] * vx + m.up[1] * vy - m.back[1],
      m.right[2] * vx + m.up[2] * vy - m.back[2],
    ];
    if (Math.abs(dir[2]) < 1e-9) return null;
    const t = (z - m.eye[2]) / dir[2];
    if (t <= 0) return null;
    return [m.eye[0] + t * dir[0], m.eye[1] + t * dir[1]];
  }

  // Screen position of a point in plate units, in CSS pixels.
  project(p, width, height) {
    const v = this.last.viewProj;
    const x = v[0] * p[0] + v[4] * p[1] + v[8] * p[2] + v[12];
    const y = v[1] * p[0] + v[5] * p[1] + v[9] * p[2] + v[13];
    const w = v[3] * p[0] + v[7] * p[1] + v[11] * p[2] + v[15];
    return [((x / w + 1) / 2) * width, ((1 - y / w) / 2) * height];
  }
}
