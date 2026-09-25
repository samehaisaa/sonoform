// Every figure of the plate at once, the way Chladni printed them.
//
// The walk's steady state is known, ρ ∝ 1 / (D0 + |A|), so each figure is
// drawn by sampling it directly. It is what the sand on the plate converges
// to given long enough: the plate on screen shows the approach, and these
// show where it ends.

import { note as nameOf } from './physics.js';
import { formatHz as hz, seeded } from './util.js';

// Softer than the simulation's floor, so a figure printed small still reads.
const PRINT_FLOOR = 0.014;

function frameFor(plate, size, margin) {
  const w = plate.xmax - plate.xmin, h = plate.ymax - plate.ymin;
  const s = (size - 2 * margin) / Math.max(w, h);
  const ox = size / 2 - (s * (plate.xmin + plate.xmax)) / 2;
  const oy = size / 2 + (s * (plate.ymin + plate.ymax)) / 2;
  return { s, map: (x, y) => [ox + s * x, oy - s * y] };
}

function outlinePath(ctx, plate, map) {
  const o = plate.outline;
  ctx.beginPath();
  for (let i = 0; i < o.length; i += 2) {
    const [x, y] = map(o[i], o[i + 1]);
    if (i) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
  }
  ctx.closePath();
}

function bilinear(grid, values, x, y) {
  const { n, h, x0, y0 } = grid;
  const gx = (x - x0) / h - 0.5, gy = (y - y0) / h - 0.5;
  const ix = Math.max(0, Math.min(n - 2, Math.floor(gx))), iy = Math.max(0, Math.min(n - 2, Math.floor(gy)));
  const fx = gx - ix, fy = gy - iy, k = iy * n + ix;
  return (1 - fx) * (1 - fy) * values[k] + fx * (1 - fy) * values[k + 1] +
    (1 - fx) * fy * values[k + n] + fx * fy * values[k + n + 1];
}

// Sand at rest on a plate, drawn from the steady state rather than walked.
export function drawSand(ctx, plate, grid, values, size, { dots, ink, paper, rim, seed = 7, margin = 10, grain = 1.3 }) {
  const { map } = frameFor(plate, size, margin);
  if (paper) {
    ctx.fillStyle = paper;
    ctx.fillRect(0, 0, size, size);
  }
  outlinePath(ctx, plate, map);
  ctx.fillStyle = 'rgba(232, 226, 213, 0.035)';
  ctx.fill();
  const random = seeded(seed);
  const { x0, y0, n, h } = grid;
  const span = n * h;
  ctx.fillStyle = ink;
  let placed = 0, tries = 0;
  const limit = dots * 400;
  while (placed < dots && tries < limit) {
    tries++;
    const x = x0 + random() * span, y = y0 + random() * span;
    if (bilinear(grid, grid.sdf, x, y) >= 0) continue;
    const a = Math.abs(bilinear(grid, values, x, y));
    if (random() > PRINT_FLOOR / (PRINT_FLOOR + a)) continue;
    const [px, py] = map(x, y);
    const r = grain * (0.6 + 0.8 * random());
    ctx.globalAlpha = 0.55 + 0.45 * random();
    ctx.fillRect(px - r / 2, py - r / 2, r, r);
    placed++;
  }
  ctx.globalAlpha = 1;
  if (rim) {
    outlinePath(ctx, plate, map);
    ctx.strokeStyle = rim;
    ctx.lineWidth = Math.max(1, size / 260);
    ctx.stroke();
  }
}

export class Atlas {
  constructor(app) {
    this.app = app;
    this.el = document.getElementById('atlas');
    document.getElementById('atlasClose').onclick = () => this.close();
    document.getElementById('atlasSave').onclick = () => this.poster();
  }

  get isOpen() {
    return this.el.classList.contains('open');
  }

  toggle() {
    if (this.isOpen) this.close();
    else this.open();
  }

  close() {
    this.el.classList.remove('open');
    this.el.setAttribute('aria-hidden', 'true');
    this._job = null;
  }

  open() {
    const app = this.app;
    this.el.classList.add('open');
    this.el.setAttribute('aria-hidden', 'false');
    const entry = app.manifest.plates.find((p) => p.key === app.state.plate);
    document.getElementById('atlasTitle').textContent = `${entry.label} plate`;
    document.getElementById('atlasSub').textContent =
      `${app.material().label} · ${Math.round(app.state.span * 1000)} mm across · ${(app.state.thickness * 1000).toFixed(1)} mm thick · ${app.notes.length} notes`;
    const box = document.getElementById('atlasGrid');
    box.innerHTML = '';
    const size = 420;
    const figures = app.notes.map((n) => {
      const fig = document.createElement('figure');
      fig.className = 'fig';
      const c = document.createElement('canvas');
      c.width = size;
      c.height = size;
      const name = nameOf(n.hz);
      fig.innerHTML = '';
      fig.append(c);
      const cap = document.createElement('figcaption');
      cap.innerHTML = `<span>${n.index + 1} · <b>${name.label}</b></span><span>${hz(n.hz)} Hz</span>`;
      fig.append(cap);
      fig.onclick = () => {
        this.close();
        app.selectNote(n.index);
      };
      box.append(fig);
      return { n, c };
    });
    // one figure per frame, so opening the atlas never stalls the page
    const job = (this._job = {});
    let k = 0;
    const next = () => {
      if (this._job !== job || k >= figures.length) return;
      const { n, c } = figures[k++];
      const field = app.noteField(n.index, app.bowAt);
      drawSand(c.getContext('2d'), app.plate, app.grid, field.grid, size, {
        dots: 9000, ink: '#e8e2d5', paper: '#1f201d', rim: 'rgba(232,226,213,0.32)', seed: 11 + n.index, grain: 1.35,
      });
      requestAnimationFrame(next);
    };
    requestAnimationFrame(next);
  }

  // Every figure of this sheet on one sheet, captioned, bone on slate.
  poster() {
    const app = this.app;
    const notes = app.notes;
    const cols = 4, rows = Math.ceil(notes.length / cols);
    const W = 2400, cell = 520, gap = 40, top = 430, foot = 220;
    const H = top + rows * (cell + 90) + foot;
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#161715';
    ctx.fillRect(0, 0, W, H);
    const entry = app.manifest.plates.find((p) => p.key === app.state.plate);
    const font = '"Zen Kaku Gothic New", "Hiragino Sans", system-ui, sans-serif';
    // canvas letter-spacing is not everywhere yet, so the wordmark is spaced by hand
    const spaced = (text, x, y, gap) => {
      let w = 0;
      for (const ch of text) w += ctx.measureText(ch).width + gap;
      let at = x - (w - gap) / 2;
      ctx.textAlign = 'left';
      for (const ch of text) {
        ctx.fillText(ch, at, y);
        at += ctx.measureText(ch).width + gap;
      }
      ctx.textAlign = 'center';
    };
    ctx.fillStyle = '#e8e2d5';
    ctx.textAlign = 'center';
    ctx.font = `300 40px ${font}`;
    spaced('sonoform', W / 2, 140, 16);
    ctx.font = `300 96px ${font}`;
    ctx.fillText(`${entry.label} plate`, W / 2, 270);
    ctx.fillStyle = '#8d897f';
    ctx.font = `400 34px ${font}`;
    ctx.fillText(`${app.material().label} · ${Math.round(app.state.span * 1000)} mm across · ${(app.state.thickness * 1000).toFixed(1)} mm thick · free at every edge`, W / 2, 346);
    const x0 = (W - (cols * cell + (cols - 1) * gap)) / 2;
    notes.forEach((n, i) => {
      const col = i % cols, row = Math.floor(i / cols);
      const x = x0 + col * (cell + gap), y = top + row * (cell + 90);
      const tile = document.createElement('canvas');
      tile.width = cell;
      tile.height = cell;
      const field = app.noteField(n.index, app.bowAt);
      drawSand(tile.getContext('2d'), app.plate, app.grid, field.grid, cell, {
        dots: 16000, ink: '#e8e2d5', paper: '#1f201d', rim: 'rgba(232,226,213,0.32)', seed: 3 + i, grain: 1.7,
      });
      ctx.drawImage(tile, x, y);
      const name = nameOf(n.hz);
      ctx.textAlign = 'left';
      ctx.fillStyle = '#e8e2d5';
      ctx.font = `500 32px ${font}`;
      ctx.fillText(`${i + 1} · ${name.label}`, x + 4, y + cell + 44);
      ctx.textAlign = 'right';
      ctx.fillStyle = '#8d897f';
      ctx.font = `400 26px ${font}`;
      ctx.fillText(`${hz(n.hz)} Hz`, x + cell - 4, y + cell + 44);
    });
    ctx.textAlign = 'center';
    ctx.fillStyle = '#8d897f';
    ctx.font = `400 30px ${font}`;
    ctx.fillText('Each figure is where sand comes to rest on the bowed plate, the steady state of grains that bounce where the metal moves.', W / 2, H - 140);
    ctx.font = `400 24px ${font}`;
    ctx.fillText('D ∇⁴w = ρh ω² w, solved with Argyris finite elements  ·  samehaisaa.github.io/sonoform', W / 2, H - 84);
    c.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `sonoform-${app.state.plate}-${app.state.metal}.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    });
  }
}
