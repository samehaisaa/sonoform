// A link that opens this plate, this note, the way it was being looked at.

import { note as nameOf } from './physics.js';
import { formatHz as hz } from './util.js';

const VIEWS = ['sand', 'strobe', 'hologram'];

export function readHash(manifest, defaults) {
  const q = new URLSearchParams(location.hash.slice(1));
  const out = {};
  const plate = q.get('plate');
  if (plate && manifest.plates.some((p) => p.key === plate)) out.plate = plate;
  const metal = q.get('metal');
  if (metal && manifest.materials.some((m) => m.key === metal)) out.metal = metal;
  const h = Number(q.get('h'));
  if (h >= 0.5 && h <= 6) out.thickness = h / 1000;
  const L = Number(q.get('L'));
  if (L >= 80 && L <= 600) out.span = L / 1000;
  const note = Number(q.get('note'));
  if (Number.isInteger(note) && note >= 1) out.note = note - 1;
  const view = q.get('view');
  if (VIEWS.includes(view)) out.view = view;
  return { ...defaults, ...out };
}

let pending = null;
export function writeHash(state) {
  clearTimeout(pending);
  pending = setTimeout(() => {
    const q = new URLSearchParams({
      plate: state.plate,
      metal: state.metal,
      h: (state.thickness * 1000).toFixed(1),
      L: String(Math.round(state.span * 1000)),
      note: String(state.note + 1),
      view: state.view,
    });
    history.replaceState(null, '', `#${q}`);
  }, 250);
}

export async function copyLink() {
  try {
    await navigator.clipboard.writeText(location.href);
    return true;
  } catch {
    return false;
  }
}

// The current frame with a caption, as a PNG. The WebGL canvas does not keep
// its image between frames, so this draws one and copies it at once.
export function saveImage(app, name) {
  const gl = app.canvas;
  app.renderer.render(app._scene(app.camera.last, 0));
  const out = document.createElement('canvas');
  out.width = gl.width;
  out.height = gl.height;
  const ctx = out.getContext('2d');
  ctx.drawImage(gl, 0, 0);
  const s = app.renderer.scale;
  const n = app.notes[app.state.note];
  const note = nameOf(n.hz);
  const entry = app.manifest.plates.find((p) => p.key === app.state.plate);
  const font = '"Zen Kaku Gothic New", "Hiragino Sans", system-ui, sans-serif';
  ctx.fillStyle = 'rgba(232,226,213,0.86)';
  ctx.font = `300 ${16 * s}px ${font}`;
  let at = 32 * s;
  for (const ch of 'sonoform') {
    ctx.fillText(ch, at, out.height - 62 * s);
    at += ctx.measureText(ch).width + 5.5 * s;
  }
  ctx.font = `400 ${13 * s}px ${font}`;
  ctx.fillStyle = 'rgba(141,137,127,0.95)';
  ctx.fillText(`${entry.label} · ${app.material().label} · ${Math.round(app.state.span * 1000)} mm · ${note.label} · ${hz(n.hz)} Hz`, 32 * s, out.height - 36 * s);
  out.toBlob((blob) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `sonoform-${name}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  });
}
