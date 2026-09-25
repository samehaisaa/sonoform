// The interface around the plate. The notes are stepping stones, the bow is
// a ring to hold, and the ways of looking are three small glyphs. Plates,
// sheets and sizes are in the tray; everything else is in the menu.

import { note as nameOf } from './physics.js';
import { writeHash } from './share.js';
import { formatHz as hz, seeded } from './util.js';

const $ = (id) => document.getElementById(id);
const VIEWS = ['sand', 'strobe', 'hologram'];
const VIEW_NAMES = { sand: 'sand', strobe: 'stroboscope', hologram: 'hologram' };
const TOURED = 'sonoform.toured';

function outlinePath(points) {
  // plate units, y up, into a 0..100 box with y down
  const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const s = 88 / Math.max(x1 - x0, y1 - y0);
  const ox = 50 - (s * (x0 + x1)) / 2, oy = 50 + (s * (y0 + y1)) / 2;
  return points.map(([x, y], i) => `${i ? 'L' : 'M'}${(ox + s * x).toFixed(1)} ${(oy - s * y).toFixed(1)}`).join('') + 'Z';
}

// A flat river stone: a closed curve through a wobbly ellipse.
function pebble(cx, cy, rx, ry, rot, rnd) {
  const p1 = rnd() * 6.283, p2 = rnd() * 6.283;
  const a1 = 0.05 + 0.06 * rnd(), a2 = 0.02 + 0.04 * rnd();
  const pts = [];
  for (let k = 0; k < 12; k++) {
    const t = (k / 12) * 2 * Math.PI;
    const r = 1 + a1 * Math.sin(2 * t + p1) + a2 * Math.sin(3 * t + p2);
    const x = Math.cos(t) * rx * r, y = Math.sin(t) * ry * r;
    pts.push([cx + x * Math.cos(rot) - y * Math.sin(rot), cy + x * Math.sin(rot) + y * Math.cos(rot)]);
  }
  const f = (v) => v.toFixed(2);
  const n = pts.length;
  let d = `M${f(pts[0][0])},${f(pts[0][1])}`;
  for (let i = 0; i < n; i++) {
    const p0 = pts[(i - 1 + n) % n], a = pts[i], b = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
    d += `C${f(a[0] + (b[0] - p0[0]) / 6)},${f(a[1] + (b[1] - p0[1]) / 6)} ${f(b[0] - (p3[0] - a[0]) / 6)},${f(b[1] - (p3[1] - a[1]) / 6)} ${f(b[0])},${f(b[1])}`;
  }
  return `${d}Z`;
}

function seedOf(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

export class UI {
  constructor(app, manifest, finishes) {
    this.app = app;
    this.manifest = manifest;
    this.finishes = finishes;
    this.rimHint = 0;
    this.nodeHint = 0;
    // the marks along the rim that say where a note can be bowed: faint at
    // rest, clear while the pointer is over the edge
    this._rimTarget = 0.22;
    this._toastTimer = null;
    this._hintUntil = 0;
    this._hover = -1;
    this._build();
  }

  _build() {
    const app = this.app;
    const plates = $('plates');
    for (const p of this.manifest.plates) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'plate-btn';
      b.dataset.key = p.key;
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-label', p.label);
      b.title = p.label;
      b.innerHTML = `<svg viewBox="0 0 100 100" aria-hidden="true"><path d="${outlinePath(p.outline)}"/></svg>`;
      b.onclick = () => app.setPlate(p.key);
      plates.append(b);
    }
    const metals = $('metals');
    for (const m of this.manifest.materials) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'metal-btn';
      b.dataset.key = m.key;
      b.setAttribute('role', 'radio');
      b.innerHTML = `<i style="background:${this.finishes[m.key].swatch}"></i>${m.label}`;
      b.onclick = () => app.setMetal(m.key);
      metals.append(b);
    }

    const thick = $('thick'), span = $('span');
    const dims = () => {
      const h = Number(thick.value) / 1000, L = Number(span.value) / 1000;
      $('thickOut').textContent = `${(h * 1000).toFixed(1)} mm`;
      $('spanOut').textContent = `${Math.round(L * 1000)} mm`;
      app.setDimensions(h, L);
      this._thinCheck();
      this.now();
    };
    thick.addEventListener('input', dims);
    span.addEventListener('input', dims);

    // the ring: hold to bow, a short press keeps it going
    const bow = $('bow');
    let pressed = 0;
    bow.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      bow.setPointerCapture(e.pointerId);
      pressed = performance.now();
      if (app.bowing && app.latched) {
        app.setBow(false);
        pressed = 0;
        return;
      }
      app.setBow(true);
    });
    const release = () => {
      if (!pressed) return;
      const quick = performance.now() - pressed < 260;
      pressed = 0;
      if (quick) app.setBow(true, { latch: true });
      else app.setBow(false);
    };
    bow.addEventListener('pointerup', release);
    bow.addEventListener('pointercancel', release);
    bow.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') app.toggleLatch();
    });

    for (const b of $('views').querySelectorAll('button')) b.onclick = () => this.setView(b.dataset.view);
    $('nodes').onclick = () => this.toggleNodes();
    $('pour').onclick = () => app.pour();
    $('btnSound').onclick = () => this.toggleMute();
    $('now').onclick = () => this.toggleTray();
    $('trayClose').onclick = () => this.toggleTray(false);
    $('btnMenu').onclick = () => this.toggleMenu();
    const menuItem = (id, fn) => {
      $(id).onclick = () => {
        this.toggleMenu(false);
        fn();
      };
    };
    menuItem('btnInfo', () => this.toggleSheet());
    menuItem('btnAtlas', () => app.atlas.toggle());
    menuItem('btnShare', () => app.shareLink());
    menuItem('btnShot', () => app.saveImage());
    menuItem('btnZen', () => app.toggleZen());
    $('btnStar').addEventListener('click', () => this.toggleMenu(false));
    $('sheetClose').onclick = () => this.toggleSheet(false);
    $('replayTour').onclick = () => {
      this.toggleSheet(false);
      app.tour.start();
    };
    // anything outside the menu or the tray puts them away
    document.addEventListener('pointerdown', (e) => {
      if (!$('menu').hidden && !e.target.closest('#menu, #btnMenu')) this.toggleMenu(false);
      if ($('tray').classList.contains('open') && !e.target.closest('#tray, #now')) this.toggleTray(false);
    });

    // the stones' label follows the pointer, and comes back to the note
    const ladder = $('ladder');
    ladder.addEventListener('pointerover', (e) => {
      const s = e.target.closest('.stone');
      if (s) this._showLabel(Number(s.dataset.index), true);
    });
    ladder.addEventListener('pointerleave', () => this._showLabel(this.app.state.note, false));
    ladder.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault();
        e.stopPropagation();
        const i = this.app.state.note + (e.key === 'ArrowRight' ? 1 : -1);
        this.app.selectNote(i);
        const el = this.stones && this.stones[this.app.state.note];
        if (el) el.focus();
      }
    });

    const toured = localStorage.getItem(TOURED) === '1';
    this.toured = toured;
    $('begin').onclick = () => this.clickBegin();
    $('skipTour').textContent = toured ? 'the tour again' : 'skip the tour';
    $('skipTour').onclick = () => {
      if (!$('begin').disabled) this._enter(toured);
    };
  }

  ready() {
    const begin = $('begin');
    begin.disabled = false;
    $('beginLabel').textContent = this.toured ? 'enter' : 'begin';
  }

  clickBegin() {
    if ($('begin').disabled) return;
    this._enter(!this.toured);
  }

  _enter(tour) {
    $('intro').classList.add('gone');
    this.app.begin(tour);
    this.resize();
  }

  markToured() {
    this.toured = true;
    try {
      localStorage.setItem(TOURED, '1');
    } catch {
      /* private windows may refuse; the tour just plays again */
    }
  }

  // --- the plate on screen ------------------------------------------------

  plate() {
    const app = this.app;
    for (const b of $('plates').children) {
      const on = b.dataset.key === app.state.plate;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
    }
    for (const b of $('metals').children) {
      const on = b.dataset.key === app.state.metal;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
    }
    const entry = this.manifest.plates.find((p) => p.key === app.state.plate);
    $('plateNote').textContent = entry.note;
    $('metalNote').textContent = app.material().note;
    $('thick').value = (app.state.thickness * 1000).toFixed(1);
    $('span').value = Math.round(app.state.span * 1000);
    $('thickOut').textContent = `${(app.state.thickness * 1000).toFixed(1)} mm`;
    $('spanOut').textContent = `${Math.round(app.state.span * 1000)} mm`;
    this._thinCheck();
    this.now();
    this._stones();
    this.selectNote(app.state.note);
    this.setView(app.state.view, { quiet: true });
    $('nodes').setAttribute('aria-pressed', String(app.state.nodes));
    this.facts();
    this.explain();
  }

  now() {
    const app = this.app;
    const entry = this.manifest.plates.find((p) => p.key === app.state.plate);
    $('nowPlate').textContent = entry.label;
    $('nowSheet').textContent = app.material().label;
    $('nowSize').textContent = `${Math.round(app.state.span * 1000)} mm`;
  }

  _thinCheck() {
    const ratio = this.app.state.span / this.app.state.thickness;
    const warn = $('thinWarn');
    warn.hidden = ratio >= 20;
    if (ratio < 20) {
      warn.textContent = `The sheet is now 1/${Math.round(ratio)} as thick as it is wide. Thin-plate theory starts to read the higher notes sharp here.`;
    }
  }

  // One stone per note, along a path. Lower notes are larger stones, as a
  // larger bell rings lower, and a note shared by two figures is two stones.
  _stones() {
    const app = this.app;
    const box = $('ladder');
    box.innerHTML = '';
    const notes = app.notes;
    const small = window.innerWidth <= 760;
    this._small = small;
    const rmax = small ? 15 : 19, gap = small ? 10 : 12, height = 72, arc = small ? 5 : 11;
    const rnd = seeded(seedOf(`${app.plate.key}:${notes.length}`));
    const lowest = notes[0].hz;
    const sizes = notes.map((n) => rmax * Math.min(1, Math.max(0.5, (lowest / n.hz) ** 0.22)));
    const xs = [];
    let x = 10;
    sizes.forEach((r, i) => {
      x += (i ? gap : 0) + r;
      xs.push(x);
      x += r;
    });
    const width = x + 10;
    box.style.width = `${width}px`;
    box.style.height = `${height}px`;
    this.stones = notes.map((n, i) => {
      const r = sizes[i];
      const t = notes.length > 1 ? i / (notes.length - 1) : 0.5;
      const cx = xs[i];
      const cy = height - 24 - arc * Math.sin(Math.PI * t) + (rnd() - 0.5) * 4;
      const ry = r * (0.56 + 0.1 * rnd()), rot = (rnd() - 0.5) * 0.7;
      const w = r * 2.8, h = ry * 2 + 16;
      const face = pebble(0, 0, r, ry, rot, rnd);
      const twin = n.members.length > 1 ? pebble(r * 0.68, -ry * 0.42, r * 0.6, ry * 0.62, rot + 0.6, rnd) : '';
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'stone';
      b.dataset.index = String(i);
      b.setAttribute('role', 'option');
      b.style.left = `${cx}px`;
      b.style.top = `${cy}px`;
      b.style.width = `${w}px`;
      b.style.height = `${h}px`;
      const shade = (d) => `<path class="shade" d="${d}" transform="translate(0 3)" filter="url(#stone-soft)"/>`;
      b.innerHTML = `<svg viewBox="${(-w / 2).toFixed(1)} ${(-h / 2).toFixed(1)} ${w.toFixed(1)} ${h.toFixed(1)}" aria-hidden="true">` +
        `${twin ? shade(twin) : ''}${shade(face)}${twin ? `<path class="face" d="${twin}"/>` : ''}<path class="face" d="${face}"/></svg>`;
      b.onclick = () => app.selectNote(i);
      box.append(b);
      return { el: b, x: cx, y: cy - ry - 12, n };
    });
    this.notes();
  }

  // Labels only: after a change of size or thickness nothing else moves.
  notes() {
    const app = this.app;
    if (!this.stones) return;
    app.notes.forEach((n, i) => {
      const f = app.acoustics.hz[n.members[0]];
      n.hz = f;
      const name = nameOf(f);
      const s = this.stones[i];
      s.name = name.label;
      s.hz = `${hz(f)} Hz`;
      s.el.setAttribute('aria-label', `${name.label}, ${hz(f)} hertz${n.members.length > 1 ? ', two figures' : ''}`);
      s.el.title = `${name.label} ${name.offset} · ${hz(f)} Hz${n.members.length > 1 ? ' · two figures' : ''}`;
    });
    this._showLabel(this._hover >= 0 ? this._hover : app.state.note, this._hover >= 0);
    this.explain();
    this.facts();
  }

  _showLabel(i, hover) {
    if (!this.stones || !this.stones[i]) return;
    this._hover = hover ? i : -1;
    const s = this.stones[i];
    const label = $('noteLabel');
    const box = $('ladder');
    label.style.left = `${box.offsetLeft + s.x}px`;
    label.style.top = `${box.offsetTop + s.y}px`;
    label.classList.toggle('hover', hover && i !== this.app.state.note);
    $('noteName').textContent = s.name || '';
    $('noteHz').textContent = s.hz || '';
  }

  selectNote(i) {
    if (!this.stones) return;
    this.stones.forEach((s, k) => {
      s.el.classList.toggle('on', k === i);
      s.el.setAttribute('aria-selected', String(k === i));
      s.el.tabIndex = k === i ? 0 : -1;
    });
    this._showLabel(i, false);
    // on a phone the path scrolls, so bring the note into view
    const path = $('path');
    if (path.scrollWidth > path.clientWidth) {
      const s = this.stones[i];
      path.scrollTo({ left: $('ladder').offsetLeft + s.x - path.clientWidth / 2, behavior: 'smooth' });
    }
  }

  // A short line over the stones, only when there is something to say.
  explain(extra, seconds = 3.2) {
    const app = this.app;
    const el = $('status');
    if (extra) {
      // each phrase stays whole when a narrow screen wraps the line
      el.replaceChildren(...extra.split(' · ').flatMap((part, i) => {
        const span = document.createElement('span');
        span.textContent = part;
        return i ? [' · ', span] : [span];
      }));
      this._hintUntil = performance.now() + seconds * 1000;
      el.classList.add('show');
      return;
    }
    if (performance.now() < this._hintUntil) return;
    if (app.field && app.bowing && app.field.drive < 0.08) {
      el.textContent = 'slide the bow along the edge';
      el.classList.add('show');
    } else {
      el.classList.remove('show');
    }
  }

  hint() {
    this.explain('hold the ring · tap the plate · drag to turn', 7);
  }

  bowState() {
    const app = this.app;
    const b = $('bow');
    b.classList.toggle('active', app.bowing);
    b.classList.toggle('latched', app.bowing && app.latched);
    b.classList.remove('invite');
    b.setAttribute('aria-pressed', String(app.bowing));
    $('bowLabel').textContent = app.bowing && app.latched ? 'stop' : 'hold';
    this.explain();
  }

  setView(view, { quiet = false } = {}) {
    if (!VIEWS.includes(view)) return;
    this.app.state.view = view;
    for (const b of $('views').children) {
      const on = b.dataset.view === view;
      b.classList.toggle('on', on);
      b.setAttribute('aria-checked', String(on));
    }
    if (!quiet) {
      this.explain(VIEW_NAMES[view], 1.8);
      this.app.emit('view', view);
    }
    writeHash(this.app.state);
  }

  cycleView() {
    const i = VIEWS.indexOf(this.app.state.view);
    this.setView(VIEWS[(i + 1) % VIEWS.length]);
  }

  toggleNodes() {
    const s = this.app.state;
    s.nodes = !s.nodes;
    $('nodes').setAttribute('aria-pressed', String(s.nodes));
    this.app.emit('nodes', s.nodes);
  }

  toggleMute() {
    const a = this.app.audio;
    a.setMuted(!a.muted);
    $('btnSound').classList.toggle('muted', a.muted);
    $('btnSound').setAttribute('aria-label', a.muted ? 'Unmute' : 'Mute');
  }

  toggleTray(open) {
    const tray = $('tray');
    const on = open ?? !tray.classList.contains('open');
    tray.classList.toggle('open', on);
    tray.setAttribute('aria-hidden', String(!on));
    $('now').setAttribute('aria-expanded', String(on));
    if (on) {
      this.toggleMenu(false);
      this.toggleSheet(false);
    }
  }

  toggleMenu(open) {
    const menu = $('menu');
    const on = open ?? menu.hidden;
    menu.hidden = !on;
    $('btnMenu').setAttribute('aria-expanded', String(on));
    $('btnMenu').classList.toggle('on', on);
  }

  toggleSheet(open) {
    const sheet = $('sheet');
    const on = open ?? !sheet.classList.contains('open');
    sheet.classList.toggle('open', on);
    sheet.setAttribute('aria-hidden', String(!on));
    document.body.classList.toggle('sheet-open', on);
    $('btnInfo').classList.toggle('active', on);
    if (on) {
      this.toggleTray(false);
      this.checks();
    }
  }

  closeAll() {
    this.toggleSheet(false);
    this.toggleTray(false);
    this.toggleMenu(false);
    this.app.atlas.close();
  }

  busy(on) {
    document.body.classList.toggle('busy', on);
  }

  toast(text, ms = 2400) {
    const t = $('toast');
    t.textContent = text;
    t.classList.add('show');
    clearTimeout(this._toastTimer);
    this._toastTimer = setTimeout(() => t.classList.remove('show'), ms);
  }

  rimHover(on) {
    this._rimTarget = on ? 1 : 0.22;
  }

  frame(dt) {
    const k = 1 - Math.exp(-dt / 0.25);
    this.rimHint += (this._rimTarget - this.rimHint) * k;
    if (this._hintUntil && performance.now() > this._hintUntil) {
      this._hintUntil = 0;
      this.explain();
    }
  }

  // Centre the plate between the top line and the stones, and back the
  // camera off until it fits there.
  layout({ now = false } = {}) {
    const app = this.app;
    const W = window.innerWidth, H = window.innerHeight;
    const insets = { top: 0, bottom: 0, left: 0, right: 0 };
    const root = document.documentElement.style;
    const dock = $('dock').getBoundingClientRect();
    // the note's name stands above the stones, outside the dock's own box
    const floor = dock.top - 58;
    root.setProperty('--ui-h', `${Math.max(0, H - floor)}px`);
    if (!document.body.classList.contains('intro-on')) {
      const top = $('top').getBoundingClientRect();
      insets.top = top.bottom + 6;
      insets.bottom = H - floor + 6;
      insets.left = insets.right = Math.min(48, W * 0.03);
    }
    app.camera.frame(W, H, insets, now);
    const aspect = W / Math.max(H, 1);
    if (app.radius) app.camera.fly({ dist: app.camera.fit(app.radius, aspect), glide: now ? 0.2 : 0.9 });
  }

  resize() {
    clearTimeout(this._resize);
    this._resize = setTimeout(() => {
      // the stones are sized for the screen, so a phone turned sideways gets new ones
      if (this.stones && (window.innerWidth <= 760) !== this._small) {
        this._small = window.innerWidth <= 760;
        this._stones();
        this.selectNote(this.app.state.note);
      }
      this.layout();
      this._showLabel(this._hover >= 0 ? this._hover : this.app.state.note, this._hover >= 0);
    }, 60);
  }

  // --- the physics panel's live numbers ------------------------------------

  facts() {
    const app = this.app;
    if (!app.plate || !app.notes) return;
    const p = app.plate;
    const m = app.material();
    const omegas = app.notes.slice(0, 8).map((n) => p.omega[n.members[0]].toFixed(2)).join(', ');
    $('plateFacts').innerHTML =
      `<b>This plate</b>: ${p.nt} Argyris triangles, ${p.dofs.toLocaleString('en-US')} unknowns, ${p.modes} modes in ${app.notes.length} notes.<br>` +
      `Rigid-body motions at <b>${p.kernel.toExponential(1)}</b> of the first flexible eigenvalue, so the free edge really is free.<br>` +
      `Ω = ωL²√(ρh/D) for its first notes at ν = ${m.poisson}: ${omegas}.<br>` +
      `${m.label}: E = ${(m.young / 1e9).toFixed(0)} GPa, ρ = ${m.density} kg/m³, loss factor η = ${m.lossFactor.toExponential(0)}.`;
  }

  async checks() {
    if (this._checked) return;
    this._checked = true;
    let v;
    try {
      const res = await fetch('data/verification.json');
      v = await res.json();
    } catch {
      $('checks').innerHTML = '<p class="dim">The benchmarks could not be loaded.</p>';
      return;
    }
    const pct = (a, b) => `${(((a - b) / b) * 100).toFixed(3)} %`;
    const sq = v.square;
    let html = `<table><caption>The free square at ν = ${sq.poisson}, against ${sq.source}.</caption><tr><th>mode</th><th>here</th><th>Leissa</th><th>diff</th></tr>`;
    sq.computed.forEach((c, i) => {
      html += `<tr><td>${i + 1}</td><td>${c.toFixed(4)}</td><td>${sq.reference[i].toFixed(3)}</td><td class="good">${pct(c, sq.reference[i])}</td></tr>`;
    });
    html += '</table>';
    const d = v.disc;
    html += `<table><caption>The free disc at ν = ${d.poisson}, against ${d.source}.</caption><tr><th>mode</th><th>here</th><th>exact</th><th>diff</th></tr>`;
    for (const r of d.rows) {
      html += `<tr><td>${r.diameters} diam., ${r.circles} circ.</td><td>${r.computed.toFixed(4)}</td><td>${r.reference.toFixed(4)}</td><td class="good">${pct(r.computed, r.reference)}</td></tr>`;
    }
    html += '</table>';
    html += `<p class="dim">Solved when this site was built, by the code that solved every plate on it. The disc's remaining gap is its outline: a polygon of 140 sides stands in for the circle.</p>`;
    $('checks').innerHTML = html;
  }
}
