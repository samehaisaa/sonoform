// The first minute. Each step asks for one thing and waits until it has
// happened, so nobody reads about the sand without having moved it. One
// short line each.

import { formatHz as hz } from './util.js';

const $ = (id) => document.getElementById(id);

const STEPS = [
  {
    text: 'In 1787 Chladni drew a violin bow along a plate of sand.',
    wait: 5,
  },
  {
    text: 'Hold the ring, and keep holding.',
    spot: () => $('bow').querySelector('.ring'),
    enter: (app) => {
      app.ui.setView('sand', { quiet: true });
      if (app.state.note !== 0) app.selectNote(0, { quiet: true });
      if (app.field.drive < 0.3) app.moveBowToBest();
      $('bow').classList.add('invite');
    },
    until: (app, s) => s.bowTime > 4 || (s.bowTime > 2.5 && app.figure > 0.3),
  },
  {
    text: 'The sand leaves wherever the plate moves.',
    until: (app, s) => (s.t > 4.5 && app.figure > 0.42) || s.t > 10,
  },
  {
    text: 'What stays is where the metal is still, at {hz}.',
    wait: 6,
  },
  {
    text: 'Step to another stone, and hold again.',
    spot: () => $('ladder'),
    enter: (app, s) => {
      s.startNote = app.state.note;
    },
    note: (app, s) => {
      s.bowTime = 0;
      if (app.field.drive < 0.25) app.moveBowToBest();
    },
    until: (app, s) => app.state.note !== s.startNote && s.bowTime > 3.5,
  },
  {
    text: 'Now tap the plate itself.',
    until: (app, s) => s.struck,
  },
  {
    text: 'A tap rings every note at once. Nothing settles.',
    wait: 6,
  },
  {
    text: 'Look through the stroboscope, and hold.',
    spot: () => document.querySelector('[data-view=strobe]'),
    until: (app, s) => app.state.view === 'strobe' && s.bowTime > 3.5,
    view: (app, s) => (s.bowTime = 0),
  },
  {
    text: 'Slowed to one swing every two seconds. The still lines never move.',
    until: (app, s) => s.t > 7,
  },
  {
    text: 'Or as a hologram.',
    spot: () => document.querySelector('[data-view=hologram]'),
    until: (app, s) => app.state.view === 'hologram' && s.bowTime > 3.5,
    view: (app, s) => (s.bowTime = 0),
  },
  {
    text: 'Each dark fringe is another sixth of a micron of swing.',
    wait: 6,
  },
  {
    text: 'The rest is yours.',
    next: 'done',
    leave: (app) => app.ui.setView('sand', { quiet: true }),
  },
];

export class Tour {
  constructor(app) {
    this.app = app;
    this.active = false;
    this.index = -1;
    this.s = null;
    this.caption = $('caption');
    this.text = $('captionText');
    this.nextButton = $('captionNext');
    this.nextButton.onclick = () => this.advance();
    $('captionSkip').onclick = () => this.stop();
    app.on('note', () => this._event('note'));
    app.on('view', () => this._event('view'));
    app.on('strike', () => {
      if (this.s) this.s.struck = true;
    });
  }

  start() {
    this.active = true;
    document.body.classList.add('touring');
    this.go(0);
  }

  stop() {
    if (!this.active) return;
    const step = STEPS[this.index];
    if (step && step.leave) step.leave(this.app);
    this.active = false;
    this.index = -1;
    this._spot(null);
    this.caption.classList.remove('show');
    document.body.classList.remove('touring');
    $('bow').classList.remove('invite');
    this.app.ui.markToured();
    this.app.ui.explain();
  }

  advance() {
    if (!this.active) return;
    const step = STEPS[this.index];
    if (step && step.leave) step.leave(this.app);
    if (this.index + 1 >= STEPS.length) this.stop();
    else this.go(this.index + 1);
  }

  go(i) {
    this.index = i;
    const step = STEPS[i];
    this.s = { t: 0, bowTime: 0 };
    this.caption.classList.remove('show');
    clearTimeout(this._fade);
    this._fade = setTimeout(() => {
      const f = this.app.notes ? `${hz(this.app.notes[this.app.state.note].hz)} Hz` : '';
      this.text.textContent = step.text.replaceAll('{hz}', f);
      this.nextButton.textContent = step.next || 'next';
      this.caption.classList.add('show');
    }, 420);
    this._spot(step.spot ? step.spot() : null);
    if (step.enter) step.enter(this.app, this.s);
  }

  _spot(el) {
    if (this._lit) this._lit.classList.remove('spotlight');
    this._lit = el;
    if (el) el.classList.add('spotlight');
  }

  _event(kind) {
    if (!this.active) return;
    const step = STEPS[this.index];
    if (step && step[kind]) step[kind](this.app, this.s);
  }

  update(dt) {
    if (!this.active || !this.s) return;
    const s = this.s;
    s.t += dt;
    if (this.app.bowing) s.bowTime += dt;
    const step = STEPS[this.index];
    if (s.t < 0.6) return;
    if ((step.wait && s.t > step.wait) || (step.until && step.until(this.app, s))) this.advance();
  }
}
