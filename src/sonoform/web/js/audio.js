// The room the plate is heard in, and the voices in it.
//
// A bowed plate at resonance radiates essentially one frequency: its Q is in
// the thousands, so the harmonics of the stick-slip are a thousand times
// weaker than the note. Each voice is therefore a pure sine. The level at
// each ear follows what the plate radiates toward it, and when the bow comes
// off, the note decays at the mode's own rate. Changing note does not glide.
// The old mode rings down while the new one builds, as it would on a bench.
//
// The room is a synthetic reverberation, three seconds to fall 60 dB. It is
// the only part of the sound that does not come from the plate.

const VOICES = 3;

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.muted = false;
    this.voices = [];
  }

  get ready() {
    return !!this.ctx && this.ctx.state === 'running';
  }

  // Browsers only allow sound after a gesture, so this is called from one.
  unlock() {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return false;
    if (!this.ctx) {
      this.ctx = new Ctx({ latencyHint: 'interactive' });
      this._build();
    }
    if (this.ctx.state === 'suspended') this.ctx.resume();
    return true;
  }

  _build() {
    const ctx = this.ctx;
    this.master = ctx.createGain();
    this.master.gain.value = this.muted ? 0 : 1;
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -14;
    limiter.knee.value = 10;
    limiter.ratio.value = 4;
    limiter.attack.value = 0.003;
    limiter.release.value = 0.3;
    this.master.connect(limiter).connect(ctx.destination);

    this.dry = ctx.createGain();
    this.dry.gain.value = 0.85;
    this.dry.connect(this.master);
    this.reverb = ctx.createConvolver();
    this.reverb.buffer = this._room(3.1);
    this.wet = ctx.createGain();
    this.wet.gain.value = 0.34;
    this.reverb.connect(this.wet).connect(this.master);

    for (let i = 0; i < VOICES; i++) {
      const osc = ctx.createOscillator();
      osc.type = 'sine';
      osc.frequency.value = 220;
      const left = ctx.createGain(), right = ctx.createGain();
      left.gain.value = 0;
      right.gain.value = 0;
      const dl = ctx.createDelay(0.01), dr = ctx.createDelay(0.01);
      const merge = ctx.createChannelMerger(2);
      osc.connect(left).connect(dl).connect(merge, 0, 0);
      osc.connect(right).connect(dr).connect(merge, 0, 1);
      merge.connect(this.dry);
      merge.connect(this.reverb);
      osc.start();
      this.voices.push({ osc, left, right, dl, dr, note: null, used: 0 });
    }
  }

  // Exponentially decaying stereo noise, darkening as it falls, with a few
  // early reflections in front.
  _room(seconds) {
    const ctx = this.ctx, rate = ctx.sampleRate;
    const n = Math.floor(seconds * rate);
    const buffer = ctx.createBuffer(2, n, rate);
    const tau = seconds / 6.91;
    for (let c = 0; c < 2; c++) {
      const d = buffer.getChannelData(c);
      let low = 0;
      let seed = 1234567 + 999 * c;
      const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x3fffffff) - 1;
      for (let i = 0; i < n; i++) {
        const t = i / rate;
        const cutoff = 0.5 * Math.exp(-t / (0.7 * seconds)) + 0.04;
        low += cutoff * (rand() - low);
        d[i] = low * Math.exp(-t / tau) * Math.min(1, t / 0.012);
      }
      for (const [at, g] of [[0.011, 0.5], [0.019, 0.34], [0.031, 0.28], [0.043, 0.2]]) {
        const i = Math.floor((at + 0.002 * c) * rate);
        if (i < n) d[i] += g * (c ? -1 : 1);
      }
    }
    return buffer;
  }

  setMuted(muted) {
    this.muted = muted;
    if (this.master) this.master.gain.setTargetAtTime(muted ? 0 : 1, this.ctx.currentTime, 0.05);
  }

  // The voice sounding `note`, taking the quietest free one if none is.
  voice(note) {
    let v = this.voices.find((x) => x.note === note);
    if (!v) {
      v = this.voices.reduce((a, b) => (a.used < b.used ? a : b));
      v.note = note;
      v.left.gain.cancelScheduledValues(this.ctx.currentTime);
      v.right.gain.cancelScheduledValues(this.ctx.currentTime);
      v.left.gain.setValueAtTime(0, this.ctx.currentTime);
      v.right.gain.setValueAtTime(0, this.ctx.currentTime);
    }
    v.used = this.ctx.currentTime;
    return v;
  }

  // Levels and inter-aural delays for one sounding note, every frame.
  sing(note, hz, left, right, delayL, delayR) {
    if (!this.ctx) return;
    const v = this.voice(note);
    const t = this.ctx.currentTime;
    v.osc.frequency.setTargetAtTime(hz, t, 0.004);
    v.left.gain.setTargetAtTime(left, t, 0.025);
    v.right.gain.setTargetAtTime(right, t, 0.025);
    v.dl.delayTime.setTargetAtTime(delayL, t, 0.05);
    v.dr.delayTime.setTargetAtTime(delayR, t, 0.05);
  }

  // Let every voice not listed fall silent.
  release(keep) {
    if (!this.ctx) return;
    const t = this.ctx.currentTime;
    for (const v of this.voices) {
      if (keep.has(v.note)) continue;
      v.left.gain.setTargetAtTime(0, t, 0.05);
      v.right.gain.setTargetAtTime(0, t, 0.05);
      if (v.note !== null && t - v.used > 1) v.note = null;
    }
  }

  // A strike, already synthesised, one array per ear.
  play(channels, rate, gain) {
    if (!this.ctx) return;
    const buffer = this.ctx.createBuffer(2, channels[0].length, rate);
    for (let c = 0; c < 2; c++) {
      const d = buffer.getChannelData(c), s = channels[c];
      for (let i = 0; i < s.length; i++) d[i] = s[i] * gain;
    }
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.dry);
    src.connect(this.reverb);
    src.start();
  }
}
