// Small shared helpers. Nothing here knows about plates.

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

// Base64 to bytes. atob exists in browsers and in Node 16+, so the same
// decoder serves the page and the tests.
export function decodeBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

export function float32From(text) {
  const bytes = decodeBase64(text);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

export function indexFrom(text, kind) {
  const bytes = decodeBase64(text);
  return kind === 'u32'
    ? new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
    : new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
}

// A seeded generator, so a poured plate can be poured again identically and
// the tests are reproducible. Mulberry32: tiny, fast, and uniform enough for
// sand.
export function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal deviates by the Marsaglia polar method, two at a time.
// A uniform step has the wrong variance by a factor of three, which silently
// slows the sand by that much; this is the reason the class exists.
export class Normal {
  constructor(random = Math.random) {
    this.random = random;
    this.spare = 0;
    this.hasSpare = false;
  }

  next() {
    if (this.hasSpare) {
      this.hasSpare = false;
      return this.spare;
    }
    let u, v, s;
    do {
      u = this.random() * 2 - 1;
      v = this.random() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    this.spare = v * f;
    this.hasSpare = true;
    return u * f;
  }
}

// 1342.64 -> "1 342.6", a narrow space between the thousands. The unit is
// left to the caller.
export function formatHz(hz) {
  const [whole, frac] = hz.toFixed(1).split('.');
  return `${whole.replace(/\B(?=(\d{3})+(?!\d))/g, '\u202f')}.${frac}`;
}

export function formatNumber(x, digits = 0) {
  return x.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}
