// The look of the page: a dry garden at dusk.
//
// Slate, bone-white sand, and gravel raked in rings around the plate. There
// is no accent colour; the sand is the only light thing in the frame. Colours
// are written as the eye sees them (sRGB) and handed to the renderer in
// linear light.

export function linear(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
}

export const PALETTE = {
  night: '#161715',
  ground: '#1f201d',
  slate: '#2a2b28',
  wet: '#3c3d39',
  bone: '#e8e2d5',
  ash: '#8d897f',
};

// Chladni plates are usually painted dark so the sand shows, which is what
// the top face is here. The bare metal only shows at the cut edge.
// f0 and rough are the metal at the rim, face and sheen the painted top.
export const FINISHES = {
  brass: { f0: [0.9, 0.62, 0.27], rough: 0.3, glass: 0, face: ['#4a4841', '#383731'], speck: 0.55, sheen: 0.25, swatch: '#8b7d5c' },
  copper: { f0: [0.93, 0.52, 0.38], rough: 0.32, glass: 0, face: ['#4c4541', '#3a3431'], speck: 0.55, sheen: 0.25, swatch: '#8a6a58' },
  aluminium: { f0: [0.9, 0.91, 0.92], rough: 0.38, glass: 0, face: ['#56575a', '#434446'], speck: 0.45, sheen: 0.35, swatch: '#9c9d9e' },
  steel: { f0: [0.52, 0.53, 0.55], rough: 0.28, glass: 0, face: ['#46484b', '#35373a'], speck: 0.4, sheen: 0.4, swatch: '#6b6e72' },
  glass: { f0: [0.04, 0.04, 0.04], rough: 0.05, glass: 1, face: ['#2f3432', '#222725'], speck: 0.1, sheen: 1.0, swatch: '#4c5a54' },
};

// Everything the renderer needs that does not change from frame to frame.
export const SCENE = {
  sky: { top: linear('#131412'), bottom: linear('#1f201d') },
  ground: { z: -0.2, a: linear('#2c2d2a'), b: linear('#35362f'), fog: linear('#1f201d'), fogNear: 1.2, fogFar: 3.4, shadow: 0.3, blur: 0.15 },
  ambient: linear('#5a5a55'),
  sand: { a: linear('#ece6d9'), b: linear('#cbc2b0') },
  lines: linear('#e8e2d5'),
  hologram: { lo: linear('#1b1c1a'), hi: linear('#e8e2d5') },
  grade: { exposure: 1.02, tonemap: 0.55, saturation: 0.82, contrast: 1.04, vignette: 0.5, grain: 0.02, bloom: 0.1 },
};

export function finish(key) {
  const f = FINISHES[key] || FINISHES.brass;
  if (!f.linear) f.linear = { a: linear(f.face[0]), b: linear(f.face[1]) };
  return f;
}
