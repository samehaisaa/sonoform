// GLSL for the renderer. WebGL2, ES 3.00.

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
`;

// The exact solution. Each element's 21 coefficients live in six RGBA32F
// texels, 64 elements to a row, and the fragment shader evaluates the
// quintic itself, so nodal lines and fringes are drawn from the solution and
// not from a mesh of samples of it.
const QUINTIC = `
uniform sampler2D uCoef;
float quintic(int e, vec2 q, out vec2 g) {
  int row = e / 64;
  int col = (e - row * 64) * 6;
  vec4 c0 = texelFetch(uCoef, ivec2(col, row), 0);
  vec4 c1 = texelFetch(uCoef, ivec2(col + 1, row), 0);
  vec4 c2 = texelFetch(uCoef, ivec2(col + 2, row), 0);
  vec4 c3 = texelFetch(uCoef, ivec2(col + 3, row), 0);
  vec4 c4 = texelFetch(uCoef, ivec2(col + 4, row), 0);
  vec4 c5 = texelFetch(uCoef, ivec2(col + 5, row), 0);
  float x = q.x, y = q.y;
  float x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
  float y2 = y * y, y3 = y2 * y, y4 = y3 * y, y5 = y4 * y;
  g.x = c0.y + 2.0 * c0.w * x + c1.x * y + 3.0 * c1.z * x2 + 2.0 * c1.w * x * y + c2.x * y2
      + 4.0 * c2.z * x3 + 3.0 * c2.w * x2 * y + 2.0 * c3.x * x * y2 + c3.y * y3
      + 5.0 * c3.w * x4 + 4.0 * c4.x * x3 * y + 3.0 * c4.y * x2 * y2 + 2.0 * c4.z * x * y3 + c4.w * y4;
  g.y = c0.z + c1.x * x + 2.0 * c1.y * y + c1.w * x2 + 2.0 * c2.x * x * y + 3.0 * c2.y * y2
      + c2.w * x3 + 2.0 * c3.x * x2 * y + 3.0 * c3.y * x * y2 + 4.0 * c3.z * y3
      + c4.x * x4 + 2.0 * c4.y * x3 * y + 3.0 * c4.z * x2 * y2 + 4.0 * c4.w * x * y3 + 5.0 * c5.x * y4;
  return c0.x + c0.y * x + c0.z * y + c0.w * x2 + c1.x * x * y + c1.y * y2
       + c1.z * x3 + c1.w * x2 * y + c2.x * x * y2 + c2.y * y3
       + c2.z * x4 + c2.w * x3 * y + c3.x * x2 * y2 + c3.y * x * y3 + c3.z * y4
       + c3.w * x5 + c4.x * x4 * y + c4.y * x3 * y2 + c4.z * x2 * y3 + c4.w * x * y4 + c5.x * y5;
}
`;

// J0 by the rational and asymptotic forms of Numerical Recipes' bessj0.
// Time-averaged holography records J0 of the local amplitude, so the fringes
// below are the hologram's own intensity.
const BESSEL = `
float besselJ0(float x) {
  float ax = abs(x);
  if (ax < 8.0) {
    float y = x * x;
    float a = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
      + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
    float b = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
      + y * (59272.64853 + y * (267.8532712 + y))));
    return a / b;
  }
  float z = 8.0 / ax;
  float y = z * z;
  float xx = ax - 0.785398164;
  float p = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
    + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  float q = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5
    + y * (0.7621095161e-6 - y * 0.934935152e-7)));
  return sqrt(0.636619772 / ax) * (cos(xx) * p - z * sin(xx) * q);
}
`;

const NOISE = `
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x),
             mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { s += a * noise(p); p = p * 2.03 + 17.1; a *= 0.5; }
  return s;
}
`;

// Soft boxes around the plate. The painted face barely reflects them, but
// the bare metal at the cut edge does, and so does the bow.
const LIGHTING = `
const float PI = 3.14159265;
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform vec3 uFillDir;
uniform vec3 uFillColor;
uniform float uEnvTurn;

float softbox(vec3 r, vec3 dir, vec2 size, float blur) {
  float w = dot(r, dir);
  if (w <= 0.0) return 0.0;
  vec3 u = normalize(cross(dir, vec3(0.0, 0.0, 1.0)));
  vec3 v = cross(u, dir);
  vec2 p = vec2(dot(r, u), dot(r, v)) / w;
  vec2 d = abs(p) - size;
  return (1.0 - smoothstep(-blur, blur, d.x)) * (1.0 - smoothstep(-blur, blur, d.y));
}

vec3 turn(vec3 v, float a) {
  float c = cos(a), s = sin(a);
  return vec3(c * v.x - s * v.y, s * v.x + c * v.y, v.z);
}

vec3 studio(vec3 r, float rough) {
  float blur = 0.015 + rough * 0.55;
  vec3 col = mix(vec3(0.004), vec3(0.016, 0.016, 0.015), smoothstep(-0.2, 0.9, r.z));
  col += vec3(1.1, 1.05, 0.96) * softbox(r, turn(normalize(vec3(0.0, 0.93, 0.36)), uEnvTurn), vec2(1.0, 0.24), blur + 0.08);
  col += vec3(0.42, 0.42, 0.44) * softbox(r, turn(normalize(vec3(0.93, -0.1, 0.34)), uEnvTurn), vec2(0.5, 0.2), blur + 0.05);
  col += vec3(0.36, 0.34, 0.31) * softbox(r, turn(normalize(vec3(-0.9, -0.3, 0.3)), uEnvTurn), vec2(0.45, 0.18), blur + 0.05);
  col += vec3(0.07, 0.068, 0.064) * smoothstep(-0.1, 0.7, r.z);
  col += vec3(0.12, 0.115, 0.11) * softbox(r, normalize(vec3(0.0, 0.08, 1.0)), vec2(0.9, 0.9), 0.6);
  return col;
}

float ggx(float ndh, float a) {
  float a2 = a * a;
  float d = ndh * ndh * (a2 - 1.0) + 1.0;
  return a2 / (PI * d * d);
}

float smith(float ndv, float ndl, float a) {
  float k = 0.5 * a;
  return (ndv / (ndv * (1.0 - k) + k)) * (ndl / (ndl * (1.0 - k) + k)) / max(4.0 * ndv * ndl, 1e-4);
}

vec3 schlick(vec3 f0, float c) {
  return f0 + (1.0 - f0) * pow(1.0 - c, 5.0);
}

vec3 directLight(vec3 n, vec3 v, vec3 l, vec3 color, vec3 f0, float a) {
  vec3 h = normalize(l + v);
  float ndl = max(dot(n, l), 0.0);
  float ndv = max(dot(n, v), 1e-4);
  return ggx(max(dot(n, h), 0.0), a) * smith(ndv, ndl, a) * schlick(f0, max(dot(v, h), 0.0)) * ndl * color;
}

// Bare metal, or glass when uGlass is one.
uniform vec3 uF0;
uniform float uRough;
uniform float uGlass;

vec3 surface(vec3 n, vec3 v, float rough) {
  float a = max(rough * rough, 0.002);
  float ndv = max(dot(n, v), 1e-4);
  vec3 f0 = mix(uF0, vec3(0.04), uGlass);
  vec3 f = schlick(f0, ndv);
  vec3 r = reflect(-v, n);
  vec3 col = studio(r, rough) * f * mix(1.0, 0.6, rough);
  col += 0.012 * directLight(n, v, normalize(uKeyDir), uKeyColor, f0, max(a, 0.08));
  col += 0.012 * directLight(n, v, normalize(uFillDir), uFillColor, f0, max(a, 0.08));
  col += uGlass * (1.0 - f) * vec3(0.006, 0.009, 0.008);
  return col;
}
`;

// The dusk behind everything, drawn first.
export const SKY_VS = HEADER + `
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const SKY_FS = HEADER + `
in vec2 vUv;
uniform vec3 uTop, uBottom;
out vec4 outColor;
void main() {
  outColor = vec4(mix(uBottom, uTop, smoothstep(0.1, 1.0, vUv.y)), 1.0);
}
`;

export const PLATE_VS = HEADER + QUINTIC + `
layout(location = 0) in vec2 aXY;
layout(location = 1) in vec2 aLocal;
layout(location = 2) in float aScale;
layout(location = 3) in float aElem;
uniform mat4 uViewProj;
uniform float uDisp;
out vec3 vWorld;
out vec2 vLocal;
flat out float vScale;
flat out int vElem;
void main() {
  int e = int(aElem + 0.5);
  vec2 g;
  float a = uDisp != 0.0 ? quintic(e, aLocal, g) : 0.0;
  vec3 p = vec3(aXY, uDisp * a);
  vWorld = p;
  vLocal = aLocal;
  vScale = aScale;
  vElem = e;
  gl_Position = uViewProj * vec4(p, 1.0);
}
`;

export const PLATE_FS = HEADER + QUINTIC + BESSEL + NOISE + `
in vec3 vWorld;
in vec2 vLocal;
flat in float vScale;
flat in int vElem;
uniform vec3 uEye;
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform vec3 uAmbient;
uniform float uDisp;
uniform int uView;       // 0 sand, 1 strobe, 2 hologram
uniform float uHoloK;    // 4π A / λ at the antinode
uniform float uNodes;    // opacity of the drawn nodal set
uniform float uField;    // how much of the field to show: zero before any note
uniform float uSigned;   // the strobe's current sign and size
uniform vec3 uFaceA, uFaceB;
uniform float uSpeck, uSheen;
uniform vec3 uHoloLo, uHoloHi, uLine;
out vec4 outColor;

void main() {
  vec2 g;
  float A = quintic(vElem, vLocal, g) * uField;
  g *= uField / vScale;
  vec3 n = normalize(vec3(-uDisp * g, 1.0));
  vec3 v = normalize(uEye - vWorld);
  vec3 l = normalize(uKeyDir);

  // a painted face, a little lighter toward the far edge where it catches
  // the sky, flecked like slate
  vec2 toward = normalize(uEye.xy + 1e-5);
  float far = clamp(0.5 - 0.7 * dot(vWorld.xy, toward), 0.0, 1.0);
  float speck = fbm(vWorld.xy * 22.0) * 0.6 + hash(floor(vWorld.xy * 600.0)) * 0.4;
  float ndl = max(dot(n, l) * 0.6 + 0.4, 0.0);
  vec3 col = mix(uFaceB, uFaceA, far) * (uAmbient + uKeyColor * 0.42 * ndl);
  col *= 1.0 - uSpeck * 0.35 * (speck - 0.5);
  vec3 h = normalize(l + v);
  col += uKeyColor * 0.02 * uSheen * pow(max(dot(n, h), 0.0), 24.0);
  vec3 r = reflect(-v, n);
  col += uSheen * 0.06 * vec3(0.9, 0.88, 0.84) * smoothstep(0.0, 0.8, r.z) * pow(1.0 - max(dot(n, v), 0.0), 3.0);

  if (uView == 1) {
    // the stroboscope: a shade lighter where the sheet is above its rest
    col *= 1.0 + 0.1 * A * uSigned;
  } else if (uView == 2) {
    // a hologram, photographed on black and white film as the first ones were
    float j = besselJ0(uHoloK * abs(A));
    float intensity = pow(j * j, 0.9);
    float grain = 1.0 + 0.25 * (hash(floor(gl_FragCoord.xy * 0.8)) * 2.0 - 1.0);
    col = mix(uHoloLo, uHoloHi, clamp(intensity * grain, 0.0, 1.0));
  }

  if (uNodes > 0.0) {
    // distance to the zero set in pixels, from the screen-space derivative,
    // so the line is one width everywhere and exact where it runs
    float w = length(vec2(dFdx(A), dFdy(A)));
    float d = abs(A) / max(w, 1e-7);
    float line = 1.0 - smoothstep(0.55, 1.7, d);
    col = mix(col, uLine, line * uNodes);
  }
  outColor = vec4(col, 1.0);
}
`;

export const RIM_VS = HEADER + QUINTIC + `
layout(location = 0) in vec2 aXY;
layout(location = 1) in vec2 aLocal;
layout(location = 2) in float aScale;
layout(location = 3) in float aElem;
layout(location = 4) in vec2 aNormal;
layout(location = 5) in float aBottom;
uniform mat4 uViewProj;
uniform float uDisp;
uniform float uThick;
out vec3 vWorld;
out vec3 vNormal;
out float vBottom;
void main() {
  vec2 g;
  float a = uDisp != 0.0 ? quintic(int(aElem + 0.5), aLocal, g) : 0.0;
  vec3 p = vec3(aXY, uDisp * a - aBottom * uThick);
  vWorld = p;
  vNormal = vec3(aNormal, 0.0);
  vBottom = aBottom;
  gl_Position = uViewProj * vec4(p, 1.0);
}
`;

export const RIM_FS = HEADER + LIGHTING + `
in vec3 vWorld;
in vec3 vNormal;
in float vBottom;
uniform vec3 uEye;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  vec3 v = normalize(uEye - vWorld);
  vec3 col = surface(n, v, min(uRough + 0.12, 1.0)) * 0.8;
  // cut glass shows its green at the edge
  col = mix(col, col * 0.35 + vec3(0.02, 0.07, 0.05), uGlass);
  col *= mix(1.0, 0.55, vBottom);
  outColor = vec4(col, 1.0);
}
`;

export const GROUND_VS = HEADER + `
layout(location = 0) in vec2 aXY;
uniform mat4 uViewProj;
uniform float uZ;
out vec2 vXY;
void main() {
  vXY = aXY;
  gl_Position = uViewProj * vec4(aXY, uZ, 1.0);
}
`;

// Gravel raked in rings that follow the outline, the way it is raked round a
// rock, with the plate's shadow lying across it.
export const GROUND_FS = HEADER + NOISE + `
in vec2 vXY;
uniform sampler2D uDist;   // signed distance to the outline, negative inside
uniform vec3 uDistBox;     // x0, y0, size
uniform vec3 uA, uB, uFog;
uniform float uShadowK, uBlur, uFogNear, uFogFar;
uniform vec2 uCast;        // how far the key light carries the shadow sideways
out vec4 outColor;
float sdf(vec2 p) {
  vec2 uv = (p - uDistBox.xy) / uDistBox.z;
  if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0) return 2.0;
  return texture(uDist, uv).r;
}
void main() {
  float d = max(sdf(vXY), 0.0);
  float ring = sin(6.2831853 * d / 0.034);
  float rake = smoothstep(0.35, 1.0, ring) * exp(-d * 0.9) * smoothstep(0.0, 0.02, d) * smoothstep(1.55, 0.95, d);
  vec3 col = mix(uA, uB, rake * 0.9);
  col *= 1.0 + 0.04 * (hash(floor(vXY * 320.0)) - 0.5) + 0.06 * (noise(vXY * 6.0) + 0.5 * noise(vXY * 13.0) - 0.75);
  float shade = 1.0 - smoothstep(-uBlur, uBlur * 1.6, sdf(vXY + uCast));
  col *= 1.0 - uShadowK * shade;
  col = mix(col, uFog, smoothstep(uFogNear, uFogFar, length(vXY)));
  outColor = vec4(col, 1.0);
}
`;

// Grains. Each carries its position, the height it stands at (a pour, or a
// pile), the bowed field under it for the stroboscope, and how hard the plate
// moves under it, which decides whether it hops.
export const SAND_VS = HEADER + `
layout(location = 0) in vec2 aXY;
layout(location = 1) in float aZ;
layout(location = 2) in float aA;
layout(location = 3) in float aAct;
layout(location = 4) in vec3 aTrait;
uniform mat4 uViewProj;
uniform float uTime;
uniform float uDisp;
uniform float uHop;
uniform float uGrain;
uniform float uPixel;
uniform float uShadow;     // 1 for the contact shadow pass
uniform vec3 uKeyDir;
uniform vec3 uSandA, uSandB;
out vec3 vColor;
out float vAlpha;
void main() {
  float phase = fract(uTime * (2.3 + 1.9 * aTrait.x) + aTrait.x * 7.31);
  float arc = 4.0 * phase * (1.0 - phase);
  float hop = uHop * smoothstep(0.015, 0.5, aAct) * arc * (0.3 + 0.7 * fract(aTrait.x * 13.7));
  float size = uGrain * aTrait.y;
  float surface = uDisp * aA;
  float lift = aZ + hop + 0.5 * size;
  vec3 p;
  if (uShadow > 0.5) {
    // where the key light puts this grain's shadow on the sheet
    vec2 fall = -uKeyDir.xy / max(uKeyDir.z, 0.2) * lift;
    p = vec3(aXY + fall, surface + 0.0004);
    vAlpha = 0.5 * exp(-aZ * 30.0) / (1.0 + 60.0 * hop);
  } else {
    p = vec3(aXY, surface + lift);
    vAlpha = 1.0;
  }
  vec4 clip = uViewProj * vec4(p, 1.0);
  gl_Position = clip;
  float scale = uShadow > 0.5 ? 1.9 : 1.0;
  gl_PointSize = clamp(size * scale * uPixel / clip.w, 1.0, 14.0);
  // a few dark grains, and small differences in tone between the rest
  float t = aTrait.z;
  vColor = t < 0.05 ? uSandA * 0.28 : mix(uSandA, uSandB, smoothstep(0.1, 1.0, fract(t * 7.0)) * 0.6);
  vColor *= 0.9 + 0.2 * fract(t * 31.0);
}
`;

export const SAND_FS = HEADER + `
in vec3 vColor;
in float vAlpha;
uniform float uShadow;
uniform vec3 uKeyView;
out vec4 outColor;
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(d, d);
  if (r2 > 1.0) discard;
  if (uShadow > 0.5) {
    float a = vAlpha * (1.0 - r2) * (1.0 - r2);
    outColor = vec4(0.0, 0.0, 0.0, a);
    return;
  }
  vec3 n = vec3(d.x, -d.y, sqrt(1.0 - r2));
  float ndl = max(dot(n, uKeyView), 0.0);
  outColor = vec4(vColor * (0.24 + 0.76 * ndl), 1.0);
}
`;

// The bow, drawn fading upward: only the part near the plate is in the
// story, and a full-length stick would own the frame.
export const BOW_VS = HEADER + `
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uViewProj;
uniform mat4 uModel;
out vec3 vWorld;
out vec3 vNormal;
out vec3 vColor;
out float vAlong;
void main() {
  vec4 w = uModel * vec4(aPos, 1.0);
  vWorld = w.xyz;
  vNormal = mat3(uModel) * aNormal;
  vColor = aColor;
  vAlong = aPos.z;
  gl_Position = uViewProj * w;
}
`;

export const BOW_FS = HEADER + `
in vec3 vWorld;
in vec3 vNormal;
in vec3 vColor;
in float vAlong;
uniform vec3 uEye;
uniform vec3 uKeyDir;
uniform vec3 uKeyColor;
uniform float uFade;
out vec4 outColor;
void main() {
  vec3 n = normalize(vNormal);
  vec3 v = normalize(uEye - vWorld);
  vec3 l = normalize(uKeyDir);
  float ndl = max(dot(n, l), 0.0);
  vec3 h = normalize(l + v);
  float spec = pow(max(dot(n, h), 0.0), 60.0);
  vec3 col = vColor * (0.05 + ndl * uKeyColor * 0.32) + spec * 0.12 * uKeyColor;
  float fade = 1.0 - smoothstep(0.04, 0.34, vAlong);
  outColor = vec4(col, fade * uFade);
}
`;

// Flat sprites lying on the plate: the glow where the bow touches, the ring
// a tap leaves, the marks along the rim where a note can be bowed.
export const SPRITE_VS = HEADER + `
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 aSprite;   // x, y, radius, kind
layout(location = 2) in vec4 aTint;     // r, g, b, strength
uniform mat4 uViewProj;
uniform float uZ;
out vec2 vUv;
out vec4 vTint;
flat out float vKind;
void main() {
  vUv = aCorner;
  vTint = aTint;
  vKind = aSprite.w;
  vec2 p = aSprite.xy + aCorner * aSprite.z;
  gl_Position = uViewProj * vec4(p, uZ, 1.0);
}
`;

export const SPRITE_FS = HEADER + `
in vec2 vUv;
in vec4 vTint;
flat in float vKind;
out vec4 outColor;
void main() {
  float r = length(vUv);
  float a;
  if (vKind < 0.5) {
    a = exp(-r * r * 4.0);                                   // glow
  } else if (vKind < 1.5) {
    a = exp(-pow((r - 0.8) / 0.08, 2.0));                    // ring
  } else {
    a = exp(-r * r * 9.0);                                   // dot
  }
  outColor = vec4(vTint.rgb * vTint.a * a, 0.0);
}
`;

export const SCREEN_VS = HEADER + `
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const BRIGHT_FS = HEADER + `
in vec2 vUv;
uniform sampler2D uSource;
uniform float uThreshold;
out vec4 outColor;
void main() {
  vec3 c = texture(uSource, vUv).rgb;
  float l = max(c.r, max(c.g, c.b));
  float knee = 0.5 * uThreshold;
  float soft = clamp(l - uThreshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee + 1e-5);
  float k = max(soft, l - uThreshold) / max(l, 1e-5);
  outColor = vec4(c * k, 1.0);
}
`;

export const DOWN_FS = HEADER + `
in vec2 vUv;
uniform sampler2D uSource;
uniform vec2 uTexel;
out vec4 outColor;
void main() {
  vec3 c = texture(uSource, vUv + uTexel * vec2(-1, -1)).rgb
         + texture(uSource, vUv + uTexel * vec2(1, -1)).rgb
         + texture(uSource, vUv + uTexel * vec2(-1, 1)).rgb
         + texture(uSource, vUv + uTexel * vec2(1, 1)).rgb;
  outColor = vec4(c * 0.25, 1.0);
}
`;

export const UP_FS = HEADER + `
in vec2 vUv;
uniform sampler2D uSource;
uniform vec2 uTexel;
out vec4 outColor;
void main() {
  vec3 c = 4.0 * texture(uSource, vUv).rgb;
  c += 2.0 * (texture(uSource, vUv + uTexel * vec2(1, 0)).rgb + texture(uSource, vUv - uTexel * vec2(1, 0)).rgb
            + texture(uSource, vUv + uTexel * vec2(0, 1)).rgb + texture(uSource, vUv - uTexel * vec2(0, 1)).rgb);
  c += texture(uSource, vUv + uTexel).rgb + texture(uSource, vUv - uTexel).rgb
     + texture(uSource, vUv + uTexel * vec2(1, -1)).rgb + texture(uSource, vUv + uTexel * vec2(-1, 1)).rgb;
  outColor = vec4(c / 16.0, 1.0);
}
`;

// One grade for the whole frame: a soft tonemap, a little less colour than
// the camera saw, a vignette, and film grain.
export const COMPOSITE_FS = HEADER + NOISE + `
in vec2 vUv;
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform float uBloomStrength;
uniform float uExposure;
uniform float uTonemap;
uniform float uSaturation;
uniform float uContrast;
uniform float uVignette;
uniform float uGrain;
uniform float uAspect;
uniform float uTime;
uniform float uFlash;
out vec4 outColor;
vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
  vec3 c = texture(uScene, vUv).rgb + uBloomStrength * texture(uBloom, vUv).rgb;
  c *= uExposure * (1.0 + uFlash);
  c = mix(clamp(c, 0.0, 1.0), aces(c * 1.35), uTonemap);
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(vec3(l), c, uSaturation);
  vec2 q = (vUv - 0.5) * vec2(uAspect, 1.0);
  c *= mix(1.0 - uVignette, 1.0, smoothstep(1.05, 0.25, length(q)));
  c = pow(max(c, 0.0), vec3(1.0 / 2.2));
  c = (c - 0.5) * uContrast + 0.5;
  c += (hash(gl_FragCoord.xy + fract(uTime * 37.0) * 91.0) - 0.5) * uGrain;
  outColor = vec4(c, 1.0);
}
`;
