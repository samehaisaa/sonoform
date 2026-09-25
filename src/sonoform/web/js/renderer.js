// Draws the plate, its sand, the bow and the garden around them.
//
// All of it is WebGL2 with no library. The plate's fragment shader evaluates
// the solved quintic itself, so what the stroboscope bends, what the
// hologram fringes and where the nodal lines run are the solution, pixel by
// pixel. Colour is computed in linear light into a floating-point target
// where the hardware allows, then graded once.

import { attachAttributes, mat4, multisample, program, release, target, texture, vertexArray } from './gl.js';
import * as S from './shaders.js';
import { SCENE } from './stone.js';

const LEVEL = 5; // each element drawn as 25 triangles, enough for a smooth bend
const FLOOR_Z = SCENE.ground.z; // the gravel the plate floats above, in plate units
const ROW = 64; // elements per row of the coefficient texture
const DIST_N = 256; // texels across the distance field the gravel is raked from
const DIST_EXTENT = 2.2; // plate units from the centre it covers

export class Renderer {
  constructor(canvas) {
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      alpha: false,
      depth: false,
      stencil: false,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    if (!gl) throw new Error('This page needs WebGL2, which this browser does not offer.');
    this.gl = gl;
    this.canvas = canvas;
    this.hdr = !!gl.getExtension('EXT_color_buffer_float');
    this.samples = Math.max(1, Math.min(4, gl.getParameter(gl.MAX_SAMPLES)));
    this.scale = Math.min(window.devicePixelRatio || 1, 2);
    const p = (vs, fs, label) => program(gl, vs, fs, label);
    this.prog = {
      sky: p(S.SKY_VS, S.SKY_FS, 'sky'),
      plate: p(S.PLATE_VS, S.PLATE_FS, 'plate'),
      rim: p(S.RIM_VS, S.RIM_FS, 'rim'),
      ground: p(S.GROUND_VS, S.GROUND_FS, 'ground'),
      sand: p(S.SAND_VS, S.SAND_FS, 'sand'),
      bow: p(S.BOW_VS, S.BOW_FS, 'bow'),
      sprite: p(S.SPRITE_VS, S.SPRITE_FS, 'sprite'),
      bright: p(S.SCREEN_VS, S.BRIGHT_FS, 'bright'),
      down: p(S.SCREEN_VS, S.DOWN_FS, 'down'),
      up: p(S.SCREEN_VS, S.UP_FS, 'up'),
      composite: p(S.SCREEN_VS, S.COMPOSITE_FS, 'composite'),
    };
    this.empty = gl.createVertexArray();
    this._ground();
    this._bow();
    this._sprites();
    this.targets = null;
    this.resize();
  }

  // --- geometry that depends on the plate -------------------------------

  setPlate(plate) {
    const gl = this.gl;
    this.plate = plate;
    const n = LEVEL;
    const per = ((n + 1) * (n + 2)) / 2;
    const verts = new Float32Array(plate.nt * per * 6);
    const idx = [];
    const p = plate.points, t = plate.triangles, f = plate.frames;
    let v = 0;
    for (let e = 0; e < plate.nt; e++) {
      const a = t[3 * e], b = t[3 * e + 1], c = t[3 * e + 2];
      const ax = p[2 * a], ay = p[2 * a + 1];
      const ux = p[2 * b] - ax, uy = p[2 * b + 1] - ay;
      const wx = p[2 * c] - ax, wy = p[2 * c + 1] - ay;
      const cx = f[3 * e], cy = f[3 * e + 1], s = f[3 * e + 2];
      const base = e * per;
      const at = {};
      let k = 0;
      for (let i = 0; i <= n; i++) {
        for (let j = 0; j <= n - i; j++) {
          const x = ax + (i / n) * ux + (j / n) * wx;
          const y = ay + (i / n) * uy + (j / n) * wy;
          verts.set([x, y, (x - cx) / s, (y - cy) / s, s, e], v * 6);
          at[i * 16 + j] = base + k;
          k++;
          v++;
        }
      }
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n - i; j++) {
          idx.push(at[i * 16 + j], at[(i + 1) * 16 + j], at[i * 16 + j + 1]);
          if (i + j < n - 1) idx.push(at[(i + 1) * 16 + j], at[(i + 1) * 16 + j + 1], at[i * 16 + j + 1]);
        }
      }
    }
    const index = v < 65536 ? new Uint16Array(idx) : new Uint32Array(idx);
    if (this.top) gl.deleteVertexArray(this.top.vao);
    this.top = vertexArray(gl, verts, [
      { size: 2, location: 0 }, { size: 2, location: 1 }, { size: 1, location: 2 }, { size: 1, location: 3 },
    ], gl.STATIC_DRAW, index);

    // the edge of the sheet, a band from the top face down through its thickness
    const o = plate.outline, nb = o.length / 2;
    const rim = new Float32Array(nb * 6 * 9);
    let r = 0;
    const put = (i, e, bottom) => {
      const x = o[2 * i], y = o[2 * i + 1];
      rim.set([x, y, (x - f[3 * e]) / f[3 * e + 2], (y - f[3 * e + 1]) / f[3 * e + 2], f[3 * e + 2], e,
        plate.rimNormals[2 * i], plate.rimNormals[2 * i + 1], bottom], r * 9);
      r++;
    };
    for (let i = 0; i < nb; i++) {
      const j = (i + 1) % nb, e = plate.edgeElement[i];
      put(i, e, 0); put(j, e, 0); put(j, e, 1);
      put(i, e, 0); put(j, e, 1); put(i, e, 1);
    }
    if (this.rim) gl.deleteVertexArray(this.rim.vao);
    this.rim = vertexArray(gl, rim, [
      { size: 2, location: 0 }, { size: 2, location: 1 }, { size: 1, location: 2 }, { size: 1, location: 3 },
      { size: 2, location: 4 }, { size: 1, location: 5 },
    ]);

    // the solution, six texels per element
    const rows = Math.ceil(plate.nt / ROW);
    if (this.coef) gl.deleteTexture(this.coef);
    this.coefData = new Float32Array(ROW * 6 * rows * 4);
    this.coef = texture(gl, {
      width: ROW * 6, height: rows, internal: gl.RGBA32F, format: gl.RGBA, type: gl.FLOAT,
      data: this.coefData, filter: gl.NEAREST,
    });
    this.coefRows = rows;
  }

  // Upload the field to draw: one mode, or a bow-selected combination.
  setField(field) {
    const gl = this.gl, nt = this.plate.nt, out = this.coefData;
    out.fill(0);
    for (let e = 0; e < nt; e++) {
      const row = Math.floor(e / ROW), col = (e % ROW) * 6;
      const o = (row * ROW * 6 + col) * 4;
      for (let k = 0; k < 21; k++) out[o + k] = field[21 * e + k];
    }
    gl.bindTexture(gl.TEXTURE_2D, this.coef);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, ROW * 6, this.coefRows, gl.RGBA, gl.FLOAT, out);
  }

  // Signed distance to the outline, negative on the plate, for the rings
  // raked round it and for its shadow.
  setGround(plate) {
    const gl = this.gl, n = DIST_N, size = 2 * DIST_EXTENT, h = size / n, x0 = -DIST_EXTENT, y0 = -DIST_EXTENT;
    const o = plate.outline, nb = o.length / 2;
    const data = new Float32Array(n * n);
    for (let j = 0; j < n; j++) {
      const y = y0 + (j + 0.5) * h;
      for (let i = 0; i < n; i++) {
        const x = x0 + (i + 0.5) * h;
        let best = Infinity, inside = false;
        for (let a = 0, b = nb - 1; a < nb; b = a++) {
          const ax = o[2 * a], ay = o[2 * a + 1], bx = o[2 * b], by = o[2 * b + 1];
          const dx = bx - ax, dy = by - ay;
          let t = ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy);
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          const ex = x - ax - t * dx, ey = y - ay - t * dy;
          const d2 = ex * ex + ey * ey;
          if (d2 < best) best = d2;
          if ((ay > y) !== (by > y) && x < ax + ((y - ay) * dx) / dy) inside = !inside;
        }
        data[j * n + i] = inside ? -Math.sqrt(best) : Math.sqrt(best);
      }
    }
    if (this.dist) gl.deleteTexture(this.dist);
    this.dist = texture(gl, { width: n, height: n, internal: gl.R16F, format: gl.RED, type: gl.FLOAT, data });
    this.distBox = [x0, y0, size];
  }

  setSand(sand) {
    const gl = this.gl;
    if (this.sand) gl.deleteVertexArray(this.sand.vao);
    this.sand = vertexArray(gl, sand.pack(), [
      { size: 2, location: 0 }, { size: 1, location: 1 }, { size: 1, location: 2 }, { size: 1, location: 3 },
    ], gl.DYNAMIC_DRAW);
    attachAttributes(gl, this.sand, sand.traits, [{ size: 3, location: 4 }]);
    this.sand.count = sand.count;
  }

  updateSand(packed, count) {
    const gl = this.gl;
    this.sand.drawn = count;
    // a length of zero would mean "to the end" to bufferSubData, not nothing
    if (!count) return;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.sand.buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, packed, 0, count * 5);
  }

  // --- fixed geometry ----------------------------------------------------

  _ground() {
    const r = 6;
    this.floor = vertexArray(this.gl, new Float32Array([-r, -r, r, -r, r, r, -r, -r, r, r, -r, r]), [{ size: 2, location: 0 }]);
  }

  // A violin bow near its head, in its own frame: z along the hair, x across
  // the ribbon, y out from the plate toward the stick. Lengths in plate units
  // for an 18 cm plate: the ribbon is a centimetre wide and the stick stands
  // two centimetres off it. The bow is held from above, so the frog and the
  // hand are up out of the frame and the head hangs below the plate.
  _bow() {
    const hair = [0.4, 0.39, 0.36], wood = [0.09, 0.075, 0.065], bone = [0.62, 0.6, 0.55];
    const data = [];
    // a prism along z with a regular cross-section, so the stick reads round
    const prism = (cx, cy, r, sides, z0, z1, color) => {
      for (let k = 0; k < sides; k++) {
        const a0 = (2 * Math.PI * k) / sides, a1 = (2 * Math.PI * (k + 1)) / sides, am = 0.5 * (a0 + a1);
        const p0 = [cx + r * Math.cos(a0), cy + r * Math.sin(a0)], p1 = [cx + r * Math.cos(a1), cy + r * Math.sin(a1)];
        const n = [Math.cos(am), Math.sin(am), 0];
        for (const v of [[p0, z0], [p1, z0], [p1, z1], [p0, z0], [p1, z1], [p0, z1]]) data.push(v[0][0], v[0][1], v[1], ...n, ...color);
      }
    };
    const slab = (x0, x1, y0, y1, z0, z1, color) => {
      const q = (a, b, c, d, n) => { for (const v of [a, b, c, a, c, d]) data.push(...v, ...n, ...color); };
      q([x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1], [0, -1, 0]);
      q([x1, y1, z0], [x0, y1, z0], [x0, y1, z1], [x1, y1, z1], [0, 1, 0]);
      q([x0, y1, z0], [x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [-1, 0, 0]);
      q([x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1], [1, 0, 0]);
      q([x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [x1, y0, z0], [0, 0, -1]);
    };
    slab(-0.02, 0.02, 0.0, 0.0025, -0.2, 1.4, hair);
    prism(0, 0.1, 0.0085, 8, -0.25, 1.4, wood);
    slab(-0.013, 0.013, 0.0, 0.1, -0.285, -0.2, wood);
    slab(-0.0135, 0.0135, 0.004, 0.096, -0.292, -0.284, bone);
    this.bowMesh = vertexArray(this.gl, new Float32Array(data), [
      { size: 3, location: 0 }, { size: 3, location: 1 }, { size: 3, location: 2 },
    ]);
  }

  _sprites() {
    const gl = this.gl;
    this.spriteData = new Float32Array(256 * 8);
    this.spriteVao = gl.createVertexArray();
    gl.bindVertexArray(this.spriteVao);
    const corners = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, corners);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.spriteBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.spriteBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.spriteData.byteLength, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 32, 0);
    gl.vertexAttribDivisor(1, 1);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 4, gl.FLOAT, false, 32, 16);
    gl.vertexAttribDivisor(2, 1);
    gl.bindVertexArray(null);
  }

  // --- targets -----------------------------------------------------------

  resize() {
    const gl = this.gl;
    const rect = this.canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width * this.scale));
    const h = Math.max(1, Math.round(rect.height * this.scale));
    if (this.targets && this.targets.w === w && this.targets.h === h) return;
    this.canvas.width = w;
    this.canvas.height = h;
    if (this.targets) {
      release(gl, this.targets.ms);
      release(gl, this.targets.scene);
      this.targets.chain.forEach((t) => release(gl, t));
    }
    const internal = this.hdr ? gl.RGBA16F : gl.RGBA8;
    const type = this.hdr ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
    const ms = multisample(gl, w, h, this.samples, internal);
    const scene = target(gl, w, h, { internal, format: gl.RGBA, type });
    const chain = [];
    let cw = w, ch = h;
    for (let i = 0; i < 6; i++) {
      cw = Math.max(1, cw >> 1);
      ch = Math.max(1, ch >> 1);
      chain.push(target(gl, cw, ch, { internal, format: gl.RGBA, type }));
    }
    this.targets = { w, h, ms, scene, chain };
  }

  // --- drawing -----------------------------------------------------------

  render(s) {
    const gl = this.gl;
    this.resize();
    const { w, h, ms, scene, chain } = this.targets;
    const P = this.prog;

    gl.bindFramebuffer(gl.FRAMEBUFFER, ms.fbo);
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.disable(gl.BLEND);

    // the dusk behind everything
    gl.disable(gl.DEPTH_TEST);
    gl.useProgram(P.sky.handle);
    gl.uniform3fv(P.sky.uniforms.uTop, SCENE.sky.top);
    gl.uniform3fv(P.sky.uniforms.uBottom, SCENE.sky.bottom);
    gl.bindVertexArray(this.empty);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);

    const lights = (prog) => {
      gl.uniform3fv(prog.uniforms.uKeyDir, s.keyDir);
      gl.uniform3fv(prog.uniforms.uKeyColor, s.keyColor);
      if (prog.uniforms.uFillDir) gl.uniform3fv(prog.uniforms.uFillDir, s.fillDir);
      if (prog.uniforms.uFillColor) gl.uniform3fv(prog.uniforms.uFillColor, s.fillColor);
      if (prog.uniforms.uEnvTurn) gl.uniform1f(prog.uniforms.uEnvTurn, s.envTurn || 0);
      if (prog.uniforms.uF0) gl.uniform3fv(prog.uniforms.uF0, s.material.f0);
      if (prog.uniforms.uRough) gl.uniform1f(prog.uniforms.uRough, s.material.rough);
      if (prog.uniforms.uGlass) gl.uniform1f(prog.uniforms.uGlass, s.material.glass);
    };

    // raked gravel, with the plate's shadow on it
    if (this.dist) {
      const u = P.ground.uniforms, G = SCENE.ground;
      gl.useProgram(P.ground.handle);
      gl.uniformMatrix4fv(u.uViewProj, false, s.viewProj);
      gl.uniform1f(u.uZ, FLOOR_Z);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.dist);
      gl.uniform1i(u.uDist, 1);
      gl.uniform3fv(u.uDistBox, this.distBox);
      gl.uniform3fv(u.uA, G.a);
      gl.uniform3fv(u.uB, G.b);
      gl.uniform3fv(u.uFog, G.fog);
      gl.uniform1f(u.uShadowK, G.shadow);
      gl.uniform1f(u.uBlur, G.blur);
      gl.uniform1f(u.uFogNear, G.fogNear);
      gl.uniform1f(u.uFogFar, G.fogFar);
      const k = s.keyDir;
      gl.uniform2f(u.uCast, (k[0] / Math.max(k[2], 0.3)) * -FLOOR_Z, (k[1] / Math.max(k[2], 0.3)) * -FLOOR_Z);
      gl.bindVertexArray(this.floor.vao);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    if (this.top) {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.coef);

      gl.useProgram(P.plate.handle);
      const u = P.plate.uniforms;
      gl.uniform1i(u.uCoef, 0);
      gl.uniformMatrix4fv(u.uViewProj, false, s.viewProj);
      gl.uniform3fv(u.uEye, s.eye);
      gl.uniform1f(u.uDisp, s.disp);
      gl.uniform1i(u.uView, s.view);
      gl.uniform1f(u.uHoloK, s.holoK);
      gl.uniform1f(u.uNodes, s.nodes);
      gl.uniform1f(u.uField, s.field);
      gl.uniform1f(u.uSigned, s.signed);
      gl.uniform3fv(u.uAmbient, SCENE.ambient);
      gl.uniform3fv(u.uFaceA, s.material.linear.a);
      gl.uniform3fv(u.uFaceB, s.material.linear.b);
      gl.uniform1f(u.uSpeck, s.material.speck);
      gl.uniform1f(u.uSheen, s.material.sheen);
      gl.uniform3fv(u.uHoloLo, SCENE.hologram.lo);
      gl.uniform3fv(u.uHoloHi, SCENE.hologram.hi);
      gl.uniform3fv(u.uLine, SCENE.lines);
      lights(P.plate);
      gl.bindVertexArray(this.top.vao);
      gl.drawElements(gl.TRIANGLES, this.top.count, this.top.indexType, 0);

      gl.useProgram(P.rim.handle);
      gl.uniform1i(P.rim.uniforms.uCoef, 0);
      gl.uniformMatrix4fv(P.rim.uniforms.uViewProj, false, s.viewProj);
      gl.uniform3fv(P.rim.uniforms.uEye, s.eye);
      gl.uniform1f(P.rim.uniforms.uDisp, s.disp);
      gl.uniform1f(P.rim.uniforms.uThick, s.thickness);
      lights(P.rim);
      gl.bindVertexArray(this.rim.vao);
      gl.drawArrays(gl.TRIANGLES, 0, this.rim.count);
    }

    // sand: shadows first, darkening what is under them, then the grains
    if (this.sand && this.sand.drawn && s.sandVisible !== false) {
      const u = P.sand.uniforms;
      gl.useProgram(P.sand.handle);
      gl.uniformMatrix4fv(u.uViewProj, false, s.viewProj);
      gl.uniform1f(u.uTime, s.time);
      gl.uniform1f(u.uDisp, s.disp);
      gl.uniform1f(u.uHop, s.hop);
      gl.uniform1f(u.uGrain, s.grain);
      gl.uniform1f(u.uPixel, s.pixel * this.scale);
      gl.uniform3fv(u.uKeyDir, s.keyDir);
      gl.uniform3fv(u.uKeyView, s.keyView);
      gl.uniform3fv(u.uSandA, SCENE.sand.a);
      gl.uniform3fv(u.uSandB, SCENE.sand.b);
      gl.bindVertexArray(this.sand.vao);

      gl.uniform1f(u.uShadow, 1);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ZERO, gl.ONE_MINUS_SRC_ALPHA);
      gl.drawArrays(gl.POINTS, 0, this.sand.drawn);

      gl.disable(gl.BLEND);
      gl.depthMask(true);
      gl.uniform1f(u.uShadow, 0);
      gl.drawArrays(gl.POINTS, 0, this.sand.drawn);
    }

    // glows and rings lying on the sheet
    if (s.sprites && s.sprites.length) {
      const n = Math.min(256, s.sprites.length);
      const d = this.spriteData;
      for (let i = 0; i < n; i++) d.set(s.sprites[i], i * 8);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.spriteBuffer);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, d, 0, n * 8);
      gl.useProgram(P.sprite.handle);
      gl.uniformMatrix4fv(P.sprite.uniforms.uViewProj, false, s.viewProj);
      gl.uniform1f(P.sprite.uniforms.uZ, 0.0015);
      gl.depthMask(false);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.bindVertexArray(this.spriteVao);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, n);
      gl.depthMask(true);
      gl.disable(gl.BLEND);
    }

    if (s.bow && s.bow.fade > 0.01) {
      const u = P.bow.uniforms;
      gl.useProgram(P.bow.handle);
      gl.uniformMatrix4fv(u.uViewProj, false, s.viewProj);
      gl.uniformMatrix4fv(u.uModel, false, s.bow.model);
      gl.uniform3fv(u.uEye, s.eye);
      gl.uniform3fv(u.uKeyDir, s.keyDir);
      gl.uniform3fv(u.uKeyColor, s.keyColor);
      gl.uniform1f(u.uFade, s.bow.fade);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.bowMesh.vao);
      gl.drawArrays(gl.TRIANGLES, 0, this.bowMesh.count);
      gl.disable(gl.BLEND);
    }
    gl.bindVertexArray(null);

    // resolve, bloom, tonemap
    gl.disable(gl.DEPTH_TEST);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, ms.fbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, scene.fbo);
    gl.blitFramebuffer(0, 0, w, h, 0, 0, w, h, gl.COLOR_BUFFER_BIT, gl.NEAREST);

    gl.bindVertexArray(this.empty);
    const pass = (prog, dst, src, setup) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst ? dst.fbo : null);
      gl.viewport(0, 0, dst ? dst.width : w, dst ? dst.height : h);
      gl.useProgram(prog.handle);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, src.color);
      gl.uniform1i(prog.uniforms.uSource, 0);
      if (setup) setup(prog.uniforms);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    pass(P.bright, chain[0], scene, (u) => gl.uniform1f(u.uThreshold, this.hdr ? 1.35 : 0.85));
    for (let i = 1; i < chain.length; i++) {
      pass(P.down, chain[i], chain[i - 1], (u) => gl.uniform2f(u.uTexel, 1 / chain[i - 1].width, 1 / chain[i - 1].height));
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = chain.length - 1; i > 0; i--) {
      pass(P.up, chain[i - 1], chain[i], (u) => gl.uniform2f(u.uTexel, 1 / chain[i].width, 1 / chain[i].height));
    }
    gl.disable(gl.BLEND);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, w, h);
    const c = P.composite;
    gl.useProgram(c.handle);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, scene.color);
    gl.uniform1i(c.uniforms.uScene, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, chain[0].color);
    gl.uniform1i(c.uniforms.uBloom, 1);
    const g = SCENE.grade;
    gl.uniform1f(c.uniforms.uBloomStrength, s.bloom ?? g.bloom);
    gl.uniform1f(c.uniforms.uExposure, s.exposure * g.exposure);
    gl.uniform1f(c.uniforms.uTonemap, g.tonemap);
    gl.uniform1f(c.uniforms.uSaturation, g.saturation);
    gl.uniform1f(c.uniforms.uContrast, g.contrast);
    gl.uniform1f(c.uniforms.uVignette, g.vignette);
    gl.uniform1f(c.uniforms.uGrain, g.grain);
    gl.uniform1f(c.uniforms.uAspect, w / h);
    gl.uniform1f(c.uniforms.uTime, s.time);
    gl.uniform1f(c.uniforms.uFlash, s.flash || 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }
}

export { mat4 };
