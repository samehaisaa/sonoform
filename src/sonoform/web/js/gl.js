// Thin WebGL2 helpers. Everything the renderer builds goes through these.

export function compile(gl, type, source, label) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`${label || 'shader'}: ${log}`);
  }
  return shader;
}

// A linked program with its uniform locations looked up once.
export function program(gl, vertex, fragment, label) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vertex, `${label} vertex`));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fragment, `${label} fragment`));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(`${label}: ${gl.getProgramInfoLog(p)}`);
  }
  const uniforms = {};
  const count = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(p, i);
    const name = info.name.replace(/\[0\]$/, '');
    uniforms[name] = gl.getUniformLocation(p, info.name);
  }
  return { handle: p, uniforms, label };
}

// Vertex array from interleaved float attributes: [{ size, location }].
export function vertexArray(gl, data, layout, usage = gl.STATIC_DRAW, index = null) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, usage);
  const stride = layout.reduce((s, a) => s + a.size, 0) * 4;
  let offset = 0;
  for (const a of layout) {
    gl.enableVertexAttribArray(a.location);
    gl.vertexAttribPointer(a.location, a.size, gl.FLOAT, false, stride, offset);
    offset += a.size * 4;
  }
  let indexBuffer = null;
  if (index) {
    indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, index, gl.STATIC_DRAW);
  }
  gl.bindVertexArray(null);
  return {
    vao,
    buffer,
    indexBuffer,
    count: index ? index.length : data.length / (stride / 4),
    indexType: index instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT,
    stride,
  };
}

// A second buffer of per-vertex attributes on an existing vertex array.
export function attachAttributes(gl, target, data, layout, usage = gl.STATIC_DRAW) {
  gl.bindVertexArray(target.vao);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, data, usage);
  const stride = layout.reduce((s, a) => s + a.size, 0) * 4;
  let offset = 0;
  for (const a of layout) {
    gl.enableVertexAttribArray(a.location);
    gl.vertexAttribPointer(a.location, a.size, gl.FLOAT, false, stride, offset);
    offset += a.size * 4;
  }
  gl.bindVertexArray(null);
  return buffer;
}

export function texture(gl, { width, height, internal, format, type, data = null, filter = gl.LINEAR, wrap = gl.CLAMP_TO_EDGE }) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, width, height, 0, format, type, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, wrap);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, wrap);
  return t;
}

// Render target with a colour texture, and optionally depth.
export function target(gl, width, height, { internal, format, type, depth = false }) {
  const fbo = gl.createFramebuffer();
  const color = texture(gl, { width, height, internal, format, type });
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
  let depthBuffer = null;
  if (depth) {
    depthBuffer = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, width, height);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthBuffer);
  }
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, color, depthBuffer, width, height, ok };
}

// Multisampled colour and depth, resolved into a target by blitting.
export function multisample(gl, width, height, samples, internal) {
  const fbo = gl.createFramebuffer();
  const color = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, color);
  gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, internal, width, height);
  const depth = gl.createRenderbuffer();
  gl.bindRenderbuffer(gl.RENDERBUFFER, depth);
  gl.renderbufferStorageMultisample(gl.RENDERBUFFER, samples, gl.DEPTH_COMPONENT24, width, height);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.RENDERBUFFER, color);
  gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depth);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return { fbo, color, depth, width, height, ok };
}

export function release(gl, t) {
  if (!t) return;
  if (t.fbo) gl.deleteFramebuffer(t.fbo);
  if (t.color && t.color instanceof WebGLTexture) gl.deleteTexture(t.color);
  if (t.color && t.color instanceof WebGLRenderbuffer) gl.deleteRenderbuffer(t.color);
  if (t.depth) gl.deleteRenderbuffer(t.depth);
  if (t.depthBuffer) gl.deleteRenderbuffer(t.depthBuffer);
}

// Column-major 4x4 helpers, just what the renderer needs.
export const mat4 = {
  multiply(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        o[c * 4 + r] =
          a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] +
          a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
      }
    }
    return o;
  },
  // offsetX and offsetY shift the image in normalised device units, which
  // moves the centre of projection without turning the camera.
  perspective(fovy, aspect, near, far, offsetY = 0, offsetX = 0) {
    const f = 1 / Math.tan(fovy / 2);
    const m = new Float32Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[8] = -offsetX;
    m[9] = -offsetY;
    m[10] = (far + near) / (near - far);
    m[11] = -1;
    m[14] = (2 * far * near) / (near - far);
    return m;
  },
  lookAt(eye, at, up) {
    let zx = eye[0] - at[0], zy = eye[1] - at[1], zz = eye[2] - at[2];
    let l = Math.hypot(zx, zy, zz);
    zx /= l; zy /= l; zz /= l;
    let xx = up[1] * zz - up[2] * zy, xy = up[2] * zx - up[0] * zz, xz = up[0] * zy - up[1] * zx;
    l = Math.hypot(xx, xy, xz) || 1;
    xx /= l; xy /= l; xz /= l;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    const m = new Float32Array(16);
    m[0] = xx; m[4] = xy; m[8] = xz;
    m[1] = yx; m[5] = yy; m[9] = yz;
    m[2] = zx; m[6] = zy; m[10] = zz;
    m[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
    m[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
    m[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
    m[15] = 1;
    return { m, right: [xx, xy, xz], up: [yx, yy, yz], back: [zx, zy, zz] };
  },
};
