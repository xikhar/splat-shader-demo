import * as THREE from 'three';

/**
 * Custom WebGL2 Anisotropic 3D Gaussian Splat Renderer.
 *
 * Uses EWA screen-space covariance projection in the vertex shader + fast
 * 2-pass 16-bit radix sort for back-to-front alpha compositing.
 *
 * Also supports a companion "core depth + surface-id" pass so screen-space NPR
 * post-processing shaders (Obra Dinn 1-bit, Ligne Claire, Ukiyo-e, Kuwahara)
 * receive a clean linear depth + object-boundary buffer alongside the soft
 * alpha-blended splat color buffer.
 */

const SPLAT_COLOR_VERT = /* glsl */ `
precision highp float;
precision highp int;

attribute float aSplatIndex;

uniform sampler2D uPosScaleTex;   // RGBA32F: xyz = pos, w = groupId / flags
uniform sampler2D uColorOpTex;    // RGBA32F: rgb = color, a = opacity
uniform sampler2D uCovATex;       // RGBA32F: (c00, c01, c02, c11) upper 3D covariance
uniform sampler2D uCovBTex;       // RGBA32F: (c12, c22, animPhase, animKind)
uniform int uTexWidth;
uniform vec2 uViewport;
uniform vec2 uFocal;
uniform float uSplatScale;
uniform float uTime;
uniform float uFloaterStrength;

varying vec2 vLocal;
varying vec4 vColor;

ivec2 idxToUV(int idx) {
  int y = idx / uTexWidth;
  int x = idx - y * uTexWidth;
  return ivec2(x, y);
}

void main() {
  int idx = int(aSplatIndex + 0.5);
  ivec2 uv = idxToUV(idx);

  vec4 posGroup = texelFetch(uPosScaleTex, uv, 0);
  vec4 colOp = texelFetch(uColorOpTex, uv, 0);
  vec4 covA = texelFetch(uCovATex, uv, 0);
  vec4 covB = texelFetch(uCovBTex, uv, 0);

  vec3 pos = posGroup.xyz;
  float animPhase = covB.z;
  float animKind = covB.w;
  float opacity = colOp.a;

  // Optional subtle world-alive motion & toggleable reconstruction floaters
  if (animKind > 0.5 && animKind < 1.5) {
    // 1.0 = Reconstruction floater artifact (controlled by uFloaterStrength)
    opacity *= uFloaterStrength;
    if (opacity < 0.01) {
      gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
      return;
    }
    pos.y += sin(uTime * 0.7 + animPhase) * 0.06;
  } else if (animKind >= 1.5 && animKind < 2.5) {
    // 2.0 = Wind / banner / foliage sway
    pos.x += sin(uTime * 1.9 + animPhase + pos.y * 0.8) * 0.08;
    pos.z += cos(uTime * 1.5 + animPhase) * 0.06;
  } else if (animKind >= 2.5 && animKind < 3.5) {
    // 3.0 = Falling / waterfall / rain / blossom drift
    float cycle = fract(animPhase * 0.15915 + uTime * 0.24);
    pos.y -= (cycle - 0.5) * 1.4;
    pos.x += sin(uTime * 1.6 + animPhase) * 0.14;
  } else if (animKind >= 3.5 && animKind < 4.5) {
    // 4.0 = Orbiting / beacon rotation around world Y
    float ang = uTime * 0.55;
    float cs = cos(ang), sn = sin(ang);
    pos.xz = mat2(cs, -sn, sn, cs) * pos.xz;
  } else if (animKind >= 4.5) {
    // 5.0 = Bobbing water / boat / wave crest
    pos.y += sin(uTime * 1.8 + pos.x * 0.7 + pos.z * 0.5 + animPhase) * 0.09;
  }

  vec4 viewPos = modelViewMatrix * vec4(pos, 1.0);
  if (viewPos.z > -0.15) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  vec4 clipPos = projectionMatrix * viewPos;
  if (abs(clipPos.x) > clipPos.w * 1.35 || abs(clipPos.y) > clipPos.w * 1.35) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  // Reconstruct symmetric 3D covariance matrix Sigma (scaled by uSplatScale^2)
  float s2 = uSplatScale * uSplatScale;
  mat3 Vrk = mat3(
    covA.x, covA.y, covA.z,
    covA.y, covA.w, covB.x,
    covA.z, covB.x, covB.y
  ) * s2;

  // View-space Jacobian for perspective projection (EWA splatting)
  float invZ = 1.0 / (-viewPos.z);
  float tx = clamp(viewPos.x * invZ, -1.35, 1.35) * (-viewPos.z);
  float ty = clamp(viewPos.y * invZ, -1.35, 1.35) * (-viewPos.z);
  float tz = -viewPos.z;

  mat3 J = mat3(
    uFocal.x / tz, 0.0, -(uFocal.x * tx) / (tz * tz),
    0.0, uFocal.y / tz, -(uFocal.y * ty) / (tz * tz),
    0.0, 0.0, 0.0
  );

  mat3 W = mat3(modelViewMatrix);
  mat3 T = J * W;
  mat3 cov2dMat = T * Vrk * transpose(T);

  // Low-pass filter (0.35px) prevents sub-pixel aliasing
  float a = cov2dMat[0][0] + 0.35;
  float b = cov2dMat[0][1];
  float d = cov2dMat[1][1] + 0.35;

  float trace = a + d;
  float det = max(0.05, a * d - b * b);
  float mid = 0.5 * trace;
  float disc = sqrt(max(0.0001, mid * mid - det));
  float lambda1 = max(0.1, mid + disc);
  float lambda2 = max(0.1, mid - disc);

  vec2 eigVec1 = normalize(vec2(b, lambda1 - a) + vec2(1e-5, 0.0));
  vec2 eigVec2 = vec2(-eigVec1.y, eigVec1.x);

  // 2.6 sigma quad radius clamped to avoid giant full-screen overdraw
  float rad1 = min(sqrt(lambda1) * 2.6, max(uViewport.x, uViewport.y) * 0.28);
  float rad2 = min(sqrt(lambda2) * 2.6, max(uViewport.x, uViewport.y) * 0.28);

  vec2 quad = position.xy; // -1..1
  vec2 offsetPx = eigVec1 * (quad.x * rad1) + eigVec2 * (quad.y * rad2);
  vec2 ndcOffset = (offsetPx / uViewport) * 2.0;

  gl_Position = vec4(clipPos.xy / clipPos.w + ndcOffset, clipPos.z / clipPos.w, 1.0);
  vLocal = quad * 2.6;
  vColor = vec4(colOp.rgb, opacity);
}
`;

const SPLAT_COLOR_FRAG = /* glsl */ `
precision highp float;

varying vec2 vLocal;
varying vec4 vColor;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > 6.76) discard;
  float gauss = exp(-0.5 * r2);
  float alpha = min(0.99, gauss * vColor.a);
  if (alpha < 0.015) discard;
  gl_FragColor = vec4(vColor.rgb, alpha);
}
`;

// Companion Depth + GroupID pass (writes hard core depth so NPR shaders get razor-sharp silhouettes!)
const SPLAT_DEPTH_VERT = /* glsl */ `
precision highp float;
precision highp int;

attribute float aSplatIndex;

uniform sampler2D uPosScaleTex;
uniform sampler2D uColorOpTex;
uniform sampler2D uCovATex;
uniform sampler2D uCovBTex;
uniform int uTexWidth;
uniform vec2 uViewport;
uniform vec2 uFocal;
uniform float uSplatScale;
uniform float uTime;
uniform float uNear;
uniform float uFar;

varying vec2 vLocal;
varying float vLinearDepth;
varying float vGroupId;

ivec2 idxToUV(int idx) {
  int y = idx / uTexWidth;
  int x = idx - y * uTexWidth;
  return ivec2(x, y);
}

void main() {
  int idx = int(aSplatIndex + 0.5);
  ivec2 uv = idxToUV(idx);

  vec4 posGroup = texelFetch(uPosScaleTex, uv, 0);
  vec4 colOp = texelFetch(uColorOpTex, uv, 0);
  vec4 covA = texelFetch(uCovATex, uv, 0);
  vec4 covB = texelFetch(uCovBTex, uv, 0);

  float animKind = covB.w;
  // Skip reconstruction floaters, rain/petal particles, orbiting light beams, and low-opacity haze
  // in the depth pass so silhouette outlines stay razor-sharp on solid architecture & terrain!
  if (
    (animKind > 0.5 && animKind < 1.5) ||
    (animKind >= 2.5 && animKind < 3.5) ||
    colOp.a < 0.52 ||
    posGroup.w < 0.08
  ) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  vec3 pos = posGroup.xyz;
  float animPhase = covB.z;
  if (animKind >= 1.5 && animKind < 2.5) {
    pos.x += sin(uTime * 1.9 + animPhase + pos.y * 0.8) * 0.08;
    pos.z += cos(uTime * 1.5 + animPhase) * 0.06;
  } else if (animKind >= 2.5 && animKind < 3.5) {
    float cycle = fract(animPhase * 0.15915 + uTime * 0.24);
    pos.y -= (cycle - 0.5) * 1.4;
    pos.x += sin(uTime * 1.6 + animPhase) * 0.14;
  } else if (animKind >= 3.5 && animKind < 4.5) {
    float ang = uTime * 0.55;
    float cs = cos(ang), sn = sin(ang);
    pos.xz = mat2(cs, -sn, sn, cs) * pos.xz;
  } else if (animKind >= 4.5) {
    pos.y += sin(uTime * 1.8 + pos.x * 0.7 + pos.z * 0.5 + animPhase) * 0.09;
  }

  vec4 viewPos = modelViewMatrix * vec4(pos, 1.0);
  if (viewPos.z > -0.15) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  vec4 clipPos = projectionMatrix * viewPos;
  float s2 = uSplatScale * uSplatScale;
  mat3 Vrk = mat3(
    covA.x, covA.y, covA.z,
    covA.y, covA.w, covB.x,
    covA.z, covB.x, covB.y
  ) * s2;

  float tz = -viewPos.z;
  float tx = clamp(viewPos.x / tz, -1.35, 1.35) * tz;
  float ty = clamp(viewPos.y / tz, -1.35, 1.35) * tz;
  mat3 J = mat3(
    uFocal.x / tz, 0.0, -(uFocal.x * tx) / (tz * tz),
    0.0, uFocal.y / tz, -(uFocal.y * ty) / (tz * tz),
    0.0, 0.0, 0.0
  );
  mat3 W = mat3(modelViewMatrix);
  mat3 T = J * W;
  mat3 cov2dMat = T * Vrk * transpose(T);

  float a = cov2dMat[0][0] + 0.35;
  float b = cov2dMat[0][1];
  float d = cov2dMat[1][1] + 0.35;
  float mid = 0.5 * (a + d);
  float det = max(0.05, a * d - b * b);
  float disc = sqrt(max(0.0001, mid * mid - det));
  float lambda1 = max(0.1, mid + disc);
  float lambda2 = max(0.1, mid - disc);

  vec2 eigVec1 = normalize(vec2(b, lambda1 - a) + vec2(1e-5, 0.0));
  vec2 eigVec2 = vec2(-eigVec1.y, eigVec1.x);

  // Core silhouette footprint (1.55 sigma)
  float rad1 = min(sqrt(lambda1) * 1.55, max(uViewport.x, uViewport.y) * 0.18);
  float rad2 = min(sqrt(lambda2) * 1.55, max(uViewport.x, uViewport.y) * 0.18);

  vec2 quad = position.xy;
  vec2 offsetPx = eigVec1 * (quad.x * rad1) + eigVec2 * (quad.y * rad2);
  vec2 ndcOffset = (offsetPx / uViewport) * 2.0;

  gl_Position = vec4(clipPos.xy / clipPos.w + ndcOffset, clipPos.z / clipPos.w, 1.0);
  vLocal = quad;
  vLinearDepth = clamp((-viewPos.z - uNear) / (uFar - uNear), 0.0, 1.0);
  vGroupId = posGroup.w;
}
`;

const SPLAT_DEPTH_FRAG = /* glsl */ `
precision highp float;

varying vec2 vLocal;
varying float vLinearDepth;
varying float vGroupId;

void main() {
  if (dot(vLocal, vLocal) > 1.0) discard;
  // R = linear depth (0..1), G = semantic group ID (0..1), B = valid geometry flag (1.0)
  gl_FragColor = vec4(vLinearDepth, vGroupId, 1.0, 1.0);
}
`;

// Hard-disc surfel pass (Gaussian falloff switched off) for Cobanov-style crisp Pixel Art!
const SPLAT_CRISP_VERT = /* glsl */ `
precision highp float;
precision highp int;

attribute float aSplatIndex;

uniform sampler2D uPosScaleTex;
uniform sampler2D uColorOpTex;
uniform sampler2D uCovATex;
uniform sampler2D uCovBTex;
uniform int uTexWidth;
uniform vec2 uCrispViewport;
uniform vec2 uCrispFocal;
uniform float uSplatScale;
uniform float uTime;

varying vec2 vLocal;
varying vec3 vCol;

ivec2 idxToUV(int idx) {
  int y = idx / uTexWidth;
  int x = idx - y * uTexWidth;
  return ivec2(x, y);
}

void main() {
  int idx = int(aSplatIndex + 0.5);
  ivec2 uv = idxToUV(idx);

  vec4 posGroup = texelFetch(uPosScaleTex, uv, 0);
  vec4 colOp = texelFetch(uColorOpTex, uv, 0);
  vec4 covA = texelFetch(uCovATex, uv, 0);
  vec4 covB = texelFetch(uCovBTex, uv, 0);

  float animKind = covB.w;
  // Exclude floaters and faint translucent mist/beams (translucent glow composites from colorRT!)
  if ((animKind > 0.5 && animKind < 1.5) || colOp.a < 0.48) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  vec3 pos = posGroup.xyz;
  float animPhase = covB.z;
  if (animKind >= 1.5 && animKind < 2.5) {
    pos.x += sin(uTime * 1.9 + animPhase + pos.y * 0.8) * 0.08;
    pos.z += cos(uTime * 1.5 + animPhase) * 0.06;
  } else if (animKind >= 2.5 && animKind < 3.5) {
    float cycle = fract(animPhase * 0.15915 + uTime * 0.24);
    pos.y -= (cycle - 0.5) * 1.4;
    pos.x += sin(uTime * 1.6 + animPhase) * 0.14;
  } else if (animKind >= 3.5 && animKind < 4.5) {
    float ang = uTime * 0.55;
    float cs = cos(ang), sn = sin(ang);
    pos.xz = mat2(cs, -sn, sn, cs) * pos.xz;
  } else if (animKind >= 4.5) {
    pos.y += sin(uTime * 1.8 + pos.x * 0.7 + pos.z * 0.5 + animPhase) * 0.09;
  }

  vec4 viewPos = modelViewMatrix * vec4(pos, 1.0);
  if (viewPos.z > -0.15) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  vec4 clipPos = projectionMatrix * viewPos;
  float s2 = uSplatScale * uSplatScale;
  mat3 Vrk = mat3(
    covA.x, covA.y, covA.z,
    covA.y, covA.w, covB.x,
    covA.z, covB.x, covB.y
  ) * s2;

  float tz = -viewPos.z;
  float tx = clamp(viewPos.x / tz, -1.35, 1.35) * tz;
  float ty = clamp(viewPos.y / tz, -1.35, 1.35) * tz;
  mat3 J = mat3(
    uCrispFocal.x / tz, 0.0, -(uCrispFocal.x * tx) / (tz * tz),
    0.0, uCrispFocal.y / tz, -(uCrispFocal.y * ty) / (tz * tz),
    0.0, 0.0, 0.0
  );
  mat3 W = mat3(modelViewMatrix);
  mat3 T = J * W;
  mat3 cov2dMat = T * Vrk * transpose(T);

  float a = cov2dMat[0][0] + 0.28;
  float b = cov2dMat[0][1];
  float d = cov2dMat[1][1] + 0.28;
  float mid = 0.5 * (a + d);
  float det = max(0.05, a * d - b * b);
  float disc = sqrt(max(0.0001, mid * mid - det));
  float lambda1 = max(0.25, mid + disc);
  float lambda2 = max(0.25, mid - disc);

  vec2 eigVec1 = normalize(vec2(b, lambda1 - a) + vec2(1e-5, 0.0));
  vec2 eigVec2 = vec2(-eigVec1.y, eigVec1.x);

  // Solid surfel disc radius (1.58 sigma -> clean gapless coverage without fuzzy halos)
  float rad1 = min(sqrt(lambda1) * 1.58, max(uCrispViewport.x, uCrispViewport.y) * 0.2);
  float rad2 = min(sqrt(lambda2) * 1.58, max(uCrispViewport.x, uCrispViewport.y) * 0.2);

  vec2 quad = position.xy;
  vec2 offsetPx = eigVec1 * (quad.x * rad1) + eigVec2 * (quad.y * rad2);
  vec2 ndcOffset = (offsetPx / uCrispViewport) * 2.0;

  gl_Position = vec4(clipPos.xy / clipPos.w + ndcOffset, clipPos.z / clipPos.w, 1.0);
  vLocal = quad;
  vCol = colOp.rgb;
}
`;

const SPLAT_CRISP_FRAG = /* glsl */ `
precision highp float;

varying vec2 vLocal;
varying vec3 vCol;

void main() {
  if (dot(vLocal, vLocal) > 1.0) discard;
  gl_FragColor = vec4(vCol, 1.0);
}
`;

/**
 * Compute upper 6 entries of symmetric 3D covariance Sigma = R * S * S^T * R^T
 */
export function computeCov3D(sx, sy, sz, qx, qy, qz, qw) {
  // Normalize quaternion
  const len = Math.hypot(qx, qy, qz, qw) || 1;
  const x = qx / len;
  const y = qy / len;
  const z = qz / len;
  const w = qw / len;

  const r00 = 1 - 2 * (y * y + z * z);
  const r01 = 2 * (x * y - z * w);
  const r02 = 2 * (x * z + y * w);
  const r10 = 2 * (x * y + z * w);
  const r11 = 1 - 2 * (x * x + z * z);
  const r12 = 2 * (y * z - x * w);
  const r20 = 2 * (x * z - y * w);
  const r21 = 2 * (y * z + x * w);
  const r22 = 1 - 2 * (x * x + y * y);

  // M = R * S
  const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
  const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
  const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

  // Sigma = M * M^T
  const c00 = m00 * m00 + m01 * m01 + m02 * m02;
  const c01 = m00 * m10 + m01 * m11 + m02 * m12;
  const c02 = m00 * m20 + m01 * m21 + m02 * m22;
  const c11 = m10 * m10 + m11 * m11 + m12 * m12;
  const c12 = m10 * m20 + m11 * m21 + m12 * m22;
  const c22 = m20 * m20 + m21 * m21 + m22 * m22;
  return [c00, c01, c02, c11, c12, c22];
}

export class SplatCloud {
  /**
   * @param {object} data
   * @param {number} data.count
   * @param {Float32Array} data.positions   length count * 3
   * @param {Float32Array} data.scales      length count * 3
   * @param {Float32Array} data.quaternions length count * 4 (xyzw)
   * @param {Float32Array} data.colors      length count * 3 (0..1)
   * @param {Float32Array} data.opacities   length count
   * @param {Float32Array} [data.groupIds]  length count (0..1)
   * @param {Float32Array} [data.animKinds] length count (0=static,1=floater,2=sway,3=fall,4=orbit,5=bob)
   * @param {Float32Array} [data.animPhases] length count
   */
  constructor(data) {
    this.count = data.count;
    this.positions = data.positions;
    this.texWidth = Math.min(2048, Math.max(64, Math.ceil(Math.sqrt(this.count))));
    this.texHeight = Math.ceil(this.count / this.texWidth);
    const totalTexels = this.texWidth * this.texHeight;

    const posBuf = new Float32Array(totalTexels * 4);
    const colBuf = new Float32Array(totalTexels * 4);
    const covABuf = new Float32Array(totalTexels * 4);
    const covBBuf = new Float32Array(totalTexels * 4);

    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      const i4 = i * 4;
      posBuf[i4] = data.positions[i3];
      posBuf[i4 + 1] = data.positions[i3 + 1];
      posBuf[i4 + 2] = data.positions[i3 + 2];
      posBuf[i4 + 3] = data.groupIds ? data.groupIds[i] : 0.5;

      colBuf[i4] = data.colors[i3];
      colBuf[i4 + 1] = data.colors[i3 + 1];
      colBuf[i4 + 2] = data.colors[i3 + 2];
      colBuf[i4 + 3] = data.opacities[i];

      const [c00, c01, c02, c11, c12, c22] = computeCov3D(
        data.scales[i3],
        data.scales[i3 + 1],
        data.scales[i3 + 2],
        data.quaternions[i4],
        data.quaternions[i4 + 1],
        data.quaternions[i4 + 2],
        data.quaternions[i4 + 3]
      );
      covABuf[i4] = c00;
      covABuf[i4 + 1] = c01;
      covABuf[i4 + 2] = c02;
      covABuf[i4 + 3] = c11;

      covBBuf[i4] = c12;
      covBBuf[i4 + 1] = c22;
      covBBuf[i4 + 2] = data.animPhases ? data.animPhases[i] : (i * 0.61803) % 6.283;
      covBBuf[i4 + 3] = data.animKinds ? data.animKinds[i] : 0;
    }

    this.posTex = this._makeDataTex(posBuf);
    this.colTex = this._makeDataTex(colBuf);
    this.covATex = this._makeDataTex(covABuf);
    this.covBTex = this._makeDataTex(covBBuf);

    // Base quad geometry (-1..1)
    const baseGeo = new THREE.PlaneGeometry(2, 2);
    this.geometry = new THREE.InstancedBufferGeometry();
    this.geometry.index = baseGeo.index;
    this.geometry.attributes.position = baseGeo.attributes.position;
    this.geometry.instanceCount = this.count;

    this.sortedIndices = new Float32Array(this.count);
    for (let i = 0; i < this.count; i++) this.sortedIndices[i] = i;
    this.indexAttr = new THREE.InstancedBufferAttribute(this.sortedIndices, 1);
    this.indexAttr.setUsage(THREE.DynamicDrawUsage);
    this.geometry.setAttribute('aSplatIndex', this.indexAttr);

    // Scratch buffers for 16-bit radix sort
    this._keys = new Uint32Array(this.count);
    this._idxA = new Uint32Array(this.count);
    this._idxB = new Uint32Array(this.count);
    this._counts = new Uint32Array(256);
    this._lastViewDir = new THREE.Vector3(99, 99, 99);
    this._lastCamPos = new THREE.Vector3(999, 999, 999);

    const sharedUniforms = {
      uPosScaleTex: { value: this.posTex },
      uColorOpTex: { value: this.colTex },
      uCovATex: { value: this.covATex },
      uCovBTex: { value: this.covBTex },
      uTexWidth: { value: this.texWidth },
      uViewport: { value: new THREE.Vector2(1280, 720) },
      uFocal: { value: new THREE.Vector2(800, 800) },
      uCrispViewport: { value: new THREE.Vector2(640, 360) },
      uCrispFocal: { value: new THREE.Vector2(400, 400) },
      uSplatScale: { value: 1.0 },
      uTime: { value: 0.0 },
      uFloaterStrength: { value: 0.55 },
      uNear: { value: 0.2 },
      uFar: { value: 85.0 },
    };
    this.uniforms = sharedUniforms;

    this.colorMaterial = new THREE.ShaderMaterial({
      vertexShader: SPLAT_COLOR_VERT,
      fragmentShader: SPLAT_COLOR_FRAG,
      uniforms: sharedUniforms,
      transparent: true,
      depthTest: false,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });

    this.depthMaterial = new THREE.ShaderMaterial({
      vertexShader: SPLAT_DEPTH_VERT,
      fragmentShader: SPLAT_DEPTH_FRAG,
      uniforms: sharedUniforms,
      transparent: false,
      depthTest: true,
      depthWrite: true,
    });

    this.crispMaterial = new THREE.ShaderMaterial({
      vertexShader: SPLAT_CRISP_VERT,
      fragmentShader: SPLAT_CRISP_FRAG,
      uniforms: sharedUniforms,
      transparent: false,
      depthTest: true,
      depthWrite: true,
    });

    this.colorMesh = new THREE.Mesh(this.geometry, this.colorMaterial);
    this.colorMesh.frustumCulled = false;

    this.depthMesh = new THREE.Mesh(this.geometry, this.depthMaterial);
    this.depthMesh.frustumCulled = false;

    this.crispMesh = new THREE.Mesh(this.geometry, this.crispMaterial);
    this.crispMesh.frustumCulled = false;
  }

  _makeDataTex(data) {
    const tex = new THREE.DataTexture(
      data,
      this.texWidth,
      this.texHeight,
      THREE.RGBAFormat,
      THREE.FloatType
    );
    tex.minFilter = THREE.NearestFilter;
    tex.magFilter = THREE.NearestFilter;
    tex.generateMipmaps = false;
    tex.needsUpdate = true;
    return tex;
  }

  updateUniforms(camera, width, height, time, splatScale, floaterStrength, crispW = 640, crispH = 360) {
    const fovRad = THREE.MathUtils.degToRad(camera.fov);
    const fy = (height * 0.5) / Math.tan(fovRad * 0.5);
    const fx = fy;
    const cfy = (crispH * 0.5) / Math.tan(fovRad * 0.5);
    this.uniforms.uViewport.value.set(width, height);
    this.uniforms.uFocal.value.set(fx, fy);
    this.uniforms.uCrispViewport.value.set(crispW, crispH);
    this.uniforms.uCrispFocal.value.set(cfy, cfy);
    this.uniforms.uTime.value = time;
    this.uniforms.uSplatScale.value = splatScale;
    this.uniforms.uFloaterStrength.value = floaterStrength;
    this.uniforms.uNear.value = camera.near;
    this.uniforms.uFar.value = Math.min(camera.far, 90.0);
  }

  /**
   * Fast 2-pass 8-bit radix sort (16-bit depth keys), back-to-front.
   */
  sort(camera, force = false) {
    const viewDir = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    if (
      !force &&
      viewDir.dot(this._lastViewDir) > 0.9994 &&
      camera.position.distanceToSquared(this._lastCamPos) < 0.02
    ) {
      return false;
    }
    this._lastViewDir.copy(viewDir);
    this._lastCamPos.copy(camera.position);

    const m = camera.matrixWorldInverse.elements;
    const m2 = m[2], m6 = m[6], m10 = m[10], m14 = m[14];
    const pos = this.positions;
    const count = this.count;

    let minZ = Infinity;
    let maxZ = -Infinity;
    const tmpZ = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const i3 = i * 3;
      // Camera looks down -Z, so more negative = farther away.
      // We want FARTHEST drawn FIRST (smallest z first -> ascending sort on viewZ).
      const z = m2 * pos[i3] + m6 * pos[i3 + 1] + m10 * pos[i3 + 2] + m14;
      tmpZ[i] = z;
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }

    const range = maxZ - minZ || 1.0;
    const invRange = 65535.0 / range;
    const keys = this._keys;
    const idxA = this._idxA;
    const idxB = this._idxB;
    const counts = this._counts;

    // Pass 1 (low byte)
    counts.fill(0);
    for (let i = 0; i < count; i++) {
      const k = ((tmpZ[i] - minZ) * invRange) | 0;
      keys[i] = k;
      counts[k & 255]++;
    }
    let sum = 0;
    for (let b = 0; b < 256; b++) {
      const c = counts[b];
      counts[b] = sum;
      sum += c;
    }
    for (let i = 0; i < count; i++) {
      const b = keys[i] & 255;
      idxA[counts[b]++] = i;
    }

    // Pass 2 (high byte)
    counts.fill(0);
    for (let i = 0; i < count; i++) {
      const k = keys[idxA[i]];
      counts[(k >> 8) & 255]++;
    }
    sum = 0;
    for (let b = 0; b < 256; b++) {
      const c = counts[b];
      counts[b] = sum;
      sum += c;
    }
    for (let i = 0; i < count; i++) {
      const idx = idxA[i];
      const b = (keys[idx] >> 8) & 255;
      idxB[counts[b]++] = idx;
    }

    const out = this.sortedIndices;
    for (let i = 0; i < count; i++) out[i] = idxB[i];
    this.indexAttr.needsUpdate = true;
    return true;
  }

  dispose() {
    this.geometry.dispose();
    this.colorMaterial.dispose();
    this.depthMaterial.dispose();
    this.crispMaterial.dispose();
    this.posTex.dispose();
    this.colTex.dispose();
    this.covATex.dispose();
    this.covBTex.dispose();
  }
}
