import * as THREE from 'three';

/**
 * Deterministic PRNG (Mulberry32) so every sample world builds identically.
 */
function createRng(seed) {
  let a = seed >>> 0;
  return function rand() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function hexToRgb(hex) {
  return [
    ((hex >> 16) & 255) / 255,
    ((hex >> 8) & 255) / 255,
    (hex & 255) / 255,
  ];
}

function mixRgb(a, b, t) {
  return [
    a[0] + (b[0] - a[0]) * t,
    a[1] + (b[1] - a[1]) * t,
    a[2] + (b[2] - a[2]) * t,
  ];
}

const _up = new THREE.Vector3(0, 0, 1);
const _nVec = new THREE.Vector3();
const _quat = new THREE.Quaternion();

/**
 * Builds dense, surface-aligned anisotropic 3D Gaussian splat clouds with
 * directional key + cool sky fill + warm bounce lighting baked per splat.
 */
class SplatBuilder {
  constructor(capacity, rng, lightDir, skyTint, bounceTint) {
    this.cap = capacity;
    this.count = 0;
    this.rng = rng;
    this.lightDir = new THREE.Vector3(...lightDir).normalize();
    this.skyTint = skyTint;
    this.bounceTint = bounceTint;

    this.positions = new Float32Array(capacity * 3);
    this.scales = new Float32Array(capacity * 3);
    this.quaternions = new Float32Array(capacity * 4);
    this.colors = new Float32Array(capacity * 3);
    this.opacities = new Float32Array(capacity);
    this.groupIds = new Float32Array(capacity);
    this.animKinds = new Float32Array(capacity);
    this.animPhases = new Float32Array(capacity);
  }

  shade(baseRgb, nx, ny, nz, ao = 1.0, emissive = 0.0) {
    const ndl = clamp01(nx * this.lightDir.x + ny * this.lightDir.y + nz * this.lightDir.z);
    const sky = clamp01(ny * 0.5 + 0.5);
    const bounce = clamp01(-ny * 0.5 + 0.5);
    const jitter = (this.rng() - 0.5) * 0.065;

    const r =
      baseRgb[0] * (0.28 + 0.85 * ndl + 0.22 * sky * this.skyTint[0] + 0.14 * bounce * this.bounceTint[0]) * ao +
      baseRgb[0] * emissive +
      jitter;
    const g =
      baseRgb[1] * (0.28 + 0.85 * ndl + 0.22 * sky * this.skyTint[1] + 0.14 * bounce * this.bounceTint[1]) * ao +
      baseRgb[1] * emissive +
      jitter * 0.9;
    const b =
      baseRgb[2] * (0.28 + 0.85 * ndl + 0.22 * sky * this.skyTint[2] + 0.14 * bounce * this.bounceTint[2]) * ao +
      baseRgb[2] * emissive +
      jitter * 0.8;

    return [clamp01(r), clamp01(g), clamp01(b)];
  }

  push(x, y, z, sx, sy, sz, nx, ny, nz, rgb, opacity, groupId, animKind = 0) {
    if (this.count >= this.cap) return;
    const i = this.count++;
    const i3 = i * 3;
    const i4 = i * 4;

    this.positions[i3] = x;
    this.positions[i3 + 1] = y;
    this.positions[i3 + 2] = z;

    this.scales[i3] = sx;
    this.scales[i3 + 1] = sy;
    this.scales[i3 + 2] = sz;

    // Align local +Z (thin axis sz) along surface normal so splats lie flat on surfaces!
    _nVec.set(nx, ny, nz);
    if (_nVec.lengthSq() < 1e-5) {
      _quat.identity();
    } else {
      _nVec.normalize();
      _quat.setFromUnitVectors(_up, _nVec);
    }
    this.quaternions[i4] = _quat.x;
    this.quaternions[i4 + 1] = _quat.y;
    this.quaternions[i4 + 2] = _quat.z;
    this.quaternions[i4 + 3] = _quat.w;

    this.colors[i3] = rgb[0];
    this.colors[i3 + 1] = rgb[1];
    this.colors[i3 + 2] = rgb[2];
    this.opacities[i] = opacity;
    this.groupIds[i] = groupId;
    this.animKinds[i] = animKind;
    this.animPhases[i] = this.rng() * Math.PI * 2;
  }

  /** Surface-aligned anisotropic splats on a 3D box */
  addBox(cx, cy, cz, w, h, d, baseRgb, opts = {}) {
    const density = opts.density ?? 18;
    const groupId = opts.groupId ?? 0.5;
    const rotY = opts.rotY ?? 0;
    const splatR = opts.splatR ?? 0.14;
    const emissive = opts.emissive ?? 0;
    const cs = Math.cos(rotY);
    const sn = Math.sin(rotY);

    const faces = [
      { n: [0, 1, 0], u: [w, 0, 0], v: [0, 0, d], c: [0, h * 0.5, 0], area: w * d },
      { n: [0, -1, 0], u: [w, 0, 0], v: [0, 0, d], c: [0, -h * 0.5, 0], area: w * d * 0.4 },
      { n: [1, 0, 0], u: [0, h, 0], v: [0, 0, d], c: [w * 0.5, 0, 0], area: h * d },
      { n: [-1, 0, 0], u: [0, h, 0], v: [0, 0, d], c: [-w * 0.5, 0, 0], area: h * d },
      { n: [0, 0, 1], u: [w, 0, 0], v: [0, h, 0], c: [0, 0, d * 0.5], area: w * h },
      { n: [0, 0, -1], u: [w, 0, 0], v: [0, h, 0], c: [0, 0, -d * 0.5], area: w * h },
    ];

    for (const f of faces) {
      const nCount = Math.max(8, Math.round(f.area * density));
      const rx = f.n[0] * cs + f.n[2] * sn;
      const ry = f.n[1];
      const rz = -f.n[0] * sn + f.n[2] * cs;

      for (let i = 0; i < nCount; i++) {
        const uu = this.rng() - 0.5;
        const vv = this.rng() - 0.5;
        const lx = f.c[0] + f.u[0] * uu + f.v[0] * vv;
        const ly = f.c[1] + f.u[1] * uu + f.v[1] * vv;
        const lz = f.c[2] + f.u[2] * uu + f.v[2] * vv;

        const wx = cx + lx * cs + lz * sn;
        const wy = cy + ly;
        const wz = cz - lx * sn + lz * cs;

        const ao = clamp01(0.74 + 0.26 * ((ly + h * 0.5) / Math.max(h, 0.2)));
        const col = this.shade(baseRgb, rx, ry, rz, ao, emissive);
        const r = splatR * (0.82 + this.rng() * 0.45);
        this.push(wx, wy, wz, r * 1.15, r * 0.95, r * 0.28, rx, ry, rz, col, 0.94, groupId, opts.animKind ?? 0);
      }
    }
  }

  /** Cylinder / tapered column or lighthouse tower */
  addCylinder(cx, baseY, cz, rBottom, rTop, height, baseRgb, opts = {}) {
    const density = opts.density ?? 20;
    const groupId = opts.groupId ?? 0.55;
    const splatR = opts.splatR ?? 0.13;
    const emissive = opts.emissive ?? 0;
    const avgR = (rBottom + rTop) * 0.5;
    const area = 2 * Math.PI * avgR * height;
    const count = Math.max(24, Math.round(area * density));
    const slope = (rBottom - rTop) / Math.max(height, 0.01);

    for (let i = 0; i < count; i++) {
      const t = this.rng();
      const ang = this.rng() * Math.PI * 2;
      const rad = rBottom + (rTop - rBottom) * t;
      const wx = cx + Math.cos(ang) * rad;
      const wy = baseY + t * height;
      const wz = cz + Math.sin(ang) * rad;

      const nx = Math.cos(ang);
      const ny = slope;
      const nz = Math.sin(ang);
      const ao = 0.78 + 0.22 * t;
      const col = this.shade(baseRgb, nx, ny, nz, ao, emissive);
      const s = splatR * (0.85 + this.rng() * 0.4);
      this.push(wx, wy, wz, s * 1.05, s * 1.15, s * 0.28, nx, ny, nz, col, 0.95, groupId, opts.animKind ?? 0);
    }

    if (opts.capTop) {
      const capCount = Math.max(12, Math.round(Math.PI * rTop * rTop * density * 1.1));
      for (let i = 0; i < capCount; i++) {
        const rr = Math.sqrt(this.rng()) * rTop;
        const ang = this.rng() * Math.PI * 2;
        const col = this.shade(baseRgb, 0, 1, 0, 1.0, emissive);
        this.push(
          cx + Math.cos(ang) * rr,
          baseY + height,
          cz + Math.sin(ang) * rr,
          splatR,
          splatR,
          splatR * 0.25,
          0, 1, 0,
          col,
          0.95,
          groupId
        );
      }
    }
  }

  /** Organic blob / foliage / boulder / lantern globe */
  addBlob(cx, cy, cz, rx, ry, rz, count, baseRgb, opts = {}) {
    const groupId = opts.groupId ?? 0.6;
    const splatR = opts.splatR ?? 0.16;
    const emissive = opts.emissive ?? 0;
    const animKind = opts.animKind ?? 0;
    const opacity = opts.opacity ?? 0.93;
    const shell = opts.shell ?? 0.82;

    for (let i = 0; i < count; i++) {
      // Uniform point on unit sphere
      const u = this.rng() * 2 - 1;
      const phi = this.rng() * Math.PI * 2;
      const root = Math.sqrt(1 - u * u);
      const dx = root * Math.cos(phi);
      const dy = u;
      const dz = root * Math.sin(phi);
      const rad = shell + (1 - shell) * Math.cbrt(this.rng());

      const wx = cx + dx * rx * rad;
      const wy = cy + dy * ry * rad;
      const wz = cz + dz * rz * rad;

      const ao = clamp01(0.55 + 0.45 * (dy * 0.5 + 0.5));
      const col = this.shade(baseRgb, dx, dy, dz, ao, emissive);
      const s = splatR * (0.78 + this.rng() * 0.5);
      this.push(wx, wy, wz, s * 1.25, s * 0.9, s * 0.42, dx, dy, dz, col, opacity, groupId, animKind);
    }
  }

  /** Curved arch (monolith portal or arched bridge) */
  addArch(cx, cy, cz, radius, tubeW, tubeD, startAng, endAng, count, baseRgb, opts = {}) {
    const groupId = opts.groupId ?? 0.72;
    const splatR = opts.splatR ?? 0.15;
    const rotY = opts.rotY ?? 0;
    const cs = Math.cos(rotY);
    const sn = Math.sin(rotY);

    for (let i = 0; i < count; i++) {
      const ang = startAng + this.rng() * (endAng - startAng);
      const rOff = (this.rng() - 0.5) * tubeW;
      const dOff = (this.rng() - 0.5) * tubeD;
      const r = radius + rOff;

      const lx = Math.cos(ang) * r;
      const ly = Math.sin(ang) * r;
      const lz = dOff;

      const wx = cx + lx * cs + lz * sn;
      const wy = cy + ly;
      const wz = cz - lx * sn + lz * cs;

      const nLx = Math.abs(rOff) > Math.abs(dOff) * 0.6 ? Math.sign(rOff) * Math.cos(ang) : 0;
      const nLy = Math.abs(rOff) > Math.abs(dOff) * 0.6 ? Math.sign(rOff) * Math.sin(ang) : 0.35;
      const nLz = Math.abs(rOff) > Math.abs(dOff) * 0.6 ? 0 : Math.sign(dOff || 1);

      const rx = nLx * cs + nLz * sn;
      const rz = -nLx * sn + nLz * cs;
      const col = this.shade(baseRgb, rx, nLy, rz, 0.92, opts.emissive ?? 0);
      const s = splatR * (0.85 + this.rng() * 0.4);
      this.push(wx, wy, wz, s * 1.1, s * 1.0, s * 0.35, rx, nLy, rz, col, 0.95, groupId, opts.animKind ?? 0);
    }
  }

  /** Terrain heightfield with analytical normals */
  addTerrain(xMin, xMax, zMin, zMax, count, heightFn, colorFn, opts = {}) {
    const groupId = opts.groupId ?? 0.2;
    const splatR = opts.splatR ?? 0.26;
    const eps = 0.12;

    for (let i = 0; i < count; i++) {
      const x = xMin + this.rng() * (xMax - xMin);
      const z = zMin + this.rng() * (zMax - zMin);
      const y = heightFn(x, z);
      if (y === null) continue;

      const hx = heightFn(x + eps, z) ?? y;
      const hz = heightFn(x, z + eps) ?? y;
      const nx = -(hx - y) / eps;
      const ny = 1.0;
      const nz = -(hz - y) / eps;
      const nLen = Math.hypot(nx, ny, nz) || 1;

      const [baseRgb, extraAo = 1.0, animKind = 0, op = 0.94] = colorFn(x, y, z, ny / nLen);
      const col = this.shade(baseRgb, nx / nLen, ny / nLen, nz / nLen, extraAo);
      const s = splatR * (0.8 + this.rng() * 0.55);
      this.push(x, y, z, s * 1.3, s * 1.15, s * 0.24, nx / nLen, ny / nLen, nz / nLen, col, op, groupId, animKind);
    }
  }

  /** Realistic 3DGS reconstruction "floaters" near surfaces — toggleable in UI to show how NPR tames artifacts! */
  addFloaters(centers, count, palette, radius = 3.2) {
    for (let i = 0; i < count; i++) {
      const c = centers[Math.floor(this.rng() * centers.length)];
      const ang = this.rng() * Math.PI * 2;
      const dist = (0.35 + this.rng() * 0.85) * radius;
      const wx = c[0] + Math.cos(ang) * dist;
      const wy = c[1] + (this.rng() - 0.3) * radius * 0.85;
      const wz = c[2] + Math.sin(ang) * dist;
      const base = palette[Math.floor(this.rng() * palette.length)];
      // Deliberately smeared, semi-transparent elongated blobs like real raw 3DGS floaters
      const sx = 0.22 + this.rng() * 0.55;
      const sy = 0.08 + this.rng() * 0.2;
      const sz = 0.08 + this.rng() * 0.18;
      this.push(
        wx, wy, wz,
        sx, sy, sz,
        this.rng() - 0.5, this.rng() - 0.5, this.rng() - 0.5,
        base,
        0.22 + this.rng() * 0.24,
        0.02,
        1.0 // animKind 1 = floater controlled by uFloaterStrength
      );
    }
  }

  build() {
    return {
      count: this.count,
      positions: this.positions.subarray(0, this.count * 3),
      scales: this.scales.subarray(0, this.count * 3),
      quaternions: this.quaternions.subarray(0, this.count * 4),
      colors: this.colors.subarray(0, this.count * 3),
      opacities: this.opacities.subarray(0, this.count),
      groupIds: this.groupIds.subarray(0, this.count),
      animKinds: this.animKinds.subarray(0, this.count),
      animPhases: this.animPhases.subarray(0, this.count),
    };
  }
}

/* -------------------------------------------------------------------------- */
/*  WORLD 1: MOEBIUS DESERT MONOLITH & FLOATING ARCH                          */
/* -------------------------------------------------------------------------- */
function buildMoebiusWorld() {
  const rng = createRng(197601);
  const b = new SplatBuilder(56000, rng, [0.68, 0.62, 0.38], [0.76, 0.88, 1.0], [1.0, 0.68, 0.45]);

  const sandLight = hexToRgb(0xe8c493);
  const sandWarm = hexToRgb(0xd99266);
  const mesaOchre = hexToRgb(0xc96b4b);
  const stoneCream = hexToRgb(0xe8dcc4);
  const tealGlyph = hexToRgb(0x2ca6a4);
  const obsidian = hexToRgb(0x2b2d42);
  const bannerCoral = hexToRgb(0xe25845);
  const sunGold = hexToRgb(0xffd27d);
  const skyLavender = hexToRgb(0xc4b0d9);

  // Rolling desert dunes
  const duneH = (x, z) => {
    const r = Math.hypot(x, z);
    const plateau = r < 5.2 ? Math.max(0, 1 - r / 5.2) * 0.45 : 0;
    return (
      Math.sin(x * 0.22 + z * 0.14) * 0.95 +
      Math.cos(z * 0.28 - x * 0.1) * 0.65 +
      Math.sin(x * 0.7 + z * 0.5) * 0.18 +
      plateau -
      0.6
    );
  };

  b.addTerrain(
    -18, 18, -18, 18, 15500,
    duneH,
    (x, y, z) => {
      const ripple = 0.5 + 0.5 * Math.sin(x * 1.6 + z * 0.7);
      const col = mixRgb(sandWarm, sandLight, clamp01(0.35 + y * 0.25 + ripple * 0.25));
      return [col, 0.96, 0, 0.95];
    },
    { groupId: 0.16, splatR: 0.32 }
  );

  // Stepped sandstone altar dais
  b.addBox(0, 0.1, 0, 7.6, 0.65, 7.6, stoneCream, { density: 28, groupId: 0.32, splatR: 0.16 });
  b.addBox(0, 0.62, 0, 5.4, 0.5, 5.4, sandLight, { density: 30, groupId: 0.36, splatR: 0.15 });
  b.addBox(0, 1.05, 0, 3.6, 0.42, 3.6, stoneCream, { density: 32, groupId: 0.4, splatR: 0.14 });

  // Twin towering monolith pylons + monumental portal arch
  b.addBox(-2.45, 4.1, 0, 1.35, 6.4, 1.5, mesaOchre, { density: 32, groupId: 0.68, splatR: 0.15 });
  b.addBox(2.45, 4.1, 0, 1.35, 6.4, 1.5, mesaOchre, { density: 32, groupId: 0.68, splatR: 0.15 });
  // Glowing teal carved inlay stripes on the pylons
  b.addBox(-2.45, 4.2, 0.76, 0.32, 4.6, 0.12, tealGlyph, { density: 42, groupId: 0.85, splatR: 0.1, emissive: 0.45 });
  b.addBox(2.45, 4.2, 0.76, 0.32, 4.6, 0.12, tealGlyph, { density: 42, groupId: 0.85, splatR: 0.1, emissive: 0.45 });

  // Great stone arch spanning the monolith
  b.addArch(0, 6.6, 0, 2.55, 1.2, 1.45, 0, Math.PI, 3200, stoneCream, { groupId: 0.72, splatR: 0.16 });
  // Inner teal energy portal halo
  b.addArch(0, 6.5, 0, 1.85, 0.25, 0.25, 0, Math.PI * 2, 1100, tealGlyph, { groupId: 0.92, splatR: 0.11, emissive: 0.72 });

  // Levitating obsidian octahedron/crystal & orbiting golden ring inside portal
  b.addBlob(0, 4.3, 0, 0.72, 1.25, 0.72, 950, obsidian, { groupId: 0.88, splatR: 0.12, animKind: 5.0 });
  for (let i = 0; i < 900; i++) {
    const ang = (i / 900) * Math.PI * 2;
    const rx = Math.cos(ang) * 1.7;
    const rz = Math.sin(ang) * 1.7;
    const ry = 4.3 + Math.sin(ang * 2) * 0.32;
    const col = b.shade(sunGold, Math.cos(ang), 0.5, Math.sin(ang), 1.0, 0.4);
    b.push(rx, ry, rz, 0.1, 0.1, 0.08, Math.cos(ang), 0.4, Math.sin(ang), col, 0.96, 0.94, 4.0);
  }

  // Ruined colonnade pillars on left & right + fluttering Moebius coral banners
  const pillars = [
    [-5.8, -2.5, 4.6], [-7.4, 1.8, 3.3], [-4.8, 4.4, 2.6],
    [5.8, -2.2, 5.1], [7.2, 2.4, 3.8], [5.0, 4.8, 2.2],
  ];
  for (const [px, pz, ph] of pillars) {
    const baseY = duneH(px, pz) - 0.3;
    b.addCylinder(px, baseY, pz, 0.48, 0.42, ph, stoneCream, { density: 28, groupId: 0.54, splatR: 0.13, capTop: true });
  }

  // Fluttering banners hanging from the right/left tall pillars
  const bannerAnchors = [[-5.8, duneH(-5.8, -2.5) + 4.1, -2.5], [5.8, duneH(5.8, -2.2) + 4.6, -2.2]];
  for (const [bx, by, bz] of bannerAnchors) {
    for (let i = 0; i < 650; i++) {
      const u = rng();
      const v = rng();
      const wx = bx + 0.5 + u * 2.2;
      const wy = by - v * 2.4 - u * 0.35;
      const wz = bz + Math.sin(u * 4.5) * 0.25;
      const col = b.shade(bannerCoral, 0, 0.2, 1, 0.95);
      b.push(wx, wy, wz, 0.12, 0.12, 0.04, 0, 0.2, 1, col, 0.95, 0.82, 2.0);
    }
  }

  // Distant stepped desert mesas & twin celestial Moebius moons
  b.addBox(-11.5, 2.6, -12.5, 6.2, 5.8, 4.8, mesaOchre, { density: 14, groupId: 0.25, splatR: 0.26, rotY: 0.35 });
  b.addBox(12.0, 3.2, -13.0, 7.2, 6.8, 5.2, mixRgb(mesaOchre, skyLavender, 0.28), { density: 14, groupId: 0.25, splatR: 0.28, rotY: -0.4 });
  b.addBlob(-6.5, 11.2, -16.5, 2.1, 2.1, 1.4, 900, sunGold, { groupId: 0.98, splatR: 0.24, emissive: 0.65 });
  b.addBlob(4.5, 12.8, -17.5, 1.15, 1.15, 0.9, 450, skyLavender, { groupId: 0.98, splatR: 0.2, emissive: 0.45 });

  // Drifting golden desert dust motes
  for (let i = 0; i < 600; i++) {
    b.push(
      (rng() - 0.5) * 16,
      0.8 + rng() * 8,
      (rng() - 0.5) * 16,
      0.06, 0.06, 0.06,
      0, 1, 0,
      sunGold,
      0.65,
      0.04,
      2.0
    );
  }

  // Realistic raw 3DGS floaters around thin silhouettes (tamed by NPR shaders!)
  b.addFloaters(
    [[0, 4.8, 0], [-5.8, 3.2, -2.5], [5.8, 3.5, -2.2], [0, 7.2, 0]],
    1400,
    [sandLight, tealGlyph, bannerCoral, mesaOchre],
    3.6
  );

  return {
    id: 'moebius-desert',
    title: 'Moebius Monolith Ruin',
    shortTitle: 'Moebius Ruin',
    subtitle: 'Ligne claire sandstone arch, orbiting crystal & desert wind',
    defaultShader: 'ligne-claire',
    skyTop: [0.49, 0.67, 0.78],
    skyBottom: [0.95, 0.83, 0.68],
    camera: { pos: [0.8, 3.6, 12.2], target: [0, 3.4, 0] },
    emitters: [
      {
        id: 'portal-hum',
        label: 'Portal Resonance',
        position: [0, 4.3, 0],
        color: '#2ca6a4',
        synthPreset: 'portalHum',
      },
      {
        id: 'silk-wind',
        label: 'Banner Wind',
        position: [6.2, 4.0, -2.2],
        color: '#e25845',
        synthPreset: 'desertWind',
      },
      {
        id: 'crystal-chime',
        label: 'Orbiting Ring Chime',
        position: [-4.6, 3.6, 1.8],
        color: '#f5b841',
        synthPreset: 'crystalChimes',
      },
    ],
    splats: b.build(),
  };
}

/* -------------------------------------------------------------------------- */
/*  WORLD 2: UKIYO-E MISTY SHRINE & TORII WATERFALL                           */
/* -------------------------------------------------------------------------- */
function buildUkiyoeWorld() {
  const rng = createRng(183102);
  const b = new SplatBuilder(56000, rng, [-0.55, 0.72, 0.42], [0.75, 0.86, 0.96], [0.95, 0.72, 0.56]);

  const indigoWater = hexToRgb(0x1f4968);
  const foamCream = hexToRgb(0xf2ead7);
  const mossRock = hexToRgb(0x4a5d4e);
  const slateRock = hexToRgb(0x3a404a);
  const vermilion = hexToRgb(0xd43d2a);
  const sumiTimber = hexToRgb(0x232024);
  const pineDark = hexToRgb(0x214a3b);
  const pineSage = hexToRgb(0x4c7c63);
  const sakuraPink = hexToRgb(0xf4a8b8);
  const lanternGlow = hexToRgb(0xffcb69);

  // Rocky stream basin & misty pond
  const basinH = (x, z) => {
    const streamDist = Math.abs(x + Math.sin(z * 0.25) * 1.1);
    const cliffBack = z < -5.0 ? Math.pow((-5.0 - z) * 0.42, 1.5) * 2.1 : 0;
    const banks = streamDist > 2.7 ? Math.pow(streamDist - 2.7, 1.25) * 0.58 : -0.45;
    return banks + cliffBack + Math.sin(x * 0.6 + z * 0.5) * 0.22;
  };

  // Water surface
  b.addTerrain(
    -15, 15, -12, 14, 7200,
    (x, z) => (basinH(x, z) < 0.05 ? 0.0 : null),
    (x, _y, z) => {
      const foam = z < -3.2 && Math.abs(x) < 2.6 ? clamp01(1 - Math.abs(z + 4.5) * 0.45) : 0;
      const wave = 0.5 + 0.5 * Math.sin(z * 2.2 + Math.cos(x * 1.4));
      const col = mixRgb(indigoWater, foamCream, clamp01(foam * 0.75 + wave * 0.16));
      return [col, 0.95, 5.0, 0.92];
    },
    { groupId: 0.12, splatR: 0.26 }
  );

  // Mossy rocky crags & mountain cliff wall
  b.addTerrain(
    -16, 16, -13, 14, 13000,
    (x, z) => {
      const h = basinH(x, z);
      return h >= -0.05 ? h : null;
    },
    (_x, y, _z, upDot) => {
      const col = mixRgb(slateRock, mossRock, clamp01(upDot * 1.15 - 0.15 + y * 0.04));
      return [col, 0.92, 0, 0.95];
    },
    { groupId: 0.24, splatR: 0.29 }
  );

  // Cascading waterfall curtain down the back cliff
  for (let i = 0; i < 3600; i++) {
    const u = (rng() - 0.5) * 3.1;
    const v = rng();
    const wy = 0.1 + v * 6.8;
    const wz = -5.6 - v * 1.1 + Math.sin(u * 2.2) * 0.25;
    const col = mixRgb(indigoWater, foamCream, 0.45 + rng() * 0.5);
    b.push(u, wy, wz, 0.15, 0.34, 0.12, 0, 0.15, 1, col, 0.85, 0.44, 3.0);
  }

  // Iconic Vermilion Torii Gate standing in the shallow pool / stone landing
  const toriiZ = -1.6;
  b.addCylinder(-2.2, -0.3, toriiZ, 0.34, 0.28, 5.2, vermilion, { density: 34, groupId: 0.76, splatR: 0.13 });
  b.addCylinder(2.2, -0.3, toriiZ, 0.34, 0.28, 5.2, vermilion, { density: 34, groupId: 0.76, splatR: 0.13 });
  // Stone bases
  b.addCylinder(-2.2, -0.4, toriiZ, 0.48, 0.42, 0.75, sumiTimber, { density: 30, groupId: 0.72, splatR: 0.13 });
  b.addCylinder(2.2, -0.4, toriiZ, 0.48, 0.42, 0.75, sumiTimber, { density: 30, groupId: 0.72, splatR: 0.13 });
  // Lower tie beam (nuki) & upper curved kasagi lintel
  b.addBox(0, 3.9, toriiZ, 5.8, 0.36, 0.36, vermilion, { density: 34, groupId: 0.78, splatR: 0.12 });
  b.addBox(0, 4.95, toriiZ, 6.7, 0.42, 0.52, vermilion, { density: 36, groupId: 0.8, splatR: 0.13 });
  b.addBox(0, 5.32, toriiZ, 7.1, 0.28, 0.62, sumiTimber, { density: 36, groupId: 0.82, splatR: 0.12 });
  b.addBox(0, 4.45, toriiZ, 0.52, 0.85, 0.22, sumiTimber, { density: 40, groupId: 0.84, splatR: 0.1 });

  // Arched vermilion wooden footbridge in foreground
  b.addArch(0, -1.6, 3.2, 3.2, 0.35, 1.5, Math.PI * 0.24, Math.PI * 0.76, 2200, vermilion, { groupId: 0.65, splatR: 0.14 });
  b.addArch(0, -0.85, 2.52, 3.25, 0.16, 0.16, Math.PI * 0.24, Math.PI * 0.76, 650, sumiTimber, { groupId: 0.68, splatR: 0.1 });
  b.addArch(0, -0.85, 3.88, 3.25, 0.16, 0.16, Math.PI * 0.24, Math.PI * 0.76, 650, sumiTimber, { groupId: 0.68, splatR: 0.1 });

  // Glowing stone lanterns (Toro) on left & right banks
  const lanterns = [[-4.1, 1.9], [4.0, 1.5], [-3.6, -3.5]];
  for (const [lx, lz] of lanterns) {
    const ly = Math.max(0.2, basinH(lx, lz));
    b.addCylinder(lx, ly, lz, 0.22, 0.18, 1.45, slateRock, { density: 28, groupId: 0.58, splatR: 0.1 });
    b.addBlob(lx, ly + 1.72, lz, 0.32, 0.34, 0.32, 340, lanternGlow, { groupId: 0.94, splatR: 0.11, emissive: 0.85 });
    b.addBox(lx, ly + 2.15, lz, 0.92, 0.26, 0.92, sumiTimber, { density: 28, groupId: 0.62, splatR: 0.11 });
  }

  // Sculptural Japanese pine & sakura trees framing the gorge
  const trees = [
    { x: -6.2, z: -0.8, h: 4.5, colA: pineDark, colB: pineSage },
    { x: 6.0, z: -1.4, h: 4.9, colA: sakuraPink, colB: mixRgb(sakuraPink, foamCream, 0.45) },
    { x: -5.4, z: -5.5, h: 4.2, colA: pineDark, colB: pineSage },
    { x: 5.5, z: -5.2, h: 4.6, colA: pineDark, colB: pineSage },
  ];
  for (const t of trees) {
    const baseY = Math.max(0.3, basinH(t.x, t.z));
    b.addCylinder(t.x, baseY, t.z, 0.36, 0.2, t.h, sumiTimber, { density: 24, groupId: 0.5, splatR: 0.13 });
    const pads = [
      [0, t.h + 0.3, 0, 2.2, 0.8, 1.9],
      [-1.3, t.h - 0.5, 0.6, 1.6, 0.65, 1.5],
      [1.2, t.h - 0.3, -0.5, 1.7, 0.68, 1.5],
    ];
    for (const [ox, oy, oz, rx, ry, rz] of pads) {
      b.addBlob(t.x + ox, baseY + oy, t.z + oz, rx, ry, rz, 750, t.colA, { groupId: 0.86, splatR: 0.19, animKind: 2.0 });
      b.addBlob(t.x + ox, baseY + oy + 0.35, t.z + oz, rx * 0.82, ry * 0.6, rz * 0.82, 420, t.colB, { groupId: 0.88, splatR: 0.17, animKind: 2.0 });
    }
  }

  // Distant tiered pagoda silhouette on the upper ridge
  b.addBox(4.8, 7.6, -10.5, 2.6, 2.2, 2.6, slateRock, { density: 16, groupId: 0.35, splatR: 0.2 });
  b.addBox(4.8, 8.8, -10.5, 3.6, 0.42, 3.6, sumiTimber, { density: 18, groupId: 0.38, splatR: 0.2 });
  b.addBox(4.8, 9.8, -10.5, 1.9, 1.5, 1.9, vermilion, { density: 16, groupId: 0.4, splatR: 0.19 });
  b.addBox(4.8, 10.7, -10.5, 2.8, 0.38, 2.8, sumiTimber, { density: 18, groupId: 0.42, splatR: 0.19 });

  // Drifting sakura petals
  for (let i = 0; i < 650; i++) {
    b.push(
      (rng() - 0.5) * 14,
      1.2 + rng() * 7.5,
      (rng() - 0.5) * 12,
      0.07, 0.05, 0.04,
      0.2, 0.9, 0.2,
      sakuraPink,
      0.86,
      0.9,
      3.0
    );
  }

  // Reconstruction floaters (mist wisps & edge artifacts)
  b.addFloaters(
    [[0, 2.5, -4.8], [-2.2, 3.5, -1.6], [2.2, 3.5, -1.6], [0, 1.5, 3.2]],
    1300,
    [foamCream, vermilion, sakuraPink, indigoWater],
    3.2
  );

  return {
    id: 'ukiyoe-shrine',
    title: 'Ukiyo-e Misty Torii Gorge',
    shortTitle: 'Ukiyo-e Shrine',
    subtitle: 'Vermilion shrine gate, waterfall cascade & drifting sakura',
    defaultShader: 'ukiyo-e',
    skyTop: [0.32, 0.48, 0.62],
    skyBottom: [0.95, 0.87, 0.74],
    camera: { pos: [1.4, 3.0, 11.4], target: [0, 2.6, -1.8] },
    emitters: [
      {
        id: 'waterfall',
        label: 'Waterfall Cascade',
        position: [0, 3.2, -5.8],
        color: '#4ea8de',
        synthPreset: 'waterfall',
      },
      {
        id: 'temple-bell',
        label: 'Shrine Bell & Chime',
        position: [0, 4.6, -1.6],
        color: '#d43d2a',
        synthPreset: 'templeBell',
      },
      {
        id: 'lantern-breeze',
        label: 'Pine Breeze & Crickets',
        position: [4.2, 2.0, 1.5],
        color: '#ffcb69',
        synthPreset: 'nightCrickets',
      },
    ],
    splats: b.build(),
  };
}

/* -------------------------------------------------------------------------- */
/*  WORLD 3: OBRA DINN STORM LANTERN LIGHTHOUSE & CLIFF PIER                  */
/* -------------------------------------------------------------------------- */
function buildLighthouseWorld() {
  const rng = createRng(180703);
  const b = new SplatBuilder(56000, rng, [0.55, 0.65, 0.52], [0.64, 0.72, 0.82], [0.85, 0.78, 0.65]);

  const seaDark = hexToRgb(0x162432);
  const seaMid = hexToRgb(0x284354);
  const seaFoam = hexToRgb(0xe8ece8);
  const cliffRock = hexToRgb(0x3b4046);
  const cliffLight = hexToRgb(0x6e7378);
  const towerWhite = hexToRgb(0xe5dfd1);
  const towerBand = hexToRgb(0x272a30);
  const timber = hexToRgb(0x5a4738);
  const lanternAmber = hexToRgb(0xffe184);
  const hullCream = hexToRgb(0xc7bca8);

  // Choppy North Atlantic storm swell
  b.addTerrain(
    -18, 18, -16, 16, 11500,
    (x, z) => {
      const wave =
        Math.sin(x * 0.45 + z * 0.3) * 0.42 +
        Math.cos(z * 0.65 - x * 0.2) * 0.28 +
        Math.sin(x * 1.3 + z * 1.1) * 0.1;
      return wave - 0.4;
    },
    (x, y, z) => {
      const distRock = Math.hypot(x + 2.2, z + 2.5);
      const crest = clamp01((y + 0.15) * 1.45);
      const surfRing = distRock > 3.2 && distRock < 5.8 ? 0.55 : 0;
      const col = mixRgb(mixRgb(seaDark, seaMid, crest), seaFoam, clamp01(crest * 0.48 + surfRing * 0.65));
      return [col, 0.95, 5.0, 0.94];
    },
    { groupId: 0.14, splatR: 0.3 }
  );

  // Craggy basalt sea stack supporting the lighthouse (centered at origin so beam pivots cleanly)
  b.addBlob(0.0, 0.5, 0.0, 4.3, 2.3, 4.1, 4600, cliffRock, { groupId: 0.28, splatR: 0.26, shell: 0.72 });
  b.addBlob(1.4, 0.2, 1.9, 2.4, 1.5, 2.2, 1600, cliffLight, { groupId: 0.3, splatR: 0.22, shell: 0.75 });
  b.addBlob(-2.5, 0.0, 1.6, 2.1, 1.4, 2.3, 1400, cliffRock, { groupId: 0.28, splatR: 0.22, shell: 0.74 });

  // Striped stone Lighthouse Tower (high contrast stripes read AMAZINGLY in 1-bit dither!)
  const lx = 0.0;
  const lz = 0.0;
  b.addCylinder(lx, 2.0, lz, 1.65, 1.45, 2.4, towerWhite, { density: 34, groupId: 0.72, splatR: 0.14 });
  b.addCylinder(lx, 4.4, lz, 1.45, 1.28, 1.7, towerBand, { density: 34, groupId: 0.75, splatR: 0.14 });
  b.addCylinder(lx, 6.1, lz, 1.28, 1.12, 2.1, towerWhite, { density: 34, groupId: 0.72, splatR: 0.14 });
  b.addCylinder(lx, 8.2, lz, 1.12, 1.05, 1.1, towerBand, { density: 34, groupId: 0.75, splatR: 0.13 });

  // Gallery deck, bright glowing lantern room, cupola dome
  b.addCylinder(lx, 9.3, lz, 1.55, 1.55, 0.32, towerBand, { density: 36, groupId: 0.82, splatR: 0.12, capTop: true });
  b.addCylinder(lx, 9.62, lz, 0.92, 0.92, 1.45, lanternAmber, { density: 44, groupId: 0.96, splatR: 0.12, emissive: 0.95 });
  b.addBlob(lx, 11.15, lz, 1.12, 0.78, 1.12, 700, towerBand, { groupId: 0.84, splatR: 0.12 });

  // Rotating volumetric-style lighthouse light beam cones!
  for (let i = 0; i < 1600; i++) {
    const side = i % 2 === 0 ? 1 : -1;
    const t = 0.08 + rng() * 0.92;
    const dist = 1.1 + t * 11.5;
    const spread = 0.25 + t * 1.85;
    const a = rng() * Math.PI * 2;
    const r = Math.sqrt(rng()) * spread;
    // Offset relative to world origin for Y-axis orbit (we place tower near origin or rotate local)
    const wx = side * dist;
    const wy = 10.35 + Math.sin(a) * r * 0.65 - t * 0.8;
    const wz = Math.cos(a) * r;
    b.push(wx, wy, wz, 0.36 + t * 0.45, 0.24 + t * 0.3, 0.22, 0, 1, 0, lanternAmber, 0.18 * (1.05 - t * 0.7), 0.98, 4.0);
  }

  // Keeper's cottage attached to the rock shelf
  b.addBox(-2.8, 2.2, -1.2, 2.7, 1.9, 2.3, towerWhite, { density: 28, groupId: 0.64, splatR: 0.14, rotY: 0.22 });
  b.addBox(-2.8, 3.35, -1.2, 3.1, 0.55, 2.7, towerBand, { density: 28, groupId: 0.66, splatR: 0.14, rotY: 0.22 });
  // Glowing cottage window
  b.addBox(-1.55, 2.25, -0.18, 0.55, 0.65, 0.15, lanternAmber, { density: 42, groupId: 0.95, splatR: 0.09, emissive: 0.9, rotY: 0.22 });

  // Weathered timber pier & pilings extending toward viewer
  b.addBox(3.4, 1.15, 4.4, 2.1, 0.28, 7.2, timber, { density: 30, groupId: 0.52, splatR: 0.13, rotY: -0.24 });
  for (let p = -3; p <= 3; p += 1.5) {
    const pxL = 2.45 - p * 0.24;
    const pxR = 4.35 - p * 0.24;
    const pz = 4.4 + p;
    b.addCylinder(pxL, -0.7, pz, 0.16, 0.15, 2.45, towerBand, { density: 28, groupId: 0.56, splatR: 0.1 });
    b.addCylinder(pxR, -0.7, pz, 0.16, 0.15, 2.45, towerBand, { density: 28, groupId: 0.56, splatR: 0.1 });
  }

  // Hanging iron lantern post at the end of the pier
  b.addCylinder(2.1, 1.25, 7.6, 0.1, 0.08, 2.7, towerBand, { density: 30, groupId: 0.62, splatR: 0.09 });
  b.addBox(2.45, 3.85, 7.6, 0.85, 0.1, 0.1, towerBand, { density: 30, groupId: 0.62, splatR: 0.08 });
  b.addBlob(2.8, 3.45, 7.6, 0.22, 0.28, 0.22, 320, lanternAmber, { groupId: 0.96, splatR: 0.09, emissive: 0.92, animKind: 2.0 });

  // Bobbing wooden rowboat moored beside the pier
  b.addBlob(6.2, 0.18, 4.9, 0.95, 0.45, 2.15, 1300, hullCream, { groupId: 0.8, splatR: 0.13, animKind: 5.0 });
  b.addBox(6.2, 0.48, 4.9, 1.25, 0.14, 0.42, timber, { density: 30, groupId: 0.82, splatR: 0.1, animKind: 5.0 });

  // Slanted storm rain streaks (subtle atmospheric streaks, excluded from depth outline pass)
  for (let i = 0; i < 420; i++) {
    b.push(
      (rng() - 0.5) * 18,
      1.0 + rng() * 11.5,
      (rng() - 0.5) * 18,
      0.022, 0.24, 0.022,
      0, 0, 1,
      seaFoam,
      0.34,
      0.04,
      3.0
    );
  }

  // Reconstruction floaters (sea spray & splat edge artifacts)
  b.addFloaters(
    [[0.0, 5.5, 0.0], [3.4, 1.8, 4.8], [6.2, 0.8, 4.9]],
    1100,
    [seaFoam, cliffLight, lanternAmber, seaMid],
    3.4
  );

  return {
    id: 'obra-dinn-lighthouse',
    title: 'Storm Lantern Lighthouse',
    shortTitle: 'Storm Lighthouse',
    subtitle: 'High-contrast striped lighthouse, timber pier & Atlantic swell',
    defaultShader: 'obra-dinn',
    skyTop: [0.11, 0.15, 0.2],
    skyBottom: [0.46, 0.51, 0.54],
    camera: { pos: [9.2, 5.6, 17.8], target: [1.0, 4.8, 1.6] },
    emitters: [
      {
        id: 'foghorn',
        label: 'Lighthouse Foghorn',
        position: [0.0, 9.8, 0.0],
        color: '#ffe184',
        synthPreset: 'foghorn',
      },
      {
        id: 'surf',
        label: 'Crashing Tidebreak',
        position: [1.4, 0.4, 2.6],
        color: '#6ec5e9',
        synthPreset: 'oceanSurf',
      },
      {
        id: 'pier-creak',
        label: 'Pier Timber & Boat',
        position: [5.6, 1.2, 5.1],
        color: '#c9a67a',
        synthPreset: 'timberCreak',
      },
    ],
    splats: b.build(),
  };
}

export const SAMPLE_WORLD_BUILDERS = [
  { id: 'moebius-desert', label: 'Moebius Ruin', tag: 'Ligne Claire', build: buildMoebiusWorld },
  { id: 'ukiyoe-shrine', label: 'Ukiyo-e Shrine', tag: 'Woodblock', build: buildUkiyoeWorld },
  { id: 'obra-dinn-lighthouse', label: 'Storm Lighthouse', tag: '1-Bit Dither', build: buildLighthouseWorld },
];
