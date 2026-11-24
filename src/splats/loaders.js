import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

/**
 * Real 3D loaders for browser import (.ply and .glb/.gltf):
 * 1. Binary & ASCII .ply files:
 *    - 3D Gaussian Splats (`speridlabs/idol`, standard 3DGS: x,y,z, f_dc_0..2, opacity, scale_0..2, rot_0..3)
 *    - Colored point clouds (`microsoft/moge-3`: x,y,z, red,green,blue)
 * 2. .glb / .gltf files:
 *    - Colored point-cloud GLBs (`speridlabs/idol` model_glb)
 *    - Textured or vertex-colored 3D mesh GLBs (`speridlabs/idol/max`, Meshy, Hyper3D, Blender, etc.)
 *      sampled into surface-aligned anisotropic surfel splats!
 * 3. RGB image + float32 log-depth `.npy` lifter
 */

const SH_C0 = 0.28209479177387814;

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Parse a .npy float32 2D array (from `speridlabs/iris-z/small/text-to-rgbd` depth_map).
 */
export function parseNpyFloat32(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== 'NUMPY') {
    throw new Error('Invalid .npy magic header');
  }
  const major = bytes[6];
  const view = new DataView(arrayBuffer);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const headerStr = new TextDecoder().decode(bytes.slice(headerOffset, headerOffset + headerLen));

  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]+)\)/);
  if (!shapeMatch) throw new Error('Could not parse .npy shape');
  const dims = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);

  const height = dims[0];
  const width = dims[1];
  const dataOffset = headerOffset + headerLen;

  let data;
  if (headerStr.includes('<f4') || headerStr.includes('float32')) {
    data = new Float32Array(arrayBuffer, dataOffset, width * height);
  } else if (headerStr.includes('<f8') || headerStr.includes('float64')) {
    const f64 = new Float64Array(arrayBuffer, dataOffset, width * height);
    data = Float32Array.from(f64);
  } else {
    throw new Error(`Unsupported .npy dtype in header: ${headerStr}`);
  }

  return { width, height, data };
}

/**
 * Lift an RGB image + aligned depth buffer (from `speridlabs/iris-z/small/text-to-rgbd`)
 * into a dense anisotropic 3D Gaussian splat cloud!
 */
export async function liftRgbDepthToSplats(imageUrl, depthNpyBuffer, opts = {}) {
  const maxDim = opts.maxDim ?? 240;
  const fovDeg = opts.fovDeg ?? 52;

  const img = await loadImage(imageUrl);
  const npy = depthNpyBuffer ? parseNpyFloat32(depthNpyBuffer) : null;

  const aspect = img.width / img.height;
  const gridW = aspect >= 1 ? maxDim : Math.max(64, Math.round(maxDim * aspect));
  const gridH = aspect >= 1 ? Math.max(64, Math.round(maxDim / aspect)) : maxDim;

  const canvas = document.createElement('canvas');
  canvas.width = gridW;
  canvas.height = gridH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, gridW, gridH);
  const pixels = ctx.getImageData(0, 0, gridW, gridH).data;

  // Compute min/max depth from npy (Iris-Z returns relative log depth; sky ~ 1.0)
  let dMin = Infinity;
  let dMax = -Infinity;
  const sampledDepth = new Float32Array(gridW * gridH);

  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      let d;
      if (npy) {
        const sx = Math.min(npy.width - 1, Math.floor((x / gridW) * npy.width));
        const sy = Math.min(npy.height - 1, Math.floor((y / gridH) * npy.height));
        d = npy.data[sy * npy.width + sx];
      } else {
        const idx = (y * gridW + x) * 4;
        d = 1.0 - (pixels[idx] * 0.299 + pixels[idx + 1] * 0.587 + pixels[idx + 2] * 0.114) / 255;
      }
      sampledDepth[y * gridW + x] = d;
      if (Number.isFinite(d) && d < 0.995) {
        if (d < dMin) dMin = d;
        if (d > dMax) dMax = d;
      }
    }
  }
  if (!Number.isFinite(dMin) || dMax <= dMin) {
    dMin = 0;
    dMax = 1;
  }

  const count = gridW * gridH;
  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const quaternions = new Float32Array(count * 4);
  const colors = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
  const groupIds = new Float32Array(count);
  const animKinds = new Float32Array(count);

  const tanHalf = Math.tan((fovDeg * Math.PI) / 360);

  let outIdx = 0;
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const pIdx = y * gridW + x;
      const rawD = sampledDepth[pIdx];
      const isSky = rawD >= 0.992;
      const normD = isSky ? 1.0 : Math.max(0, Math.min(1, (rawD - dMin) / (dMax - dMin)));

      // Map depth smoothly into walkable world Z (near = 3.2m, far = 26m, sky = 36m)
      const zDist = isSky ? 36.0 : 3.2 + Math.pow(normD, 1.35) * 23.0;
      const nx = ((x + 0.5) / gridW) * 2 - 1;
      const ny = 1 - ((y + 0.5) / gridH) * 2;

      const wx = nx * zDist * tanHalf * aspect;
      const wy = ny * zDist * tanHalf + 1.2;
      const wz = 6.5 - zDist;

      // Pixel footprint at this depth -> crisp overlapping Gaussian splat
      const pixelWorld = (2.0 * zDist * tanHalf) / gridH;
      const sxy = pixelWorld * (isSky ? 1.55 : 1.08);
      const sz = pixelWorld * 0.55;

      const i3 = outIdx * 3;
      const i4 = outIdx * 4;
      positions[i3] = wx;
      positions[i3 + 1] = wy;
      positions[i3 + 2] = wz;

      scales[i3] = sxy;
      scales[i3 + 1] = sxy;
      scales[i3 + 2] = sz;

      quaternions[i4] = 0;
      quaternions[i4 + 1] = 0;
      quaternions[i4 + 2] = 0;
      quaternions[i4 + 3] = 1;

      colors[i3] = pixels[pIdx * 4] / 255;
      colors[i3 + 1] = pixels[pIdx * 4 + 1] / 255;
      colors[i3 + 2] = pixels[pIdx * 4 + 2] / 255;

      opacities[outIdx] = isSky ? 0.92 : 0.96;
      // Quantize depth bands slightly into groupIds so foreground/midground/bg separate nicely
      groupIds[outIdx] = isSky ? 0.05 : Math.min(0.95, 0.15 + Math.floor(normD * 6) * 0.13);
      animKinds[outIdx] = 0;
      outIdx++;
    }
  }

  return {
    count: outIdx,
    positions,
    scales,
    quaternions,
    colors,
    opacities,
    groupIds,
    animKinds,
  };
}

/**
 * Parse a binary_little_endian or ASCII .ply file:
 * - Automatically detects 3DGS splats (`speridlabs/idol`: f_dc_0, scale_0, rot_0, opacity)
 * - Or colored point clouds (`microsoft/moge-3`: x, y, z, red, green, blue)
 */
export function parsePlyToSplats(arrayBuffer, maxSplats = 120000) {
  const bytes = new Uint8Array(arrayBuffer);
  const headerProbe = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 8192)));
  const endMarker = 'end_header';
  const endIdx = headerProbe.indexOf(endMarker);
  if (endIdx < 0) throw new Error('Invalid PLY file: missing end_header');

  let bodyOffset = endIdx + endMarker.length;
  if (bytes[bodyOffset] === 13) bodyOffset++;
  if (bytes[bodyOffset] === 10) bodyOffset++;

  const headerLines = headerProbe.slice(0, endIdx).split(/\r?\n/);
  let format = 'ascii';
  let vertexCount = 0;
  let inVertex = false;
  const props = [];

  const typeSize = {
    char: 1, int8: 1, uchar: 1, uint8: 1,
    short: 2, int16: 2, ushort: 2, uint16: 2,
    int: 4, int32: 4, uint: 4, uint32: 4,
    float: 4, float32: 4, double: 8, float64: 8,
  };

  for (const raw of headerLines) {
    const line = raw.trim();
    if (line.startsWith('format ')) {
      format = line.split(/\s+/)[1];
    } else if (line.startsWith('element ')) {
      const parts = line.split(/\s+/);
      inVertex = parts[1] === 'vertex';
      if (inVertex) vertexCount = parseInt(parts[2], 10);
    } else if (inVertex && line.startsWith('property ')) {
      const parts = line.split(/\s+/);
      if (parts[1] === 'list') continue;
      props.push({ type: parts[1], name: parts[2], size: typeSize[parts[1]] || 4 });
    }
  }

  if (!vertexCount) throw new Error('PLY file contains 0 vertices');

  const propOffset = {};
  const propType = {};
  let stride = 0;
  for (const p of props) {
    propOffset[p.name] = stride;
    propType[p.name] = p.type;
    stride += p.size;
  }

  const isGaussian = 'f_dc_0' in propOffset && 'scale_0' in propOffset;
  const step = Math.max(1, Math.ceil(vertexCount / maxSplats));
  const outCount = Math.ceil(vertexCount / step);

  const positions = new Float32Array(outCount * 3);
  const scales = new Float32Array(outCount * 3);
  const quaternions = new Float32Array(outCount * 4);
  const colors = new Float32Array(outCount * 3);
  const opacities = new Float32Array(outCount);
  const groupIds = new Float32Array(outCount);
  const animKinds = new Float32Array(outCount);

  const readField = (dv, base, name, fallback = 0) => {
    if (!(name in propOffset)) return fallback;
    const off = base + propOffset[name];
    switch (propType[name]) {
      case 'float':
      case 'float32':
        return dv.getFloat32(off, true);
      case 'double':
      case 'float64':
        return dv.getFloat64(off, true);
      case 'uchar':
      case 'uint8':
        return dv.getUint8(off);
      case 'char':
      case 'int8':
        return dv.getInt8(off);
      case 'ushort':
      case 'uint16':
        return dv.getUint16(off, true);
      case 'short':
      case 'int16':
        return dv.getInt16(off, true);
      case 'int':
      case 'int32':
        return dv.getInt32(off, true);
      case 'uint':
      case 'uint32':
        return dv.getUint32(off, true);
      default:
        return dv.getFloat32(off, true);
    }
  };

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  if (format === 'binary_little_endian') {
    const dv = new DataView(arrayBuffer, bodyOffset);
    let o = 0;
    for (let i = 0; i < vertexCount; i += step) {
      const base = i * stride;
      const x = readField(dv, base, 'x');
      const y = readField(dv, base, 'y');
      const z = readField(dv, base, 'z');
      const i3 = o * 3;
      const i4 = o * 4;

      // Flip Y/Z from OpenCV convention commonly used by 3DGS & MoGe
      positions[i3] = x;
      positions[i3 + 1] = -y;
      positions[i3 + 2] = -z;

      if (x < minX) minX = x;
      if (-y < minY) minY = -y;
      if (-z < minZ) minZ = -z;
      if (x > maxX) maxX = x;
      if (-y > maxY) maxY = -y;
      if (-z > maxZ) maxZ = -z;

      if (isGaussian) {
        const s0 = Math.min(0.6, Math.exp(readField(dv, base, 'scale_0', -4)));
        const s1 = Math.min(0.6, Math.exp(readField(dv, base, 'scale_1', -4)));
        const s2 = Math.min(0.6, Math.exp(readField(dv, base, 'scale_2', -4)));
        scales[i3] = s0;
        scales[i3 + 1] = s1;
        scales[i3 + 2] = s2;

        // 3DGS stores rot_0..3 as (w, x, y, z)
        quaternions[i4] = readField(dv, base, 'rot_1', 0);
        quaternions[i4 + 1] = readField(dv, base, 'rot_2', 0);
        quaternions[i4 + 2] = readField(dv, base, 'rot_3', 0);
        quaternions[i4 + 3] = readField(dv, base, 'rot_0', 1);

        colors[i3] = Math.max(0, Math.min(1, 0.5 + SH_C0 * readField(dv, base, 'f_dc_0', 0)));
        colors[i3 + 1] = Math.max(0, Math.min(1, 0.5 + SH_C0 * readField(dv, base, 'f_dc_1', 0)));
        colors[i3 + 2] = Math.max(0, Math.min(1, 0.5 + SH_C0 * readField(dv, base, 'f_dc_2', 0)));
        opacities[o] = sigmoid(readField(dv, base, 'opacity', 2));
      } else {
        // Colored point cloud (e.g. microsoft/moge-3)
        const r = readField(dv, base, 'red', 180);
        const g = readField(dv, base, 'green', 180);
        const b = readField(dv, base, 'blue', 180);
        const norm = r > 1 || g > 1 || b > 1 ? 1 / 255 : 1;
        colors[i3] = r * norm;
        colors[i3 + 1] = g * norm;
        colors[i3 + 2] = b * norm;
        scales[i3] = 0.045;
        scales[i3 + 1] = 0.045;
        scales[i3 + 2] = 0.045;
        quaternions[i4 + 3] = 1;
        opacities[o] = 0.94;
      }
      groupIds[o] = 0.5;
      animKinds[o] = 0;
      o++;
    }
  } else {
    // ASCII fallback
    const text = new TextDecoder().decode(bytes.subarray(bodyOffset));
    const lines = text.split(/\r?\n/);
    const propIdx = {};
    props.forEach((p, idx) => (propIdx[p.name] = idx));
    let o = 0;
    for (let i = 0; i < Math.min(vertexCount, lines.length); i += step) {
      const vals = lines[i].trim().split(/\s+/).map(Number);
      if (vals.length < 3 || Number.isNaN(vals[0])) continue;
      const i3 = o * 3;
      const i4 = o * 4;
      const x = vals[propIdx.x ?? 0];
      const y = -vals[propIdx.y ?? 1];
      const z = -vals[propIdx.z ?? 2];
      positions[i3] = x;
      positions[i3 + 1] = y;
      positions[i3 + 2] = z;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
      const r = vals[propIdx.red ?? 3] ?? 200;
      const g = vals[propIdx.green ?? 4] ?? 200;
      const b = vals[propIdx.blue ?? 5] ?? 200;
      const norm = r > 1 || g > 1 || b > 1 ? 1 / 255 : 1;
      colors[i3] = r * norm;
      colors[i3 + 1] = g * norm;
      colors[i3 + 2] = b * norm;
      scales[i3] = scales[i3 + 1] = scales[i3 + 2] = 0.045;
      quaternions[i4 + 3] = 1;
      opacities[o] = 0.95;
      groupIds[o] = 0.5;
      o++;
    }
  }

  // Normalize bounding box cleanly into ~10m hero scale centered at origin
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.01);
  const fitScale = 10.0 / extent;
  const pointSplatSize = (10.0 / Math.sqrt(outCount)) * 1.65;

  for (let i = 0; i < outCount; i++) {
    const i3 = i * 3;
    positions[i3] = (positions[i3] - cx) * fitScale;
    positions[i3 + 1] = (positions[i3 + 1] - cy) * fitScale + 1.5;
    positions[i3 + 2] = (positions[i3 + 2] - cz) * fitScale;
    if (isGaussian) {
      scales[i3] *= fitScale;
      scales[i3 + 1] *= fitScale;
      scales[i3 + 2] *= fitScale;
    } else {
      scales[i3] = scales[i3 + 1] = scales[i3 + 2] = pointSplatSize;
    }
  }

  return {
    count: outCount,
    isGaussian,
    positions,
    scales,
    quaternions,
    colors,
    opacities,
    groupIds,
    animKinds,
  };
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${url}`));
    img.src = url;
  });
}

function extractTexturePixels(texture) {
  const img = texture?.image;
  if (!img || !img.width || !img.height) return null;
  const maxTex = 1024;
  const scale = Math.min(1, maxTex / Math.max(img.width, img.height));
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, w, h);
  return { width: w, height: h, data: ctx.getImageData(0, 0, w, h).data };
}

function linearToSrgb(v) {
  const x = Math.max(0, Math.min(1, v));
  return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

/**
 * Parse a .glb or .gltf ArrayBuffer into a ready-to-render SplatData object.
 * Supports BOTH:
 *  - Colored point-cloud GLBs (e.g. `speridlabs/idol` `model_glb`)
 *  - Textured / vertex-colored triangle-mesh GLBs (sampled into surface-aligned surfels!)
 */
export async function parseGlbToSplats(arrayBuffer, targetSplats = 65000) {
  const loader = new GLTFLoader();
  const gltf = await new Promise((resolve, reject) => {
    loader.parse(arrayBuffer, '', resolve, reject);
  });

  gltf.scene.updateMatrixWorld(true);

  const pointClouds = [];
  const meshes = [];
  gltf.scene.traverse((obj) => {
    if (obj.isPoints && obj.geometry?.attributes?.position) {
      pointClouds.push(obj);
    } else if (obj.isMesh && obj.geometry?.attributes?.position) {
      meshes.push(obj);
    }
  });

  // --- Branch A: Point-Cloud GLB (e.g. speridlabs/idol model_glb) ---
  if (pointClouds.length > 0 && meshes.length === 0) {
    let totalPts = 0;
    for (const p of pointClouds) totalPts += p.geometry.attributes.position.count;
    if (!totalPts) throw new Error('GLB point cloud contains 0 vertices');

    const step = Math.max(1, Math.ceil(totalPts / Math.max(targetSplats, 100000)));
    const outCount = Math.ceil(totalPts / step);

    const positions = new Float32Array(outCount * 3);
    const scales = new Float32Array(outCount * 3);
    const quaternions = new Float32Array(outCount * 4);
    const colors = new Float32Array(outCount * 3);
    const opacities = new Float32Array(outCount);
    const groupIds = new Float32Array(outCount);
    const animKinds = new Float32Array(outCount);

    const v = new THREE.Vector3();
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let o = 0;
    let globalIdx = 0;

    for (let pIdx = 0; pIdx < pointClouds.length; pIdx++) {
      const pts = pointClouds[pIdx];
      const posAttr = pts.geometry.attributes.position;
      const colAttr = pts.geometry.attributes.color;
      const matColor = pts.material?.color;
      const gId = 0.25 + ((pIdx * 0.19) % 0.65);

      for (let i = 0; i < posAttr.count; i++, globalIdx++) {
        if (globalIdx % step !== 0 || o >= outCount) continue;
        v.fromBufferAttribute(posAttr, i).applyMatrix4(pts.matrixWorld);
        const i3 = o * 3;
        const i4 = o * 4;
        positions[i3] = v.x;
        positions[i3 + 1] = v.y;
        positions[i3 + 2] = v.z;

        if (v.x < minX) minX = v.x;
        if (v.y < minY) minY = v.y;
        if (v.z < minZ) minZ = v.z;
        if (v.x > maxX) maxX = v.x;
        if (v.y > maxY) maxY = v.y;
        if (v.z > maxZ) maxZ = v.z;

        if (colAttr) {
          // glTF vertex colors are stored in linear space -> convert cleanly to display sRGB
          colors[i3] = linearToSrgb(colAttr.getX(i));
          colors[i3 + 1] = linearToSrgb(colAttr.getY(i));
          colors[i3 + 2] = linearToSrgb(colAttr.getZ(i));
        } else if (matColor) {
          colors[i3] = linearToSrgb(matColor.r);
          colors[i3 + 1] = linearToSrgb(matColor.g);
          colors[i3 + 2] = linearToSrgb(matColor.b);
        } else {
          colors[i3] = colors[i3 + 1] = colors[i3 + 2] = 0.82;
        }

        quaternions[i4 + 3] = 1;
        opacities[o] = 0.95;
        groupIds[o] = gId;
        animKinds[o] = 0;
        o++;
      }
    }

    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    const cz = (minZ + maxZ) * 0.5;
    const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.001);
    const fitScale = 8.5 / extent;
    const splatSize = Math.max(0.035, (8.5 / Math.sqrt(Math.max(o, 1))) * 1.9);

    for (let i = 0; i < o; i++) {
      const i3 = i * 3;
      positions[i3] = (positions[i3] - cx) * fitScale;
      positions[i3 + 1] = (positions[i3 + 1] - cy) * fitScale + 2.2;
      positions[i3 + 2] = (positions[i3 + 2] - cz) * fitScale;
      scales[i3] = scales[i3 + 1] = scales[i3 + 2] = splatSize;
    }

    return {
      count: o,
      kind: 'glb-points',
      positions: positions.subarray(0, o * 3),
      scales: scales.subarray(0, o * 3),
      quaternions: quaternions.subarray(0, o * 4),
      colors: colors.subarray(0, o * 3),
      opacities: opacities.subarray(0, o),
      groupIds: groupIds.subarray(0, o),
      animKinds: animKinds.subarray(0, o),
    };
  }

  // --- Branch B: Textured / Vertex-Colored Triangle-Mesh GLB -> Surface-Aligned Surfels ---
  if (meshes.length === 0) {
    throw new Error('GLB file contains no meshes or point clouds');
  }

  const meshEntries = [];
  let grandTotalArea = 0;
  const pA = new THREE.Vector3();
  const pB = new THREE.Vector3();
  const pC = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const cross = new THREE.Vector3();

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  for (let mIdx = 0; mIdx < meshes.length; mIdx++) {
    const mesh = meshes[mIdx];
    const geo = mesh.geometry;
    const posAttr = geo.attributes.position;
    if (!geo.attributes.normal) geo.computeVertexNormals();
    const normAttr = geo.attributes.normal;
    const uvAttr = geo.attributes.uv;
    const colAttr = geo.attributes.color;
    const index = geo.index;

    const triCount = index ? Math.floor(index.count / 3) : Math.floor(posAttr.count / 3);
    if (triCount <= 0) continue;

    const mat = Array.isArray(mesh.material) ? mesh.material[0] : mesh.material;
    const texData = mat?.map ? extractTexturePixels(mat.map) : null;
    const baseColor = mat?.color
      ? [linearToSrgb(mat.color.r), linearToSrgb(mat.color.g), linearToSrgb(mat.color.b)]
      : [0.85, 0.83, 0.8];
    const isUnlit = Boolean(mat?.isMeshBasicMaterial);

    const worldMat = mesh.matrixWorld;
    const normMat = new THREE.Matrix3().getNormalMatrix(worldMat);
    const cdf = new Float64Array(triCount);
    let meshArea = 0;

    for (let t = 0; t < triCount; t++) {
      const ia = index ? index.getX(t * 3) : t * 3;
      const ib = index ? index.getX(t * 3 + 1) : t * 3 + 1;
      const ic = index ? index.getX(t * 3 + 2) : t * 3 + 2;

      pA.fromBufferAttribute(posAttr, ia).applyMatrix4(worldMat);
      pB.fromBufferAttribute(posAttr, ib).applyMatrix4(worldMat);
      pC.fromBufferAttribute(posAttr, ic).applyMatrix4(worldMat);

      for (const pt of [pA, pB, pC]) {
        if (pt.x < minX) minX = pt.x;
        if (pt.y < minY) minY = pt.y;
        if (pt.z < minZ) minZ = pt.z;
        if (pt.x > maxX) maxX = pt.x;
        if (pt.y > maxY) maxY = pt.y;
        if (pt.z > maxZ) maxZ = pt.z;
      }

      e1.subVectors(pB, pA);
      e2.subVectors(pC, pA);
      const area = cross.crossVectors(e1, e2).length() * 0.5;
      meshArea += area;
      cdf[t] = meshArea;
    }

    if (meshArea > 0) {
      grandTotalArea += meshArea;
      meshEntries.push({
        posAttr,
        normAttr,
        uvAttr,
        colAttr,
        index,
        triCount,
        worldMat,
        normMat,
        texData,
        baseColor,
        isUnlit,
        cdf,
        meshArea,
        groupId: 0.2 + ((mIdx * 0.17) % 0.7),
      });
    }
  }

  if (grandTotalArea <= 0) throw new Error('GLB mesh has zero surface area');

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const extent = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 0.001);
  const fitScale = 8.5 / extent;
  const scaledTotalArea = grandTotalArea * fitScale * fitScale;

  const count = targetSplats;
  const positions = new Float32Array(count * 3);
  const scales = new Float32Array(count * 3);
  const quaternions = new Float32Array(count * 4);
  const colors = new Float32Array(count * 3);
  const opacities = new Float32Array(count);
  const groupIds = new Float32Array(count);
  const animKinds = new Float32Array(count);

  const surfelR = Math.max(0.03, Math.sqrt(scaledTotalArea / (Math.PI * count)) * 2.05);
  const upZ = new THREE.Vector3(0, 0, 1);
  const nA = new THREE.Vector3();
  const nB = new THREE.Vector3();
  const nC = new THREE.Vector3();
  const nInterp = new THREE.Vector3();
  const quat = new THREE.Quaternion();
  const lightDir = new THREE.Vector3(0.58, 0.72, 0.42).normalize();

  // Deterministic PRNG for clean repeatable sampling
  let seed = 20261125;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  let out = 0;
  for (let m = 0; m < meshEntries.length; m++) {
    const me = meshEntries[m];
    const mCount =
      m === meshEntries.length - 1
        ? count - out
        : Math.max(1, Math.round((me.meshArea / grandTotalArea) * count));

    for (let s = 0; s < mCount && out < count; s++) {
      // Binary search triangle CDF by area
      const targetA = rand() * me.meshArea;
      let lo = 0;
      let hi = me.triCount - 1;
      while (lo < hi) {
        const mid = (lo + hi) >>> 1;
        if (me.cdf[mid] < targetA) lo = mid + 1;
        else hi = mid;
      }
      const t = lo;
      const ia = me.index ? me.index.getX(t * 3) : t * 3;
      const ib = me.index ? me.index.getX(t * 3 + 1) : t * 3 + 1;
      const ic = me.index ? me.index.getX(t * 3 + 2) : t * 3 + 2;

      // Uniform barycentric coords on triangle
      const r1 = Math.sqrt(rand());
      const r2 = rand();
      const wa = 1 - r1;
      const wb = r1 * (1 - r2);
      const wc = r1 * r2;

      pA.fromBufferAttribute(me.posAttr, ia).applyMatrix4(me.worldMat);
      pB.fromBufferAttribute(me.posAttr, ib).applyMatrix4(me.worldMat);
      pC.fromBufferAttribute(me.posAttr, ic).applyMatrix4(me.worldMat);

      const wx = (pA.x * wa + pB.x * wb + pC.x * wc - cx) * fitScale;
      const wy = (pA.y * wa + pB.y * wb + pC.y * wc - cy) * fitScale + 2.2;
      const wz = (pA.z * wa + pB.z * wb + pC.z * wc - cz) * fitScale;

      nA.fromBufferAttribute(me.normAttr, ia).applyMatrix3(me.normMat);
      nB.fromBufferAttribute(me.normAttr, ib).applyMatrix3(me.normMat);
      nC.fromBufferAttribute(me.normAttr, ic).applyMatrix3(me.normMat);
      nInterp.set(
        nA.x * wa + nB.x * wb + nC.x * wc,
        nA.y * wa + nB.y * wb + nC.y * wc,
        nA.z * wa + nB.z * wb + nC.z * wc
      );
      if (nInterp.lengthSq() < 1e-6) nInterp.set(0, 1, 0);
      else nInterp.normalize();

      // Base color from texture -> vertex colors -> material color
      let cr = me.baseColor[0];
      let cg = me.baseColor[1];
      let cb = me.baseColor[2];

      if (me.texData && me.uvAttr) {
        const u = me.uvAttr.getX(ia) * wa + me.uvAttr.getX(ib) * wb + me.uvAttr.getX(ic) * wc;
        const v = me.uvAttr.getY(ia) * wa + me.uvAttr.getY(ib) * wb + me.uvAttr.getY(ic) * wc;
        const fu = ((u % 1) + 1) % 1;
        const fv = ((v % 1) + 1) % 1;
        const px = Math.min(me.texData.width - 1, Math.floor(fu * me.texData.width));
        const py = Math.min(me.texData.height - 1, Math.floor(fv * me.texData.height));
        const p4 = (py * me.texData.width + px) * 4;
        cr = (me.texData.data[p4] / 255) * cr;
        cg = (me.texData.data[p4 + 1] / 255) * cg;
        cb = (me.texData.data[p4 + 2] / 255) * cb;
      } else if (me.colAttr) {
        cr = linearToSrgb(me.colAttr.getX(ia) * wa + me.colAttr.getX(ib) * wb + me.colAttr.getX(ic) * wc);
        cg = linearToSrgb(me.colAttr.getY(ia) * wa + me.colAttr.getY(ib) * wb + me.colAttr.getY(ic) * wc);
        cb = linearToSrgb(me.colAttr.getZ(ia) * wa + me.colAttr.getZ(ib) * wb + me.colAttr.getZ(ic) * wc);
      }

      // Gentle directional key + sky fill shading so meshes have crisp readable values in NPR shaders
      if (!me.isUnlit) {
        const ndl = Math.max(0, nInterp.dot(lightDir));
        const sky = nInterp.y * 0.5 + 0.5;
        const shadeR = 0.42 + 0.68 * ndl + 0.14 * sky * 0.85;
        const shadeG = 0.42 + 0.68 * ndl + 0.14 * sky * 0.92;
        const shadeB = 0.42 + 0.68 * ndl + 0.14 * sky * 1.05;
        cr = Math.min(1, cr * shadeR);
        cg = Math.min(1, cg * shadeG);
        cb = Math.min(1, cb * shadeB);
      }

      quat.setFromUnitVectors(upZ, nInterp);

      const i3 = out * 3;
      const i4 = out * 4;
      positions[i3] = wx;
      positions[i3 + 1] = wy;
      positions[i3 + 2] = wz;

      const sJitter = surfelR * (0.88 + rand() * 0.3);
      scales[i3] = sJitter;
      scales[i3 + 1] = sJitter;
      scales[i3 + 2] = sJitter * 0.25;

      quaternions[i4] = quat.x;
      quaternions[i4 + 1] = quat.y;
      quaternions[i4 + 2] = quat.z;
      quaternions[i4 + 3] = quat.w;

      colors[i3] = cr;
      colors[i3 + 1] = cg;
      colors[i3 + 2] = cb;
      opacities[out] = 0.96;
      groupIds[out] = me.groupId;
      animKinds[out] = 0;
      out++;
    }
  }

  return {
    count: out,
    kind: 'glb-mesh',
    positions: positions.subarray(0, out * 3),
    scales: scales.subarray(0, out * 3),
    quaternions: quaternions.subarray(0, out * 4),
    colors: colors.subarray(0, out * 3),
    opacities: opacities.subarray(0, out),
    groupIds: groupIds.subarray(0, out),
    animKinds: animKinds.subarray(0, out),
  };
}
