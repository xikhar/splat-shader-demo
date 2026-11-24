import * as THREE from 'three';

/**
 * Full-screen NPR Post-Processing Suite for 3D Gaussian Splats.
 *
 * Modes:
 *  0 = raw         (Unprocessed soft Gaussian Splat baseline)
 *  1 = obra-dinn   (True 2-color 1-bit Bayer dither + view-ray stabilization + depth ink)
 *  2 = kuwahara    (Oriented 4-quadrant luminance-variance Kuwahara oil paint)
 *  3 = ukiyo-e     (Woodblock print: bokashi wash, 5-pigment palette, sumi Sobel lines, washi paper)
 *  4 = ligne-claire (Moebius clear-line: technical pen depth/group contours + flat pastel cel + hatching)
 *  5 = risograph   (3-ink spot separation: fluorescent pink, teal, deep indigo with drum misregistration)
 */

export const NPR_MODES = [
  {
    id: 'raw',
    index: 0,
    name: 'Raw Splat',
    short: 'Raw',
    note: 'Unprocessed 3D Gaussian splats — notice soft edges and semi-transparent floaters.',
  },
  {
    id: 'obra-dinn',
    index: 1,
    name: 'Obra Dinn 1-Bit',
    short: '1-Bit Dither',
    note: 'Strict 2-color output (zero intermediate greys) with view-direction-stabilized Bayer dither + depth silhouettes.',
  },
  {
    id: 'kuwahara',
    index: 2,
    name: 'Kuwahara Oil',
    short: 'Kuwahara',
    note: 'Lowest-variance quadrant painterly filter melts splat overlap noise into crisp-edged brushstrokes.',
  },
  {
    id: 'ukiyo-e',
    index: 3,
    name: 'Ukiyo-e Woodblock',
    short: 'Ukiyo-e',
    note: 'Sumi-e keylines from splat depth, 5-pigment woodblock quantization, bokashi sky & washi paper tooth.',
  },
  {
    id: 'ligne-claire',
    index: 4,
    name: 'Ligne Claire',
    short: 'Moebius',
    note: 'Uniform technical-pen depth/group contours over flat cel pastel bands with shadow hatching.',
  },
  {
    id: 'risograph',
    index: 5,
    name: 'Risograph Zine',
    short: 'Risograph',
    note: '3-ink spot-color overprint (fluo pink, teal, indigo) with angled halftone dots & drum misregistration.',
  },
  {
    id: 'pixel-art',
    index: 6,
    name: 'Pixel Art Splat',
    short: 'Pixel Art',
    note: 'Hard-disc surfels at 2× art resolution, 2×2 majority-vote palette snapping & self-tinted 1-px outlines.',
  },
];

export const MAX_PIXEL_PALETTE = 24;

function hexPalette(hexList) {
  return hexList.map((h) => [((h >> 16) & 255) / 255, ((h >> 8) & 255) / 255, (h & 255) / 255]);
}

export const PIXEL_PALETTES = {
  indie24: {
    label: 'Indie (24)',
    colors: hexPalette([
      0x14151e, 0x26283b, 0x3f435c, 0x6f758e, 0xb3b7c4, 0xf2ece1,
      0x542322, 0x99342a, 0xd44d34, 0xe88a54, 0xf2c679, 0xffe8a3,
      0x1c3a32, 0x2b5e4c, 0x528f68, 0x8eb880,
      0x17314a, 0x255275, 0x3b869e, 0x74b8c4, 0xb5dce2,
      0x6b3a5c, 0xc9738e, 0xf4b3c2,
    ]),
  },
  sweetie16: {
    label: 'Sweetie 16',
    colors: hexPalette([
      0x1a1c2c, 0x5d275d, 0xb13e53, 0xef7d57,
      0xffcd75, 0xa7f070, 0x38b764, 0x257179,
      0x29366f, 0x3b5dc9, 0x41a6f6, 0x73eff7,
      0xf4f4f4, 0x94b0c2, 0x566c86, 0x333c57,
    ]),
  },
  pico8: {
    label: 'PICO-8 (16)',
    colors: hexPalette([
      0x000000, 0x1d2b53, 0x7e2553, 0x008751,
      0xab5236, 0x5f574f, 0xc2c3c7, 0xfff1e8,
      0xff004d, 0xffa300, 0xffec27, 0x00e436,
      0x29adff, 0x83769c, 0xff77a8, 0xffccaa,
    ]),
  },
  twilight8: {
    label: 'Twilight (8)',
    colors: hexPalette([
      0x151421, 0x2d2a4a, 0x59416b, 0x9b5a6f,
      0xd9806c, 0xf2b880, 0xf7e6c4, 0x548c9c,
    ]),
  },
  gameboy: {
    label: 'Game Boy DMG (4)',
    colors: hexPalette([0x0f380f, 0x306230, 0x8bac0f, 0x9bbc0f]),
  },
};

export const OBRA_PALETTES = {
  macintosh: { label: 'Mac Cream / Ink', light: [0.92, 0.90, 0.82], dark: [0.11, 0.12, 0.13] },
  amber: { label: 'IBM Amber', light: [1.0, 0.72, 0.28], dark: [0.11, 0.06, 0.02] },
  lcd: { label: 'LCD Sage', light: [0.79, 0.87, 0.71], dark: [0.12, 0.17, 0.13] },
  sumi: { label: 'Washi / Sumi', light: [0.96, 0.93, 0.86], dark: [0.08, 0.07, 0.06] },
};

const POST_VERT = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

const POST_FRAG = /* glsl */ `
precision highp float;

varying vec2 vUv;

uniform sampler2D tColor;
uniform sampler2D tDepth;      // R = linear depth (0..1), G = semantic groupId, B = valid flag
uniform sampler2D tCrisp;      // 2x-art-resolution hard-disc surfels (RGB = color, A = solid flag)
uniform vec2 uCrispSize;       // pixel dimensions of tCrisp
uniform vec2 uResolution;
uniform int uMode;
uniform float uSplitX;
uniform bool uSplitEnabled;
uniform float uOutlineStrength;
uniform float uKuwaharaRadius; // 2.0, 4.0, or 6.0
uniform float uDitherScale;    // 1.0..3.0
uniform bool uStabilizeDither; // Spherical view-ray mapping (Obra Dinn style!)
uniform float uPixelSize;      // 3.0, 4.0, 6.0, 8.0 display px per art pixel
uniform float uPixelDither;    // 0.0 (majority vote) .. 1.0 (2-color Bayer dither)
uniform vec3 uPixelPalette[${MAX_PIXEL_PALETTE}];
uniform int uPixelPaletteSize;
uniform vec3 uObraLight;
uniform vec3 uObraDark;
uniform vec3 uSkyTop;
uniform vec3 uSkyBottom;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec3 uCamForward;
uniform float uTanHalfFov;
uniform float uAspect;
uniform float uTime;

float luma(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float hash12(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

// 4x4 Bayer matrix normalized to (0..1)
float bayer4(vec2 p) {
  vec2 c = mod(floor(p), 4.0);
  int x = int(c.x);
  int y = int(c.y);
  int idx = y * 4 + x;
  float m = 0.0;
  if (idx == 0) m = 0.0;  else if (idx == 1) m = 8.0;  else if (idx == 2) m = 2.0;  else if (idx == 3) m = 10.0;
  else if (idx == 4) m = 12.0; else if (idx == 5) m = 4.0;  else if (idx == 6) m = 14.0; else if (idx == 7) m = 6.0;
  else if (idx == 8) m = 3.0;  else if (idx == 9) m = 11.0; else if (idx == 10) m = 1.0; else if (idx == 11) m = 9.0;
  else if (idx == 12) m = 15.0; else if (idx == 13) m = 7.0; else if (idx == 14) m = 13.0; else m = 5.0;
  return (m + 0.5) / 16.0;
}

// 8x8 Bayer built recursively from 4x4
float bayer8(vec2 p) {
  vec2 q = floor(p);
  float coarse = bayer4(floor(q * 0.5));
  vec2 sub = mod(q, 2.0);
  float fine = (sub.y * 2.0 + sub.x);
  // 2x2 Bayer order: 0, 2, 3, 1
  float b2 = (fine < 0.5) ? 0.0 : (fine < 1.5) ? 2.0 : (fine < 2.5) ? 3.0 : 1.0;
  return (coarse * 16.0 + b2 * 0.25) / 16.0;
}

// Composite soft Gaussian splat RGBA over smooth sky gradient
vec3 sampleScene(vec2 uv) {
  vec4 s = texture2D(tColor, uv);
  vec3 sky = mix(uSkyBottom, uSkyTop, smoothstep(0.05, 0.95, uv.y));
  return mix(sky, s.rgb, clamp(s.a, 0.0, 1.0));
}

// Clean silhouette & structural edge detector using splat core depth + groupId + luminance
float computeEdge(vec2 uv, float thickness) {
  vec2 px = thickness / uResolution;

  vec4 dC  = texture2D(tDepth, uv);
  vec4 dL  = texture2D(tDepth, uv + vec2(-px.x, 0.0));
  vec4 dR  = texture2D(tDepth, uv + vec2( px.x, 0.0));
  vec4 dU  = texture2D(tDepth, uv + vec2(0.0,  px.y));
  vec4 dD  = texture2D(tDepth, uv + vec2(0.0, -px.y));
  vec4 dUL = texture2D(tDepth, uv + vec2(-px.x,  px.y));
  vec4 dUR = texture2D(tDepth, uv + vec2( px.x,  px.y));
  vec4 dDL = texture2D(tDepth, uv + vec2(-px.x, -px.y));
  vec4 dDR = texture2D(tDepth, uv + vec2( px.x, -px.y));

  // Sky pixels have valid flag B == 0 -> treat depth as 1.0
  float zL  = mix(1.0, dL.r,  dL.b);
  float zR  = mix(1.0, dR.r,  dR.b);
  float zU  = mix(1.0, dU.r,  dU.b);
  float zD  = mix(1.0, dD.r,  dD.b);
  float zUL = mix(1.0, dUL.r, dUL.b);
  float zUR = mix(1.0, dUR.r, dUR.b);
  float zDL = mix(1.0, dDL.r, dDL.b);
  float zDR = mix(1.0, dDR.r, dDR.b);

  float gxD = -zUL - 2.0 * zL - zDL + zUR + 2.0 * zR + zDR;
  float gyD = -zDL - 2.0 * zD - zDR + zUL + 2.0 * zU + zUR;
  float depthEdge = smoothstep(0.045, 0.14, sqrt(gxD * gxD + gyD * gyD));

  // Semantic group-ID discontinuities catch crisp object-on-object borders even at equal depth!
  float gDiff =
    abs(dC.g - dL.g) + abs(dC.g - dR.g) + abs(dC.g - dU.g) + abs(dC.g - dD.g);
  float groupEdge = smoothstep(0.16, 0.34, gDiff) * max(dC.b, max(dL.b, dR.b));

  // Gentle luminance Sobel for strong painted contrast edges (e.g., lighthouse stripes)
  float lL = luma(sampleScene(uv + vec2(-px.x, 0.0)));
  float lR = luma(sampleScene(uv + vec2( px.x, 0.0)));
  float lU = luma(sampleScene(uv + vec2(0.0,  px.y)));
  float lD = luma(sampleScene(uv + vec2(0.0, -px.y)));
  float lumaEdge = smoothstep(0.24, 0.48, abs(lR - lL) + abs(lU - lD)) * dC.b;

  return clamp(max(max(depthEdge, groupEdge * 0.9), lumaEdge * 0.72), 0.0, 1.0);
}

// Oriented 4-quadrant Kuwahara filter (supports radius 2..6 cleanly)
vec3 runKuwahara(vec2 uv, float radiusF) {
  vec2 px = 1.0 / uResolution;
  int r = int(clamp(radiusF, 2.0, 6.0));

  // Compute local luminance gradient angle to orient quadrants along contours!
  float lL = luma(sampleScene(uv + vec2(-px.x * 2.0, 0.0)));
  float lR = luma(sampleScene(uv + vec2( px.x * 2.0, 0.0)));
  float lU = luma(sampleScene(uv + vec2(0.0,  px.y * 2.0)));
  float lD = luma(sampleScene(uv + vec2(0.0, -px.y * 2.0)));
  float gx = lR - lL;
  float gy = lU - lD;
  float angle = (gx * gx + gy * gy > 1e-4) ? atan(gy, gx) : 0.0;
  float cs = cos(angle);
  float sn = sin(angle);
  mat2 rot = mat2(cs, -sn, sn, cs);

  vec3 mean[4];
  float var[4];

  for (int q = 0; q < 4; q++) {
    vec3 sum = vec3(0.0);
    float sumL = 0.0;
    float sumL2 = 0.0;
    float count = 0.0;

    float sx = (q == 0 || q == 2) ? -1.0 : 1.0;
    float sy = (q < 2) ? -1.0 : 1.0;

    for (int iy = 0; iy <= 6; iy++) {
      if (iy > r) break;
      for (int ix = 0; ix <= 6; ix++) {
        if (ix > r) break;
        vec2 off = rot * vec2(float(ix) * sx, float(iy) * sy);
        vec3 c = sampleScene(uv + off * px);
        float lum = luma(c);
        sum += c;
        sumL += lum;
        sumL2 += lum * lum;
        count += 1.0;
      }
    }

    mean[q] = sum / count;
    float mL = sumL / count;
    var[q] = max(0.0, sumL2 / count - mL * mL);
  }

  float minVar = var[0];
  vec3 best = mean[0];
  for (int q = 1; q < 4; q++) {
    if (var[q] < minVar) {
      minVar = var[q];
      best = mean[q];
    }
  }
  return best;
}

// Mode 1: Obra Dinn 1-Bit Dither (Zero grey intermediate colors!)
vec3 renderObraDinn(vec2 uv) {
  // Subtle pixel grid quantization when uDitherScale > 1
  vec2 stepPx = vec2(max(1.0, floor(uDitherScale + 0.5)));
  vec2 qUv = (floor(gl_FragCoord.xy / stepPx) * stepPx + 0.5 * stepPx) / uResolution;

  vec3 col = sampleScene(qUv);
  float lum = luma(col);
  // Slight S-curve contrast boost for crisp woodcut/1-bit readability
  lum = smoothstep(0.07, 0.93, lum);

  vec2 ditherCoord;
  if (uStabilizeDither) {
    // View-direction-stabilized spherical projection (avoids swimming dither during camera rotation!)
    vec2 ndc = qUv * 2.0 - 1.0;
    vec3 ray = normalize(
      uCamForward +
      uCamRight * (ndc.x * uAspect * uTanHalfFov) +
      uCamUp * (ndc.y * uTanHalfFov)
    );
    float az = atan(ray.z, ray.x); // -pi..pi
    float el = asin(clamp(ray.y, -1.0, 1.0)); // -pi/2..pi/2
    float density = (uResolution.y / stepPx.y) * 0.82;
    ditherCoord = vec2(az * density * 0.5, el * density);
  } else {
    ditherCoord = gl_FragCoord.xy / stepPx;
  }

  float thresh = bayer8(ditherCoord);
  float bit = step(thresh, lum);

  // Crisp 1-bit depth silhouette outlines (switches to contrasting ink!)
  float edge = computeEdge(qUv, max(1.1, stepPx.x * 0.85));
  if (edge * uOutlineStrength > 0.42) {
    // Dark ink on bright regions, light rim on deep shadows (classic Obra Dinn legibility trick)
    bit = lum > 0.22 ? 0.0 : 1.0;
  }

  return mix(uObraDark, uObraLight, bit);
}

// Mode 2: Anisotropic Kuwahara Oil / Arcane Painterly
vec3 renderKuwahara(vec2 uv) {
  vec3 paint = runKuwahara(uv, uKuwaharaRadius);

  // Boost pigment saturation slightly & add warm glaze depth
  float l = luma(paint);
  paint = mix(vec3(l), paint, 1.18);
  paint = pow(clamp(paint, 0.0, 1.0), vec3(0.96));

  // Subtle woven linen canvas tooth
  float weaveX = sin(gl_FragCoord.x * 1.45) * 0.5 + 0.5;
  float weaveY = sin(gl_FragCoord.y * 1.45) * 0.5 + 0.5;
  float canvasGrain = (weaveX * 0.5 + weaveY * 0.5 - 0.5) * 0.045 + (hash12(floor(gl_FragCoord.xy)) - 0.5) * 0.02;
  paint += canvasGrain;

  // Delicate painted dark-umber edge accent
  float edge = computeEdge(uv, 1.35) * uOutlineStrength * 0.58;
  vec3 umber = vec3(0.11, 0.09, 0.08);
  return mix(paint, umber, clamp(edge, 0.0, 0.85));
}

// Mode 3: Ukiyo-e Woodblock Print
vec3 renderUkiyoe(vec2 uv) {
  // Smooth splat noise with fast radius-2 Kuwahara + subtle woodblock color plate misregistration
  vec2 regShift = vec2(1.4, -0.9) / uResolution;
  vec3 cBase = runKuwahara(uv + regShift, 2.0);
  vec4 dInfo = texture2D(tDepth, uv);

  // Bokashi sky wash (warm horizontal woodblock gradient on sky)
  if (dInfo.b < 0.5) {
    float band = floor(uv.y * 14.0) / 14.0;
    vec3 bokashiTop = vec3(0.15, 0.31, 0.45);
    vec3 bokashiMid = vec3(0.84, 0.78, 0.65);
    vec3 bokashiBot = vec3(0.95, 0.87, 0.72);
    vec3 skyWash = mix(bokashiBot, mix(bokashiMid, bokashiTop, smoothstep(0.42, 0.95, uv.y)), smoothstep(0.05, 0.75, uv.y));
    cBase = mix(cBase, skyWash, 0.62 + 0.08 * sin(band * 20.0));
  } else {
    // Quantize into 5 traditional woodblock tonal levels while keeping rich vermilion/indigo chroma
    float lum = luma(cBase);
    float qLum = floor(lum * 5.0 + 0.45) / 5.0;
    vec3 chroma = cBase / max(lum, 0.04);
    // Warm up highlights toward handmade washi paper
    vec3 washiTint = vec3(0.96, 0.91, 0.80);
    cBase = clamp(chroma * (0.14 + qLum * 0.86), 0.0, 1.0);
    cBase = mix(cBase, washiTint, smoothstep(0.76, 1.0, qLum) * 0.35);
  }

  // Washi paper woodgrain & organic fiber flecks
  float fiber = sin(uv.x * 160.0 + sin(uv.y * 28.0) * 3.0) * 0.018 + (hash12(floor(gl_FragCoord.xy * 0.7)) - 0.5) * 0.045;
  cBase = clamp(cBase + fiber, 0.0, 1.0);

  // Sumi-e black keyblock outline (aligned to true unshifted UV so color slightly misregisters!)
  float edge = computeEdge(uv, 1.45) * uOutlineStrength;
  vec3 sumi = vec3(0.08, 0.07, 0.08);
  return mix(cBase, sumi, smoothstep(0.18, 0.65, edge));
}

// Mode 4: Ligne Claire / Moebius Clear-Line
vec3 renderLigneClaire(vec2 uv) {
  // Smooth splat artifacts into clean flat architectural fills
  vec3 base = runKuwahara(uv, 3.0);
  vec4 dInfo = texture2D(tDepth, uv);

  float lum = luma(base);
  // Flat 4-band cel/gouache value stepping
  float cel = floor(lum * 4.2 + 0.35) / 4.2;
  vec3 flatCol = (lum > 0.03) ? (base * (cel / lum)) : base;

  // Moebius warm pastel paper lift (softens muddy shadows into luminous graphic tones)
  vec3 paper = vec3(0.96, 0.92, 0.85);
  flatCol = mix(flatCol, mix(vec3(luma(flatCol)), flatCol, 0.92), 0.25);
  flatCol = mix(flatCol, paper, 0.14);

  // Fine parallel technical-pen hatching inside deep shadow regions
  if (dInfo.b > 0.5 && lum < 0.34) {
    float hatchCoord = (gl_FragCoord.x + gl_FragCoord.y * 0.72) * 0.85;
    float hatch = smoothstep(0.68, 0.86, sin(hatchCoord * 1.4));
    flatCol = mix(flatCol, vec3(0.16, 0.14, 0.16), hatch * 0.35 * smoothstep(0.34, 0.12, lum));
  }

  // Crisp uniform Rotring ink lines
  float edge = computeEdge(uv, 1.15) * uOutlineStrength;
  vec3 ink = vec3(0.11, 0.10, 0.12);
  return mix(flatCol, ink, smoothstep(0.22, 0.56, edge));
}

// Mode 5: Risograph 3-Ink Zine Print
vec3 renderRisograph(vec2 uv) {
  // Three offset spot plates (Fluorescent Pink, Teal, Deep Indigo) on warm newsprint stock
  vec2 shiftPink = vec2(2.4, -1.6) / uResolution;
  vec2 shiftTeal = vec2(-2.1, 1.8) / uResolution;

  vec3 cKey  = runKuwahara(uv, 2.0);
  vec3 cPink = sampleScene(uv + shiftPink);
  vec3 cTeal = sampleScene(uv + shiftTeal);

  // Extract ink densities
  float pinkAmt = clamp((cPink.r - cPink.g * 0.65) * 1.45 + (1.0 - luma(cPink)) * 0.22, 0.0, 1.0);
  float tealAmt = clamp((cTeal.b * 0.7 + cTeal.g * 0.55 - cTeal.r * 0.65) * 1.45 + (1.0 - luma(cTeal)) * 0.25, 0.0, 1.0);
  float blackAmt = clamp(smoothstep(0.62, 0.08, luma(cKey)) * 0.92, 0.0, 1.0);

  // Rotated halftone dot matrices per ink drum (classic riso angles: 15°, 75°, 45°)
  float dotScale = 4.2;
  vec2 p15 = mat2(0.966, -0.259, 0.259, 0.966) * gl_FragCoord.xy / dotScale;
  vec2 p75 = mat2(0.259, -0.966, 0.966, 0.259) * gl_FragCoord.xy / dotScale;
  vec2 p45 = mat2(0.707, -0.707, 0.707, 0.707) * gl_FragCoord.xy / dotScale;

  float dPink  = 1.0 - smoothstep(0.0, 0.72, length(fract(p15) - 0.5) * 1.42 / max(0.08, sqrt(pinkAmt)));
  float dTeal  = 1.0 - smoothstep(0.0, 0.72, length(fract(p75) - 0.5) * 1.42 / max(0.08, sqrt(tealAmt)));
  float dBlack = 1.0 - smoothstep(0.0, 0.72, length(fract(p45) - 0.5) * 1.42 / max(0.08, sqrt(blackAmt)));

  // Subtractive multiply overprint on warm cream zine paper
  vec3 paper = vec3(0.96, 0.94, 0.89);
  vec3 inkPink  = vec3(1.00, 0.28, 0.53);
  vec3 inkTeal  = vec3(0.06, 0.66, 0.68);
  vec3 inkBlack = vec3(0.14, 0.13, 0.24);

  vec3 outCol = paper;
  outCol *= mix(vec3(1.0), inkPink,  clamp(dPink  * 0.86, 0.0, 0.92));
  outCol *= mix(vec3(1.0), inkTeal,  clamp(dTeal  * 0.84, 0.0, 0.92));
  outCol *= mix(vec3(1.0), inkBlack, clamp(dBlack * 0.88, 0.0, 0.94));

  // Add edge line onto the dark indigo key plate
  float edge = computeEdge(uv, 1.35) * uOutlineStrength;
  outCol = mix(outCol, inkBlack, smoothstep(0.26, 0.64, edge) * 0.85);

  // Soy-ink drum grain
  float grain = (hash12(floor(gl_FragCoord.xy)) - 0.5) * 0.055;
  return clamp(outCol + grain, 0.0, 1.0);
}

// --- Mode 6: Cobanov-style Pixel Art Gaussian Splats ---
// Hard-disc surfels at 2x art resolution -> 2x2 supersampled coverage -> perceptual majority-vote
// palette snapping -> self-tinted darkened 1-art-pixel outlines + optional 2-color Bayer dither!

int nearestPixelPal(vec3 col) {
  int best = 0;
  float bestD = 1e9;
  for (int i = 0; i < ${MAX_PIXEL_PALETTE}; i++) {
    if (i >= uPixelPaletteSize) break;
    vec3 e = col - uPixelPalette[i];
    vec3 w = e * vec3(0.30, 0.59, 0.11);
    float d = dot(w, w) + 0.25 * dot(e, e);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

vec3 snapPixelDither(vec3 col, vec2 cell) {
  vec3 c1 = col;
  vec3 c2 = col;
  float d1 = 1e9;
  float d2 = 1e9;
  for (int i = 0; i < ${MAX_PIXEL_PALETTE}; i++) {
    if (i >= uPixelPaletteSize) break;
    vec3 p = uPixelPalette[i];
    vec3 e = col - p;
    vec3 w = e * vec3(0.30, 0.59, 0.11);
    float d = dot(w, w) + 0.25 * dot(e, e);
    if (d < d1) {
      d2 = d1; c2 = c1;
      d1 = d;  c1 = p;
    } else if (d < d2) {
      d2 = d;  c2 = p;
    }
  }
  vec3 axis = c2 - c1;
  float t = clamp(dot(col - c1, axis) / max(dot(axis, axis), 1e-5), 0.0, 0.5);
  return bayer4(cell) < t * uPixelDither * 1.6 ? c2 : c1;
}

// Self-tinted darkened silhouette ink (never vanishes into dark backgrounds)
vec3 selfTintedInk(vec3 body) {
  vec3 tone = body * 0.58;
  float l = max(luma(tone), 0.001);
  tone = min(vec3(1.0), tone * max(1.0, 0.10 / l));
  return mix(body, tone, 0.72);
}

vec4 fetchCrispSub(vec2 subCoord) {
  vec2 clamped = clamp(subCoord, vec2(0.0), uCrispSize - vec2(1.0));
  return texture2D(tCrisp, (clamped + 0.5) / uCrispSize);
}

float crispCellCoverage(vec2 cell) {
  vec2 base = cell * 2.0;
  float c = 0.0;
  c += step(0.5, fetchCrispSub(base + vec2(0.0, 0.0)).a);
  c += step(0.5, fetchCrispSub(base + vec2(1.0, 0.0)).a);
  c += step(0.5, fetchCrispSub(base + vec2(0.0, 1.0)).a);
  c += step(0.5, fetchCrispSub(base + vec2(1.0, 1.0)).a);
  return c * 0.25;
}

vec3 renderPixelArt() {
  vec2 cell = floor(gl_FragCoord.xy / uPixelSize);
  vec2 cellCenterUv = ((cell + 0.5) * uPixelSize) / uResolution;
  vec2 base = cell * 2.0;

  vec4 smp[4];
  smp[0] = fetchCrispSub(base + vec2(0.0, 0.0));
  smp[1] = fetchCrispSub(base + vec2(1.0, 0.0));
  smp[2] = fetchCrispSub(base + vec2(0.0, 1.0));
  smp[3] = fetchCrispSub(base + vec2(1.0, 1.0));

  vec3 solidSum = vec3(0.0);
  int solidCount = 0;
  for (int i = 0; i < 4; i++) {
    if (smp[i].a > 0.5) {
      solidSum += smp[i].rgb;
      solidCount++;
    }
  }

  // Solid art pixel (>= 2 of 4 hard-disc samples covered)
  if (solidCount >= 2) {
    vec3 col;
    if (uPixelDither <= 0.02) {
      // Majority-vote palette snapping: zero muddy blended border colors!
      int lab[4];
      for (int i = 0; i < 4; i++) {
        lab[i] = smp[i].a > 0.5 ? nearestPixelPal(smp[i].rgb) : -1;
      }
      int bestIdx = 0;
      int bestScore = -1;
      for (int i = 0; i < 4; i++) {
        if (lab[i] < 0) continue;
        int score = 0;
        for (int j = 0; j < 4; j++) {
          if (lab[j] == lab[i]) score++;
        }
        if (score > bestScore) {
          bestScore = score;
          bestIdx = lab[i];
        }
      }
      col = uPixelPalette[bestIdx];
    } else {
      col = snapPixelDither(solidSum / float(solidCount), cell);
    }

    // Crisp 1-art-pixel self-tinted outlines (silhouette + foreground depth steps)
    if (uOutlineStrength > 0.05) {
      int nSolid =
        int(crispCellCoverage(cell + vec2( 1.0,  0.0)) >= 0.5) +
        int(crispCellCoverage(cell + vec2(-1.0,  0.0)) >= 0.5) +
        int(crispCellCoverage(cell + vec2( 0.0,  1.0)) >= 0.5) +
        int(crispCellCoverage(cell + vec2( 0.0, -1.0)) >= 0.5);
      // Silhouette edge (skip isolated 1-px sparkles/embers where nSolid == 0!)
      bool isEdge = (nSolid > 0 && nSolid < 4);

      // Internal foreground depth-step crease (foreground pixel only -> crisp 1-px line!)
      if (!isEdge && uOutlineStrength > 0.35) {
        vec2 stepUv = uPixelSize / uResolution;
        vec4 dC = texture2D(tDepth, cellCenterUv);
        if (dC.b > 0.5) {
          float dR = texture2D(tDepth, cellCenterUv + vec2( stepUv.x, 0.0)).r;
          float dL = texture2D(tDepth, cellCenterUv + vec2(-stepUv.x, 0.0)).r;
          float dU = texture2D(tDepth, cellCenterUv + vec2(0.0,  stepUv.y)).r;
          float dD = texture2D(tDepth, cellCenterUv + vec2(0.0, -stepUv.y)).r;
          float thresh = mix(0.11, 0.042, clamp(uOutlineStrength * 0.65, 0.0, 1.0));
          float farJump = max(max(dR - dC.r, dL - dC.r), max(dU - dC.r, dD - dC.r));
          if (farJump > thresh) isEdge = true;
        }
      }

      if (isEdge) {
        vec3 inkCol = selfTintedInk(col);
        col = uPixelPalette[nearestPixelPal(inkCol)];
      }
    }
    return col;
  }

  // Background art pixel: sky gradient + translucent beams/halos snapped cleanly into palette
  vec3 bg = sampleScene(cellCenterUv);
  return uPixelDither > 0.02
    ? snapPixelDither(bg, cell)
    : uPixelPalette[nearestPixelPal(bg)];
}

void main() {
  vec3 rawCol = sampleScene(vUv);

  // Determine whether this fragment shows Raw Splat (left of compare split) or NPR Shader (right)
  bool showRaw = (uMode == 0) || (uSplitEnabled && vUv.x < uSplitX);

  vec3 finalCol;
  if (showRaw) {
    finalCol = rawCol;
  } else if (uMode == 1) {
    finalCol = renderObraDinn(vUv);
  } else if (uMode == 2) {
    finalCol = renderKuwahara(vUv);
  } else if (uMode == 3) {
    finalCol = renderUkiyoe(vUv);
  } else if (uMode == 4) {
    finalCol = renderLigneClaire(vUv);
  } else if (uMode == 5) {
    finalCol = renderRisograph(vUv);
  } else {
    finalCol = renderPixelArt();
  }

  // Draw crisp split-screen divider line
  if (uSplitEnabled && uMode != 0) {
    float distPx = abs(vUv.x - uSplitX) * uResolution.x;
    if (distPx < 1.25) {
      finalCol = vec3(0.98, 0.96, 0.92);
    } else if (distPx < 2.4) {
      finalCol = mix(finalCol, vec3(0.08, 0.08, 0.09), 0.7);
    }
  }

  gl_FragColor = vec4(finalCol, 1.0);
}
`;

export class NPRPostProcessor {
  constructor(width, height) {
    this.width = width;
    this.height = height;
    this.pixelSize = 4;

    this.colorRT = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: false,
    });

    this.depthRT = new THREE.WebGLRenderTarget(width, height, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.HalfFloatType,
      depthBuffer: true,
    });

    const crispW = Math.max(64, Math.ceil(width / this.pixelSize) * 2);
    const crispH = Math.max(64, Math.ceil(height / this.pixelSize) * 2);
    this.crispRT = new THREE.WebGLRenderTarget(crispW, crispH, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
    });

    const palArr = Array.from({ length: MAX_PIXEL_PALETTE }, () => new THREE.Vector3(0, 0, 0));

    this.uniforms = {
      tColor: { value: this.colorRT.texture },
      tDepth: { value: this.depthRT.texture },
      tCrisp: { value: this.crispRT.texture },
      uCrispSize: { value: new THREE.Vector2(crispW, crispH) },
      uResolution: { value: new THREE.Vector2(width, height) },
      uMode: { value: 4 },
      uSplitX: { value: 0.36 },
      uSplitEnabled: { value: true },
      uOutlineStrength: { value: 1.0 },
      uKuwaharaRadius: { value: 4.0 },
      uDitherScale: { value: 1.0 },
      uStabilizeDither: { value: true },
      uPixelSize: { value: this.pixelSize },
      uPixelDither: { value: 0.0 },
      uPixelPalette: { value: palArr },
      uPixelPaletteSize: { value: PIXEL_PALETTES.indie24.colors.length },
      uObraLight: { value: new THREE.Vector3(...OBRA_PALETTES.macintosh.light) },
      uObraDark: { value: new THREE.Vector3(...OBRA_PALETTES.macintosh.dark) },
      uSkyTop: { value: new THREE.Vector3(0.49, 0.67, 0.78) },
      uSkyBottom: { value: new THREE.Vector3(0.95, 0.83, 0.68) },
      uCamRight: { value: new THREE.Vector3(1, 0, 0) },
      uCamUp: { value: new THREE.Vector3(0, 1, 0) },
      uCamForward: { value: new THREE.Vector3(0, 0, -1) },
      uTanHalfFov: { value: 0.55 },
      uAspect: { value: width / height },
      uTime: { value: 0 },
    };

    this.setPixelPalette('indie24');

    this.scene = new THREE.Scene();
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const quad = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        vertexShader: POST_VERT,
        fragmentShader: POST_FRAG,
        uniforms: this.uniforms,
        depthTest: false,
        depthWrite: false,
      })
    );
    this.scene.add(quad);
  }

  setPixelPalette(key) {
    const pal = PIXEL_PALETTES[key] || PIXEL_PALETTES.indie24;
    const arr = this.uniforms.uPixelPalette.value;
    const count = Math.min(MAX_PIXEL_PALETTE, pal.colors.length);
    for (let i = 0; i < count; i++) {
      arr[i].set(...pal.colors[i]);
    }
    this.uniforms.uPixelPaletteSize.value = count;
  }

  setPixelSize(px) {
    this.pixelSize = Math.max(2, Math.min(12, Math.round(px)));
    this.uniforms.uPixelSize.value = this.pixelSize;
    this._resizeCrisp();
  }

  _resizeCrisp() {
    const crispW = Math.max(64, Math.ceil(this.width / this.pixelSize) * 2);
    const crispH = Math.max(64, Math.ceil(this.height / this.pixelSize) * 2);
    this.crispRT.setSize(crispW, crispH);
    this.uniforms.uCrispSize.value.set(crispW, crispH);
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.colorRT.setSize(width, height);
    this.depthRT.setSize(width, height);
    this._resizeCrisp();
    this.uniforms.uResolution.value.set(width, height);
    this.uniforms.uAspect.value = width / Math.max(1, height);
  }

  updateCameraBasis(camera, time) {
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize();
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize();
    this.uniforms.uCamRight.value.copy(right);
    this.uniforms.uCamUp.value.copy(up);
    this.uniforms.uCamForward.value.copy(forward);
    this.uniforms.uTanHalfFov.value = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5));
    this.uniforms.uTime.value = time;
  }

  render(renderer, colorScene, depthScene, crispScene, camera, time) {
    this.updateCameraBasis(camera, time);

    // Pass 1: Soft alpha-blended 3D Gaussian Splats (transparent background so sky composites cleanly)
    renderer.setRenderTarget(this.colorRT);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, false, false);
    renderer.render(colorScene, camera);

    // Pass 2: Hard core splat depth + semantic group IDs (writes depth buffer)
    renderer.setRenderTarget(this.depthRT);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, true, false);
    renderer.render(depthScene, camera);

    // Pass 2b (Pixel Art mode only): 2x-art-resolution hard-disc surfels for 2x2 majority vote!
    if (this.uniforms.uMode.value === 6 && crispScene) {
      renderer.setRenderTarget(this.crispRT);
      renderer.setClearColor(0x000000, 0);
      renderer.clear(true, true, false);
      renderer.render(crispScene, camera);
    }

    // Pass 3: Full-screen NPR post-process + split-screen compare divider
    renderer.setRenderTarget(null);
    renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.colorRT.dispose();
    this.depthRT.dispose();
    this.crispRT.dispose();
  }
}
