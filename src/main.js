import './style.css';
import * as THREE from 'three';
import { SplatCloud } from './splats/SplatCloud.js';
import { SAMPLE_WORLD_BUILDERS } from './splats/sampleWorlds.js';
import { parsePlyToSplats, parseGlbToSplats } from './splats/loaders.js';
import {
  NPRPostProcessor,
  NPR_MODES,
  OBRA_PALETTES,
  PIXEL_PALETTES,
} from './shaders/nprPostProcess.js';
import { SpatialAudioEngine } from './audio/spatialAudio.js';

const elApp = document.getElementById('app');
const canvas = document.getElementById('viewport');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  powerPreference: 'high-performance',
  preserveDrawingBuffer: false,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
renderer.setSize(window.innerWidth, window.innerHeight);

const camera = new THREE.PerspectiveCamera(54, window.innerWidth / window.innerHeight, 0.2, 120);
const colorScene = new THREE.Scene();
const depthScene = new THREE.Scene();
const crispScene = new THREE.Scene();

const post = new NPRPostProcessor(
  Math.floor(window.innerWidth * renderer.getPixelRatio()),
  Math.floor(window.innerHeight * renderer.getPixelRatio())
);
const audioEngine = new SpatialAudioEngine();

// App state
const state = {
  worlds: new Map(),
  activeWorld: null,
  activeCloud: null,
  modeId: 'ligne-claire',
  splitEnabled: true,
  splitX: 0.38,
  floaterStrength: 0.55,
  outlineStrength: 1.0,
  kuwaharaRadius: 4.0,
  obraPalette: 'macintosh',
  stabilizeDither: true,
  pixelSize: 4,
  pixelPalette: 'indie24',
  pixelDither: 0.0,
  showPins: true,
  autoOrbit: false,
  uiHidden: false,
};

// Smooth Orbit + WASD Walk Camera Controller
const camCtrl = {
  target: new THREE.Vector3(0, 3.2, 0),
  yaw: 0.08,
  pitch: 0.08,
  distance: 12.2,
  keys: new Set(),
  dragging: false,
  lastX: 0,
  lastY: 0,
};

function setCameraFromSpec(spec) {
  const pos = new THREE.Vector3(...spec.pos);
  const tgt = new THREE.Vector3(...spec.target);
  const offset = pos.clone().sub(tgt);
  camCtrl.target.copy(tgt);
  camCtrl.distance = Math.max(2.0, offset.length());
  camCtrl.yaw = Math.atan2(offset.x, offset.z);
  camCtrl.pitch = Math.asin(Math.max(-0.85, Math.min(0.85, offset.y / camCtrl.distance)));
  syncCameraTransform();
}

function syncCameraTransform() {
  const cp = Math.cos(camCtrl.pitch);
  const sp = Math.sin(camCtrl.pitch);
  const sy = Math.sin(camCtrl.yaw);
  const cy = Math.cos(camCtrl.yaw);
  camera.position.set(
    camCtrl.target.x + sy * cp * camCtrl.distance,
    camCtrl.target.y + sp * camCtrl.distance,
    camCtrl.target.z + cy * cp * camCtrl.distance
  );
  camera.lookAt(camCtrl.target);
  camera.updateMatrixWorld();
}

// DOM refs
const elWorldTitle = document.getElementById('world-title');
const elWorldSub = document.getElementById('world-subtitle');
const elStatSplats = document.getElementById('stat-splats');
const elStatFps = document.getElementById('stat-fps');
const elWorldTabs = document.getElementById('world-tabs');
const elHudBottom = document.getElementById('hud-bottom');
const elShaderTabs = document.getElementById('shader-tabs');
const elShaderNote = document.getElementById('shader-note');
const elSplitHandle = document.getElementById('split-handle');
const elSplitRightLabel = document.getElementById('split-right-label');
const elPinsLayer = document.getElementById('audio-pins-layer');
const elBtnSplit = document.getElementById('btn-split');
const elBtnAutoOrbit = document.getElementById('btn-auto-orbit');
const elBtnAudio = document.getElementById('btn-audio');
const elAudioDot = document.getElementById('audio-dot');
const elAudioBtnLabel = document.getElementById('audio-btn-label');
const elBtnPins = document.getElementById('btn-pins');
const elBtnHideUi = document.getElementById('btn-hide-ui');
const elBtnShowUi = document.getElementById('btn-show-ui');
const elSliderFloaters = document.getElementById('slider-floaters');
const elValFloaters = document.getElementById('val-floaters');
const elSliderOutline = document.getElementById('slider-outline');
const elValOutline = document.getElementById('val-outline');
const elCtrlKuwahara = document.getElementById('ctrl-kuwahara');
const elCtrlObra = document.getElementById('ctrl-obra');
const elCtrlPixel = document.getElementById('ctrl-pixel');
const elSelectObraPalette = document.getElementById('select-obra-palette');
const elSelectPixelPalette = document.getElementById('select-pixel-palette');
const elSliderPixelDither = document.getElementById('slider-pixel-dither');
const elValPixelDither = document.getElementById('val-pixel-dither');
const elBtnStabilize = document.getElementById('btn-stabilize-dither');
const elInputImport = document.getElementById('input-import-file');
const elDropOverlay = document.getElementById('drop-overlay');

/* -------------------------------------------------------------------------- */
/*  WORLD LOADING                                                             */
/* -------------------------------------------------------------------------- */
function mountWorld(world) {
  if (state.activeCloud) {
    colorScene.remove(state.activeCloud.colorMesh);
    depthScene.remove(state.activeCloud.depthMesh);
    crispScene.remove(state.activeCloud.crispMesh);
    state.activeCloud.dispose();
  }

  state.activeWorld = world;
  const cloud = new SplatCloud(world.splats);
  state.activeCloud = cloud;

  colorScene.add(cloud.colorMesh);
  depthScene.add(cloud.depthMesh);
  crispScene.add(cloud.crispMesh);

  post.uniforms.uSkyTop.value.set(...world.skyTop);
  post.uniforms.uSkyBottom.value.set(...world.skyBottom);

  setCameraFromSpec(world.camera);
  cloud.sort(camera, true);

  audioEngine.setWorldEmitters(world.emitters || []);

  elWorldTitle.textContent = world.title;
  elWorldSub.textContent = world.subtitle;
  elStatSplats.textContent = `${world.splats.count.toLocaleString()} splats`;

  setShaderMode(world.defaultShader || 'ligne-claire');
  renderWorldTabs();
}

function loadSampleWorld(id) {
  let w = state.worlds.get(id);
  if (!w) {
    const entry = SAMPLE_WORLD_BUILDERS.find((x) => x.id === id) || SAMPLE_WORLD_BUILDERS[0];
    w = entry.build();
    state.worlds.set(w.id, w);
  }
  mountWorld(w);
}

function renderWorldTabs() {
  elWorldTabs.innerHTML = '';
  for (const entry of SAMPLE_WORLD_BUILDERS) {
    const btn = document.createElement('button');
    btn.className = `world-tab ${state.activeWorld?.id === entry.id ? 'active' : ''}`;
    btn.textContent = entry.label;
    btn.onclick = () => loadSampleWorld(entry.id);
    elWorldTabs.appendChild(btn);
  }
  for (const [id, w] of state.worlds.entries()) {
    if (SAMPLE_WORLD_BUILDERS.some((b) => b.id === id)) continue;
    const btn = document.createElement('button');
    btn.className = `world-tab ${state.activeWorld?.id === id ? 'active' : ''}`;
    btn.textContent = w.shortTitle || 'Imported';
    btn.onclick = () => mountWorld(w);
    elWorldTabs.appendChild(btn);
  }
}

/* -------------------------------------------------------------------------- */
/*  IMPORT .PLY / .GLB / .GLTF                                                */
/* -------------------------------------------------------------------------- */
async function importModelFile(file) {
  const lower = file.name.toLowerCase();
  const isPly = lower.endsWith('.ply');
  const isGlb = lower.endsWith('.glb') || lower.endsWith('.gltf');
  if (!isPly && !isGlb) {
    alert('Please import a .ply, .glb, or .gltf file.');
    return;
  }

  try {
    const buf = await file.arrayBuffer();
    const splats = isPly ? parsePlyToSplats(buf, 110000) : await parseGlbToSplats(buf, 65000);

    let subtitle;
    if (isPly) {
      subtitle = splats.isGaussian
        ? 'Imported 3DGS Gaussian Splat (.ply) — anisotropic covariance active'
        : 'Imported colored 3D Point Cloud (.ply)';
    } else {
      subtitle =
        splats.kind === 'glb-points'
          ? 'Imported Point-Cloud GLB (e.g. Idol model_glb) rendered as 3D splats'
          : 'Imported 3D Mesh GLB converted to surface-aligned anisotropic surfels';
    }

    const baseName = file.name.replace(/\.(ply|glb|gltf)$/i, '');
    const id = `import-${Date.now()}`;
    const importedWorld = {
      id,
      title: baseName,
      shortTitle: baseName.slice(0, 14),
      subtitle,
      defaultShader: state.modeId === 'raw' ? 'pixel-art' : state.modeId,
      skyTop: [0.14, 0.17, 0.23],
      skyBottom: [0.68, 0.66, 0.62],
      camera: { pos: [0, 2.8, 10.8], target: [0, 2.2, 0] },
      emitters: [
        {
          id: 'import-focus',
          label: 'Model Focus',
          position: [0, 2.2, 0],
          color: '#e8a44a',
          synthPreset: 'portalHum',
        },
      ],
      splats,
    };
    state.worlds.set(id, importedWorld);
    mountWorld(importedWorld);
  } catch (err) {
    alert(`Could not import ${file.name}: ${err.message}`);
  }
}

/* -------------------------------------------------------------------------- */
/*  NPR SHADER & UI MODE CONTROLS                                             */
/* -------------------------------------------------------------------------- */
function setUiHidden(hidden) {
  state.uiHidden = Boolean(hidden);
  elApp.classList.toggle('ui-hidden', state.uiHidden);
  if (!state.uiHidden) {
    elHudBottom.classList.remove('is-expanded');
  }
}

function setShaderMode(modeId) {
  const mode = NPR_MODES.find((m) => m.id === modeId) || NPR_MODES[0];
  state.modeId = mode.id;
  post.uniforms.uMode.value = mode.index;
  elShaderNote.textContent = mode.note;
  elSplitRightLabel.textContent = mode.short.toUpperCase();

  document.querySelectorAll('.shader-btn').forEach((b) => {
    b.classList.toggle('active', b.dataset.mode === mode.id);
  });

  elCtrlKuwahara.classList.toggle('hidden', mode.id !== 'kuwahara');
  elCtrlObra.classList.toggle('hidden', mode.id !== 'obra-dinn');
  elCtrlPixel.classList.toggle('hidden', mode.id !== 'pixel-art');
  syncSplitHandleUI();
}

function syncSplitHandleUI() {
  const showHandle = state.splitEnabled && state.modeId !== 'raw';
  post.uniforms.uSplitEnabled.value = showHandle;
  post.uniforms.uSplitX.value = state.splitX;
  elSplitHandle.classList.toggle('hidden', !showHandle);
  elSplitHandle.style.left = `${(state.splitX * 100).toFixed(2)}%`;
  elBtnSplit.classList.toggle('active', state.splitEnabled);
}

function initUI() {
  // Shader tabs (1..7)
  NPR_MODES.forEach((m, idx) => {
    const btn = document.createElement('button');
    btn.className = 'shader-btn';
    btn.dataset.mode = m.id;
    btn.innerHTML = `<kbd>${idx + 1}</kbd> ${m.short}`;
    btn.onclick = () => setShaderMode(m.id);
    elShaderTabs.appendChild(btn);
  });

  // Hide / Show UI controls
  elBtnHideUi.onclick = () => setUiHidden(true);
  elBtnShowUi.onclick = () => setUiHidden(false);

  // Keep compact bottom dock smoothly expanded while interacting with dropdowns/sliders in Hidden UI
  elHudBottom.addEventListener('pointerenter', () => {
    if (state.uiHidden) elHudBottom.classList.add('is-expanded');
  });
  elHudBottom.addEventListener('pointerleave', () => {
    if (!state.uiHidden) return;
    const active = document.activeElement;
    if (active && active.tagName === 'SELECT' && elHudBottom.contains(active)) return;
    elHudBottom.classList.remove('is-expanded');
  });
  elHudBottom.querySelectorAll('select').forEach((sel) => {
    sel.addEventListener('change', () => {
      sel.blur();
      if (state.uiHidden && !elHudBottom.matches(':hover')) {
        elHudBottom.classList.remove('is-expanded');
      }
    });
    sel.addEventListener('blur', () => {
      if (state.uiHidden && !elHudBottom.matches(':hover')) {
        elHudBottom.classList.remove('is-expanded');
      }
    });
  });

  // Obra Dinn palette dropdown
  for (const [key, pal] of Object.entries(OBRA_PALETTES)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = pal.label;
    elSelectObraPalette.appendChild(opt);
  }
  elSelectObraPalette.onchange = () => {
    const pal = OBRA_PALETTES[elSelectObraPalette.value];
    if (!pal) return;
    state.obraPalette = elSelectObraPalette.value;
    post.uniforms.uObraLight.value.set(...pal.light);
    post.uniforms.uObraDark.value.set(...pal.dark);
  };

  // Stabilized dither toggle
  elBtnStabilize.onclick = () => {
    state.stabilizeDither = !state.stabilizeDither;
    post.uniforms.uStabilizeDither.value = state.stabilizeDither;
    elBtnStabilize.classList.toggle('active', state.stabilizeDither);
  };

  // Kuwahara radii (2 / 4 / 6 px)
  document.querySelectorAll('#kuwahara-radii button').forEach((btn) => {
    btn.onclick = () => {
      const r = Number(btn.dataset.r);
      state.kuwaharaRadius = r;
      post.uniforms.uKuwaharaRadius.value = r;
      document.querySelectorAll('#kuwahara-radii button').forEach((x) => x.classList.toggle('active', x === btn));
    };
  });

  // Pixel Art controls
  const syncPixelSize = () => {
    const dpr = renderer.getPixelRatio();
    post.setPixelSize(Math.max(2, Math.round(state.pixelSize * dpr)));
  };
  syncPixelSize();

  document.querySelectorAll('#pixel-sizes button').forEach((btn) => {
    btn.onclick = () => {
      state.pixelSize = Number(btn.dataset.px);
      syncPixelSize();
      document.querySelectorAll('#pixel-sizes button').forEach((x) => x.classList.toggle('active', x === btn));
    };
  });

  for (const [key, pal] of Object.entries(PIXEL_PALETTES)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = pal.label;
    elSelectPixelPalette.appendChild(opt);
  }
  elSelectPixelPalette.onchange = () => {
    state.pixelPalette = elSelectPixelPalette.value;
    post.setPixelPalette(state.pixelPalette);
  };

  elSliderPixelDither.oninput = () => {
    state.pixelDither = Number(elSliderPixelDither.value);
    post.uniforms.uPixelDither.value = state.pixelDither;
    elValPixelDither.textContent = `${Math.round(state.pixelDither * 100)}%`;
  };

  // Sliders
  elSliderFloaters.oninput = () => {
    state.floaterStrength = Number(elSliderFloaters.value);
    elValFloaters.textContent = `${Math.round(state.floaterStrength * 100)}%`;
  };
  elSliderOutline.oninput = () => {
    state.outlineStrength = Number(elSliderOutline.value);
    post.uniforms.uOutlineStrength.value = state.outlineStrength;
    elValOutline.textContent = `${Math.round(state.outlineStrength * 100)}%`;
  };

  // Split compare, auto-orbit, reset
  elBtnSplit.onclick = () => {
    state.splitEnabled = !state.splitEnabled;
    syncSplitHandleUI();
  };
  elBtnAutoOrbit.onclick = () => {
    state.autoOrbit = !state.autoOrbit;
    elBtnAutoOrbit.classList.toggle('active', state.autoOrbit);
  };
  document.getElementById('btn-reset-cam').onclick = () => {
    if (state.activeWorld) setCameraFromSpec(state.activeWorld.camera);
  };

  // Split divider drag interaction
  let draggingSplit = false;
  elSplitHandle.addEventListener('pointerdown', (e) => {
    e.stopPropagation();
    draggingSplit = true;
    elSplitHandle.setPointerCapture(e.pointerId);
  });
  window.addEventListener('pointermove', (e) => {
    if (!draggingSplit) return;
    state.splitX = Math.max(0.06, Math.min(0.94, e.clientX / window.innerWidth));
    syncSplitHandleUI();
  });
  window.addEventListener('pointerup', () => {
    draggingSplit = false;
  });

  // Spatial audio controls
  elBtnAudio.onclick = async () => {
    const live = await audioEngine.toggle();
    elAudioDot.classList.toggle('live', live);
    elBtnAudio.classList.toggle('active', live);
    elAudioBtnLabel.textContent = live ? 'Audio Live' : 'Spatial Audio';
  };
  elBtnPins.onclick = () => {
    state.showPins = !state.showPins;
    elBtnPins.classList.toggle('active', state.showPins);
    elPinsLayer.style.display = state.showPins ? 'block' : 'none';
  };

  // File picker import (.ply / .glb / .gltf)
  elInputImport.onchange = (e) => {
    const file = e.target.files?.[0];
    if (file) importModelFile(file);
    elInputImport.value = '';
  };

  // Drag & drop import anywhere on viewport
  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    elDropOverlay.classList.remove('hidden');
  });
  window.addEventListener('dragleave', (e) => {
    if (e.relatedTarget === null) elDropOverlay.classList.add('hidden');
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    elDropOverlay.classList.add('hidden');
    const file = e.dataTransfer?.files?.[0];
    if (file) importModelFile(file);
  });
}

/* -------------------------------------------------------------------------- */
/*  INPUT & CAMERA NAVIGATION                                                 */
/* -------------------------------------------------------------------------- */
canvas.addEventListener('pointerdown', (e) => {
  camCtrl.dragging = true;
  camCtrl.lastX = e.clientX;
  camCtrl.lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener('pointermove', (e) => {
  if (!camCtrl.dragging) return;
  const dx = e.clientX - camCtrl.lastX;
  const dy = e.clientY - camCtrl.lastY;
  camCtrl.lastX = e.clientX;
  camCtrl.lastY = e.clientY;
  camCtrl.yaw -= dx * 0.0055;
  camCtrl.pitch = Math.max(-0.55, Math.min(1.15, camCtrl.pitch + dy * 0.0045));
});

canvas.addEventListener('pointerup', () => {
  camCtrl.dragging = false;
});

canvas.addEventListener(
  'wheel',
  (e) => {
    e.preventDefault();
    camCtrl.distance = Math.max(2.2, Math.min(28.0, camCtrl.distance * (1 + e.deltaY * 0.001)));
  },
  { passive: false }
);

window.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName)) return;
  const k = e.key.toLowerCase();
  if (['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'].includes(k)) {
    camCtrl.keys.add(k);
  }
  if (k >= '1' && k <= '7') {
    const mode = NPR_MODES[Number(k) - 1];
    if (mode) setShaderMode(mode.id);
  }
  if (k === ' ') {
    e.preventDefault();
    state.splitEnabled = !state.splitEnabled;
    syncSplitHandleUI();
  }
  if (k === 'h') {
    setUiHidden(!state.uiHidden);
  }
  if (k === 'r' && state.activeWorld) {
    setCameraFromSpec(state.activeWorld.camera);
  }
});

window.addEventListener('keyup', (e) => {
  camCtrl.keys.delete(e.key.toLowerCase());
});

window.addEventListener('resize', () => {
  const dpr = Math.min(window.devicePixelRatio || 1, 1.75);
  renderer.setPixelRatio(dpr);
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  post.resize(Math.floor(window.innerWidth * dpr), Math.floor(window.innerHeight * dpr));
  post.setPixelSize(Math.max(2, Math.round(state.pixelSize * dpr)));
});

/* -------------------------------------------------------------------------- */
/*  3D SPATIAL AUDIO PIN OVERLAY                                              */
/* -------------------------------------------------------------------------- */
const _proj = new THREE.Vector3();

function updateAudioHUD(telemetry) {
  if (!state.showPins || state.uiHidden) return;
  const worldKey = `${state.activeWorld?.id || 'none'}:${telemetry.map((t) => t.id).join('|')}`;

  if (elPinsLayer.dataset.worldKey !== worldKey) {
    elPinsLayer.dataset.worldKey = worldKey;
    elPinsLayer.innerHTML = telemetry
      .map(
        (t) =>
          `<div class="audio-pin" data-id="${t.id}">
            <span class="audio-pin-dot" style="background:${t.color};box-shadow:0 0 8px ${t.color}"></span>
            <span>${t.label}</span>
            <span class="audio-pin-meta"></span>
          </div>`
      )
      .join('');
  }

  const dockRect = elHudBottom.getBoundingClientRect();
  const safeTop = 128;
  const safeBottom = dockRect ? dockRect.top - 12 : window.innerHeight - 165;

  telemetry.forEach((t, idx) => {
    const el = elPinsLayer.children[idx];
    if (!el) return;
    _proj.set(...t.position).project(camera);
    if (_proj.z > 1 || _proj.z < -1 || Math.abs(_proj.x) > 1.05 || Math.abs(_proj.y) > 1.05) {
      el.style.display = 'none';
      return;
    }
    el.style.display = 'flex';
    const halfW = (el.offsetWidth || 180) * 0.5 + 12;
    const rawX = (_proj.x * 0.5 + 0.5) * window.innerWidth;
    const rawY = (-_proj.y * 0.5 + 0.5) * window.innerHeight;
    const x = Math.max(halfW, Math.min(window.innerWidth - halfW, rawX));
    const y = Math.max(safeTop, Math.min(safeBottom, rawY));
    el.style.left = `${x.toFixed(1)}px`;
    el.style.top = `${y.toFixed(1)}px`;
    const panChar = t.panHint < -0.22 ? 'L' : t.panHint > 0.22 ? 'R' : 'C';
    const khz = (t.cutoffHz / 1000).toFixed(1);
    el.querySelector('.audio-pin-meta').textContent = `${t.distance.toFixed(1)}m · ${panChar} · ${khz}kHz`;
  });
}

/* -------------------------------------------------------------------------- */
/*  MAIN RENDER LOOP                                                          */
/* -------------------------------------------------------------------------- */
initUI();
loadSampleWorld('moebius-desert');

let prevTime = performance.now();
let fpsAccum = 0;
let fpsFrames = 0;

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, (now - prevTime) * 0.001);
  prevTime = now;
  const elapsed = now * 0.001;

  fpsAccum += dt;
  fpsFrames++;
  if (fpsAccum >= 0.45) {
    elStatFps.textContent = `${Math.round(fpsFrames / fpsAccum)} fps`;
    fpsAccum = 0;
    fpsFrames = 0;
  }

  if (state.autoOrbit && !camCtrl.dragging) {
    camCtrl.yaw += dt * 0.22;
  }

  const moveSpeed = 6.2 * dt;
  const fwd = new THREE.Vector3(-Math.sin(camCtrl.yaw), 0, -Math.cos(camCtrl.yaw));
  const right = new THREE.Vector3(Math.cos(camCtrl.yaw), 0, -Math.sin(camCtrl.yaw));
  if (camCtrl.keys.has('w') || camCtrl.keys.has('arrowup')) camCtrl.target.addScaledVector(fwd, moveSpeed);
  if (camCtrl.keys.has('s') || camCtrl.keys.has('arrowdown')) camCtrl.target.addScaledVector(fwd, -moveSpeed);
  if (camCtrl.keys.has('d') || camCtrl.keys.has('arrowright')) camCtrl.target.addScaledVector(right, moveSpeed);
  if (camCtrl.keys.has('a') || camCtrl.keys.has('arrowleft')) camCtrl.target.addScaledVector(right, -moveSpeed);
  if (camCtrl.keys.has('e')) camCtrl.target.y += moveSpeed * 0.7;
  if (camCtrl.keys.has('q')) camCtrl.target.y -= moveSpeed * 0.7;

  syncCameraTransform();

  if (state.activeCloud) {
    const dpr = renderer.getPixelRatio();
    const w = Math.floor(window.innerWidth * dpr);
    const h = Math.floor(window.innerHeight * dpr);
    const cw = post.crispRT.width;
    const ch = post.crispRT.height;
    state.activeCloud.updateUniforms(camera, w, h, elapsed, 0.72, state.floaterStrength, cw, ch);
    state.activeCloud.sort(camera, false);
    post.render(renderer, colorScene, depthScene, crispScene, camera, elapsed);
  }

  const telemetry = audioEngine.update(camera);
  updateAudioHUD(telemetry);
}

requestAnimationFrame(animate);
