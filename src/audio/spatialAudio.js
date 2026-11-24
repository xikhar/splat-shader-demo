import * as THREE from 'three';

/**
 * Spatial 3D Web Audio Engine with HRTF PannerNodes + directional/distance low-pass occlusion.
 *
 * Works immediately offline via bespoke Web Audio procedural soundscapes per emitter,
 * AND supports hot-swapping any emitter to a generated MP3 loop from `elevenlabs/sound-effects/v3`.
 */

function createNoiseBuffer(ctx, seconds = 4) {
  const len = ctx.sampleRate * seconds;
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const data = buf.getChannelData(0);
  // Pink-ish noise (Paul Kellet's filter)
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < len; i++) {
    const white = Math.random() * 2 - 1;
    b0 = 0.99886 * b0 + white * 0.0555179;
    b1 = 0.99332 * b1 + white * 0.0750759;
    b2 = 0.96900 * b2 + white * 0.1538520;
    b3 = 0.86650 * b3 + white * 0.3104856;
    b4 = 0.55000 * b4 + white * 0.5329522;
    b5 = -0.7616 * b5 - white * 0.0168980;
    data[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.05;
    b6 = white * 0.115926;
  }
  return buf;
}

class SpatialEmitter {
  constructor(ctx, masterGain, noiseBuf, spec) {
    this.ctx = ctx;
    this.spec = spec;
    this.pos = new THREE.Vector3(...spec.position);
    this.nodes = [];
    this.timers = [];
    this.sourceLabel = 'Procedural Synth';
    this.telemetry = { distance: 0, level: 0, cutoffHz: 14000, panHint: 0 };

    // Audio chain: source -> trimGain -> occlusionFilter (lowpass) -> HRTF Panner -> masterGain
    this.trimGain = ctx.createGain();
    this.trimGain.gain.value = 0.65;

    this.occlusion = ctx.createBiquadFilter();
    this.occlusion.type = 'lowpass';
    this.occlusion.frequency.value = 12000;
    this.occlusion.Q.value = 0.8;

    this.panner = ctx.createPanner();
    this.panner.panningModel = 'HRTF';
    this.panner.distanceModel = 'inverse';
    this.panner.refDistance = 3.2;
    this.panner.maxDistance = 42;
    this.panner.rolloffFactor = 1.25;
    this.panner.positionX.value = this.pos.x;
    this.panner.positionY.value = this.pos.y;
    this.panner.positionZ.value = this.pos.z;

    this.trimGain.connect(this.occlusion);
    this.occlusion.connect(this.panner);
    this.panner.connect(masterGain);

    this.noiseBuf = noiseBuf;
    this._startSynth(spec.synthPreset);
  }

  _clearSources() {
    for (const id of this.timers) clearInterval(id);
    this.timers = [];
    for (const n of this.nodes) {
      try {
        if (typeof n.stop === 'function') n.stop();
      } catch {}
      try {
        n.disconnect();
      } catch {}
    }
    this.nodes = [];
  }

  _loopNoise() {
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuf;
    src.loop = true;
    src.start();
    this.nodes.push(src);
    return src;
  }

  _startSynth(preset) {
    this._clearSources();
    const ctx = this.ctx;
    const out = this.trimGain;

    if (preset === 'portalHum') {
      // Warm harmonic fifth drone with slow resonant filter sweep
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 420;
      filter.Q.value = 4.2;

      const lfo = ctx.createOscillator();
      const lfoGain = ctx.createGain();
      lfo.frequency.value = 0.14;
      lfoGain.gain.value = 190;
      lfo.connect(lfoGain).connect(filter.frequency);
      lfo.start();

      [82.41, 123.47, 164.81, 246.94].forEach((freq, i) => {
        const osc = ctx.createOscillator();
        osc.type = i % 2 === 0 ? 'sawtooth' : 'triangle';
        osc.frequency.value = freq;
        const g = ctx.createGain();
        g.gain.value = i < 2 ? 0.14 : 0.06;
        osc.connect(g).connect(filter);
        osc.start();
        this.nodes.push(osc, g);
      });
      filter.connect(out);
      this.nodes.push(filter, lfo, lfoGain);
    } else if (preset === 'desertWind') {
      const noise = this._loopNoise();
      const bp = ctx.createBiquadFilter();
      bp.type = 'bandpass';
      bp.frequency.value = 540;
      bp.Q.value = 3.2;

      const lfo = ctx.createOscillator();
      const lfoG = ctx.createGain();
      lfo.frequency.value = 0.21;
      lfoG.gain.value = 280;
      lfo.connect(lfoG).connect(bp.frequency);
      lfo.start();

      const g = ctx.createGain();
      g.gain.value = 0.9;
      noise.connect(bp).connect(g).connect(out);
      this.nodes.push(bp, lfo, lfoG, g);
    } else if (preset === 'crystalChimes') {
      const notes = [523.25, 659.25, 783.99, 987.77, 1174.66, 1318.51];
      const ping = () => {
        if (ctx.state !== 'running') return;
        const now = ctx.currentTime;
        const f = notes[Math.floor(Math.random() * notes.length)];
        const osc = ctx.createOscillator();
        const overtone = ctx.createOscillator();
        const env = ctx.createGain();
        osc.type = 'sine';
        overtone.type = 'sine';
        osc.frequency.setValueAtTime(f, now);
        overtone.frequency.setValueAtTime(f * 2.756, now);
        env.gain.setValueAtTime(0.0001, now);
        env.gain.exponentialRampToValueAtTime(0.14, now + 0.02);
        env.gain.exponentialRampToValueAtTime(0.0001, now + 2.4);
        osc.connect(env);
        overtone.connect(env);
        env.connect(out);
        osc.start(now);
        overtone.start(now);
        osc.stop(now + 2.5);
        overtone.stop(now + 2.5);
      };
      ping();
      this.timers.push(setInterval(ping, 1850));
    } else if (preset === 'waterfall') {
      const noise = this._loopNoise();
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 1850;
      const hp = ctx.createBiquadFilter();
      hp.type = 'highpass';
      hp.frequency.value = 140;
      const g = ctx.createGain();
      g.gain.value = 1.1;
      noise.connect(hp).connect(lp).connect(g).connect(out);
      this.nodes.push(lp, hp, g);
    } else if (preset === 'templeBell') {
      const strike = () => {
        if (ctx.state !== 'running') return;
        const now = ctx.currentTime;
        const base = 174.61; // F3 bronze temple bell
        [1, 2.02, 2.98, 4.16].forEach((ratio, idx) => {
          const osc = ctx.createOscillator();
          const env = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(base * ratio, now);
          const amp = 0.16 / (idx + 1);
          const decay = 4.6 / (idx * 0.55 + 1);
          env.gain.setValueAtTime(0.0001, now);
          env.gain.exponentialRampToValueAtTime(amp, now + 0.025);
          env.gain.exponentialRampToValueAtTime(0.0001, now + decay);
          osc.connect(env).connect(out);
          osc.start(now);
          osc.stop(now + decay + 0.1);
        });
      };
      strike();
      this.timers.push(setInterval(strike, 6200));
    } else if (preset === 'nightCrickets') {
      const chirp = () => {
        if (ctx.state !== 'running') return;
        const now = ctx.currentTime;
        for (let b = 0; b < 3; b++) {
          const t0 = now + b * 0.11;
          const osc = ctx.createOscillator();
          const env = ctx.createGain();
          osc.type = 'sine';
          osc.frequency.setValueAtTime(4250 + Math.random() * 180, t0);
          env.gain.setValueAtTime(0.0001, t0);
          env.gain.exponentialRampToValueAtTime(0.055, t0 + 0.015);
          env.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.085);
          osc.connect(env).connect(out);
          osc.start(t0);
          osc.stop(t0 + 0.095);
        }
      };
      chirp();
      this.timers.push(setInterval(chirp, 2100));
    } else if (preset === 'foghorn') {
      const blow = () => {
        if (ctx.state !== 'running') return;
        const now = ctx.currentTime;
        const filter = ctx.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.setValueAtTime(340, now);
        const env = ctx.createGain();
        env.gain.setValueAtTime(0.0001, now);
        env.gain.linearRampToValueAtTime(0.24, now + 0.65);
        env.gain.setValueAtTime(0.22, now + 2.1);
        env.gain.exponentialRampToValueAtTime(0.0001, now + 3.4);
        [65.41, 98.0].forEach((f) => {
          const osc = ctx.createOscillator();
          osc.type = 'sawtooth';
          osc.frequency.setValueAtTime(f, now);
          osc.connect(filter);
          osc.start(now);
          osc.stop(now + 3.5);
        });
        filter.connect(env).connect(out);
      };
      blow();
      this.timers.push(setInterval(blow, 7500));
    } else if (preset === 'oceanSurf') {
      const noise = this._loopNoise();
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 620;
      const g = ctx.createGain();
      g.gain.value = 0.65;

      const swell = ctx.createOscillator();
      const swellFreqG = ctx.createGain();
      const swellAmpG = ctx.createGain();
      swell.frequency.value = 0.16;
      swellFreqG.gain.value = 420;
      swellAmpG.gain.value = 0.35;
      swell.connect(swellFreqG).connect(lp.frequency);
      swell.connect(swellAmpG).connect(g.gain);
      swell.start();

      noise.connect(lp).connect(g).connect(out);
      this.nodes.push(lp, g, swell, swellFreqG, swellAmpG);
    } else {
      // timberCreak / ambient fallback
      const creak = () => {
        if (ctx.state !== 'running') return;
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const bp = ctx.createBiquadFilter();
        const env = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(95 + Math.random() * 40, now);
        osc.frequency.exponentialRampToValueAtTime(62, now + 0.7);
        bp.type = 'bandpass';
        bp.frequency.value = 320;
        bp.Q.value = 6;
        env.gain.setValueAtTime(0.0001, now);
        env.gain.linearRampToValueAtTime(0.11, now + 0.18);
        env.gain.exponentialRampToValueAtTime(0.0001, now + 0.78);
        osc.connect(bp).connect(env).connect(out);
        osc.start(now);
        osc.stop(now + 0.85);
      };
      creak();
      this.timers.push(setInterval(creak, 3100));
    }
  }

  /**
   * Hot-swap this spatial emitter to an ElevenLabs SFX v3 generated MP3 loop!
   */
  async attachAudioUrl(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Audio fetch failed (${res.status})`);
    const arrayBuf = await res.arrayBuffer();
    const audioBuf = await this.ctx.decodeAudioData(arrayBuf);
    this._clearSources();
    const src = this.ctx.createBufferSource();
    src.buffer = audioBuf;
    src.loop = true;
    src.connect(this.trimGain);
    src.start();
    this.nodes.push(src);
    this.sourceLabel = 'ElevenLabs SFX v3';
  }

  updateTelemetry(camPos, camForward, camRight) {
    const toEmitter = this.pos.clone().sub(camPos);
    const dist = Math.max(0.4, toEmitter.length());
    const dir = toEmitter.clone().divideScalar(dist);

    // Directional occlusion: sounds behind the listener or occluded far away get muffled low-pass
    const facing = dir.dot(camForward); // -1..1
    const panHint = dir.dot(camRight);  // -1 (L) .. +1 (R)
    const distMuffle = Math.max(0, Math.min(1, (dist - 3.0) / 22.0));
    const backMuffle = facing < 0 ? Math.abs(facing) * 0.65 : 0;
    const occlusionAmt = Math.min(0.92, distMuffle * 0.55 + backMuffle);
    const cutoffHz = Math.round(14000 * Math.pow(0.045, occlusionAmt));

    this.occlusion.frequency.setTargetAtTime(cutoffHz, this.ctx.currentTime, 0.08);

    const level = Math.min(1, 3.2 / (3.2 + 1.25 * Math.max(0, dist - 3.2)));
    this.telemetry = { distance: dist, level, cutoffHz, panHint };
  }

  dispose() {
    this._clearSources();
    this.trimGain.disconnect();
    this.occlusion.disconnect();
    this.panner.disconnect();
  }
}

export class SpatialAudioEngine {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.noiseBuf = null;
    this.emitters = [];
    this.enabled = false;
    this._pendingSpecs = [];
  }

  async toggle() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx();
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.value = 0.0;
      this.masterGain.connect(this.ctx.destination);
      this.noiseBuf = createNoiseBuffer(this.ctx, 4);
      this._rebuildEmitters(this._pendingSpecs);
    }

    if (this.ctx.state === 'suspended') {
      await this.ctx.resume();
    }

    this.enabled = !this.enabled;
    this.masterGain.gain.setTargetAtTime(this.enabled ? 0.85 : 0.0, this.ctx.currentTime, 0.12);
    return this.enabled;
  }

  setWorldEmitters(specs) {
    this._pendingSpecs = specs;
    if (this.ctx) {
      this._rebuildEmitters(specs);
    }
  }

  _rebuildEmitters(specs) {
    for (const em of this.emitters) em.dispose();
    this.emitters = specs.map((s) => new SpatialEmitter(this.ctx, this.masterGain, this.noiseBuf, s));
  }

  async attachElevenLabsUrl(emitterId, url) {
    if (!this.ctx) await this.toggle();
    const em = this.emitters.find((e) => e.spec.id === emitterId);
    if (em) await em.attachAudioUrl(url);
  }

  update(camera) {
    const camPos = camera.position;
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion).normalize();
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion).normalize();

    if (this.ctx && this.ctx.listener) {
      const L = this.ctx.listener;
      if (L.positionX) {
        L.positionX.setTargetAtTime(camPos.x, this.ctx.currentTime, 0.03);
        L.positionY.setTargetAtTime(camPos.y, this.ctx.currentTime, 0.03);
        L.positionZ.setTargetAtTime(camPos.z, this.ctx.currentTime, 0.03);
        L.forwardX.setTargetAtTime(forward.x, this.ctx.currentTime, 0.03);
        L.forwardY.setTargetAtTime(forward.y, this.ctx.currentTime, 0.03);
        L.forwardZ.setTargetAtTime(forward.z, this.ctx.currentTime, 0.03);
        L.upX.setTargetAtTime(up.x, this.ctx.currentTime, 0.03);
        L.upY.setTargetAtTime(up.y, this.ctx.currentTime, 0.03);
        L.upZ.setTargetAtTime(up.z, this.ctx.currentTime, 0.03);
      }
      for (const em of this.emitters) {
        em.updateTelemetry(camPos, forward, right);
      }
      return this.emitters.map((e) => ({
        id: e.spec.id,
        label: e.spec.label,
        color: e.spec.color,
        position: e.spec.position,
        sourceLabel: e.sourceLabel,
        ...e.telemetry,
      }));
    }

    // Even when audio is muted/unstarted, compute visual radar distances for the HUD!
    return this._pendingSpecs.map((spec) => {
      const p = new THREE.Vector3(...spec.position);
      const toE = p.clone().sub(camPos);
      const dist = Math.max(0.4, toE.length());
      const dir = toE.clone().divideScalar(dist);
      const facing = dir.dot(forward);
      const panHint = dir.dot(right);
      const distMuffle = Math.max(0, Math.min(1, (dist - 3.0) / 22.0));
      const backMuffle = facing < 0 ? Math.abs(facing) * 0.65 : 0;
      const cutoffHz = Math.round(14000 * Math.pow(0.045, Math.min(0.92, distMuffle * 0.55 + backMuffle)));
      const level = Math.min(1, 3.2 / (3.2 + 1.25 * Math.max(0, dist - 3.2)));
      return {
        id: spec.id,
        label: spec.label,
        color: spec.color,
        position: spec.position,
        sourceLabel: 'Muted (click Audio)',
        distance: dist,
        level,
        cutoffHz,
        panHint,
      };
    });
  }
}
