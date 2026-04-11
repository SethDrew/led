import { AudioEngine } from './audio';
import { Renderer } from './renderer';
import { JamSocket } from './ws';
import type { ServerMessage } from './protocol';
import { createEffect, EFFECT_NAMES, type BaseEffect, type EffectName } from './effects';
import type { Topology } from './topology/loader';
import yggdrasilData from './topology/yggdrasil-branches.json';

// --- DOM ---
const audioEl = document.getElementById('audio') as HTMLAudioElement;
const canvas3d = document.getElementById('viewport') as HTMLDivElement;
const statusEl = document.getElementById('status') as HTMLSpanElement;
const usersEl = document.getElementById('users') as HTMLSpanElement;
const effectNameEl = document.getElementById('effect-name') as HTMLSpanElement;
const trackUrlInput = document.getElementById('track-url') as HTMLInputElement;
const playBtn = document.getElementById('play-btn') as HTMLButtonElement;
const tracksBtn = document.getElementById('tracks-btn') as HTMLButtonElement;
const trackList = document.getElementById('track-list') as HTMLDivElement;
const trackItems = document.getElementById('track-items') as HTMLDivElement;
const trackTitleEl = document.getElementById('track-title') as HTMLSpanElement;
const seekBar = document.getElementById('seek-bar') as HTMLInputElement;
const timeDisplay = document.getElementById('time-display') as HTMLSpanElement;
const effectSelect = document.getElementById('effect-select') as HTMLSelectElement;

// --- Audio ---
const audio = new AudioEngine();

let audioConnected = false;
function ensureAudioConnected(): void {
  if (!audioConnected) {
    audio.connectElement(audioEl);
    audioConnected = true;
  }
}

// --- Renderer ---
const topology = yggdrasilData as Topology;
const renderer = new Renderer(canvas3d, topology);

// --- Effects ---
// Populate effect dropdown
for (const name of EFFECT_NAMES) {
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = name;
  effectSelect.appendChild(opt);
}

const effectConfig = {
  numLeds: renderer.numLeds,
  sampleRate: audio.sampleRate,
  topology,
};

let effectIdx = 0;
let currentEffect: BaseEffect = createEffect(EFFECT_NAMES[effectIdx], effectConfig);
effectNameEl.textContent = currentEffect.name;

function switchEffect(name: EffectName): void {
  const idx = EFFECT_NAMES.indexOf(name);
  if (idx === -1) return;
  effectIdx = idx;
  currentEffect = createEffect(name, effectConfig);
  effectNameEl.textContent = currentEffect.name;
  effectSelect.value = name;
}

function cycleEffect(): void {
  effectIdx = (effectIdx + 1) % EFFECT_NAMES.length;
  const name = EFFECT_NAMES[effectIdx];
  switchEffect(name);
  ws.send({ type: 'select_effect', effect: name });
}

// Effect dropdown change
effectSelect.addEventListener('change', () => {
  const name = effectSelect.value as EffectName;
  switchEffect(name);
  ws.send({ type: 'select_effect', effect: name });
});

// Click effect name label to cycle
effectNameEl.style.cursor = 'pointer';
effectNameEl.addEventListener('click', cycleEffect);

// --- WebSocket ---
const ws = new JamSocket();
const userName = localStorage.getItem('jam-username') || (() => {
  const name = prompt('Your name:') || 'anon';
  localStorage.setItem('jam-username', name);
  return name;
})();

ws.onStatus((status) => {
  statusEl.textContent = status;
});

ws.connect();

ws.onceOpen(() => {
  ws.send({ type: 'join', name: userName });
});

// --- Suppress echo: ignore server messages that were triggered by our own actions ---
let suppressPlay = false;
let suppressPause = false;
let suppressSeek = false;

// --- Audius Track Browser ---
const AUDIUS_API = 'https://discoveryprovider.audius.co/v1';
const AUDIUS_APP = 'led-jam';

interface AudiusTrack {
  id: string;
  title: string;
  user: { name: string };
  duration: number;
  artwork?: { '150x150'?: string };
}

let currentTrackTitle = '';

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

async function loadTrendingTracks(): Promise<void> {
  try {
    const res = await fetch(`${AUDIUS_API}/tracks/trending?app_name=${AUDIUS_APP}&limit=20`);
    const json = await res.json();
    const tracks: AudiusTrack[] = json.data || [];
    trackItems.innerHTML = '';
    for (const track of tracks) {
      const div = document.createElement('div');
      div.className = 'track-item';
      div.innerHTML = `
        <span><span style="color:#ccc">${track.title}</span> <span class="artist">— ${track.user.name}</span></span>
        <span class="duration">${formatDuration(track.duration)}</span>
      `;
      div.addEventListener('click', () => playAudiusTrack(track));
      trackItems.appendChild(div);
    }
  } catch (e) {
    trackItems.innerHTML = '<div style="padding:8px 16px;color:#666;font-size:11px;">Failed to load tracks</div>';
  }
}

function playAudiusTrack(track: AudiusTrack): void {
  const streamUrl = `${AUDIUS_API}/tracks/${track.id}/stream?app_name=${AUDIUS_APP}`;
  currentTrackTitle = `${track.title} — ${track.user.name}`;
  trackTitleEl.textContent = currentTrackTitle;
  ensureAudioConnected();
  audioEl.src = streamUrl;
  audioEl.play();
  trackList.style.display = 'none';
}

// Toggle track browser
tracksBtn.addEventListener('click', () => {
  const visible = trackList.style.display !== 'none';
  trackList.style.display = visible ? 'none' : 'block';
  if (!visible && trackItems.children.length === 0) {
    loadTrendingTracks();
  }
});

// Load tracks on startup
loadTrendingTracks();

// --- UI Controls ---

// Play/pause button
playBtn.addEventListener('click', () => {
  ensureAudioConnected();
  const url = trackUrlInput.value.trim();
  if (url && audioEl.src !== url) {
    audioEl.src = url;
  }
  if (audioEl.paused) {
    audioEl.play();
  } else {
    audioEl.pause();
  }
});

// Update play button text
audioEl.addEventListener('play', () => { playBtn.textContent = 'pause'; });
audioEl.addEventListener('pause', () => { playBtn.textContent = 'play'; });

// Seek bar interaction
let seeking = false;
seekBar.addEventListener('mousedown', () => { seeking = true; });
seekBar.addEventListener('touchstart', () => { seeking = true; }, { passive: true });

seekBar.addEventListener('input', () => {
  if (audioEl.duration && isFinite(audioEl.duration)) {
    suppressSeek = true;
    audioEl.currentTime = (parseFloat(seekBar.value) / 100) * audioEl.duration;
  }
});

seekBar.addEventListener('mouseup', () => { seeking = false; });
seekBar.addEventListener('touchend', () => { seeking = false; });

// Track URL: pressing Enter loads and plays
trackUrlInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    const url = trackUrlInput.value.trim();
    if (url) {
      ensureAudioConnected();
      audioEl.src = url;
      audioEl.play();
    }
  }
});

// --- Client → Server: forward local audio control events ---

audioEl.addEventListener('play', () => {
  ensureAudioConnected();
  if (suppressPlay) { suppressPlay = false; return; }
  ws.send({ type: 'play', trackUrl: audioEl.src, position: audioEl.currentTime });
});

audioEl.addEventListener('pause', () => {
  if (suppressPause) { suppressPause = false; return; }
  ws.send({ type: 'pause', position: audioEl.currentTime });
});

audioEl.addEventListener('seeked', () => {
  if (suppressSeek) { suppressSeek = false; return; }
  ws.send({ type: 'seek', position: audioEl.currentTime });
});

// --- Server → Client: handle incoming messages ---

ws.onMessage((msg: ServerMessage) => {
  switch (msg.type) {
    case 'state':
      if (msg.trackUrl) {
        audioEl.src = msg.trackUrl;
        trackUrlInput.value = msg.trackUrl;
      }
      if (msg.playing && msg.trackUrl) {
        ensureAudioConnected();
        suppressPlay = true;
        suppressSeek = true;
        audioEl.currentTime = msg.position;
        audioEl.play();
      }
      usersEl.textContent = msg.users.join(', ');
      if (msg.effect && EFFECT_NAMES.includes(msg.effect as EffectName)) {
        switchEffect(msg.effect as EffectName);
      }
      break;

    case 'play':
      ensureAudioConnected();
      if (audioEl.src !== msg.trackUrl) {
        audioEl.src = msg.trackUrl;
        trackUrlInput.value = msg.trackUrl;
      }
      suppressPlay = true;
      suppressSeek = true;
      audioEl.currentTime = msg.position;
      audioEl.play();
      break;

    case 'pause':
      suppressPause = true;
      suppressSeek = true;
      audioEl.pause();
      audioEl.currentTime = msg.position;
      break;

    case 'seek':
      suppressSeek = true;
      audioEl.currentTime = msg.position;
      break;

    case 'select_effect':
      if (EFFECT_NAMES.includes(msg.effect as EffectName)) {
        switchEffect(msg.effect as EffectName);
      }
      break;

    case 'presence':
      usersEl.textContent = msg.users.join(', ');
      break;

    case 'sync':
      // Nudge playback if drift > 500ms (small drift not noticeable, hard-seek causes glitch)
      if (Math.abs(audioEl.currentTime - msg.position) > 0.5) {
        suppressSeek = true;
        audioEl.currentTime = msg.position;
      }
      break;
  }
});

// --- Time display helper ---
function formatTime(s: number): string {
  if (!isFinite(s)) return '0:00';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

// --- Render loop ---
let lastTime = 0;

function loop(now: number): void {
  const dt = lastTime ? (now - lastTime) / 1000 : 1 / 60;
  lastTime = now;

  // Update seek bar and time display
  if (!seeking && audioEl.duration && isFinite(audioEl.duration)) {
    seekBar.value = String((audioEl.currentTime / audioEl.duration) * 100);
  }
  timeDisplay.textContent = `${formatTime(audioEl.currentTime)} / ${formatTime(audioEl.duration)}`;

  // Feed audio to effect
  const freqData = audio.getFrequencyData();
  const timeData = audio.getTimeDomainData();
  currentEffect.processAudio(freqData, timeData);

  // Get LED colors and push to renderer
  const colors = currentEffect.render(dt);
  renderer.updateLeds(colors);
  renderer.render();

  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);
