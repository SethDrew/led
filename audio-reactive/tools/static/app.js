// ── IndexedDB cache for analysis results ────────────────────────
const cacheDB = (() => {
    let db = null;
    const DB_NAME = 'led-viewer-cache';
    const DB_VERSION = 2;

    function open() {
        if (db) return Promise.resolve(db);
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, DB_VERSION);
            req.onupgradeneeded = (e) => {
                const d = req.result;
                if (!d.objectStoreNames.contains('panels')) d.createObjectStore('panels');
                if (!d.objectStoreNames.contains('audioFiles')) d.createObjectStore('audioFiles');
            };
            req.onsuccess = () => { db = req.result; resolve(db); };
            req.onerror = () => resolve(null);
        });
    }

    return {
        async get(key, store = 'panels') {
            try {
                const d = await open();
                if (!d) return null;
                return new Promise(resolve => {
                    const tx = d.transaction(store, 'readonly');
                    const req = tx.objectStore(store).get(key);
                    req.onsuccess = () => resolve(req.result || null);
                    req.onerror = () => resolve(null);
                });
            } catch { return null; }
        },
        async put(key, value, store = 'panels') {
            try {
                const d = await open();
                if (!d) return;
                const tx = d.transaction(store, 'readwrite');
                tx.objectStore(store).put(value, key);
            } catch {}
        },
        async getAll(store = 'panels') {
            try {
                const d = await open();
                if (!d) return [];
                return new Promise(resolve => {
                    const tx = d.transaction(store, 'readonly');
                    const req = tx.objectStore(store).getAll();
                    const keyReq = tx.objectStore(store).getAllKeys();
                    const results = {};
                    req.onsuccess = () => { results.values = req.result; };
                    keyReq.onsuccess = () => {
                        results.keys = keyReq.result;
                        const out = [];
                        for (let i = 0; i < results.keys.length; i++) {
                            out.push({ key: results.keys[i], value: results.values[i] });
                        }
                        resolve(out);
                    };
                    keyReq.onerror = () => resolve([]);
                });
            } catch { return []; }
        },
        async delete(key, store = 'panels') {
            try {
                const d = await open();
                if (!d) return;
                const tx = d.transaction(store, 'readwrite');
                tx.objectStore(store).delete(key);
            } catch {}
        }
    };
})();

async function cachedFetchPNG(url) {
    const cached = await cacheDB.get(url);
    if (cached) {
        return {
            blob: new Blob([cached.png], {type: 'image/png'}),
            pixelMapping: cached.pixelMapping
        };
    }

    // Bypass browser HTTP cache — our IndexedDB layer is the cache;
    // without this, Cache-Control: max-age=3600 serves stale PNGs
    // after annotations are saved and IndexedDB is cleared.
    const resp = await fetch(url, { cache: 'no-cache' });
    if (!resp.ok) return null;

    const pm = {
        xLeft: parseFloat(resp.headers.get('X-Left-Px')),
        xRight: parseFloat(resp.headers.get('X-Right-Px')),
        pngWidth: parseFloat(resp.headers.get('X-Png-Width')),
        duration: parseFloat(resp.headers.get('X-Duration')),
    };
    const blob = await resp.blob();
    const buf = await blob.arrayBuffer();

    await cacheDB.put(url, { png: buf, pixelMapping: pm });

    return { blob, pixelMapping: pm };
}

async function clearPanelCache(filePath) {
    // Clear all cached panels for a given file path
    const encoded = encodeURIComponent(filePath);
    const all = await cacheDB.getAll('panels');
    for (const { key } of all) {
        if (typeof key === 'string' && key.includes(encoded)) {
            await cacheDB.delete(key, 'panels');
        }
    }
}

// ── Auth ────────────────────────────────────────────────────────
let isAuthenticated = false;
let isPublicMode = false;
const LOCKED_TABS = new Set(['stems']);  // Only Demucs gated
const HIDDEN_TABS_PUBLIC = new Set(['effects']);
const HIDDEN_TABS_LOCAL = new Set(['welcome']);

async function checkAuth() {
    try {
        const resp = await fetch('/api/auth/status');
        const data = await resp.json();
        isAuthenticated = data.authenticated;
        isPublicMode = data.public;
        updateAuthUI();
        // Default to welcome tab on public mode (unless URL hash has a saved state)
        if (isPublicMode) {
            const saved = readHashState();
            if (!saved.tab && !saved.file) {
                currentTab = 'welcome';
                updateTabUI();
                loadPanel();
            }
        }
    } catch {}
}

function updateAuthUI() {
    const authLink = document.getElementById('authLink');
    const authArea = document.getElementById('authArea');
    if (!isPublicMode) {
        authArea.style.display = 'none';
        return;
    }
    authArea.style.display = 'block';
    if (isAuthenticated) {
        authLink.textContent = 'Signed In';
        authLink.classList.add('authed');
    } else {
        authLink.textContent = 'Sign In';
        authLink.classList.remove('authed');
    }
    // Toggle record panel content
    const recordLocal = document.getElementById('recordLocal');
    const recordPublic = document.getElementById('recordPublic');
    if (recordLocal && recordPublic) {
        recordLocal.style.display = isPublicMode ? 'none' : '';
        recordPublic.style.display = isPublicMode ? 'block' : 'none';
    }

    updateLockedTabs();
}

function updateLockedTabs() {
    // Lock/unlock tabs
    document.querySelectorAll('.tab[data-tab], .tab-dropdown-item[data-tab]').forEach(el => {
        const tab = el.dataset.tab;
        if (LOCKED_TABS.has(tab)) {
            if (isPublicMode && !isAuthenticated) {
                el.classList.add('locked');
                el.title = 'Sign in to unlock';
            } else {
                el.classList.remove('locked');
                el.title = '';
            }
        }
        if (HIDDEN_TABS_PUBLIC.has(tab)) {
            el.style.display = isPublicMode ? 'none' : '';
        }
        if (HIDDEN_TABS_LOCAL.has(tab)) {
            el.style.display = isPublicMode ? '' : 'none';
        }
    });
}

function toggleAuthBox() {
    if (isAuthenticated) {
        // Clicking "Signed In" logs out
        fetch('/api/auth/logout', {method: 'POST'}).then(() => {
            isAuthenticated = false;
            updateAuthUI();
        });
        return;
    }
    const box = document.getElementById('authBox');
    box.classList.toggle('open');
    if (box.classList.contains('open')) {
        document.getElementById('passcodeInput').focus();
    }
}

async function submitPasscode() {
    const input = document.getElementById('passcodeInput');
    const errEl = document.getElementById('authError');
    const passcode = input.value.trim();
    if (!passcode) return;

    const resp = await fetch('/api/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({passcode})
    });
    const data = await resp.json();
    if (data.ok) {
        isAuthenticated = true;
        input.value = '';
        errEl.style.display = 'none';
        document.getElementById('authBox').classList.remove('open');
        updateAuthUI();
    } else {
        errEl.textContent = data.error || 'Invalid passcode';
        errEl.style.display = 'block';
    }
}

// Close auth box on outside click
document.addEventListener('click', e => {
    const area = document.getElementById('authArea');
    if (area && !area.contains(e.target)) {
        document.getElementById('authBox').classList.remove('open');
    }
});

const audio = document.getElementById('audio');
const filePicker = document.getElementById('filePicker');
const panelImg = document.getElementById('panelImg');
const cursorLine = document.getElementById('cursorLine');
const imgContainer = document.getElementById('imgContainer');
const timeDisplay = document.getElementById('timeDisplay');
const playBtn = document.getElementById('playBtn');
const progressFill = document.getElementById('progressFill');
const progressThumb = document.getElementById('progressThumb');
const progressTrack = document.getElementById('progressTrack');
const viewer = document.getElementById('viewer');
const infoPanel = document.getElementById('infoPanel');
const volSlider = document.getElementById('volSlider');
const volIcon = document.getElementById('volIcon');

let masterVolume = 0.8;
let mutedVolume = null;  // stashed volume when muted

audio.volume = masterVolume;

volSlider.addEventListener('input', () => {
    masterVolume = parseFloat(volSlider.value);
    mutedVolume = null;
    applyVolume();
    updateVolIcon();
});

volIcon.addEventListener('click', () => {
    if (mutedVolume !== null) {
        masterVolume = mutedVolume;
        mutedVolume = null;
    } else {
        mutedVolume = masterVolume;
        masterVolume = 0;
    }
    volSlider.value = masterVolume;
    applyVolume();
    updateVolIcon();
});

function applyVolume() {
    if (hasStemAudio()) {
        audio.volume = 0;
        Object.entries(stemAudios).forEach(([name, a]) => {
            a.volume = activeStems[name] ? masterVolume : 0;
        });
    } else {
        audio.volume = masterVolume;
    }
}

function updateVolIcon() {
    volIcon.textContent = masterVolume === 0 ? '\u{1F507}' : masterVolume < 0.5 ? '\u{1F509}' : '\u{1F50A}';
}

let currentFile = null;
let currentTab = 'analysis'; // overridden to 'welcome' after checkAuth on public mode
let pixelMapping = null;   // {xLeft, xRight, pngWidth, duration}
let files = [];
let stemsPollTimer = null;

// ── Stem audio ───────────────────────────────────────────────────

let currentStemNames = [];  // set dynamically for demucs or hpss
let stemAudios = {};   // name -> Audio element
let activeStems = {};  // name -> bool

function setupStemAudio(stemNames, basePath) {
    cleanupStemAudio();
    currentStemNames = stemNames;
    stemNames.forEach(name => {
        const a = new Audio(basePath + name + '.wav');
        a.preload = 'auto';
        a.volume = masterVolume;
        stemAudios[name] = a;
        activeStems[name] = true;
    });
    audio.volume = 0;  // mute original, use stems for sound
    updateStemUI();
}

function cleanupStemAudio() {
    Object.values(stemAudios).forEach(a => { a.pause(); a.src = ''; });
    stemAudios = {};
    activeStems = {};
    currentStemNames = [];
    audio.volume = masterVolume;
    document.getElementById('stemStatus').style.display = 'none';
    document.getElementById('controlsHint').innerHTML =
        '<kbd>Space</kbd> play/pause &nbsp;' +
        '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;' +
        'Click panel to seek';
}

function toggleStem(name) {
    if (!activeStems.hasOwnProperty(name)) return;
    activeStems[name] = !activeStems[name];
    if (stemAudios[name]) stemAudios[name].volume = activeStems[name] ? masterVolume : 0;
    updateStemUI();
}

function allStemsOn() {
    currentStemNames.forEach(name => {
        activeStems[name] = true;
        if (stemAudios[name]) stemAudios[name].volume = masterVolume;
    });
    updateStemUI();
}

function updateStemUI() {
    const status = document.getElementById('stemStatus');
    status.style.display = 'block';
    status.innerHTML = currentStemNames.map((name, i) =>
        '<span class="stem-chip ' + (activeStems[name] ? 'active' : 'muted') + '" ' +
        'onclick="toggleStem(\'' + name + '\')">' +
        (i + 1) + ':' + name + '</span>'
    ).join('');
    document.getElementById('controlsHint').innerHTML =
        '<kbd>Space</kbd> play/pause &nbsp;' +
        '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;' +
        '<kbd>1</kbd>-<kbd>' + currentStemNames.length + '</kbd> toggle stems &nbsp;' +
        '<kbd>A</kbd> all on';
}

function syncStemAudios() {
    const t = audio.currentTime;
    Object.values(stemAudios).forEach(a => {
        if (Math.abs(a.currentTime - t) > 0.05) a.currentTime = t;
    });
}

function stemPlay() {
    syncStemAudios();
    Object.values(stemAudios).forEach(a => a.play());
}

function stemPause() {
    Object.values(stemAudios).forEach(a => a.pause());
}

function stemSeek() {
    const t = audio.currentTime;
    const wasPlaying = Object.values(stemAudios).some(a => !a.paused);
    Object.values(stemAudios).forEach(a => { a.currentTime = t; });
}

function hasStemAudio() {
    return Object.keys(stemAudios).length > 0;
}

// ── Feature toggles (analysis/annotations tabs) ─────────────────

let featureState = {rms: false, events: false};

function toggleFeature(name) {
    featureState[name] = !featureState[name];
    updateFeatureUI();
    loadPanel();
}

function updateFeatureUI() {
    const status = document.getElementById('stemStatus');
    if (currentTab !== 'analysis') return;
    status.style.display = 'block';
    status.innerHTML =
        '<span class="stem-chip ' + (featureState.rms ? 'active' : 'muted') + '" ' +
        'onclick="toggleFeature(\'rms\')">E:RMS overlay</span>' +
        '<span class="stem-chip ' + (featureState.events ? 'active' : 'muted') + '" ' +
        'onclick="toggleFeature(\'events\')">V:Events</span>';
    const hints = '<kbd>Space</kbd> play/pause &nbsp;' +
        '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;' +
        'Click to seek &nbsp;' +
        '<kbd>E</kbd> toggle RMS overlay &nbsp;' +
        '<kbd>V</kbd> toggle events';
    document.getElementById('controlsHint').innerHTML = hints;
}

// ── Annotation recording ─────────────────────────────────────────

let annotationTaps = [];

function recordTap() {
    if (audio.paused || !pixelMapping) return;
    const t = parseFloat(audio.currentTime.toFixed(3));
    annotationTaps.push(t);
    updateAnnotationUI();
    drawTapMarkers();
}

function drawTapMarkers() {
    const canvas = document.getElementById('tapCanvas');
    if (!canvas || !pixelMapping) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (annotationTaps.length === 0) return;

    const scale = panelImg.clientWidth / pixelMapping.pngWidth;
    ctx.strokeStyle = 'rgba(233, 69, 96, 0.8)';
    ctx.lineWidth = 2;

    annotationTaps.forEach(t => {
        const frac = t / pixelMapping.duration;
        const px = (pixelMapping.xLeft + frac * (pixelMapping.xRight - pixelMapping.xLeft)) * scale;
        ctx.beginPath();
        ctx.moveTo(px, 0);
        ctx.lineTo(px, canvas.height);
        ctx.stroke();
    });
}

function resizeTapCanvas() {
    const canvas = document.getElementById('tapCanvas');
    if (!canvas || !panelImg.clientWidth) return;
    canvas.width = panelImg.clientWidth;
    canvas.height = panelImg.clientHeight;
    drawTapMarkers();
}

async function saveAnnotation() {
    if (annotationTaps.length === 0) return;
    const layer = document.getElementById('layerInput').value.trim();
    if (!layer) { alert('Enter a layer name'); return; }

    const sortedTaps = annotationTaps.slice().sort((a, b) => a - b);

    if (isPublicMode) {
        // Public mode: save to IndexedDB only (per-user, no server YAML)
        const annKey = 'ann:' + currentFile;
        const existing = await cacheDB.get(annKey, 'audioFiles') || {};
        existing[layer] = sortedTaps;
        await cacheDB.put(annKey, existing, 'audioFiles');
        annotationTaps = [];
        updateAnnotationUI();
        drawTapMarkers();
        return;
    }

    const resp = await fetch('/api/annotations/' + encodeURIComponent(currentFile), {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ layer: layer, taps: sortedTaps })
    });

    if (resp.ok) {
        // Also cache annotations in IndexedDB for persistence
        const annKey = 'ann:' + currentFile;
        const existing = await cacheDB.get(annKey, 'audioFiles') || {};
        existing[layer] = sortedTaps;
        await cacheDB.put(annKey, existing, 'audioFiles');

        annotationTaps = [];
        updateAnnotationUI();
        drawTapMarkers();
        // Update file info to reflect new annotations
        const fileInfo = files.find(f => f.path === currentFile);
        if (fileInfo) fileInfo.has_annotations = true;
        // Clear cached panels so annotation render is fresh
        await clearPanelCache(currentFile);
        // Re-render annotations tab to show saved data
        loadPanel();
    } else {
        alert('Save failed: ' + (await resp.text()));
    }
}

function discardAnnotation() {
    annotationTaps = [];
    updateAnnotationUI();
    drawTapMarkers();
}

function updateAnnotationUI() {
    const tapCount = document.getElementById('tapCount');
    const saveBtn = document.getElementById('saveAnnBtn');
    const discardBtn = document.getElementById('discardAnnBtn');
    if (!tapCount) return;
    tapCount.textContent = annotationTaps.length + ' taps';
    saveBtn.disabled = annotationTaps.length === 0;
    discardBtn.disabled = annotationTaps.length === 0;
}

panelImg.addEventListener('load', resizeTapCanvas);
window.addEventListener('resize', resizeTapCanvas);

// Re-render when annotations widget is toggled
document.getElementById('annotationWidget').addEventListener('toggle', () => {
    if (currentTab === 'analysis' && currentFile) loadPanel();
});

// ── Recording ────────────────────────────────────────────────────

let isRecording = false;
let recordStartTime = null;
let recordTimer = null;
let levelPollTimer = null;

async function toggleRecord() {
    if (isRecording) {
        await stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    const resp = await fetch('/api/record/start', {method: 'POST'});
    const data = await resp.json();
    if (!data.ok) {
        document.getElementById('recordStatus').textContent = 'Error: ' + (data.error || 'failed');
        return;
    }
    isRecording = true;
    recordStartTime = Date.now();
    document.getElementById('recordBtn').classList.add('recording');
    document.getElementById('recordStatus').textContent = 'Recording... click to stop';
    recordTimer = setInterval(() => {
        const elapsed = (Date.now() - recordStartTime) / 1000;
        const m = Math.floor(elapsed / 60);
        const s = (elapsed - m * 60).toFixed(1).padStart(4, '0');
        document.getElementById('recordElapsed').textContent = m + ':' + s;
    }, 100);
    startLevelPolling();
}

async function stopRecording() {
    clearInterval(recordTimer);
    stopLevelPolling();
    document.getElementById('recordBtn').classList.remove('recording');
    document.getElementById('recordStatus').textContent = 'Saving...';

    const name = document.getElementById('recordName').value.trim() || '';
    const resp = await fetch('/api/record/stop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name})
    });
    const data = await resp.json();
    isRecording = false;

    if (data.ok) {
        document.getElementById('recordStatus').textContent =
            'Saved: ' + data.filename + ' (' + data.duration + 's)';
        document.getElementById('recordElapsed').textContent = '0:00.0';
        document.getElementById('recordName').value = '';
        await loadFileList();
        currentTab = 'analysis';
        updateTabUI();
        await selectFile(data.filename);
    } else {
        document.getElementById('recordStatus').textContent = 'Error: ' + (data.error || 'failed');
    }
}

function startLevelPolling() {
    stopLevelPolling();
    levelPollTimer = setInterval(pollLevel, 70);
}

function stopLevelPolling() {
    if (levelPollTimer) { clearInterval(levelPollTimer); levelPollTimer = null; }
}

async function pollLevel() {
    try {
        const resp = await fetch('/api/record/level');
        const data = await resp.json();
        if (!data.recording || !data.waveform) return;
        drawRecordWaveform(data.waveform);
        // Update level bar + dB readout
        const rms = data.rms || 0;
        const pct = Math.min(100, rms * 300);  // scale for visibility (full scale ~0.33)
        document.getElementById('recordLevelFill').style.width = pct + '%';
        const db = rms > 0 ? (20 * Math.log10(rms)).toFixed(1) : '-Inf';
        document.getElementById('recordLevelDb').textContent = db + ' dB';
    } catch (e) {}
}

function drawRecordWaveform(waveform) {
    const canvas = document.getElementById('recordWaveform');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);

    // Background
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    // Center line
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (waveform.length === 0) return;

    // Draw mirrored peak envelope (bars)
    const barW = w / waveform.length;
    ctx.fillStyle = '#e94560';
    for (let i = 0; i < waveform.length; i++) {
        const amp = Math.min(1, waveform[i] * 3);  // scale for visibility
        const barH = amp * h * 0.45;
        const x = i * barW;
        ctx.fillRect(x, h / 2 - barH, Math.max(1, barW - 0.5), barH * 2);
    }
}

// ── Browser recording (public mode) ──────────────────────────────

let browserStream = null;
let browserAudioCtx = null;
let browserProcessor = null;
let browserAnalyser = null;
let browserChunks = [];
let browserRecording = false;
let browserRecordStart = null;
let browserRecordTimer = null;
let browserAnimFrame = null;

async function populateAudioDevices() {
    const select = document.getElementById('audioDeviceSelect');
    if (!select) return;
    try {
        // Request permission first so device labels are visible
        await navigator.mediaDevices.getUserMedia({ audio: true }).then(s => s.getTracks().forEach(t => t.stop()));
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices.filter(d => d.kind === 'audioinput');
        select.innerHTML = '';
        audioInputs.forEach(d => {
            const opt = document.createElement('option');
            opt.value = d.deviceId;
            opt.textContent = d.label || ('Microphone ' + (select.options.length + 1));
            select.appendChild(opt);
        });
    } catch (e) {
        select.innerHTML = '<option>Microphone access denied</option>';
    }
}

async function toggleBrowserRecord() {
    if (browserRecording) {
        await stopBrowserRecord();
    } else {
        await startBrowserRecord();
    }
}

async function startBrowserRecord() {
    const select = document.getElementById('audioDeviceSelect');
    const deviceId = select ? select.value : undefined;
    const status = document.getElementById('browserRecordStatus');

    try {
        browserStream = await navigator.mediaDevices.getUserMedia({
            audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        });
    } catch (e) {
        status.textContent = 'Microphone access denied. Check browser permissions.';
        return;
    }

    browserAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
    const source = browserAudioCtx.createMediaStreamSource(browserStream);

    // Analyser for live waveform
    browserAnalyser = browserAudioCtx.createAnalyser();
    browserAnalyser.fftSize = 2048;
    source.connect(browserAnalyser);

    // ScriptProcessor to capture raw PCM
    browserProcessor = browserAudioCtx.createScriptProcessor(4096, 1, 1);
    browserChunks = [];
    browserProcessor.onaudioprocess = (e) => {
        if (browserRecording) {
            browserChunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
        }
    };
    source.connect(browserProcessor);
    browserProcessor.connect(browserAudioCtx.destination);

    browserRecording = true;
    browserRecordStart = Date.now();
    document.getElementById('browserRecordBtn').classList.add('recording');
    status.textContent = 'Recording... click to stop';

    browserRecordTimer = setInterval(() => {
        const elapsed = (Date.now() - browserRecordStart) / 1000;
        const m = Math.floor(elapsed / 60);
        const s = (elapsed - m * 60).toFixed(1).padStart(4, '0');
        document.getElementById('browserRecordElapsed').textContent = m + ':' + s;
    }, 100);

    // Animate waveform
    function drawLive() {
        if (!browserRecording) return;
        const canvas = document.getElementById('browserRecordWaveform');
        if (!canvas || !browserAnalyser) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        ctx.scale(dpr, dpr);

        const bufLen = browserAnalyser.fftSize;
        const data = new Float32Array(bufLen);
        browserAnalyser.getFloatTimeDomainData(data);

        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, w, h);
        ctx.strokeStyle = '#333';
        ctx.beginPath(); ctx.moveTo(0, h/2); ctx.lineTo(w, h/2); ctx.stroke();

        ctx.strokeStyle = '#e94560';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < bufLen; i++) {
            const x = (i / bufLen) * w;
            const y = (0.5 + data[i] * 0.45) * h;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Update level meter
        let sum = 0;
        for (let i = 0; i < data.length; i++) sum += data[i] * data[i];
        const rms = Math.sqrt(sum / data.length);
        const pct = Math.min(100, rms * 300);
        document.getElementById('browserLevelFill').style.width = pct + '%';
        const db = rms > 0 ? (20 * Math.log10(rms)).toFixed(1) : '-Inf';
        document.getElementById('browserLevelDb').textContent = db + ' dB';

        browserAnimFrame = requestAnimationFrame(drawLive);
    }
    drawLive();
}

async function stopBrowserRecord() {
    browserRecording = false;
    clearInterval(browserRecordTimer);
    if (browserAnimFrame) cancelAnimationFrame(browserAnimFrame);

    const status = document.getElementById('browserRecordStatus');
    const btn = document.getElementById('browserRecordBtn');
    btn.classList.remove('recording');
    status.textContent = 'Encoding WAV...';

    // Stop audio pipeline
    if (browserProcessor) { browserProcessor.disconnect(); browserProcessor = null; }
    if (browserAnalyser) { browserAnalyser.disconnect(); browserAnalyser = null; }
    if (browserAudioCtx) { browserAudioCtx.close(); browserAudioCtx = null; }
    if (browserStream) { browserStream.getTracks().forEach(t => t.stop()); browserStream = null; }

    // Encode WAV
    const wavBlob = encodeBrowserWAV(browserChunks, 44100);
    browserChunks = [];

    const nameInput = document.getElementById('browserRecordName');
    const name = (nameInput.value.trim() || 'recording_' + new Date().toISOString().slice(0,19).replace(/[:-]/g, '')) + '.wav';

    status.textContent = 'Uploading ' + name + '...';

    try {
        const data = await uploadWavBlob(wavBlob, name);
        if (data.ok) {
            const dur = (data.duration || 0).toFixed(1);
            status.textContent = 'Saved: ' + data.name + ' (' + dur + 's)';
            document.getElementById('browserRecordElapsed').textContent = '0:00.0';
            nameInput.value = '';
            await loadFileList(data.path);
        } else {
            status.textContent = 'Error: ' + (data.error || 'upload failed');
        }
    } catch (e) {
        status.textContent = 'Upload failed: ' + e.message;
    }
}

function encodeBrowserWAV(chunks, sampleRate) {
    let totalLength = 0;
    for (const c of chunks) totalLength += c.length;
    const samples = new Float32Array(totalLength);
    let offset = 0;
    for (const c of chunks) { samples.set(c, offset); offset += c.length; }

    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeStr(off, str) { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); }

    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);

    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

// Populate devices when record tab is shown
if (isPublicMode) populateAudioDevices();

// ── File picker ──────────────────────────────────────────────────

async function loadFileList(selectPath) {
    // Get user's local files from IndexedDB
    const localFiles = await cacheDB.getAll('audioFiles');
    const localPaths = new Set();
    for (const { key } of localFiles) {
        if (typeof key === 'string' && !key.startsWith('ann:')) localPaths.add(key);
    }

    // In public mode, tell the server which files we own so it only returns those
    let url = '/api/files';
    if (isPublicMode && localPaths.size > 0) {
        url += '?paths=' + encodeURIComponent([...localPaths].join(','));
    }
    const resp = await fetch(url);
    files = await resp.json();

    // Merge with locally cached files that may have been deleted from server
    const serverPaths = new Set(files.map(f => f.path));
    const serverNames = new Set(files.map(f => f.name));
    for (const { key, value } of localFiles) {
        if (typeof key === 'string' && key.startsWith('ann:')) continue;
        if (serverPaths.has(key)) continue;  // Already on server with same path
        // Check if server has this file under a sanitized name (stale IndexedDB entry)
        if (value.name && serverNames.has(value.name)) {
            cacheDB.delete(key, 'audioFiles');  // Clean up stale entry
            continue;
        }
        if (!value.name) continue;  // Skip malformed entries
        files.push({
            name: value.name,
            path: key,
            duration: 0,
            has_annotations: false,
            group: 'your files',
        });
    }

    // Group by group name
    const groups = {};
    files.forEach(f => {
        if (!groups[f.group]) groups[f.group] = [];
        groups[f.group].push(f);
    });

    filePicker.innerHTML = '';
    for (const [group, items] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = group;
        items.forEach(f => {
            const opt = document.createElement('option');
            opt.value = f.path;
            const dur = f.duration ? formatTime(f.duration) : '?';
            const ann = f.has_annotations ? ' [ann]' : '';
            opt.textContent = f.name + ' (' + dur + ')' + ann;
            optgroup.appendChild(opt);
        });
        filePicker.appendChild(optgroup);
    }

    if (files.length > 0) {
        if (selectPath) {
            selectFile(selectPath);
        } else if (currentTab === 'welcome') {
            // Don't auto-select a file on the welcome tab — add placeholder option
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = 'Select a file...';
            placeholder.disabled = true;
            placeholder.selected = true;
            filePicker.insertBefore(placeholder, filePicker.firstChild);
        } else {
            const saved = readHashState();
            const savedFile = saved.file && files.find(f => f.path === saved.file);
            if (saved.tab) {
                currentTab = saved.tab;
                updateTabUI();
            }
            selectFile(savedFile ? savedFile.path : files[0].path);
        }
    }
}

async function selectFile(path) {
    cleanupStemAudio();
    audio.pause();
    playBtn.innerHTML = '&#9654;';
    audio.currentTime = 0;
    currentFile = path;
    filePicker.value = path;

    // Clear pending taps when switching files
    annotationTaps = [];
    updateAnnotationUI();

    // Ensure file exists on server (re-upload from IndexedDB if needed)
    if (path.startsWith('uploads/')) {
        await ensureFileOnServer(path);
    }

    // Sync annotations: fetch from server, cache in IndexedDB (local mode only)
    if (!isPublicMode) {
        try {
            const annResp = await fetch('/api/annotations/' + encodeURIComponent(path));
            if (annResp.ok) {
                const annData = await annResp.json();
                if (Object.keys(annData).length > 0) {
                    await cacheDB.put('ann:' + path, annData, 'audioFiles');
                }
            }
        } catch {}
    }

    // Set audio source
    audio.src = '/audio/' + encodeURIComponent(path);
    audio.load();

    saveHashState();
    loadPanel();
}

filePicker.addEventListener('change', () => {
    if (currentTab === 'welcome') {
        currentTab = 'analysis';
        updateTabUI();
    }
    selectFile(filePicker.value);
});

// ── Tabs ─────────────────────────────────────────────────────────

const decompTabs = new Set(['stems', 'hpss', 'lab-repet', 'lab-nmf']);
const decompDropdown = document.querySelector('.tab-dropdown');
const decompToggle = document.getElementById('decompDropdownToggle');

function updateTabUI() {
    document.querySelectorAll('.tabs > .tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === currentTab);
    });
    // Decomposition dropdown: highlight toggle if a decomp sub-tab is active
    decompToggle.classList.toggle('active', decompTabs.has(currentTab));
    document.querySelectorAll('.tab-dropdown-item').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === currentTab);
    });
}

function switchTab(tabId) {
    // Block locked tabs
    if (isPublicMode && !isAuthenticated && LOCKED_TABS.has(tabId)) return;
    const prev = currentTab;
    if ((prev === 'stems' || prev === 'hpss' || prev === 'lab-repet' || prev === 'lab-nmf') && tabId !== prev) cleanupStemAudio();
    if (prev === 'effects' && tabId !== 'effects') stopEffectsPoll();
    currentTab = tabId;
    updateTabUI();
    saveHashState();
    loadPanel();
}

// Regular tab clicks
document.querySelectorAll('.tabs > .tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (tab.classList.contains('disabled')) return;
        if (tab.id === 'decompDropdownToggle') return; // handled separately
        switchTab(tab.dataset.tab);
    });
});

// Decomposition dropdown toggle
decompToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    decompDropdown.classList.toggle('open');
});

// Decomposition dropdown item clicks
document.querySelectorAll('.tab-dropdown-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.stopPropagation();
        decompDropdown.classList.remove('open');
        switchTab(item.dataset.tab);
    });
});

// Close dropdown on outside click
document.addEventListener('click', () => decompDropdown.classList.remove('open'));

// ── Panel loading ────────────────────────────────────────────────

async function loadPanel() {
    hideOverlay();

    // Annotation widget visibility (only on annotate tab, always expanded)
    const annWidget = document.getElementById('annotationWidget');
    annWidget.style.display = (currentTab === 'annotate') ? '' : 'none';
    if (currentTab === 'annotate') annWidget.open = true;

    const recordPanel = document.getElementById('recordPanel');
    const effectsPanel = document.getElementById('effectsPanel');
    const welcomePanel = document.getElementById('welcomePanel');

    if (currentTab === 'welcome') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'none';
        recordPanel.style.display = 'none';
        effectsPanel.style.display = 'none';
        welcomePanel.style.display = 'flex';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML = '';
        return;
    }
    welcomePanel.style.display = 'none';

    if (currentTab === 'reference') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'block';
        recordPanel.style.display = 'none';
        effectsPanel.style.display = 'none';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML =
            'Reference information about the analysis panels and features';
        return;
    }
    if (currentTab === 'record') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'none';
        recordPanel.style.display = 'flex';
        effectsPanel.style.display = 'none';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML =
            isPublicMode ? 'Record from microphone or system audio' : 'Record audio from BlackHole loopback';
        if (isPublicMode) populateAudioDevices();
        renderFileManager();
        return;
    }
    if (currentTab === 'effects') {
        imgContainer.style.display = 'none';
        infoPanel.style.display = 'none';
        recordPanel.style.display = 'none';
        effectsPanel.style.display = 'flex';
        cursorLine.style.display = 'none';
        document.getElementById('stemStatus').style.display = 'none';
        document.getElementById('controlsHint').innerHTML =
            'Browse, start/stop, rate, and reorder audio-reactive LED effects';
        // Only load effects list if panel is empty (first visit or tab switch).
        // File changes while already on this tab should NOT rebuild the panel —
        // effects don't depend on the current file, and rebuilding destroys the
        // detail view and live feature sparklines.
        if (!effectsPanel.querySelector('.effects-list') && !effectDetailName) {
            loadEffects();
        } else if (effectDetailName) {
            // In detail view: update the file picker to match the new file
            const fileSel = document.getElementById('effectDetailFile');
            if (fileSel && currentFile) fileSel.value = currentFile;
        }
        return;
    }
    imgContainer.style.display = 'inline-block';
    infoPanel.style.display = 'none';
    recordPanel.style.display = 'none';
    effectsPanel.style.display = 'none';

    if (!currentFile) return;

    const COMPUTE_TABS = {
        'stems': { label: 'Compute Stems (Demucs)', desc: 'Deep learning source separation — may take 30+ seconds', fn: loadStems },
        'hpss': { label: 'Compute HPSS', desc: 'Harmonic-percussive separation', fn: loadHPSS },
        'lab': { label: 'Compute Lab', desc: 'Audio feature analysis', fn: loadLab, dropdown: true },
        'lab-repet': { label: 'Compute REPET', desc: 'Repeating pattern extraction', fn: loadLabRepet },
        'lab-nmf': { label: 'Compute NMF', desc: 'Non-negative matrix factorization separation', fn: loadLabNMF },
    };

    if (currentTab === 'annotate') {
        // Annotate tab: waveform + spectrogram + annotation overlay only
        const hints = '<kbd>Space</kbd> play/pause &nbsp;' +
            '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp;' +
            'Click to seek &nbsp;' +
            '<kbd>T</kbd> tap &nbsp;<kbd>S</kbd> save';
        document.getElementById('controlsHint').innerHTML = hints;
        document.getElementById('stemStatus').style.display = 'none';

        const url = '/api/render-annotate/' + encodeURIComponent(currentFile);
        showOverlay('Rendering...');
        try {
            const result = await cachedFetchPNG(url);
            if (!result) { showOverlay('Render failed'); return; }
            pixelMapping = result.pixelMapping;
            panelImg.src = URL.createObjectURL(result.blob);
            hideOverlay();
            cursorLine.style.display = 'block';
        } catch (e) {
            showOverlay('Error: ' + e.message);
        }
        return;
    }

    if (currentTab === 'events') {
        document.getElementById('controlsHint').innerHTML =
            '<kbd>Space</kbd> play/pause &nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp; Click to seek &nbsp; ' +
            'Drops &middot; Risers &middot; Dropouts &middot; Harmonic';
        document.getElementById('stemStatus').style.display = 'none';

        const url = '/api/render-events/' + encodeURIComponent(currentFile);
        showOverlay('Rendering events...');
        try {
            const result = await cachedFetchPNG(url);
            if (!result) { showOverlay('Render failed'); return; }
            pixelMapping = result.pixelMapping;
            panelImg.src = URL.createObjectURL(result.blob);
            hideOverlay();
            cursorLine.style.display = 'block';
        } catch (e) {
            showOverlay('Error: ' + e.message);
        }
        return;
    }

    if (currentTab === 'band-analysis') {
        document.getElementById('controlsHint').innerHTML =
            '<kbd>Space</kbd> play/pause &nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp; Click to seek &nbsp; ' +
            'RT View &middot; Band Share &middot; Context Deviation &middot; RT Derivative &middot; 5s Integral';
        document.getElementById('stemStatus').style.display = 'none';

        const url = '/api/render-band-analysis/' + encodeURIComponent(currentFile);
        showOverlay('Rendering band analysis...');
        try {
            const result = await cachedFetchPNG(url);
            if (!result) { showOverlay('Render failed'); return; }
            pixelMapping = result.pixelMapping;
            panelImg.src = URL.createObjectURL(result.blob);
            hideOverlay();
            cursorLine.style.display = 'block';
        } catch (e) {
            showOverlay('Error: ' + e.message);
        }
        return;
    }

    if (currentTab === 'calculus') {
        document.getElementById('controlsHint').innerHTML =
            '<kbd>Space</kbd> play/pause &nbsp; <kbd>&larr;</kbd> <kbd>&rarr;</kbd> &plusmn;5s &nbsp; Click to seek &nbsp; ' +
            'Energy+Integral &middot; Slope &middot; Curvature &middot; Multi-Scale &middot; Onset d² &middot; Jitter &middot; Build Detector';
        document.getElementById('stemStatus').style.display = 'none';

        const url = '/api/render-calculus/' + encodeURIComponent(currentFile);
        showOverlay('Rendering calculus...');
        try {
            const result = await cachedFetchPNG(url);
            if (!result) { showOverlay('Render failed'); return; }
            pixelMapping = result.pixelMapping;
            panelImg.src = URL.createObjectURL(result.blob);
            hideOverlay();
            cursorLine.style.display = 'block';
        } catch (e) {
            showOverlay('Error: ' + e.message);
        }
        return;
    }

    if (COMPUTE_TABS[currentTab]) {
        // Demucs requires too much RAM for the public server
        if (currentTab === 'stems' && isPublicMode) {
            showDemucsUnavailable();
            return;
        }
        const info = COMPUTE_TABS[currentTab];
        showComputePrompt(info.label, info.desc, info.fn, info.dropdown);
        return;
    }

    // Show feature toggle UI for analysis tab
    updateFeatureUI();

    let url = '/api/render/' + encodeURIComponent(currentFile);
    const params = [];
    const activeFeatures = Object.entries(featureState)
        .filter(([_, v]) => v).map(([k]) => k);
    if (activeFeatures.length < 4) {
        params.push('features=' + activeFeatures.join(','));
    }
    if (params.length) url += '?' + params.join('&');

    showOverlay('Rendering...');

    try {
        const result = await cachedFetchPNG(url);
        if (!result) { showOverlay('Render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadStems() {
    if (!currentFile) return;

    // Check if stems are ready
    const statusResp = await fetch('/api/stems/status/' + encodeURIComponent(currentFile));
    const status = await statusResp.json();

    if (!status.ready) {
        showOverlay('Running demucs...');
        // Trigger demucs and poll
        fetch('/api/stems/' + encodeURIComponent(currentFile));
        pollStemsReady();
        return;
    }

    showOverlay('Rendering stems...');

    try {
        const stemUrl = '/api/stems/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(stemUrl);
        if (!result) { showOverlay('Stems render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['drums', 'bass', 'vocals', 'other'],
                       '/audio/separated/htdemucs/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

function pollStemsReady() {
    if (stemsPollTimer) clearInterval(stemsPollTimer);
    stemsPollTimer = setInterval(async () => {
        const resp = await fetch('/api/stems/status/' + encodeURIComponent(currentFile));
        const status = await resp.json();
        if (status.ready) {
            clearInterval(stemsPollTimer);
            stemsPollTimer = null;
            if (currentTab === 'stems') loadStems();
        }
    }, 2000);
}

async function loadHPSS() {
    if (!currentFile) return;
    showOverlay('Computing HPSS...');

    try {
        const hpssUrl = '/api/hpss/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(hpssUrl);
        if (!result) { showOverlay('HPSS render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['harmonic', 'percussive'],
                       '/audio/separated/hpss/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadLab() {
    if (!currentFile) return;
    const v = LAB_VARIANTS.find(x => x.value === labVariant);
    showOverlay('Computing ' + (v ? v.label : 'lab') + '...');
    const hint = labVariant === 'timbral'
        ? 'MFCC 0-3 &middot; Fine Texture &middot; Timbral Shift'
        : 'Spectral Flatness &middot; Chromagram &middot; Spectral Contrast &middot; ZCR';
    document.getElementById('controlsHint').innerHTML =
        '<kbd>Space</kbd> play/pause &nbsp; Click to seek &nbsp; ' + hint;
    document.getElementById('stemStatus').style.display = 'none';

    try {
        const labUrl = '/api/lab/' + encodeURIComponent(currentFile) + '?variant=' + labVariant;
        const result = await cachedFetchPNG(labUrl);
        if (!result) { showOverlay('Lab render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadLabNMF() {
    if (!currentFile) return;
    showOverlay('Running NMF decomposition...');
    document.getElementById('stemStatus').style.display = 'none';

    try {
        const nmfUrl = '/api/lab-nmf/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(nmfUrl);
        if (!result) { showOverlay('NMF render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['drums', 'bass', 'vocals', 'other'],
                       '/audio/separated/nmf/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

async function loadLabRepet() {
    if (!currentFile) return;
    showOverlay('Computing REPET separation...');

    try {
        const repetUrl = '/api/lab-repet/' + encodeURIComponent(currentFile);
        const result = await cachedFetchPNG(repetUrl);
        if (!result) { showOverlay('REPET render failed'); return; }
        pixelMapping = result.pixelMapping;
        panelImg.src = URL.createObjectURL(result.blob);
        hideOverlay();
        cursorLine.style.display = 'block';
        const stemName = currentFile.split('/').pop().replace('.wav', '');
        setupStemAudio(['repeating', 'non-repeating'],
                       '/audio/separated/repet/' + stemName + '/');
    } catch (e) {
        showOverlay('Error: ' + e.message);
    }
}

// ── Overlay ──────────────────────────────────────────────────────

function showOverlay(text) {
    let overlay = viewer.querySelector('.overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'overlay';
        overlay.innerHTML = '<span class="overlay-text"></span>';
        viewer.appendChild(overlay);
    }
    overlay.querySelector('.overlay-text').textContent = text;
    overlay.style.display = 'flex';
}

function hideOverlay() {
    const overlay = viewer.querySelector('.overlay');
    if (overlay) overlay.style.display = 'none';
}

const LAB_VARIANTS = [
    { value: 'timbral', label: 'Timbral Shape (MFCC)', desc: 'MFCC coefficients broken out — energy, tilt, curvature, texture, and timbral shift' },
    { value: 'misc', label: 'Misc (Chroma, Flatness, Contrast, ZCR)', desc: 'Spectral flatness, chromagram, spectral contrast, zero crossing rate' },
];
let labVariant = 'timbral';

function showComputePrompt(label, desc, fn, dropdown) {
    let overlay = viewer.querySelector('.overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'overlay';
        viewer.appendChild(overlay);
    }
    let dropdownHtml = '';
    if (dropdown) {
        const options = LAB_VARIANTS.map(v =>
            `<option value="${v.value}"${v.value === labVariant ? ' selected' : ''}>${v.label}</option>`
        ).join('');
        dropdownHtml = `<select id="labVariantSelect" style="
            margin-bottom:12px; padding:8px 12px; font-size:14px;
            background:#222; color:#eee; border:1px solid #555; border-radius:6px;
            cursor:pointer; width:100%; max-width:400px;
        ">${options}</select>`;
        desc = LAB_VARIANTS.find(v => v.value === labVariant).desc;
    }
    overlay.innerHTML = `
        <div class="compute-prompt">
            ${dropdownHtml}
            <button class="compute-btn" id="computeBtn">${label}</button>
            <p class="compute-desc" id="computeDesc">${desc}</p>
        </div>`;
    overlay.style.display = 'flex';
    if (dropdown) {
        const sel = document.getElementById('labVariantSelect');
        sel.onchange = () => {
            labVariant = sel.value;
            const v = LAB_VARIANTS.find(x => x.value === labVariant);
            document.getElementById('computeDesc').textContent = v ? v.desc : '';
        };
    }
    document.getElementById('computeBtn').onclick = () => {
        overlay.innerHTML = '<span class="overlay-text"></span>';
        fn();
    };
}

function showDemucsUnavailable() {
    let overlay = viewer.querySelector('.overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'overlay';
        viewer.appendChild(overlay);
    }
    overlay.innerHTML = `
        <div class="compute-prompt">
            <p style="font-size:1.1em;margin-bottom:12px;">Demucs source separation requires ~4GB RAM — more than the demo server provides.</p>
            <p class="compute-desc">Run locally with Docker for full Demucs support:</p>
            <pre style="background:#1a1a2e;padding:12px;border-radius:6px;margin:12px 0;font-size:0.85em;text-align:left;overflow-x:auto;">docker run -p 8080:8080 -v ~/Music:/app/audio-reactive/research/audio-segments ghcr.io/sethdrew/led-viewer</pre>
            <a href="https://github.com/SethDrew/led#run-locally-with-docker" target="_blank"
               style="color:#4fc3f7;text-decoration:underline;">Setup instructions on GitHub</a>
        </div>`;
    overlay.style.display = 'flex';
}

// ── Cursor sync ──────────────────────────────────────────────────

let lastStemSync = 0;

function updateCursor() {
    if (pixelMapping && panelImg.naturalWidth > 0) {
        const t = audio.currentTime;
        const dur = pixelMapping.duration;
        const frac = dur > 0 ? t / dur : 0;
        const scale = panelImg.clientWidth / pixelMapping.pngWidth;
        const px = (pixelMapping.xLeft + frac * (pixelMapping.xRight - pixelMapping.xLeft)) * scale;
        cursorLine.style.left = px + 'px';

        // Progress bar
        const pct = dur > 0 ? (t / dur) * 100 : 0;
        progressFill.style.width = pct + '%';
        progressThumb.style.left = pct + '%';

        // Time display
        timeDisplay.textContent = formatTime(t) + ' / ' + formatTime(dur);

        // Continuous stem sync — check every ~300ms during playback
        if (hasStemAudio() && !audio.paused) {
            const now = performance.now();
            if (now - lastStemSync > 300) {
                lastStemSync = now;
                syncStemAudios();
            }
        }
    }
    requestAnimationFrame(updateCursor);
}
requestAnimationFrame(updateCursor);

// ── Click to seek ────────────────────────────────────────────────

panelImg.addEventListener('click', (e) => {
    if (!pixelMapping) return;
    const rect = panelImg.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const scale = panelImg.clientWidth / pixelMapping.pngWidth;
    const xLeft = pixelMapping.xLeft * scale;
    const xRight = pixelMapping.xRight * scale;

    if (clickX < xLeft || clickX > xRight) return;

    const frac = (clickX - xLeft) / (xRight - xLeft);
    audio.currentTime = frac * pixelMapping.duration;
    if (hasStemAudio()) stemSeek();
});

// Progress bar seek
progressTrack.addEventListener('click', (e) => {
    if (!pixelMapping) return;
    const rect = progressTrack.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    audio.currentTime = Math.max(0, Math.min(frac * pixelMapping.duration, pixelMapping.duration));
    if (hasStemAudio()) stemSeek();
});

// ── Playback controls ────────────────────────────────────────────

function togglePlay() {
    if (audio.paused) {
        audio.play();
        if (hasStemAudio()) stemPlay();
        playBtn.innerHTML = '&#9646;&#9646;';
    } else {
        audio.pause();
        if (hasStemAudio()) stemPause();
        playBtn.innerHTML = '&#9654;';
    }
}

playBtn.addEventListener('click', togglePlay);

const loopBtn = document.getElementById('loopBtn');
let loopEnabled = false;
loopBtn.addEventListener('click', () => {
    loopEnabled = !loopEnabled;
    loopBtn.classList.toggle('active', loopEnabled);
});

audio.addEventListener('ended', () => {
    if (loopEnabled) {
        audio.currentTime = 0;
        if (hasStemAudio()) stemSeek();
        audio.play();
        if (hasStemAudio()) stemPlay();
    } else {
        playBtn.innerHTML = '&#9654;';
        if (hasStemAudio()) stemPause();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT') return;

    if (e.code === 'Space') {
        e.preventDefault();
        togglePlay();
    } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        audio.currentTime = Math.max(0, audio.currentTime - 5);
        if (hasStemAudio()) stemSeek();
    } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        if (pixelMapping) {
            audio.currentTime = Math.min(pixelMapping.duration, audio.currentTime + 5);
            if (hasStemAudio()) stemSeek();
        }
    } else if (e.code === 'KeyT' && currentTab === 'annotate' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        recordTap();
    } else if (e.code === 'KeyS' && currentTab === 'annotate' && annotationTaps.length > 0 && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        saveAnnotation();
    } else if (e.code === 'Escape' && annotationTaps.length > 0) {
        discardAnnotation();
    } else if (!hasStemAudio() && currentTab === 'analysis') {
        if (e.code === 'KeyE') {
            toggleFeature('rms');
        } else if (e.code === 'KeyV') {
            toggleFeature('events');
        }
    } else if (hasStemAudio()) {
        const digitMatch = e.code.match(/^Digit(\d)$/);
        if (digitMatch) {
            const idx = parseInt(digitMatch[1]) - 1;
            if (idx >= 0 && idx < currentStemNames.length) {
                toggleStem(currentStemNames[idx]);
            }
        } else if (e.code === 'KeyA') {
            allStemsOn();
        }
    }
});

// ── Helpers ──────────────────────────────────────────────────────

function formatTime(s) {
    if (!s || isNaN(s)) return '0:00.000';
    const m = Math.floor(s / 60);
    const sec = s - m * 60;
    return m + ':' + sec.toFixed(3).padStart(6, '0');
}

// ── URL hash state ───────────────────────────────────────────────

function saveHashState() {
    const params = new URLSearchParams();
    if (currentFile) params.set('file', currentFile);
    if (currentTab) params.set('tab', currentTab);
    history.replaceState(null, '', '#' + params.toString());
}

function readHashState() {
    const params = new URLSearchParams(location.hash.slice(1));
    return {
        file: params.get('file'),
        tab: params.get('tab'),
    };
}

// ── Effects ──────────────────────────────────────────────────────

let effectsList = [];
let deprecatedEffects = [];  // effects marked as deprecated
let palettesList = [];  // available palettes
let selectedPalette = JSON.parse(localStorage.getItem('selectedPalette') || '{}');
let selectedBrightness = JSON.parse(localStorage.getItem('selectedBrightness') || '{}');
let effectsRunning = null;  // name of running effect or null
let effectsPollTimer = null;

function paletteGradientCSS(pal) {
    const p = pal.colors;
    if (!p || p.length === 0) return '#333';
    if (p.length === 1) return 'rgb(' + p[0].join(',') + ')';
    const stops = p.map((c, i) => 'rgb(' + c.join(',') + ') ' + (i / (p.length - 1) * 100).toFixed(0) + '%');
    return 'linear-gradient(90deg, ' + stops.join(', ') + ')';
}

function getSelectedPalette(effectName, defaultPalette) {
    return selectedPalette[effectName] || defaultPalette;
}

function getBrightnessRange(effectName) {
    return selectedBrightness[effectName] || [0, 100];
}

// Close any open popover when clicking outside
document.addEventListener('click', (e) => {
    if (!e.target.closest('.popover-anchor')) {
        document.querySelectorAll('.popover-panel.open').forEach(p => p.classList.remove('open'));
    }
});

function buildPaletteRows(panel, effectName, btn, onChange) {
    panel.innerHTML = '';
    const cur = getSelectedPalette(effectName, btn._defaultPalette);
    const userPals = palettesList.filter(c => !c.is_builtin);
    const builtinPals = palettesList.filter(c => c.is_builtin);

    function selectPalette(c) {
        selectedPalette[effectName] = c.name;
        localStorage.setItem('selectedPalette', JSON.stringify(selectedPalette));
        btn.innerHTML = '<span class="popover-btn-swatch" style="background:' + paletteGradientCSS(c) + '"></span>';
        btn.title = 'Palette: ' + c.name;
        panel.querySelectorAll('.palette-popover-row').forEach(r => r.classList.remove('selected'));
        panel.classList.remove('open');
        if (onChange) onChange(c.name);
    }

    function addRow(c) {
        const row = document.createElement('div');
        row.className = 'palette-popover-row' + (c.name === cur ? ' selected' : '');
        row.innerHTML = '<span class="palette-popover-swatch" style="background:' + paletteGradientCSS(c) + '"></span>' +
            '<span class="palette-popover-name">' + c.name + '</span>';
        if (!c.is_builtin) {
            const edit = document.createElement('span');
            edit.className = 'palette-popover-edit';
            edit.textContent = '\u270E';
            edit.title = 'Edit';
            edit.addEventListener('click', (e) => {
                e.stopPropagation();
                panel.classList.remove('open');
                openPaletteEditor(c, false, () => { refreshPalettesList(panel, effectName, btn, onChange); });
            });
            row.appendChild(edit);
        }
        row.addEventListener('click', (e) => {
            if (e.target.classList.contains('palette-popover-edit')) return;
            e.stopPropagation();
            selectPalette(c);
        });
        panel.appendChild(row);
    }

    if (userPals.length > 0) {
        const hdr = document.createElement('div');
        hdr.className = 'palette-popover-section';
        hdr.textContent = 'Custom';
        panel.appendChild(hdr);
        userPals.forEach(addRow);
    }

    const hdr2 = document.createElement('div');
    hdr2.className = 'palette-popover-section';
    hdr2.textContent = 'Built-in';
    panel.appendChild(hdr2);
    builtinPals.forEach(addRow);

    // "+ New palette" button
    const newBtn = document.createElement('div');
    newBtn.className = 'palette-popover-new';
    newBtn.innerHTML = '+ New palette';
    newBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        panel.classList.remove('open');
        openPaletteEditor(null, true, () => { refreshPalettesList(panel, effectName, btn, onChange); });
    });
    panel.appendChild(newBtn);
}

async function refreshPalettesList(panel, effectName, btn, onChange) {
    try {
        const resp = await fetch('/api/palettes');
        if (resp.ok) palettesList = await resp.json();
    } catch(e) {}
    buildPaletteRows(panel, effectName, btn, onChange);
    // Update button swatch
    const cur = getSelectedPalette(effectName, btn._defaultPalette);
    const curPal = palettesList.find(c => c.name === cur) || palettesList[0];
    if (curPal) {
        btn.innerHTML = '<span class="popover-btn-swatch" style="background:' + paletteGradientCSS(curPal) + '"></span>';
        btn.title = 'Palette: ' + cur;
    }
}

function createPalettePopover(effectName, defaultPalette, onChange) {
    const anchor = document.createElement('span');
    anchor.className = 'popover-anchor';

    const btn = document.createElement('button');
    btn.className = 'popover-btn palette-btn';
    btn._defaultPalette = defaultPalette;
    const cur = getSelectedPalette(effectName, defaultPalette);
    const curPal = palettesList.find(c => c.name === cur) || palettesList[0];
    btn.innerHTML = '<span class="popover-btn-swatch" style="background:' + paletteGradientCSS(curPal) + '"></span>';
    btn.title = 'Palette: ' + cur;
    anchor.appendChild(btn);

    const panel = document.createElement('div');
    panel.className = 'popover-panel palette-popover';
    buildPaletteRows(panel, effectName, btn, onChange);
    anchor.appendChild(panel);

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.popover-panel.open').forEach(p => { if (p !== panel) p.classList.remove('open'); });
        // Rebuild rows each time to pick up any changes
        buildPaletteRows(panel, effectName, btn, onChange);
        panel.classList.toggle('open');
    });

    return anchor;
}

function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(v => Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0')).join('');
}

function hexToRgb(hex) {
    const m = hex.match(/^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [255, 255, 255];
}

function openPaletteEditor(palSpec, isNew, onDone) {
    // palSpec: existing palette object (from palettesList) or null for new
    const name = isNew ? '' : (palSpec ? palSpec.name : '');
    const palette = palSpec ? (palSpec.colors || palSpec.palette || [[255,200,100]]).map(c => [...c]) : [[255, 200, 100]];
    const gamma = palSpec ? palSpec.gamma : 0.7;
    const brightCap = palSpec ? palSpec.brightness_cap : 1.0;
    const spatialMode = palSpec ? palSpec.spatial_mode : 'uniform';
    const overlay = document.createElement('div');
    overlay.className = 'palette-editor-overlay';

    const editor = document.createElement('div');
    editor.className = 'palette-editor';
    editor.addEventListener('click', e => e.stopPropagation());

    editor.innerHTML =
        '<h3>' + (isNew ? 'New Palette' : 'Edit Palette') + '</h3>' +
        '<label>Name</label>' +
        '<input type="text" class="ce-name" value="' + name + '" placeholder="my_palette">' +
        '<label>Palette</label>' +
        '<div class="palette-editor-gradient ce-gradient"></div>' +
        '<div class="palette-editor-stops ce-stops"></div>' +
        '<div class="palette-editor-add-stop ce-add-stop">+ Add color stop</div>' +
        '<div class="palette-editor-row">' +
        '  <label>Mode</label>' +
        '  <div class="palette-editor-radio ce-mode">' +
        '    <label><input type="radio" name="ce-mode" value="uniform"' + (spatialMode === 'uniform' ? ' checked' : '') + '> Uniform</label>' +
        '    <label><input type="radio" name="ce-mode" value="fibonacci"' + (spatialMode === 'fibonacci' ? ' checked' : '') + '> Fibonacci</label>' +
        '  </div>' +
        '</div>' +
        '<div class="palette-editor-row">' +
        '  <label>Gamma</label>' +
        '  <input type="range" class="ce-gamma" min="0.1" max="2.0" step="0.05" value="' + gamma + '">' +
        '  <span class="range-val ce-gamma-val">' + gamma.toFixed(2) + '</span>' +
        '</div>' +
        '<div class="palette-editor-row">' +
        '  <label>Brightness</label>' +
        '  <input type="range" class="ce-bright" min="0.05" max="1.0" step="0.05" value="' + brightCap + '">' +
        '  <span class="range-val ce-bright-val">' + Math.round(brightCap * 100) + '%</span>' +
        '</div>' +
        '<div class="palette-editor-actions">' +
        ((!isNew && palSpec && !palSpec.is_builtin) ? '<button class="btn-delete ce-delete">Delete</button>' : '') +
        '  <button class="btn-cancel ce-cancel">Cancel</button>' +
        '  <button class="btn-save ce-save">Save</button>' +
        '</div>';

    overlay.appendChild(editor);
    document.body.appendChild(overlay);

    // State
    let stops = palette.map(c => [...c]);

    function updateGradient() {
        const grad = editor.querySelector('.ce-gradient');
        if (stops.length === 1) {
            grad.style.background = 'rgb(' + stops[0].join(',') + ')';
        } else {
            const css = stops.map((c, i) => 'rgb(' + c.join(',') + ') ' + (i / (stops.length - 1) * 100).toFixed(0) + '%');
            grad.style.background = 'linear-gradient(90deg, ' + css.join(', ') + ')';
        }
    }

    function renderStops() {
        const container = editor.querySelector('.ce-stops');
        container.innerHTML = '';
        stops.forEach((c, i) => {
            const row = document.createElement('div');
            row.className = 'palette-editor-stop';
            const colorInput = document.createElement('input');
            colorInput.type = 'color';
            colorInput.value = rgbToHex(c[0], c[1], c[2]);
            const hexLabel = document.createElement('span');
            hexLabel.className = 'stop-hex';
            hexLabel.textContent = colorInput.value;
            colorInput.addEventListener('input', () => {
                stops[i] = hexToRgb(colorInput.value);
                hexLabel.textContent = colorInput.value;
                updateGradient();
            });
            row.appendChild(colorInput);
            row.appendChild(hexLabel);
            if (stops.length > 1) {
                const rm = document.createElement('span');
                rm.className = 'stop-remove';
                rm.textContent = '\u00d7';
                rm.title = 'Remove';
                rm.addEventListener('click', () => { stops.splice(i, 1); renderStops(); updateGradient(); });
                row.appendChild(rm);
            }
            container.appendChild(row);
        });
    }

    renderStops();
    updateGradient();

    // Add stop
    editor.querySelector('.ce-add-stop').addEventListener('click', () => {
        const last = stops[stops.length - 1];
        stops.push([...last]);
        renderStops();
        updateGradient();
    });

    // Gamma/brightness sliders
    const gammaSlider = editor.querySelector('.ce-gamma');
    const gammaVal = editor.querySelector('.ce-gamma-val');
    gammaSlider.addEventListener('input', () => { gammaVal.textContent = parseFloat(gammaSlider.value).toFixed(2); });
    const brightSlider = editor.querySelector('.ce-bright');
    const brightVal = editor.querySelector('.ce-bright-val');
    brightSlider.addEventListener('input', () => { brightVal.textContent = Math.round(parseFloat(brightSlider.value) * 100) + '%'; });

    // Cancel
    editor.querySelector('.ce-cancel').addEventListener('click', () => overlay.remove());
    overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });

    // Delete
    const delBtn = editor.querySelector('.ce-delete');
    if (delBtn) {
        delBtn.addEventListener('click', async () => {
            if (!confirm('Delete palette "' + name + '"?')) return;
            try {
                await fetch('/api/palettes/delete', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name })
                });
            } catch(e) {}
            overlay.remove();
            if (onDone) onDone();
        });
    }

    // Save
    editor.querySelector('.ce-save').addEventListener('click', async () => {
        const newName = editor.querySelector('.ce-name').value.trim();
        if (!newName) { alert('Name is required'); return; }
        if (/[^a-zA-Z0-9_]/.test(newName)) { alert('Name must be alphanumeric/underscores only'); return; }
        const mode = editor.querySelector('input[name="ce-mode"]:checked').value;
        const spec = {
            name: newName,
            colors: stops,
            gamma: parseFloat(gammaSlider.value),
            brightness_cap: parseFloat(brightSlider.value),
            spatial_mode: mode,
        };
        try {
            const resp = await fetch('/api/palettes/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(spec)
            });
            const data = await resp.json();
            if (data.error) { alert(data.error); return; }
        } catch(e) { alert('Save failed: ' + e.message); return; }
        // If renamed, clear old selection
        if (!isNew && name !== newName) {
            Object.keys(selectedPalette).forEach(k => {
                if (selectedPalette[k] === name) selectedPalette[k] = newName;
            });
            localStorage.setItem('selectedPalette', JSON.stringify(selectedPalette));
        }
        overlay.remove();
        if (onDone) onDone();
    });
}

function createBrightnessPopover(effectName, onChange) {
    const anchor = document.createElement('span');
    anchor.className = 'popover-anchor';

    const btn = document.createElement('button');
    btn.className = 'popover-btn brightness-btn';
    const range = getBrightnessRange(effectName);
    btn.textContent = range[0] === 0 && range[1] === 100 ? '\u2600' : '\u2600 ' + range[0] + '-' + range[1] + '%';
    btn.title = 'Brightness: ' + range[0] + '% - ' + range[1] + '%';
    anchor.appendChild(btn);

    const panel = document.createElement('div');
    panel.className = 'popover-panel brightness-popover';
    panel.innerHTML = '<div class="brightness-popover-title">Brightness range</div>' +
        '<div class="brightness-popover-sliders">' +
        '<label>Low <input type="range" class="brt-low" min="0" max="100" value="' + range[0] + '" step="1"><span class="brt-low-val">' + range[0] + '%</span></label>' +
        '<label>High <input type="range" class="brt-high" min="0" max="100" value="' + range[1] + '" step="1"><span class="brt-high-val">' + range[1] + '%</span></label>' +
        '</div>' +
        '<div class="brightness-popover-presets">' +
        '<button data-lo="0" data-hi="100">Full</button>' +
        '<button data-lo="0" data-hi="30">Boost dim</button>' +
        '<button data-lo="50" data-hi="100">Highlights</button>' +
        '<button data-lo="10" data-hi="90">Contrast</button>' +
        '</div>';
    anchor.appendChild(panel);

    const lowSlider = panel.querySelector('.brt-low');
    const highSlider = panel.querySelector('.brt-high');
    const lowVal = panel.querySelector('.brt-low-val');
    const highVal = panel.querySelector('.brt-high-val');

    function update() {
        let lo = parseInt(lowSlider.value), hi = parseInt(highSlider.value);
        if (lo > hi) { lo = hi; lowSlider.value = lo; }
        lowVal.textContent = lo + '%';
        highVal.textContent = hi + '%';
        selectedBrightness[effectName] = [lo, hi];
        localStorage.setItem('selectedBrightness', JSON.stringify(selectedBrightness));
        btn.textContent = lo === 0 && hi === 100 ? '\u2600' : '\u2600 ' + lo + '-' + hi + '%';
        btn.title = 'Brightness: ' + lo + '% - ' + hi + '%';
        if (onChange) onChange([lo, hi]);
    }

    lowSlider.addEventListener('input', update);
    highSlider.addEventListener('input', update);

    panel.querySelectorAll('.brightness-popover-presets button').forEach(b => {
        b.addEventListener('click', (e) => {
            e.stopPropagation();
            lowSlider.value = b.dataset.lo;
            highSlider.value = b.dataset.hi;
            update();
        });
    });

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        document.querySelectorAll('.popover-panel.open').forEach(p => { if (p !== panel) p.classList.remove('open'); });
        panel.classList.toggle('open');
    });

    panel.addEventListener('click', (e) => e.stopPropagation());

    return anchor;
}
let selectorState = [];  // array of Sets, one per segment depth
let outputTargets = [];  // from /api/controllers (sculptures + controllers)
let selectedTarget = null;  // output target id or null (no LEDs)
let selectedTargetType = null;  // 'sculpture' | 'controller' | null

async function loadEffects() {
    const panel = document.getElementById('effectsPanel');
    panel.innerHTML = '<span style="color:#888;">Loading effects...</span>';

    try {
        // Load output targets and effects in parallel
        const [ctrlResp, resp] = await Promise.all([
            fetch('/api/controllers'),
            fetch('/api/effects')
        ]);
        if (ctrlResp.ok) outputTargets = await ctrlResp.json();
        if (!resp.ok) { panel.innerHTML = '<span style="color:#e94560;">Failed to load effects</span>'; return; }
        const data = await resp.json();
        effectsList = data.effects || [];
        deprecatedEffects = data.deprecated || [];
        palettesList = data.palettes || [];
        effectsRunning = data.running;
        if (data.controller !== undefined) {
            selectedTarget = data.controller;
            selectedTargetType = data.controller_type || null;
        }
        if (selectorState.length === 0) {
            const maxSegs = Math.max(...effectsList.map(e => e.name.split('_').length));
            selectorState = Array.from({length: maxSegs}, () => new Set());
        }
        renderEffectsCards();
        startEffectsPoll();
    } catch (e) {
        panel.innerHTML = '<span style="color:#e94560;">Error: ' + e.message + '</span>';
    }
}

function getFilteredEffects() {
    return effectsList.filter(eff => {
        const parts = eff.name.split('_');
        for (let d = 0; d < selectorState.length; d++) {
            if (selectorState[d].size > 0 && !selectorState[d].has(parts[d] || '')) return false;
        }
        return true;
    });
}

function buildSelector(panel) {
    const container = document.createElement('div');
    container.className = 'effects-selector';

    for (let depth = 0; depth < selectorState.length; depth++) {
        // Only show column if depth 0 or previous column has selections
        if (depth > 0 && selectorState[depth - 1].size === 0) break;

        // Get effects matching all selections at prior depths
        const matching = effectsList.filter(eff => {
            const parts = eff.name.split('_');
            for (let d = 0; d < depth; d++) {
                if (selectorState[d].size > 0 && !selectorState[d].has(parts[d] || '')) return false;
            }
            return true;
        });

        // Collect unique values at this depth with counts
        const valueCounts = {};
        matching.forEach(eff => {
            const seg = eff.name.split('_')[depth];
            if (seg) valueCounts[seg] = (valueCounts[seg] || 0) + 1;
        });

        const values = Object.keys(valueCounts).sort();
        if (values.length === 0) break;

        const col = document.createElement('div');
        col.className = 'selector-col';

        values.forEach(val => {
            const item = document.createElement('label');
            const isChecked = selectorState[depth].has(val);
            item.className = 'selector-item' + (isChecked ? ' checked' : '');

            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = isChecked;
            cb.addEventListener('change', () => {
                if (cb.checked) selectorState[depth].add(val);
                else selectorState[depth].delete(val);
                // Clear deeper selections when a parent changes
                for (let d = depth + 1; d < selectorState.length; d++) selectorState[d].clear();
                renderEffectsCards();
            });

            const count = document.createElement('span');
            count.className = 'selector-count';
            count.textContent = valueCounts[val];

            item.appendChild(cb);
            item.appendChild(document.createTextNode(val));
            item.appendChild(count);
            col.appendChild(item);
        });

        container.appendChild(col);
    }

    panel.appendChild(container);
}

// Reference data keyed by registry name: [signal, threshold, envelope, tempo]
const EFFECT_REF = {
    'impulse':           ['abs-integral 150ms', '0.30, 250ms cd',  'snap-on, decay 0.85', '\u2014'],
    'impulse_glow':      ['abs-integral 150ms', 'proportional',    'atk 0.6, decay 0.85', '\u2014'],
    'impulse_predict':   ['abs-integral 150ms', '0.30, 250ms cd',  'snap-on, decay 0.85', 'autocorr 5s'],
    'impulse_downbeat':  ['abs-integral 150ms', '0.30, 200ms cd',  'full/30% ticks',      '\u2014'],
    'impulse_breathe':   ['abs-integral 150ms', 'proportional',    'symmetric 0.74',      '\u2014'],
    'impulse_sections':  ['abs-integral 150ms', 'proportional',    'atk 0.6, decay 0.85', '\u2014'],
    'impulse_meter':     ['abs-integral 150ms', 'proportional',    'atk 0.6, decay 0.85', '\u2014'],
    'bass_pulse':        ['bass flux 20-250Hz', '0.55, 180ms cd',  'snap-on, decay 0.82', '\u2014'],
    'tempo_pulse':       ['RMS + autocorr',     'oscillator',      'raised cosine',       'autocorr 30s'],
    'rms_meter':         ['RMS raw',            'proportional',    'peak decay 0.9998',   '\u2014'],
    'longint_sections':  ['80% long RMS + 20% bass', 'proportional', '10s rolling avg',  '\u2014'],
    'impulse_snake':     ['abs-int + predict',  '0.30',            'traveling pulse',     'autocorr 5s'],
    'impulse_bands':     ['3-band abs-integral','per-band 0.30',   'Gaussian \u03c3=3\u21928', '\u2014'],
    'band_prop':         ['3-band abs-integral','proportional',    'per-band a/d',        '\u2014'],
    'band_sparkles':     ['5-band FFT',         'proportional',    '5s rolling int',      '\u2014'],
    'band_tempo_sparkles': ['abs-int + 5-band', '0.30',            'sparkle fade',        'autocorr 5s'],
    'three_voices':      ['streaming HPSS',     'proportional',    'per-voice',           '\u2014'],
    'basic_sparkles':    ['none (visual)',       '\u2014',          'random twinkle',      '\u2014'],
};

// ── Feature colors/labels (used by analyze sparklines) ──────────

const FEATURE_COLORS = {
    abs_integral: '#e94560',
    rms: '#00e676',
    centroid: '#00b0ff',
    autocorr_conf: '#ffea00',
};
const FEATURE_LABELS = {
    abs_integral: 'Abs-Integral',
    rms: 'RMS',
    centroid: 'Centroid',
    autocorr_conf: 'Autocorr',
};
function buildControllerBar(panel) {
    const bar = document.createElement('div');
    bar.className = 'controller-bar';

    const label = document.createElement('label');
    label.textContent = 'Output:';
    bar.appendChild(label);

    const select = document.createElement('select');
    // Default: no LEDs
    const noLeds = document.createElement('option');
    noLeds.value = '';
    noLeds.textContent = 'No LEDs (terminal only)';
    select.appendChild(noLeds);

    outputTargets.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.id;
        opt.dataset.type = t.type;
        const status = t.port ? t.port : 'disconnected';
        // Show: "Name (N LEDs) — Controller Name" or "Name (N LEDs) — status"
        const suffix = t.type === 'sculpture' ? ' \u2014 ' + t.controller_name : '';
        opt.textContent = t.name + ' (' + t.leds + ' LEDs)' + suffix + ' \u2014 ' + status;
        if (!t.port) opt.disabled = true;
        if (t.id === selectedTarget) opt.selected = true;
        select.appendChild(opt);
    });

    if (!selectedTarget) noLeds.selected = true;

    select.addEventListener('change', () => {
        const opt = select.selectedOptions[0];
        selectedTarget = select.value || null;
        selectedTargetType = opt && opt.dataset.type || null;
        // Persist selection
        const body = {};
        if (selectedTargetType === 'sculpture') {
            body.sculpture = selectedTarget;
        } else if (selectedTarget) {
            body.controller = selectedTarget;
        }
        fetch('/api/effects/controller', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }).catch(() => {});
    });

    bar.appendChild(select);
    panel.appendChild(bar);
}

function renderEffectsCards() {
    const panel = document.getElementById('effectsPanel');
    if (effectsList.length === 0) {
        panel.innerHTML = '<span style="color:#888;">No effects found. Check runner.py --list</span>';
        return;
    }
    panel.innerHTML = '';
    const listWrap = document.createElement('div');
    listWrap.className = 'effects-list';
    buildControllerBar(listWrap);
    buildSelector(listWrap);

    const REF_COLS = ['Signal', 'Threshold', 'Envelope', 'Tempo'];
    const wrap = document.createElement('div');
    wrap.className = 'effect-ref-wrap';

    const table = document.createElement('table');
    table.className = 'effect-ref-table';

    const thead = document.createElement('thead');
    const hr = document.createElement('tr');
    // Controls columns: drag, name, controls | separator | reference data
    ['', 'Effect', '', '', ...REF_COLS].forEach((c, i) => {
        const th = document.createElement('th');
        th.textContent = c;
        if (i === 3) th.className = 'ref-separator';
        hr.appendChild(th);
    });
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    const filtered = getFilteredEffects();

    filtered.forEach(eff => {
        const tr = document.createElement('tr');
        tr.className = eff.name === effectsRunning ? 'running' : '';
        tr.dataset.name = eff.name;

        // Drag handle
        const dragTd = document.createElement('td');
        dragTd.className = 'effect-drag-cell';
        const drag = document.createElement('span');
        drag.className = 'effect-drag';
        drag.textContent = '\u2261';
        drag.addEventListener('mousedown', () => { tr.draggable = true; });
        dragTd.appendChild(drag);
        tr.appendChild(dragTd);

        // Name cell (clickable + pencil rename)
        const nameTd = document.createElement('td');
        nameTd.className = 'effect-name-cell';
        const nameWrap = document.createElement('span');
        nameWrap.className = 'effect-name-wrap';
        const nameEl = document.createElement('span');
        nameEl.className = 'effect-name';
        nameEl.textContent = eff.display_name || eff.name;
        if (eff.description) nameEl.title = eff.description;
        if (eff.name === effectsRunning) {
            const dot = document.createElement('span');
            dot.className = 'running-dot';
            nameEl.appendChild(dot);
        }
        nameEl.addEventListener('click', (e) => { e.stopPropagation(); showEffectDetail(eff.name); });
        nameWrap.appendChild(nameEl);

        const pencil = document.createElement('button');
        pencil.className = 'effect-rename-btn';
        pencil.title = 'Rename effect';
        pencil.innerHTML = '&#9998;';
        pencil.addEventListener('click', (e) => {
            e.stopPropagation();
            const cur = eff.display_name || eff.name;
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'effect-rename-input';
            input.value = cur;
            input.placeholder = eff.name;
            nameWrap.replaceChild(input, nameEl);
            pencil.style.display = 'none';
            input.focus();
            input.select();
            const save = async () => {
                const val = input.value.trim();
                const displayName = (val === eff.name) ? '' : val;
                try {
                    await fetch('/api/effects/rename', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: eff.name, display_name: displayName })
                    });
                    eff.display_name = displayName || undefined;
                } catch (err) { console.error('rename:', err); }
                renderEffectsCards();
            };
            input.addEventListener('keydown', (ke) => {
                if (ke.key === 'Enter') save();
                if (ke.key === 'Escape') renderEffectsCards();
            });
            input.addEventListener('blur', save);
        });
        nameWrap.appendChild(pencil);

        // Notes indicator (shows icon if notes exist, click opens detail)
        const noteBtn = document.createElement('button');
        noteBtn.className = 'effect-rename-btn effect-note-btn' + (eff.notes ? ' has-notes' : '');
        noteBtn.title = eff.notes ? eff.notes : 'Add notes';
        noteBtn.innerHTML = '&#128221;';
        noteBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            showEffectDetail(eff.name, true);
        });
        nameWrap.appendChild(noteBtn);

        const archiveBtn = document.createElement('button');
        archiveBtn.className = 'effect-rename-btn effect-archive-btn';
        archiveBtn.title = 'Deprecate effect';
        archiveBtn.innerHTML = '&#128451;';  // archive box icon
        archiveBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            deprecateEffect(eff.name);
        });
        nameWrap.appendChild(archiveBtn);

        nameTd.appendChild(nameWrap);
        tr.appendChild(nameTd);

        // Controls cell: palette, brightness, stars, start/stop — right next to name
        const ctrlTd = document.createElement('td');
        ctrlTd.className = 'effect-controls-cell';

        const btn = document.createElement('button');
        btn.className = 'effect-toggle' + (eff.name === effectsRunning ? ' stop' : '');
        btn.textContent = eff.name === effectsRunning ? 'Stop' : 'Start';
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            if (eff.name === effectsRunning) stopEffect();
            else startEffect(eff.name);
        });
        ctrlTd.appendChild(btn);

        if (eff.is_signal && palettesList.length > 0) {
            ctrlTd.appendChild(createPalettePopover(eff.name, eff.default_palette));
        }
        ctrlTd.appendChild(createBrightnessPopover(eff.name));

        const stars = document.createElement('span');
        stars.className = 'effect-stars';
        for (let s = 1; s <= 5; s++) {
            const star = document.createElement('button');
            star.className = 'effect-star' + (s <= (eff.rating || 0) ? ' filled' : '');
            star.title = s + ' star' + (s > 1 ? 's' : '');
            star.addEventListener('click', (e) => { e.stopPropagation(); rateEffect(eff.name, s); });
            stars.appendChild(star);
        }
        ctrlTd.appendChild(stars);
        tr.appendChild(ctrlTd);

        // Separator cell
        const sepTd = document.createElement('td');
        sepTd.className = 'ref-separator';
        tr.appendChild(sepTd);

        // Reference data columns
        const ref = EFFECT_REF[eff.name];
        REF_COLS.forEach((_, ci) => {
            const td = document.createElement('td');
            td.className = 'ref-data';
            td.textContent = ref ? ref[ci] : '\u2014';
            tr.appendChild(td);
        });

        // Drag events on the row
        tr.addEventListener('dragstart', (e) => {
            tr.classList.add('dragging');
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', eff.name);
        });
        tr.addEventListener('dragend', () => { tr.classList.remove('dragging'); tr.draggable = false; });
        tr.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            tr.classList.add('drag-over');
        });
        tr.addEventListener('dragleave', () => tr.classList.remove('drag-over'));
        tr.addEventListener('drop', (e) => {
            e.preventDefault();
            tr.classList.remove('drag-over');
            const fromName = e.dataTransfer.getData('text/plain');
            const toName = eff.name;
            if (fromName !== toName) {
                const fromIdx = effectsList.findIndex(ef => ef.name === fromName);
                const toIdx = effectsList.findIndex(ef => ef.name === toName);
                if (fromIdx >= 0 && toIdx >= 0) {
                    const moved = effectsList.splice(fromIdx, 1)[0];
                    effectsList.splice(toIdx, 0, moved);
                    renderEffectsCards();
                    reorderEffects(effectsList.map(ef => ef.name));
                }
            }
        });

        tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    wrap.appendChild(table);
    listWrap.appendChild(wrap);

    // Deprecated effects section
    if (deprecatedEffects.length > 0) {
        const depSection = document.createElement('details');
        depSection.className = 'deprecated-section';

        const summary = document.createElement('summary');
        summary.textContent = 'Deprecated (' + deprecatedEffects.length + ')';
        depSection.appendChild(summary);

        const depWrap = document.createElement('div');
        depWrap.className = 'effect-ref-wrap deprecated-wrap';

        const depTable = document.createElement('table');
        depTable.className = 'effect-ref-table deprecated-table';

        const depThead = document.createElement('thead');
        const depHr = document.createElement('tr');
        ['Effect', 'Reason', ''].forEach(c => {
            const th = document.createElement('th');
            th.textContent = c;
            depHr.appendChild(th);
        });
        depThead.appendChild(depHr);
        depTable.appendChild(depThead);

        const depTbody = document.createElement('tbody');
        deprecatedEffects.forEach(eff => {
            const tr = document.createElement('tr');

            const nameTd = document.createElement('td');
            nameTd.className = 'effect-name-cell';
            nameTd.textContent = eff.display_name || eff.name;
            tr.appendChild(nameTd);

            const reasonTd = document.createElement('td');
            reasonTd.className = 'deprecated-reason';
            reasonTd.textContent = eff.deprecated_reason || '';
            tr.appendChild(reasonTd);

            const actionTd = document.createElement('td');
            actionTd.style.textAlign = 'right';
            const restoreBtn = document.createElement('button');
            restoreBtn.className = 'effect-toggle';
            restoreBtn.textContent = 'Restore';
            restoreBtn.title = 'Move back to active effects';
            restoreBtn.addEventListener('click', () => restoreEffect(eff.name));
            actionTd.appendChild(restoreBtn);
            tr.appendChild(actionTd);

            depTbody.appendChild(tr);
        });
        depTable.appendChild(depTbody);
        depWrap.appendChild(depTable);
        depSection.appendChild(depWrap);
        listWrap.appendChild(depSection);
    }

    panel.appendChild(listWrap);
}

async function deprecateEffect(name) {
    const reason = prompt('Why deprecate "' + name + '"? (optional)');
    if (reason === null) return;  // cancelled
    try {
        await fetch('/api/effects/deprecate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, deprecated: true, reason })
        });
        // Move from active to deprecated locally
        const idx = effectsList.findIndex(e => e.name === name);
        if (idx >= 0) {
            const eff = effectsList.splice(idx, 1)[0];
            eff.deprecated = true;
            eff.deprecated_reason = reason;
            deprecatedEffects.push(eff);
        }
        renderEffectsCards();
    } catch (e) { console.error('deprecateEffect:', e); }
}

async function restoreEffect(name) {
    try {
        await fetch('/api/effects/deprecate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, deprecated: false })
        });
        // Move from deprecated to active locally
        const idx = deprecatedEffects.findIndex(e => e.name === name);
        if (idx >= 0) {
            const eff = deprecatedEffects.splice(idx, 1)[0];
            delete eff.deprecated;
            delete eff.deprecated_reason;
            effectsList.push(eff);
        }
        renderEffectsCards();
    } catch (e) { console.error('restoreEffect:', e); }
}

async function startEffect(name) {
    try {
        const body = {};
        if (selectedTarget) {
            if (selectedTargetType === 'sculpture') {
                body.sculpture = selectedTarget;
            } else {
                body.controller = selectedTarget;
            }
        }
        const eff = effectsList.find(e => e.name === name);
        if (eff && eff.is_signal) {
            body.palette = getSelectedPalette(name, eff.default_palette);
        }
        const range = getBrightnessRange(name);
        body.brightness = range[1] / 100;
        const opts = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        };
        await fetch('/api/effects/start/' + encodeURIComponent(name), opts);
        effectsRunning = name;
        renderEffectsCards();
    } catch (e) { console.error('startEffect:', e); }
}

async function stopEffect() {
    try {
        await fetch('/api/effects/stop', { method: 'POST' });
        effectsRunning = null;
        renderEffectsCards();
    } catch (e) { console.error('stopEffect:', e); }
}

async function rateEffect(name, rating) {
    try {
        await fetch('/api/effects/rate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, rating })
        });
        const eff = effectsList.find(e => e.name === name);
        if (eff) eff.rating = rating;
        renderEffectsCards();
    } catch (e) { console.error('rateEffect:', e); }
}

async function reorderEffects(order) {
    try {
        await fetch('/api/effects/reorder', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ order })
        });
    } catch (e) { console.error('reorderEffects:', e); }
}

function startEffectsPoll() {
    stopEffectsPoll();
    effectsPollTimer = setInterval(async () => {
        if (currentTab !== 'effects') { stopEffectsPoll(); return; }
        try {
            const resp = await fetch('/api/effects');
            if (!resp.ok) return;
            const data = await resp.json();
            const newRunning = data.running;
            if (newRunning !== effectsRunning) {
                const prev = effectsRunning;
                effectsRunning = newRunning;
                // Update running indicator on rows without full rebuild
                const rows = document.querySelectorAll('.effect-ref-table tbody tr[data-name]');
                if (rows.length > 0) {
                    rows.forEach(c => {
                        const n = c.dataset.name;
                        c.classList.toggle('running', n === newRunning);
                        const toggle = c.querySelector('.effect-toggle');
                        if (toggle) {
                            toggle.classList.toggle('stop', n === newRunning);
                            toggle.textContent = (n === newRunning) ? 'Stop' : 'Start';
                        }
                    });
                } else {
                    renderEffectsCards();
                }
            }
        } catch (e) {}
    }, 2000);
}

function stopEffectsPoll() {
    if (effectsPollTimer) { clearInterval(effectsPollTimer); effectsPollTimer = null; }
}

// ── Effect Detail View ──────────────────────────────────────────

let effectDetailName = null;
let effectDetailData = null;
let effectDetailAnim = null;
let ledogramBytes = null;  // cached decoded LED data for redraw

function showEffectDetail(name, focusNotes) {
    effectDetailName = name;
    effectDetailData = null;
    stopEffectsPoll();

    const panel = document.getElementById('effectsPanel');
    panel.innerHTML = '';

    // Header
    const header = document.createElement('div');
    header.className = 'effect-detail-header';
    const backBtn = document.createElement('button');
    backBtn.className = 'effect-detail-back';
    backBtn.textContent = '\u2190 Back';
    backBtn.addEventListener('click', showEffectsList);
    header.appendChild(backBtn);
    const h2 = document.createElement('h2');
    h2.textContent = name;
    header.appendChild(h2);
    panel.appendChild(header);

    // Description (from effect registry)
    const effEntry = effectsList.find(e => e.name === name);
    if (effEntry && effEntry.description) {
        const desc = document.createElement('div');
        desc.style.cssText = 'color: #888; font-size: 13px; margin-top: -4px;';
        desc.textContent = effEntry.description;
        panel.appendChild(desc);
    }

    // Notes section (editable textarea)
    const notesWrap = document.createElement('div');
    notesWrap.className = 'effect-notes-wrap';
    const notesLabel = document.createElement('label');
    notesLabel.textContent = 'Notes';
    notesLabel.className = 'effect-notes-label';
    notesWrap.appendChild(notesLabel);
    const notesArea = document.createElement('textarea');
    notesArea.className = 'effect-notes-area';
    notesArea.placeholder = 'Add notes for future improvements...';
    notesArea.value = effEntry && effEntry.notes ? effEntry.notes : '';
    notesArea.rows = 3;
    let notesSaveTimer = null;
    notesArea.addEventListener('input', () => {
        clearTimeout(notesSaveTimer);
        notesSaveTimer = setTimeout(async () => {
            const val = notesArea.value.trim();
            try {
                await fetch('/api/effects/notes', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, notes: val })
                });
                if (effEntry) effEntry.notes = val || undefined;
            } catch (err) { console.error('notes save:', err); }
        }, 600);
    });
    notesWrap.appendChild(notesArea);
    panel.appendChild(notesWrap);
    if (focusNotes) setTimeout(() => notesArea.focus(), 50);

    // Controls: file picker + analyze button
    const controls = document.createElement('div');
    controls.className = 'effect-detail-controls';

    const fileSel = document.createElement('select');
    fileSel.id = 'effectDetailFile';
    files.forEach(f => {
        const opt = document.createElement('option');
        opt.value = f.path;
        opt.textContent = f.name;
        fileSel.appendChild(opt);
    });
    // Pre-select current file if any
    if (currentFile) fileSel.value = currentFile;
    controls.appendChild(fileSel);

    // Palette popover (signal effects only)
    if (effEntry && effEntry.is_signal && palettesList.length > 0) {
        controls.appendChild(createPalettePopover(name, effEntry.default_palette));
    }

    // Brightness popover (live-updates LED-o-gram)
    controls.appendChild(createBrightnessPopover(name, (range) => {
        const vizEl = document.getElementById('effectDetailViz');
        const ledCanvas = vizEl && vizEl.querySelectorAll('canvas')[1];
        if (ledCanvas && effectDetailData) {
            const w = vizEl.clientWidth || 860;
            const lh = Math.min(Math.max(effectDetailData.num_leds, 60), 300);
            drawLedogramCanvas(ledCanvas, effectDetailData, window.devicePixelRatio || 1, w, lh, ledogramBytes, range);
        }
    }));

    const analyzeBtn = document.createElement('button');
    analyzeBtn.textContent = 'Analyze';
    analyzeBtn.addEventListener('click', () => runEffectAnalysis(name));
    controls.appendChild(analyzeBtn);
    panel.appendChild(controls);

    // Status / loading text
    const status = document.createElement('div');
    status.className = 'effect-detail-status';
    status.id = 'effectDetailStatus';
    status.textContent = 'Select a file and click Analyze';
    panel.appendChild(status);

    // Viz container (empty until analysis runs)
    const viz = document.createElement('div');
    viz.className = 'effect-detail-viz';
    viz.id = 'effectDetailViz';
    panel.appendChild(viz);
}

function showEffectsList() {
    effectDetailName = null;
    effectDetailData = null;
    if (effectDetailAnim) { cancelAnimationFrame(effectDetailAnim); effectDetailAnim = null; }
    renderEffectsCards();
    startEffectsPoll();
}

async function runEffectAnalysis(name) {
    const fileSel = document.getElementById('effectDetailFile');
    const status = document.getElementById('effectDetailStatus');
    const viz = document.getElementById('effectDetailViz');
    if (!fileSel || !fileSel.value) return;

    status.textContent = 'Analyzing... (running effect offline)';
    viz.innerHTML = '';

    try {
        const analyzeBody = { effect: name, file: fileSel.value };
        const effEntry = effectsList.find(e => e.name === name);
        if (effEntry && effEntry.is_signal) {
            analyzeBody.palette =getSelectedPalette(name, effEntry.default_palette);
        }
        const resp = await fetch('/api/effects/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(analyzeBody)
        });
        const data = await resp.json();
        if (data.error) {
            status.textContent = 'Error: ' + data.error;
            return;
        }
        effectDetailData = data;
        status.textContent = data.num_frames + ' frames, ' + data.duration.toFixed(1) + 's, ' + data.diag_keys.length + ' diagnostics';

        // Set audio to this file for playback sync
        audio.src = '/audio/' + encodeURIComponent(fileSel.value);
        audio.load();

        renderEffectViz(viz, data);
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
    }
}

function renderEffectViz(container, data) {
    container.innerHTML = '';
    const dpr = window.devicePixelRatio || 1;
    const width = container.clientWidth || 860;

    // Decode LED data once, cache for brightness redraw
    ledogramBytes = null;
    if (data.led_data && data.num_frames > 0) {
        const raw = atob(data.led_data);
        ledogramBytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) ledogramBytes[i] = raw.charCodeAt(i);
    }

    // LED-ogram canvas
    const ledCanvas = document.createElement('canvas');
    const ledH = Math.min(Math.max(data.num_leds, 60), 300);
    ledCanvas.style.height = ledH + 'px';
    ledCanvas.style.marginTop = '2px';
    container.appendChild(ledCanvas);
    drawLedogramCanvas(ledCanvas, data, dpr, width, ledH, ledogramBytes, getBrightnessRange(effectDetailName));

    // Feature sparklines (from analyze_effect data)
    if (data.features && data.features.length > 0 && data.feature_keys) {
        const featWrap = document.createElement('div');
        featWrap.style.cssText = 'margin-top: 4px;';
        data.feature_keys.forEach(key => {
            const row = document.createElement('div');
            row.className = 'feature-row';
            const label = document.createElement('span');
            label.className = 'feature-label';
            label.textContent = FEATURE_LABELS[key] || key;
            row.appendChild(label);
            const canvas = document.createElement('canvas');
            canvas.style.cssText = 'height: 22px; cursor: crosshair;';
            row.appendChild(canvas);
            featWrap.appendChild(row);

            // Draw after append (needs layout)
            requestAnimationFrame(() => {
                const ctx = canvas.getContext('2d');
                const fw = canvas.clientWidth;
                const fh = canvas.clientHeight;
                canvas.width = fw * dpr;
                canvas.height = fh * dpr;
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.clearRect(0, 0, fw, fh);
                const n = data.features.length;
                ctx.beginPath();
                ctx.strokeStyle = FEATURE_COLORS[key] || '#888';
                ctx.lineWidth = 1.2;
                for (let i = 0; i < n; i++) {
                    const x = (i / (n - 1)) * fw;
                    const y = fh - (data.features[i][key] || 0) * fh;
                    if (i === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                ctx.stroke();
            });

            // Click-to-seek
            canvas.addEventListener('click', (e) => {
                const rect = canvas.getBoundingClientRect();
                const frac = (e.clientX - rect.left) / rect.width;
                audio.currentTime = Math.max(0, Math.min(frac * data.duration, data.duration));
            });
        });
        container.appendChild(featWrap);
    }

    // Analysis panels selector
    const panelDefs = [
        { id: 'spectrogram', label: 'Spectrogram' },
        { id: 'bands', label: 'Band Energy' },
        { id: 'rms-derivative', label: 'RMS Derivative' },
        { id: 'centroid', label: 'Center of Mass of Frequency' },
        { id: 'centroid-derivative', label: 'Centroid Derivative' },
        { id: 'band-derivative', label: 'Band Derivative' },
        { id: 'mfcc', label: 'Timbral Shape (MFCC)' },
        { id: 'novelty', label: 'Novelty' },
        { id: 'band-deviation', label: 'Band Deviation' },
        { id: 'annotations', label: 'Annotations' },
    ];
    const panelRow = document.createElement('div');
    panelRow.className = 'analysis-panel-chips';
    panelRow.innerHTML = '<span class="panel-chips-label">Analysis</span>';
    const panelContainer = document.createElement('div');
    panelContainer.className = 'analysis-panels-container';
    panelDefs.forEach(pd => {
        const chip = document.createElement('span');
        chip.className = 'panel-chip';
        chip.textContent = pd.label;
        chip.dataset.panel = pd.id;
        chip.addEventListener('click', () => toggleAnalysisPanel(chip, pd.id, panelContainer, data));
        panelRow.appendChild(chip);
    });
    container.appendChild(panelRow);
    container.appendChild(panelContainer);

    // Cursor overlay
    const cursor = document.createElement('div');
    cursor.className = 'effect-detail-cursor';
    cursor.style.left = '0px';
    container.appendChild(cursor);

    // Click-to-seek on LED-ogram
    ledCanvas.addEventListener('click', (e) => {
        const rect = ledCanvas.getBoundingClientRect();
        const frac = (e.clientX - rect.left) / rect.width;
        audio.currentTime = Math.max(0, Math.min(frac * data.duration, data.duration));
    });

    // Cursor animation
    if (effectDetailAnim) cancelAnimationFrame(effectDetailAnim);
    function animCursor() {
        if (!effectDetailData) return;
        const frac = audio.currentTime / data.duration;
        cursor.style.left = (frac * 100).toFixed(2) + '%';
        effectDetailAnim = requestAnimationFrame(animCursor);
    }
    effectDetailAnim = requestAnimationFrame(animCursor);
}

async function toggleAnalysisPanel(chip, panelId, panelContainer, data) {
    const existing = panelContainer.querySelector('[data-panel-id="' + panelId + '"]');
    if (existing) {
        existing.remove();
        chip.classList.remove('active');
        return;
    }

    const fileSel = document.getElementById('effectDetailFile');
    if (!fileSel || !fileSel.value) return;

    chip.classList.add('active');
    chip.classList.add('loading');

    const wrapper = document.createElement('div');
    wrapper.className = 'analysis-panel-img';
    wrapper.dataset.panelId = panelId;
    wrapper.innerHTML = '<span class="panel-loading">Loading ' + chip.textContent + '...</span>';
    panelContainer.appendChild(wrapper);

    try {
        const url = '/api/render-panel/' + encodeURIComponent(fileSel.value) + '?panel=' + panelId;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error('Render failed');

        const blob = await resp.blob();
        const xLeft = parseFloat(resp.headers.get('X-Left-Px') || '0');
        const xRight = parseFloat(resp.headers.get('X-Right-Px') || '1');
        const pngWidth = parseFloat(resp.headers.get('X-Png-Width') || '1');
        const duration = parseFloat(resp.headers.get('X-Duration') || '1');

        const dataWidth = xRight - xLeft;
        const img = document.createElement('img');
        img.src = URL.createObjectURL(blob);
        img.style.display = 'block';
        img.style.cursor = 'crosshair';
        // Scale image so the data area fills the wrapper width, hide axis margins
        img.style.width = (pngWidth / dataWidth * 100) + '%';
        img.style.marginLeft = -(xLeft / dataWidth * 100) + '%';

        wrapper.style.overflow = 'hidden';

        // Click anywhere in the visible (cropped) area maps directly to time
        wrapper.addEventListener('click', (e) => {
            const rect = wrapper.getBoundingClientRect();
            const frac = (e.clientX - rect.left) / rect.width;
            audio.currentTime = Math.max(0, Math.min(frac * duration, duration));
        });

        wrapper.innerHTML = '';
        wrapper.appendChild(img);
    } catch (e) {
        wrapper.innerHTML = '<span class="panel-loading" style="color:#e94560;">Error: ' + e.message + '</span>';
        chip.classList.remove('active');
    }
    chip.classList.remove('loading');
}


function drawLedogramCanvas(canvas, data, dpr, width, height, bytes, brightnessRange) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.fillStyle = '#0a0f1e';
    ctx.fillRect(0, 0, width, height);

    if (!bytes || data.num_frames === 0) return;

    const nf = data.num_frames;
    const nl = data.num_leds;

    // Brightness range as levels: [lo%, hi%] of 255
    // Pixels below lo → black, above hi → full white, between → stretched
    const br = brightnessRange || [0, 100];
    const lo = br[0] / 100 * 255;
    const hi = br[1] / 100 * 255;
    const span = hi - lo || 1;

    // Create offscreen canvas at native resolution (frames x leds)
    const offscreen = document.createElement('canvas');
    offscreen.width = nf;
    offscreen.height = nl;
    const octx = offscreen.getContext('2d');
    const imgData = octx.createImageData(nf, nl);
    const pixels = imgData.data;

    // Fill: X=time (frame), Y=LED position
    for (let f = 0; f < nf; f++) {
        for (let l = 0; l < nl; l++) {
            const srcIdx = (f * nl + l) * 3;
            const dstIdx = (l * nf + f) * 4;
            const r = bytes[srcIdx], g = bytes[srcIdx + 1], b = bytes[srcIdx + 2];
            const bright = Math.max(r, g, b);
            if (bright < lo) {
                pixels[dstIdx] = 0; pixels[dstIdx+1] = 0; pixels[dstIdx+2] = 0;
            } else {
                // Scale so that lo→0, hi→255 (clamp at 255)
                const s = Math.min(255 / span * (bright - lo), 255) / (bright || 1);
                pixels[dstIdx]     = Math.min(255, Math.round(r * s));
                pixels[dstIdx + 1] = Math.min(255, Math.round(g * s));
                pixels[dstIdx + 2] = Math.min(255, Math.round(b * s));
            }
            pixels[dstIdx + 3] = 255;
        }
    }
    octx.putImageData(imgData, 0, 0);

    // Draw scaled with nearest-neighbor for crisp pixels
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(offscreen, 0, 0, width, height);
}

// ── File upload ──────────────────────────────────────────────────

async function uploadWavBlob(blob, filename) {
    // Upload to server first to get canonical path
    const formData = new FormData();
    formData.append('file', blob, filename);
    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await resp.json();

    // Cache in IndexedDB using the server's canonical path (handles sanitization + dedup)
    if (data.ok && data.path) {
        const buf = await blob.arrayBuffer();
        await cacheDB.put(data.path, { name: data.name, wav: buf, savedAt: Date.now() }, 'audioFiles');
    }
    return data;
}

async function ensureFileOnServer(path) {
    // Check if the server has this file by trying to fetch its audio
    try {
        const resp = await fetch('/audio/' + encodeURIComponent(path), { method: 'HEAD' });
        if (resp.ok) return true;
    } catch {}

    // File missing on server — re-upload from IndexedDB
    const cached = await cacheDB.get(path, 'audioFiles');
    if (!cached) return false;

    const blob = new Blob([cached.wav], { type: 'audio/wav' });
    const progress = document.getElementById('uploadProgress');
    progress.textContent = 'Re-uploading ' + cached.name + '...';
    progress.style.display = 'block';

    try {
        const data = await uploadWavBlob(blob, cached.name);
        if (data.ok) {
            // Restore annotations from IndexedDB if any (local mode only)
            if (!isPublicMode) {
                const annKey = 'ann:' + path;
                const annotations = await cacheDB.get(annKey, 'audioFiles');
                if (annotations && Object.keys(annotations).length > 0) {
                    for (const [layer, taps] of Object.entries(annotations)) {
                        await fetch('/api/annotations/' + encodeURIComponent(data.path), {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({ layer, taps })
                        });
                    }
                }
            }
            progress.textContent = 'Ready';
            setTimeout(() => progress.style.display = 'none', 1000);
            return true;
        }
    } catch {}
    progress.style.display = 'none';
    return false;
}

async function handleFileUpload(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    const allowedExts = ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm'];
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    if (!allowedExts.includes(ext)) {
        alert('Unsupported format. Accepted: ' + allowedExts.join(', '));
        return;
    }

    const progress = document.getElementById('uploadProgress');
    const isWav = ext === '.wav';
    progress.textContent = isWav ? `Uploading ${file.name}...` : `Uploading & converting ${file.name}...`;
    progress.style.display = 'block';

    try {
        const data = await uploadWavBlob(file, file.name);
        if (data.ok) {
            progress.textContent = `Uploaded ${data.name}`;
            setTimeout(() => progress.style.display = 'none', 2000);
            await loadFileList(data.path);
        } else {
            progress.textContent = `Error: ${data.error}`;
            setTimeout(() => progress.style.display = 'none', 3000);
        }
    } catch (e) {
        progress.textContent = `Upload failed: ${e.message}`;
        setTimeout(() => progress.style.display = 'none', 3000);
    }
    document.getElementById('uploadInput').value = '';
}

// ── File manager ─────────────────────────────────────────────────

async function renderFileManager() {
    // Render into whichever file manager container is visible
    const containers = [document.getElementById('fileManager'), document.getElementById('fileManagerLocal')].filter(Boolean);
    if (containers.length === 0) return;
    for (const container of containers) await _renderFileManagerInto(container);
}

async function _renderFileManagerInto(container) {

    // Get manageable files: uploads + recordings (user clips on local, uploads on public)
    const userFiles = files.filter(f =>
        f.path.startsWith('uploads/') || f.group === 'your files' ||
        (!isPublicMode && f.group === 'user clips')
    );

    if (userFiles.length === 0) {
        container.innerHTML = '<p style="color:#666; font-size:12px;">No uploaded files yet.</p>';
        return;
    }

    container.innerHTML = '';

    // Toolbar: select all + bulk delete
    const toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex; align-items:center; gap:8px; margin-bottom:8px;';
    toolbar.innerHTML = `
        <label style="color:#888; font-size:12px; display:flex; align-items:center; gap:4px; cursor:pointer;">
            <input type="checkbox" class="fm-select-all"> Select all
        </label>
        <button class="fm-bulk-delete" style="display:none; background:none; border:1px solid #e94560; color:#e94560; font-size:11px; padding:2px 10px; border-radius:3px; cursor:pointer; margin-left:auto;">Delete selected</button>`;
    container.appendChild(toolbar);

    const selectAllCb = toolbar.querySelector('.fm-select-all');
    const bulkDeleteBtn = toolbar.querySelector('.fm-bulk-delete');

    for (const f of userFiles) {
        const item = document.createElement('div');
        item.className = 'file-item';
        const dur = f.duration ? formatTime(f.duration) : '?';
        item.innerHTML = `
            <input type="checkbox" class="fm-cb" data-path="${f.path}" data-name="${f.name}">
            <span class="file-name" title="Click to select" data-path="${f.path}">${f.name}</span>
            <span class="file-dur">${dur}</span>
            <button class="ren" data-path="${f.path}" data-name="${f.name}">rename</button>`;
        container.appendChild(item);
    }

    function updateBulkUI() {
        const checked = container.querySelectorAll('.fm-cb:checked');
        bulkDeleteBtn.style.display = checked.length > 0 ? 'inline-block' : 'none';
        bulkDeleteBtn.textContent = 'Delete selected (' + checked.length + ')';
        const allCbs = container.querySelectorAll('.fm-cb');
        selectAllCb.checked = allCbs.length > 0 && checked.length === allCbs.length;
        selectAllCb.indeterminate = checked.length > 0 && checked.length < allCbs.length;
    }

    selectAllCb.onchange = () => {
        container.querySelectorAll('.fm-cb').forEach(cb => cb.checked = selectAllCb.checked);
        updateBulkUI();
    };

    container.querySelectorAll('.fm-cb').forEach(cb => cb.onchange = updateBulkUI);

    bulkDeleteBtn.onclick = async () => {
        const checked = container.querySelectorAll('.fm-cb:checked');
        const paths = Array.from(checked).map(cb => ({ path: cb.dataset.path, name: cb.dataset.name }));
        if (paths.length === 0) return;
        if (!confirm('Delete ' + paths.length + ' file(s)? This removes them from server and browser cache.')) return;
        for (const { path } of paths) {
            await _deleteFile(path);
        }
        await loadFileList();
        renderFileManager();
    };

    // Click handlers
    container.querySelectorAll('.file-name').forEach(el => {
        el.onclick = async () => { currentTab = 'analysis'; updateTabUI(); await selectFile(el.dataset.path); };
    });
    container.querySelectorAll('.ren').forEach(el => {
        el.onclick = () => renameUserFile(el.dataset.path, el.dataset.name);
    });
}

async function _deleteFile(path) {
    // Delete from server
    try {
        await fetch('/api/files/delete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
    } catch {}
    // Delete from IndexedDB
    await cacheDB.delete(path, 'audioFiles');
    // Clear cached analysis panels
    const panelKeys = await cacheDB.getAll('panels');
    for (const { key } of panelKeys) {
        if (typeof key === 'string' && key.includes(encodeURIComponent(path))) {
            await cacheDB.delete(key, 'panels');
        }
    }
    if (currentFile === path) currentFile = null;
}

async function deleteUserFile(path, name) {
    if (!confirm('Delete "' + name + '"?')) return;
    await _deleteFile(path);
    await loadFileList();
    renderFileManager();
}

async function renameUserFile(path, oldName) {
    const newName = prompt('New name for "' + oldName + '":', oldName.replace('.wav', ''));
    if (!newName || newName === oldName.replace('.wav', '')) return;

    const finalName = newName.endsWith('.wav') ? newName : newName + '.wav';

    try {
        const resp = await fetch('/api/files/rename', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path, newName: finalName })
        });
        const data = await resp.json();

        if (data.ok) {
            // Update IndexedDB: copy to new key, delete old
            const cached = await cacheDB.get(path, 'audioFiles');
            if (cached) {
                await cacheDB.put(data.path, { ...cached, name: data.name }, 'audioFiles');
                await cacheDB.delete(path, 'audioFiles');
            }

            const wasSelected = currentFile === path;
            await loadFileList(wasSelected ? data.path : undefined);
            renderFileManager();
        } else {
            alert('Rename failed: ' + (data.error || 'unknown error'));
        }
    } catch (e) {
        alert('Rename failed: ' + e.message);
    }
}

// Drag and drop
let dragCounter = 0;
const dropOverlay = document.getElementById('dropOverlay');

document.addEventListener('dragenter', e => {
    e.preventDefault();
    dragCounter++;
    if (e.dataTransfer.types.includes('Files')) {
        dropOverlay.classList.add('active');
    }
});

document.addEventListener('dragleave', e => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
        dragCounter = 0;
        dropOverlay.classList.remove('active');
    }
});

document.addEventListener('dragover', e => e.preventDefault());

document.addEventListener('drop', e => {
    e.preventDefault();
    dragCounter = 0;
    dropOverlay.classList.remove('active');
    handleFileUpload(e.dataTransfer.files);
});

// ── Welcome ──────────────────────────────────────────────────────

function copyContact() {
    // Obfuscated email — assembled at runtime so scrapers can't find it in source
    const u = 'seth'; const d = 'sethdrew'; const t = 'com';
    const addr = u + '@' + d + '.' + t;
    navigator.clipboard.writeText(addr).then(() => {
        const el = document.getElementById('contactCopied');
        if (el) { el.style.display = 'inline'; setTimeout(() => el.style.display = 'none', 2000); }
    });
}

// ── Build time ───────────────────────────────────────────────────

(() => {
    const iso = window.__SERVER_START_ISO || '';
    if (!iso) return;
    const d = new Date(iso);
    if (isNaN(d)) return;
    const el = document.getElementById('buildTime');
    if (!el) return;
    const fmt = d.toLocaleString(undefined, {month:'short', day:'numeric', hour:'numeric', minute:'2-digit'});
    el.textContent = 'deployed ' + fmt;
})();

// ── Live reload (local dev) ───────────────────────────────────────

(function() {
    let lastHash = null;
    async function poll() {
        try {
            const r = await fetch('/api/livereload');
            const d = await r.json();
            if (lastHash === null) { lastHash = d.hash; return; }
            if (d.hash !== lastHash) location.reload();
        } catch(e) {}
    }
    setInterval(poll, 1000);
})();

// ── Init ─────────────────────────────────────────────────────────

checkAuth().then(() => loadFileList());
