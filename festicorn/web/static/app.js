// festicorn recorder UI — talks to mic_bridge via /api/* (same origin).

const RECORDER_API = '/api';
const recorderState = {
    polling: null,
    audioCtx: null,
    audioBuffer: null,
    audioSource: null,
    playing: false,
    playStart: 0,
    playOffset: 0,
    duration: 0,
    animFrame: null,
};

function recorderEl(id) { return document.getElementById(id); }

async function recorderFetch(path, method = 'GET') {
    try {
        const r = await fetch(`${RECORDER_API}${path}`, { method });
        return await r.json();
    } catch (e) {
        console.warn(`mic_bridge ${method} ${path} failed:`, e);
        return null;
    }
}

function fmtTime(s) {
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1);
    return `${m}:${sec.padStart(4, '0')}`;
}

async function recorderPollStatus() {
    const data = await recorderFetch('/status');
    if (!data) {
        recorderEl('recorderState').textContent = 'OFFLINE';
        recorderEl('recorderState').className = 'recorder-state';
        return;
    }

    const stateEl = recorderEl('recorderState');
    stateEl.textContent = data.state.toUpperCase();
    stateEl.className = 'recorder-state' + (data.state === 'recording' ? ' recording' : '');

    recorderEl('recorderTimer').textContent = fmtTime(data.timer);

    const pct = data.rms > 0 ? Math.min(100, Math.max(0, (Math.log(data.rms) / Math.log(65535)) * 100)) : 0;
    recorderEl('recorderLevelFill').style.width = pct + '%';
    const db = data.rms > 0 ? (20 * Math.log10(data.rms / 65535)).toFixed(0) : '-∞';
    recorderEl('recorderLevelDb').textContent = db + ' dB';

    const recBtn = recorderEl('recorderRecBtn');
    const stopBtn = recorderEl('recorderStopBtn');
    recBtn.classList.toggle('active', data.state === 'recording');
    stopBtn.disabled = data.state !== 'recording';

    if (data.has_recording) {
        recorderEl('recorderPlayback').style.display = '';
        recorderEl('recorderEmpty').style.display = 'none';
        recorderState.duration = data.recording_duration || 0;
        recorderEl('recorderRecInfo').textContent =
            `Duration: ${fmtTime(recorderState.duration)} · ${data.frame_count || '?'} frames`;
    } else {
        recorderEl('recorderPlayback').style.display = 'none';
        recorderEl('recorderEmpty').style.display = '';
    }
}

function recorderStartPolling() {
    if (recorderState.polling) return;
    recorderPollStatus();
    recorderState.polling = setInterval(recorderPollStatus, 200);
}

document.getElementById('recorderRecBtn').addEventListener('click', async () => {
    const data = await recorderFetch('/status');
    if (!data) {
        recorderEl('recorderState').textContent = 'OFFLINE';
        recorderEl('recorderState').className = 'recorder-state';
        alert('Recorder unavailable. Start mic_bridge and reload.');
        return;
    }
    if (data.state === 'recording') {
        await recorderFetch('/stop', 'POST');
    } else {
        const result = await recorderFetch('/record', 'POST');
        if (result) console.log('record:', result.msg);
    }
});

document.getElementById('recorderStopBtn').addEventListener('click', async () => {
    await recorderFetch('/stop', 'POST');
});

document.getElementById('recorderPlayBtn').addEventListener('click', async () => {
    if (recorderState.playing) {
        recorderStopPlayback();
        return;
    }
    const playBtn = recorderEl('recorderPlayBtn');
    playBtn.innerHTML = '&#9646;&#9646;';

    try {
        const wavResp = await fetch(`${RECORDER_API}/recording/wav`);
        if (!wavResp.ok) return;
        const buf = await wavResp.arrayBuffer();
        if (!recorderState.audioCtx) {
            recorderState.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        const ctx = recorderState.audioCtx;
        recorderState.audioBuffer = await ctx.decodeAudioData(buf);
        const src = ctx.createBufferSource();
        src.buffer = recorderState.audioBuffer;
        src.connect(ctx.destination);
        recorderFetch('/play', 'POST');
        src.start();
        recorderState.audioSource = src;
        recorderState.playing = true;
        recorderState.playStart = ctx.currentTime;
        recorderState.playOffset = 0;

        recorderDrawWaveform(recorderState.audioBuffer);

        const playheadEl = recorderEl('recorderPlayhead');
        playheadEl.style.display = 'block';
        function animate() {
            if (!recorderState.playing) return;
            const elapsed = recorderState.audioCtx.currentTime - recorderState.playStart;
            const dur = recorderState.audioBuffer.duration;
            const pct = Math.min(1, elapsed / dur);
            playheadEl.style.left = (pct * 100) + '%';
            recorderEl('recorderPlayTime').textContent =
                `${fmtTime(elapsed)} / ${fmtTime(dur)}`;
            if (pct >= 1) {
                recorderStopPlayback();
                return;
            }
            recorderState.animFrame = requestAnimationFrame(animate);
        }
        recorderState.animFrame = requestAnimationFrame(animate);

        src.onended = () => recorderStopPlayback();
    } catch (e) {
        console.error('playback error', e);
        recorderStopPlayback();
    }
});

function recorderStopPlayback() {
    recorderState.playing = false;
    if (recorderState.audioSource) {
        try { recorderState.audioSource.stop(); } catch {}
        recorderState.audioSource = null;
    }
    recorderFetch('/play/stop', 'POST');
    if (recorderState.animFrame) {
        cancelAnimationFrame(recorderState.animFrame);
        recorderState.animFrame = null;
    }
    recorderEl('recorderPlayBtn').innerHTML = '&#9654;';
    recorderEl('recorderPlayhead').style.display = 'none';
    recorderEl('recorderPlayTime').textContent =
        `0:00.0 / ${fmtTime(recorderState.duration)}`;
}

function recorderDrawWaveform(buffer) {
    const canvas = recorderEl('recorderWaveform');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.clientWidth * (window.devicePixelRatio || 1);
    canvas.height = canvas.clientHeight * (window.devicePixelRatio || 1);
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, w, h);

    const data = buffer.getChannelData(0);
    const step = Math.ceil(data.length / w);
    ctx.strokeStyle = '#e94560';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < w; i++) {
        const start = i * step;
        let min = 1, max = -1;
        for (let j = 0; j < step && start + j < data.length; j++) {
            const v = data[start + j];
            if (v < min) min = v;
            if (v > max) max = v;
        }
        const y1 = ((1 - max) / 2) * h;
        const y2 = ((1 - min) / 2) * h;
        ctx.moveTo(i, y1);
        ctx.lineTo(i, y2);
    }
    ctx.stroke();
}

document.getElementById('recorderDeleteBtn').addEventListener('click', async () => {
    await recorderFetch('/recording', 'DELETE');
    recorderStopPlayback();
    recorderPollStatus();
});

recorderStartPolling();
