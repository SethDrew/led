"""
LED Effect Preview Library

Renders a composite PNG for visual QA of LED effect output.

Input: numpy array (num_frames, num_leds, 3) uint8 + optional WAV path
Output: PIL Image of the composite preview
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import colorsys


# ── Constants ─────────────────────────────────────────────────────

WIDTH = 900
LEDOGRAM_HEIGHT = 450
BAND_ENERGY_HEIGHT = 80
WAVEFORM_HEIGHT = 60
KEYFRAME_HEIGHT = 280
METRICS_HEIGHT = 40

PX_PER_LED_Y = 3        # LED-ogram: 3px per LED vertically
PX_PER_FRAME_X = 2       # LED-ogram: 2px per frame horizontally
KEYFRAME_LED_WIDTH = 6    # Key frame strip: 6px per LED
KEYFRAME_BAR_HEIGHT = 28
KEYFRAME_GAP = 7
NUM_KEYFRAMES = 8

GAMMA = 0.6
LOW_CUTOFF = 3            # max(R,G,B) < 3 snaps to black

BG_COLOR = (0, 0, 0)
DARK_BG = (10, 15, 30)    # Slightly blue-black for band energy


# ── Gamma + cutoff ────────────────────────────────────────────────

def apply_gamma(frames: np.ndarray) -> np.ndarray:
    """Apply gamma correction and low cutoff to frames.

    Args:
        frames: (N, num_leds, 3) uint8

    Returns:
        Corrected copy, same shape/dtype.
    """
    out = frames.copy().astype(np.float32)

    # Low cutoff: where max channel < LOW_CUTOFF, snap to black
    max_channel = np.max(out, axis=-1)  # (N, num_leds)
    mask = max_channel < LOW_CUTOFF
    out[mask] = 0.0

    # Gamma: pixel = round(255 * (pixel/255)^gamma)
    nonzero = ~mask
    if np.any(nonzero):
        # Apply gamma only to non-black pixels
        out[nonzero] = np.round(255.0 * np.power(out[nonzero] / 255.0, GAMMA))

    return np.clip(out, 0, 255).astype(np.uint8)


# ── LED-ogram ─────────────────────────────────────────────────────

def render_ledogram(frames: np.ndarray, fps: float = 30.0,
                    start_time: float = 0.0) -> Image.Image:
    """Render the LED-ogram panel.

    Args:
        frames: gamma-corrected (N, num_leds, 3) uint8
        fps: frames per second
        start_time: start time offset for labels

    Returns:
        PIL Image, WIDTH x LEDOGRAM_HEIGHT
    """
    num_frames, num_leds, _ = frames.shape

    # Target: each frame = PX_PER_FRAME_X wide, each LED = PX_PER_LED_Y tall
    # Native resolution: num_frames x num_leds
    # Then scale up with nearest-neighbor

    # Build native-res image (frames wide, leds tall)
    native = np.zeros((num_leds, num_frames, 3), dtype=np.uint8)
    for f in range(num_frames):
        native[:, f, :] = frames[f]  # (num_leds, 3)

    img_native = Image.fromarray(native, 'RGB')

    # Scale to final size with nearest-neighbor
    img = img_native.resize((WIDTH, LEDOGRAM_HEIGHT), Image.NEAREST)

    # Draw time labels along bottom
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()

    duration = num_frames / fps
    # Place a label every ~3 seconds
    interval = 3.0
    t = 0.0
    while t <= duration:
        x = int(t / duration * WIDTH)
        label = f"{start_time + t:.1f}s"
        draw.text((x + 2, LEDOGRAM_HEIGHT - 14), label, fill=(200, 200, 200),
                  font=font)
        # Small tick mark
        draw.line([(x, LEDOGRAM_HEIGHT - 16), (x, LEDOGRAM_HEIGHT - 12)],
                  fill=(200, 200, 200), width=1)
        t += interval

    return img


# ── Key frames ────────────────────────────────────────────────────

def render_keyframes(frames: np.ndarray, fps: float = 30.0,
                     start_time: float = 0.0) -> Image.Image:
    """Render the key-frame strip panel.

    8 evenly-spaced horizontal bars showing individual LED colors.

    Returns:
        PIL Image, WIDTH x KEYFRAME_HEIGHT
    """
    num_frames, num_leds, _ = frames.shape

    img = Image.new('RGB', (WIDTH, KEYFRAME_HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Pick 8 evenly-spaced frame indices
    if num_frames <= NUM_KEYFRAMES:
        indices = list(range(num_frames))
    else:
        indices = [int(i * (num_frames - 1) / (NUM_KEYFRAMES - 1))
                   for i in range(NUM_KEYFRAMES)]

    for i, fi in enumerate(indices):
        y_top = i * (KEYFRAME_BAR_HEIGHT + KEYFRAME_GAP)
        frame_data = frames[fi]  # (num_leds, 3)

        # Draw each LED as a 6px wide rectangle
        for led_idx in range(num_leds):
            x = led_idx * KEYFRAME_LED_WIDTH
            if x + KEYFRAME_LED_WIDTH > WIDTH:
                break
            r, g, b = int(frame_data[led_idx, 0]), int(frame_data[led_idx, 1]), int(frame_data[led_idx, 2])
            draw.rectangle([x, y_top, x + KEYFRAME_LED_WIDTH - 1,
                            y_top + KEYFRAME_BAR_HEIGHT - 1],
                           fill=(r, g, b))

        # Timestamp label
        t = fi / fps + start_time
        label = f"{t:.2f}s"
        # Draw with slight shadow for readability
        draw.text((3, y_top + 2), label, fill=(40, 40, 40), font=font)
        draw.text((2, y_top + 1), label, fill=(255, 255, 255), font=font)

    return img


# ── Band energy ───────────────────────────────────────────────────

def render_band_energy(wav_path: str, num_frames: int, fps: float = 30.0,
                       start_time: float = 0.0) -> Image.Image:
    """Render 4-band mel energy as stacked area chart.

    Bands: sub-bass (20-80Hz), bass (80-300Hz), mid (300-4kHz), high (4k-16kHz)

    Returns:
        PIL Image, WIDTH x BAND_ENERGY_HEIGHT
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        import librosa
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        duration = num_frames / fps

        # Trim/pad audio to match the frames window
        start_sample = int(start_time * sr)
        end_sample = start_sample + int(duration * sr)
        if start_sample < len(audio):
            audio_seg = audio[start_sample:min(end_sample, len(audio))]
        else:
            audio_seg = np.zeros(int(duration * sr), dtype=np.float32)

        # Compute mel spectrogram
        n_fft = 2048
        hop_length = max(1, len(audio_seg) // num_frames)
        S = librosa.feature.melspectrogram(
            y=audio_seg, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=64, fmin=20, fmax=16000
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        # Convert back to linear scale normalized 0-1
        S_lin = np.power(10.0, S_db / 10.0)

        # Get mel frequencies
        mel_freqs = librosa.mel_frequencies(n_mels=64, fmin=20, fmax=16000)

        # Define bands by mel bin ranges
        bands = {
            'sub-bass': (20, 80),
            'bass': (80, 300),
            'mid': (300, 4000),
            'high': (4000, 16000),
        }

        band_energies = {}
        for name, (flo, fhi) in bands.items():
            mask = (mel_freqs >= flo) & (mel_freqs < fhi)
            if np.any(mask):
                band_energies[name] = np.mean(S_lin[mask, :], axis=0)
            else:
                band_energies[name] = np.zeros(S_lin.shape[1])

        # Resample each band to num_frames columns
        x = np.linspace(0, 1, num_frames)
        resampled = {}
        for name, energy in band_energies.items():
            xp = np.linspace(0, 1, len(energy))
            resampled[name] = np.interp(x, xp, energy)

    except Exception:
        # Fallback: flat zero bands
        resampled = {
            'sub-bass': np.zeros(num_frames),
            'bass': np.zeros(num_frames),
            'mid': np.zeros(num_frames),
            'high': np.zeros(num_frames),
        }

    # Plot stacked area chart
    fig, ax = plt.subplots(figsize=(WIDTH / 100, BAND_ENERGY_HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0a0f1e')

    t = np.linspace(start_time, start_time + num_frames / fps, num_frames)
    colors = ['#e94560', '#ff6b35', '#ffd740', '#4ca5ff']
    labels = ['sub-bass', 'bass', 'mid', 'high']
    values = [resampled[l] for l in labels]

    ax.stackplot(t, *values, colors=colors, alpha=0.85, labels=labels)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, None)
    ax.tick_params(colors='#666666', labelsize=6)
    ax.spines[:].set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')

    # Minimal legend
    ax.legend(loc='upper right', fontsize=5, framealpha=0.3,
              facecolor='#0a0f1e', edgecolor='none', labelcolor='#aaaaaa')

    plt.tight_layout(pad=0.2)

    # Render to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    band_img = Image.open(buf).convert('RGB')

    # Resize to exact dimensions
    band_img = band_img.resize((WIDTH, BAND_ENERGY_HEIGHT), Image.LANCZOS)
    return band_img


# ── Waveform ─────────────────────────────────────────────────────

def render_waveform(wav_path: str, num_frames: int, fps: float = 30.0,
                    start_time: float = 0.0) -> Image.Image:
    """Render audio amplitude envelope as a waveform plot.

    Shows the amplitude envelope (not raw samples) — useful at zoomed-in
    timescales where individual beats are visible.

    Returns:
        PIL Image, WIDTH x WAVEFORM_HEIGHT
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        duration = num_frames / fps

        # Trim audio to the frames window
        start_sample = int(start_time * sr)
        end_sample = start_sample + int(duration * sr)
        if start_sample < len(audio):
            audio_seg = audio[start_sample:min(end_sample, len(audio))]
        else:
            audio_seg = np.zeros(int(duration * sr), dtype=np.float32)

        # Compute amplitude envelope: split into num_frames bins, take
        # peak absolute amplitude per bin
        samples_per_frame = max(1, len(audio_seg) // num_frames)
        envelope = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            chunk = audio_seg[i * samples_per_frame:(i + 1) * samples_per_frame]
            if len(chunk) > 0:
                envelope[i] = np.max(np.abs(chunk))

    except Exception:
        envelope = np.zeros(num_frames, dtype=np.float32)

    # Plot waveform envelope
    fig, ax = plt.subplots(figsize=(WIDTH / 100, WAVEFORM_HEIGHT / 100), dpi=100)
    fig.patch.set_facecolor('#0a0f1e')
    ax.set_facecolor('#0a0f1e')

    t = np.linspace(start_time, start_time + num_frames / fps, num_frames)
    ax.fill_between(t, -envelope, envelope, color='#888888', alpha=0.7)
    ax.plot(t, envelope, color='#bbbbbb', linewidth=0.5)
    ax.plot(t, -envelope, color='#bbbbbb', linewidth=0.5)

    ax.set_xlim(t[0], t[-1])
    max_env = float(np.max(envelope)) if np.max(envelope) > 0 else 1.0
    ax.set_ylim(-max_env * 1.1, max_env * 1.1)
    ax.tick_params(colors='#666666', labelsize=5)
    ax.spines[:].set_visible(False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_yticks([])

    plt.tight_layout(pad=0.1)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    waveform_img = Image.open(buf).convert('RGB')

    # Resize to exact dimensions
    waveform_img = waveform_img.resize((WIDTH, WAVEFORM_HEIGHT), Image.LANCZOS)
    return waveform_img


# ── Metrics ───────────────────────────────────────────────────────

def compute_metrics(frames_raw: np.ndarray) -> dict:
    """Compute QA metrics from raw (non-gamma-corrected) frames.

    Args:
        frames_raw: (N, num_leds, 3) uint8 original data

    Returns:
        dict with avg_brightness, max_brightness, pct_black_frames,
        motion_score, unique_hues, verdict
    """
    num_frames = frames_raw.shape[0]

    # max channel per pixel
    max_ch = np.max(frames_raw.astype(np.float32), axis=-1)  # (N, num_leds)

    # avg_brightness: mean of max(R,G,B) across all pixels/frames
    avg_brightness = float(np.mean(max_ch))

    # max_brightness: global max
    max_brightness = int(np.max(max_ch))

    # pct_black_frames: fraction of frames where ALL LEDs below cutoff
    frame_max = np.max(max_ch, axis=1)  # (N,)
    pct_black = float(np.mean(frame_max < LOW_CUTOFF))

    # motion_score: mean frame-to-frame pixel delta
    if num_frames > 1:
        diffs = np.abs(frames_raw[1:].astype(np.float32) -
                       frames_raw[:-1].astype(np.float32))
        motion = float(np.mean(diffs))
    else:
        motion = 0.0

    # unique_hues: 12-bin hue histogram of non-black pixels
    # Flatten to (total_pixels, 3)
    flat = frames_raw.reshape(-1, 3)
    brightness = np.max(flat, axis=-1)
    nonblack = flat[brightness >= LOW_CUTOFF]

    if len(nonblack) > 0:
        # Sample up to 50k pixels for speed
        if len(nonblack) > 50000:
            indices = np.random.default_rng(42).choice(
                len(nonblack), 50000, replace=False)
            nonblack = nonblack[indices]

        hue_bins = np.zeros(12, dtype=np.int64)
        for pixel in nonblack:
            r, g, b = pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            if s > 0.05 and v > 0.04:  # Ignore near-white/grey
                bin_idx = min(int(h * 12), 11)
                hue_bins[bin_idx] += 1
        unique_hues = int(np.sum(hue_bins > 0))
    else:
        unique_hues = 0

    # Verdict
    verdict = "PASS"
    reasons = []

    if max_brightness == 0:
        verdict = "FAIL"
        reasons.append("max_brightness=0")
    elif avg_brightness < 2:
        verdict = "FAIL"
        reasons.append("avg_brightness<2")

    if verdict != "FAIL":
        if pct_black > 0.95:
            verdict = "WARN"
            reasons.append("pct_black>0.95")
        if motion < 0.5:
            verdict = "WARN"
            reasons.append("motion<0.5")
        if unique_hues < 2 and avg_brightness > 10:
            verdict = "WARN"
            reasons.append("unique_hues<2")

    return {
        'avg_brightness': round(avg_brightness, 1),
        'max_brightness': max_brightness,
        'pct_black_frames': round(pct_black, 3),
        'motion_score': round(motion, 1),
        'unique_hues': unique_hues,
        'verdict': verdict,
    }


def render_metrics_bar(metrics: dict) -> Image.Image:
    """Render the metrics bar panel.

    Returns:
        PIL Image, WIDTH x METRICS_HEIGHT
    """
    img = Image.new('RGB', (WIDTH, METRICS_HEIGHT), (15, 18, 28))
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 12)
    except (OSError, IOError):
        font = ImageFont.load_default()

    v = metrics['verdict']
    if v == 'PASS':
        verdict_color = (100, 255, 100)
    elif v == 'WARN':
        verdict_color = (255, 200, 60)
    else:
        verdict_color = (255, 80, 80)

    text = (f"bright_avg={metrics['avg_brightness']:.0f}/255  "
            f"bright_max={metrics['max_brightness']}/255  "
            f"black_frames={metrics['pct_black_frames']:.0%}  "
            f"motion={metrics['motion_score']:.1f}  "
            f"hues={metrics['unique_hues']}  ")

    draw.text((10, 12), text, fill=(200, 200, 200), font=font)
    # Verdict in color
    text_width = draw.textlength(text, font=font)
    draw.text((10 + int(text_width), 12), v, fill=verdict_color, font=font)

    return img


# ── Composite ─────────────────────────────────────────────────────

def render_preview(frames: np.ndarray, audio_path: str = None,
                   fps: float = 30.0, duration: float = 15.0,
                   start_time: float = 0.0) -> tuple:
    """Render the full composite preview image.

    Args:
        frames: (num_frames, num_leds, 3) uint8 raw RGB LED data
        audio_path: path to WAV file (for band energy plot)
        fps: frames per second
        duration: target duration in seconds
        start_time: start time offset

    Returns:
        (PIL Image, metrics dict)
    """
    num_frames, num_leds, _ = frames.shape

    # Window to requested duration
    max_frames = int(duration * fps)
    start_frame = int(start_time * fps)
    end_frame = min(start_frame + max_frames, num_frames)
    windowed_raw = frames[start_frame:end_frame]

    # Compute metrics on raw data
    metrics = compute_metrics(windowed_raw)

    # Apply gamma correction for display
    windowed = apply_gamma(windowed_raw)

    # Render each panel
    panels = []

    # 1. Band energy
    if audio_path:
        band_img = render_band_energy(audio_path, windowed.shape[0],
                                      fps=fps, start_time=start_time)
    else:
        # Blank panel
        band_img = Image.new('RGB', (WIDTH, BAND_ENERGY_HEIGHT), DARK_BG)
    panels.append(band_img)

    # 1b. Waveform
    if audio_path:
        waveform_img = render_waveform(audio_path, windowed.shape[0],
                                       fps=fps, start_time=start_time)
    else:
        waveform_img = Image.new('RGB', (WIDTH, WAVEFORM_HEIGHT), DARK_BG)
    panels.append(waveform_img)

    # 2. LED-ogram
    ledogram = render_ledogram(windowed, fps=fps, start_time=start_time)
    panels.append(ledogram)

    # 3. Key frames
    keyframes = render_keyframes(windowed, fps=fps, start_time=start_time)
    panels.append(keyframes)

    # 4. Metrics bar
    metrics_bar = render_metrics_bar(metrics)
    panels.append(metrics_bar)

    # Composite: stack vertically
    total_height = sum(p.height for p in panels)
    composite = Image.new('RGB', (WIDTH, total_height), BG_COLOR)

    y = 0
    for p in panels:
        composite.paste(p, (0, y))
        y += p.height

    return composite, metrics


# ── Zoom ─────────────────────────────────────────────────────────

ZOOM_DURATION = 2.5          # seconds of audio in zoom window
ZOOM_NUM_KEYFRAMES = 4

def find_peak_energy_window(wav_path: str, start_time: float,
                            duration: float, window_sec: float = ZOOM_DURATION) -> float:
    """Find the start time of the loudest window_sec-second window.

    Searches within [start_time, start_time + duration] and returns
    the center-aligned start time of the peak energy window.
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        seg_start = int(start_time * sr)
        seg_end = min(seg_start + int(duration * sr), len(audio))
        audio_seg = audio[seg_start:seg_end]

        if len(audio_seg) == 0:
            return start_time

        # Compute rolling energy over window_sec
        win_samples = int(window_sec * sr)
        if win_samples >= len(audio_seg):
            return start_time

        # Use cumulative sum for efficient rolling energy
        energy = audio_seg ** 2
        cumsum = np.cumsum(energy)
        cumsum = np.insert(cumsum, 0, 0)
        rolling = cumsum[win_samples:] - cumsum[:-win_samples]

        peak_idx = int(np.argmax(rolling))
        # Convert back to absolute time
        peak_center = start_time + (peak_idx + win_samples / 2) / sr
        # Align so the window is centered on peak
        zoom_start = peak_center - window_sec / 2
        # Clamp within bounds
        zoom_start = max(start_time, min(zoom_start,
                                          start_time + duration - window_sec))
        return zoom_start

    except Exception:
        return start_time


def render_zoom_preview(frames: np.ndarray, audio_path: str = None,
                        fps: float = 30.0, duration: float = 15.0,
                        start_time: float = 0.0) -> tuple:
    """Render a zoomed-in composite preview centered on peak energy.

    Same layout as render_preview (band energy + waveform + LED-ogram +
    key frames + metrics) but only 2-3 seconds of data, scaled up to
    WIDTH so each frame gets ~5-10px width.

    Args:
        frames: (num_frames, num_leds, 3) uint8 raw RGB LED data
        audio_path: path to WAV file
        fps: frames per second
        duration: total duration rendered (used to find peak window)
        start_time: global start time offset

    Returns:
        (PIL Image, metrics dict, zoom_start_time)
    """
    num_frames, num_leds, _ = frames.shape

    # Find peak energy window
    if audio_path:
        zoom_start = find_peak_energy_window(
            audio_path, start_time, duration, ZOOM_DURATION)
    else:
        # Fallback: center of the rendered window
        zoom_start = start_time + duration / 2 - ZOOM_DURATION / 2
        zoom_start = max(start_time, zoom_start)

    # Convert to frame indices relative to the frames array
    zoom_start_frame = int((zoom_start - start_time) * fps)
    zoom_end_frame = min(zoom_start_frame + int(ZOOM_DURATION * fps), num_frames)
    zoom_start_frame = max(0, zoom_start_frame)

    # Clamp to ensure we have enough frames
    if zoom_end_frame - zoom_start_frame < int(0.5 * fps):
        # Not enough frames, just use the first ZOOM_DURATION seconds
        zoom_start_frame = 0
        zoom_end_frame = min(int(ZOOM_DURATION * fps), num_frames)
        zoom_start = start_time

    windowed_raw = frames[zoom_start_frame:zoom_end_frame]
    actual_zoom_frames = windowed_raw.shape[0]

    # Compute metrics on the zoom window
    metrics = compute_metrics(windowed_raw)

    # Apply gamma correction for display
    windowed = apply_gamma(windowed_raw)

    # -- Render panels, same order as main preview --
    panels = []

    # 1. Band energy
    if audio_path:
        band_img = render_band_energy(audio_path, actual_zoom_frames,
                                      fps=fps, start_time=zoom_start)
    else:
        band_img = Image.new('RGB', (WIDTH, BAND_ENERGY_HEIGHT), DARK_BG)
    panels.append(band_img)

    # 1b. Waveform
    if audio_path:
        waveform_img = render_waveform(audio_path, actual_zoom_frames,
                                       fps=fps, start_time=zoom_start)
    else:
        waveform_img = Image.new('RGB', (WIDTH, WAVEFORM_HEIGHT), DARK_BG)
    panels.append(waveform_img)

    # 2. LED-ogram — render at native resolution then scale up
    #    At 2px/frame we'd get 120-180px wide; scale to WIDTH
    native_ledogram = np.zeros((num_leds, actual_zoom_frames, 3), dtype=np.uint8)
    for f in range(actual_zoom_frames):
        native_ledogram[:, f, :] = windowed[f]
    ledogram_native_img = Image.fromarray(native_ledogram, 'RGB')
    # Scale up to WIDTH x LEDOGRAM_HEIGHT with nearest-neighbor for crisp pixels
    ledogram = ledogram_native_img.resize((WIDTH, LEDOGRAM_HEIGHT), Image.NEAREST)

    # Draw time labels
    draw = ImageDraw.Draw(ledogram)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 10)
    except (OSError, IOError):
        font = ImageFont.load_default()

    zoom_duration_actual = actual_zoom_frames / fps
    # At this zoom level, label every 0.5 seconds
    interval = 0.5
    t = 0.0
    while t <= zoom_duration_actual:
        x = int(t / zoom_duration_actual * WIDTH) if zoom_duration_actual > 0 else 0
        label = f"{zoom_start + t:.2f}s"
        draw.text((x + 2, LEDOGRAM_HEIGHT - 14), label,
                  fill=(200, 200, 200), font=font)
        draw.line([(x, LEDOGRAM_HEIGHT - 16), (x, LEDOGRAM_HEIGHT - 12)],
                  fill=(200, 200, 200), width=1)
        t += interval

    panels.append(ledogram)

    # 3. Key frames — 4 instead of 8 for the zoom window
    zoom_keyframe_height = (ZOOM_NUM_KEYFRAMES * (KEYFRAME_BAR_HEIGHT + KEYFRAME_GAP))
    kf_img = Image.new('RGB', (WIDTH, zoom_keyframe_height), BG_COLOR)
    kf_draw = ImageDraw.Draw(kf_img)

    if actual_zoom_frames <= ZOOM_NUM_KEYFRAMES:
        kf_indices = list(range(actual_zoom_frames))
    else:
        kf_indices = [int(i * (actual_zoom_frames - 1) / (ZOOM_NUM_KEYFRAMES - 1))
                      for i in range(ZOOM_NUM_KEYFRAMES)]

    for i, fi in enumerate(kf_indices):
        y_top = i * (KEYFRAME_BAR_HEIGHT + KEYFRAME_GAP)
        frame_data = windowed[fi]
        for led_idx in range(num_leds):
            x = led_idx * KEYFRAME_LED_WIDTH
            if x + KEYFRAME_LED_WIDTH > WIDTH:
                break
            r, g, b = int(frame_data[led_idx, 0]), int(frame_data[led_idx, 1]), int(frame_data[led_idx, 2])
            kf_draw.rectangle([x, y_top, x + KEYFRAME_LED_WIDTH - 1,
                               y_top + KEYFRAME_BAR_HEIGHT - 1],
                              fill=(r, g, b))
        # Timestamp label
        t_label = fi / fps + zoom_start
        label = f"{t_label:.2f}s"
        kf_draw.text((3, y_top + 2), label, fill=(40, 40, 40), font=font)
        kf_draw.text((2, y_top + 1), label, fill=(255, 255, 255), font=font)

    panels.append(kf_img)

    # 4. Metrics bar
    metrics_bar = render_metrics_bar(metrics)
    panels.append(metrics_bar)

    # Composite: stack vertically
    total_height = sum(p.height for p in panels)
    composite = Image.new('RGB', (WIDTH, total_height), BG_COLOR)

    y = 0
    for p in panels:
        composite.paste(p, (0, y))
        y += p.height

    return composite, metrics, zoom_start
