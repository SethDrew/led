# Harmonix Set Quick Start Guide

**Last Updated:** 2026-02-06
**Status:** ✅ Dataset downloaded, ✅ Tools installed, Ready for validation

---

## What We Have

The **Harmonix Set** is now downloaded and ready to use at:
```
/Users/KO16K39/Documents/led/interactivity-research/datasets/harmonix/
```

### Dataset Contents
- **912 tracks** with human-verified beat annotations
- **56+ hours** of annotated music
- **117 rock/metal tracks** (perfect for our Opiate problem)
- **129 electronic tracks** (baseline validation)
- Beat timestamps, downbeats, segment labels (intro/verse/chorus/etc.)
- YouTube URLs for sourcing audio
- JAMS-format unified annotations

### Key Findings
- **Metal tempo range:** 59-237 BPM (widest variance, hardest genre)
- **Electronic tempo range:** 77-152 BPM (tightest, easiest baseline)
- **Rock/Prog average:** ~100-110 BPM (exactly where our Opiate issue occurs)
- **Tempo doubling risk:** Highest for rock at 100-120 BPM

---

## Installation Status

✅ **All tools installed** in `/Users/KO16K39/Documents/led/venv/`:
```bash
source venv/bin/activate
```

Installed packages:
- `librosa` — beat/tempo detection (our current tool)
- `mir_eval` — standard MIR evaluation metrics
- `jams` — JAMS annotation format support
- `yt-dlp` — YouTube audio download
- `numpy`, `scipy`, `matplotlib` — core dependencies

Still need to install separately (Python 3.10+ compatibility):
- `madmom` — alternative beat tracker (RNN-based)
  ```bash
  pip install git+https://github.com/CPJKU/madmom.git
  ```

---

## Files Created

### Documentation
1. **`harmonix_exploration.md`** — Full dataset analysis and statistics
2. **`validation_tracks.md`** — Prioritized list of 12 key validation tracks
3. **`HARMONIX_QUICKSTART.md`** — This file

### Scripts
1. **`scripts/explore_harmonix.py`** — Dataset statistics analyzer
2. **`scripts/test_jams_load.py`** — JAMS annotation loader demo

### Data
1. **`harmonix_summary.json`** — Machine-readable dataset statistics

---

## Quick Test: Load Annotations

```python
import jams
from pathlib import Path

# Load a JAMS file
jams_file = Path("/Users/KO16K39/Documents/led/interactivity-research/datasets/harmonix/dataset/jams/0010_andjusticeforall.jams")
jam = jams.load(str(jams_file))

# Get metadata
print(f"Title: {jam.file_metadata.title}")
print(f"Artist: {jam.file_metadata.artist}")
print(f"Duration: {jam.file_metadata.duration:.2f}s")

# Get beat times
beat_ann = jam.search(namespace='beat')[0]
beat_times = [obs.time for obs in beat_ann.data]
print(f"Total beats: {len(beat_times)}")
print(f"First 10: {beat_times[:10]}")

# Get segments
segment_ann = jam.search(namespace='segment_open')[0]
for obs in segment_ann.data:
    print(f"{obs.time:8.3f}s: {obs.value}")
```

**Output:**
```
Title: ...and Justice for All (LP version)
Artist: Metallica
Duration: 592.11s
Total beats: 1532
First 10: [5.567, 6.186, 6.804, 7.423, 8.041, 8.66, 9.278, 9.897, 10.515, 11.134]
0.000s: silence
5.567s: intro
135.722s: verse
...
```

---

## Priority Validation Tracks

### Top 3 for Initial Testing

1. **Daft Punk - "Around The World"** (electronic baseline)
   - File: `0012_aroundtheworld`
   - BPM: 121 (metronomic)
   - YouTube: https://www.youtube.com/watch?v=LKYPYj2XX80
   - **Why:** Should get 100% accuracy — validates pipeline

2. **Ozzy Osbourne - "No More Tears"** (similar to Opiate)
   - File: `0194_nomoretears`
   - BPM: 104 (same range as Tool)
   - YouTube: https://www.youtube.com/watch?v=fLK95K_t6nY
   - **Why:** Tests rock at our problem tempo range

3. **Metallica - "Blackened"** (fast thrash)
   - File: `0027_blackened`
   - BPM: 190 (very fast)
   - YouTube: http://www.youtube.com/watch?v=DU_ggFovJNo
   - **Why:** Tests tempo doubling at high speeds

### All 12 Priority Tracks
See: `validation_tracks.md` for complete list with testing strategy

---

## Next Steps

### Phase 1: Get Audio for Priority Tracks (< 1 hour)

**Option A: Download YouTube audio** (for development/research only)
```bash
source venv/bin/activate

# Download a single track
yt-dlp -x --audio-format wav --audio-quality 0 \
  -o "audio-segments/%(title)s.%(ext)s" \
  "https://www.youtube.com/watch?v=LKYPYj2XX80"

# Batch download from file
# Create urls.txt with one YouTube URL per line, then:
yt-dlp -x --audio-format wav --audio-quality 0 \
  -o "audio-segments/%(title)s.%(ext)s" \
  -a urls.txt
```

**Option B: Download mel-spectrograms** (dataset-provided, ~1.2GB)
```bash
cd /Users/KO16K39/Documents/led/interactivity-research/datasets/harmonix/
curl -L -o Harmonix_melspecs.tgz \
  "https://www.dropbox.com/s/zxnqlx0hxz0lsyc/Harmonix_melspecs.tgz?dl=1"
tar -xzf Harmonix_melspecs.tgz
```

### Phase 2: Build Validation Pipeline (1-2 hours)

Create script: `scripts/validate_beat_tracking.py`

```python
#!/usr/bin/env python3
"""
Beat tracking validation against Harmonix Set ground truth.
"""

import librosa
import jams
import mir_eval
import numpy as np
from pathlib import Path

def validate_beat_tracking(audio_path, jams_path):
    """
    Run beat tracking on audio and compare to ground truth.

    Returns:
        dict with f_measure, precision, recall, tempo_error
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)

    # Detect beats with librosa
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    detected_beats = librosa.frames_to_time(beat_frames, sr=sr)

    # Load ground truth from JAMS
    jam = jams.load(str(jams_path))
    beat_ann = jam.search(namespace='beat')[0]
    reference_beats = np.array([obs.time for obs in beat_ann.data])

    # Compute reference tempo
    ref_ibi = np.mean(np.diff(reference_beats))
    ref_tempo = 60.0 / ref_ibi

    # Evaluate beat tracking
    f_measure = mir_eval.beat.f_measure(
        reference_beats,
        detected_beats,
        f_measure_threshold=0.07  # 70ms tolerance
    )

    # Tempo error
    tempo_error = tempo - ref_tempo
    tempo_ratio = tempo / ref_tempo

    # Detect tempo doubling/halving
    if 1.9 <= tempo_ratio <= 2.1:
        tempo_issue = "DOUBLED"
    elif 0.45 <= tempo_ratio <= 0.55:
        tempo_issue = "HALVED"
    else:
        tempo_issue = "OK"

    return {
        'f_measure': f_measure,
        'detected_tempo': tempo,
        'reference_tempo': ref_tempo,
        'tempo_error': tempo_error,
        'tempo_ratio': tempo_ratio,
        'tempo_issue': tempo_issue,
        'num_detected_beats': len(detected_beats),
        'num_reference_beats': len(reference_beats)
    }

# Example usage
if __name__ == "__main__":
    audio = Path("audio-segments/Around_The_World.wav")
    jams_file = Path("/path/to/harmonix/dataset/jams/0012_aroundtheworld.jams")

    results = validate_beat_tracking(audio, jams_file)

    print(f"F-measure: {results['f_measure']:.3f}")
    print(f"Detected tempo: {results['detected_tempo']:.1f} BPM")
    print(f"Reference tempo: {results['reference_tempo']:.1f} BPM")
    print(f"Tempo issue: {results['tempo_issue']}")
```

### Phase 3: Run Validation (1-2 hours)

1. Test on electronic track (should get >99% F-measure)
2. Test on rock track at ~105 BPM (expect tempo doubling)
3. Test on all 12 priority tracks
4. Identify failure patterns
5. Test madmom as alternative
6. Optimize hyperparameters

### Phase 4: Document Findings (30 min)

Update `RESEARCH_FINDINGS.md` with:
- Beat tracking accuracy by genre
- Tempo doubling patterns
- Optimal hyperparameters per genre
- Recommended algorithm (librosa vs. madmom)

---

## Common JAMS Operations

### Load Beat Times
```python
import jams
jam = jams.load("file.jams")
beat_ann = jam.search(namespace='beat')[0]
beat_times = [obs.time for obs in beat_ann.data]
```

### Get Downbeats Only
```python
# Downbeats have value = 1 in beat position
# In Harmonix Set, they're marked in the beat namespace
# Filter by checking beat positions in original .txt files
```

### Load Segments
```python
segment_ann = jam.search(namespace='segment_open')[0]
segments = [(obs.time, obs.value) for obs in segment_ann.data]
# segments = [(0.0, 'silence'), (5.567, 'intro'), (135.722, 'verse'), ...]
```

### Get Track Metadata
```python
print(jam.file_metadata.title)
print(jam.file_metadata.artist)
print(jam.file_metadata.duration)
```

---

## Expected Validation Results

### Electronic Tracks (Daft Punk, Lady Gaga)
- **F-measure:** >0.99 (near perfect)
- **Tempo accuracy:** ±1 BPM
- **Failure rate:** <1%

### Simple Rock (Ozzy, Pantera)
- **F-measure:** 0.85-0.95
- **Tempo accuracy:** ±5 BPM or 2× error
- **Failure mode:** Tempo doubling likely

### Progressive Rock (Rush, Dream Theater)
- **F-measure:** 0.70-0.85
- **Tempo accuracy:** ±10 BPM
- **Failure modes:** Tempo drift, meter changes

### Thrash Metal (Metallica, Megadeth)
- **F-measure:** 0.60-0.80
- **Tempo accuracy:** 2× error common
- **Failure modes:** Tempo doubling at fast speeds

---

## Troubleshooting

### JAMS file won't load
```python
# Check file exists
from pathlib import Path
jams_path = Path("/path/to/file.jams")
print(jams_path.exists())

# Try reading as text to debug
with open(jams_path) as f:
    print(f.read()[:500])  # First 500 chars
```

### No audio files
- Dataset only includes annotations (copyright)
- Download from YouTube using provided URLs
- Or use mel-spectrograms for some algorithms

### mir_eval errors
```python
# Ensure beat times are numpy arrays
import numpy as np
reference_beats = np.array([...])
detected_beats = np.array([...])

# Check arrays are sorted and 1D
assert reference_beats.ndim == 1
assert np.all(np.diff(reference_beats) > 0)
```

---

## Resources

- **Dataset repo:** https://github.com/urinieto/harmonixset
- **Paper:** Nieto et al., ISMIR 2019
- **JAMS docs:** https://jams.readthedocs.io/
- **mir_eval docs:** https://craffel.github.io/mir_eval/
- **Our analysis:** See `harmonix_exploration.md`

---

## Summary

We now have:
- ✅ 912 professionally-annotated tracks for validation
- ✅ Tools installed (mir_eval, jams, yt-dlp)
- ✅ Scripts to load and analyze annotations
- ✅ Priority list of 12 validation tracks
- ⏭️ **Next:** Download audio and build validation pipeline

**Estimated time to first validation results:** 2-3 hours
