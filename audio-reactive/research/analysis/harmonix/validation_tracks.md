# Priority Validation Tracks for Beat Tracking

**Purpose:** Curated subset of Harmonix Set tracks for testing beat tracking algorithms, especially for rock/metal genres where librosa struggles.

---

## High-Priority Rock/Metal Tracks

These tracks are specifically chosen to test the issues we observed with Tool's "Opiate" (tempo doubling, hi-hat ambiguity, complex rhythms).

### Progressive Metal (Complex Timing, Odd Meters)

1. **Metallica - "...and Justice for All"**
   - File: `0010_andjusticeforall`
   - BPM: 172 (human-verified)
   - Time Signature: 6/4 (unusual!)
   - Duration: 9:52 (long, progressive structure)
   - Bars in 4: 65.3% (meaning 35% are NOT in 4/4!)
   - YouTube: https://www.youtube.com/watch?v=_fKAsvJrFes
   - **Why:** Odd meter, tempo changes, polyrhythms — ultimate test

2. **Dream Theater - "Constant Motion"**
   - File: `0055_constantmotion`
   - BPM: 180 (fast prog metal)
   - Time Signature: 4/4
   - Duration: 7:02
   - Bars in 4: 92.4% (some deviations)
   - YouTube: http://www.youtube.com/watch?v=jbwBsAGCMLE
   - **Why:** Fast tempo, complex drum fills, progressive structure

3. **Rush - "The Camera Eye"**
   - File: `0044_cameraeye`
   - BPM: 166 (fast prog)
   - Time Signature: 4/4
   - Duration: 10:48 (longest track, epic structure)
   - Bars in 4: 89.4%
   - YouTube: http://www.youtube.com/watch?v=fjrHJhMHyIM
   - **Why:** Long progressive journey, tempo/feel shifts

### Classic Prog Rock (Subdivision Ambiguity)

4. **Rush - "Limelight"**
   - File: `0161_limelight`
   - BPM: 130
   - Time Signature: 4/4
   - Duration: 4:24
   - Bars in 4: 45.3% (LOTS of odd meter!)
   - YouTube: https://www.youtube.com/watch?v=ZiRuj2_czzw
   - **Why:** 7/4 and mixed meters, Neil Peart drumming

5. **Rush - "Tom Sawyer"**
   - File: `0293_tomsawyer2`
   - BPM: 87 (moderate tempo)
   - Time Signature: 4/4
   - Duration: 4:54
   - Bars in 4: 96.4%
   - YouTube: http://www.youtube.com/watch?v=auLBLk95K_t6nY
   - **Why:** Iconic prog rock, complex hi-hat patterns

### Thrash Metal (Fast, Complex)

6. **Metallica - "Blackened"**
   - File: `0027_blackened`
   - BPM: 190 (very fast)
   - Time Signature: 4/4
   - Duration: 6:42
   - Bars in 4: 73.3% (27% deviations!)
   - YouTube: http://www.youtube.com/watch?v=DU_ggFovJNo
   - **Why:** Fast thrash, double-kick drums, tempo variations

7. **Judas Priest - "Screaming for Vengeance"**
   - File: `0248_screamingfor`
   - BPM: 237 (FASTEST in dataset!)
   - Time Signature: 4/4
   - Duration: 4:48
   - Bars in 4: 98.9%
   - YouTube: (check dataset)
   - **Why:** Extreme tempo, tests upper limits

### Heavy/Doom Metal (Slow, Groove-Based)

8. **Pantera - "Floods"**
   - File: `0098_floods`
   - BPM: 120 (mid-tempo groove)
   - Time Signature: 4/4
   - Duration: 7:06
   - Bars in 4: 97.7%
   - YouTube: http://www.youtube.com/watch?v=td-v6vG2Xhs
   - **Why:** Heavy groove, subdivision swings

9. **Ozzy Osbourne - "No More Tears"**
   - File: `0194_nomoretears`
   - BPM: 104 (similar to Tool's Opiate!)
   - Time Signature: 4/4
   - Duration: 7:18
   - Bars in 4: 96.1%
   - YouTube: https://www.youtube.com/watch?v=fLK95K_t6nY
   - **Why:** Tempo range similar to our problem case

10. **Mötley Crüe - "Without You"**
    - File: `0319_withoutyou`
    - BPM: 59 (SLOWEST metal track!)
    - Time Signature: 2/4
    - Duration: 4:36
    - YouTube: (check dataset)
    - **Why:** Slow ballad, tests lower tempo limits

---

## Electronic/Dance Baseline Tracks

These provide a "known good" baseline — electronic music should have near-perfect beat detection.

11. **Daft Punk - "Around The World"**
    - File: `0012_aroundtheworld`
    - BPM: 121 (classic house tempo)
    - Time Signature: 4/4
    - Duration: 2:34
    - Bars in 4: 100%
    - YouTube: https://www.youtube.com/watch?v=LKYPYj2XX80
    - **Why:** Metronomic, perfect for baseline accuracy

12. **Lady Gaga - "Bad Romance"**
    - File: `0017_badromance`
    - BPM: 119
    - Time Signature: 4/4
    - Bars in 4: 100%
    - YouTube: http://www.youtube.com/watch?v=qrO4YZeyl0I
    - **Why:** Pop perfection, should be 100% accurate

---

## Testing Strategy

### Phase 1: Electronic Baseline (Tracks 11-12)
- **Expected accuracy:** >99% beat detection
- **Purpose:** Validate that our pipeline works on "easy" tracks
- **If fails:** Pipeline implementation issue, not algorithm issue

### Phase 2: Simple Rock (Track 9)
- **Ozzy - "No More Tears"** at 104 BPM (similar to Tool)
- **Expected accuracy:** >95% (straightforward metal)
- **If fails:** Algorithm struggles with guitar-heavy timbre

### Phase 3: Progressive Rock (Tracks 4-5)
- **Rush tracks** with mixed meters and prog complexity
- **Expected accuracy:** 80-90%
- **If fails:** Meter detection issues, tempo drift

### Phase 4: Fast Metal (Tracks 6-7)
- **Metallica "Blackened"** (190 BPM) and **Judas Priest** (237 BPM)
- **Expected accuracy:** 70-85%
- **Likely issue:** Tempo doubling (algorithm locks onto 16th notes)

### Phase 5: Complex Prog/Odd Meter (Tracks 1-3)
- **Metallica "Justice"** (6/4) and **Dream Theater**
- **Expected accuracy:** 60-80%
- **Likely issues:** Meter changes, polyrhythms, tempo drift

---

## Validation Metrics

For each track, compute:

1. **Beat F-measure** (standard MIR metric)
   - Precision/recall of beat times
   - Tolerance window: ±70ms

2. **Tempo accuracy**
   - Detected BPM vs. ground truth BPM
   - Flag tempo doubling (detected = 2× ground truth)
   - Flag tempo halving (detected = 0.5× ground truth)

3. **Downbeat accuracy**
   - Detect downbeats (first beat of bar)
   - Critical for musical structure alignment

4. **Tempo stability**
   - Standard deviation of inter-beat intervals
   - Low stability = tempo drift

---

## Dataset Files

All validation tracks have:
- ✅ Beat annotations (`dataset/beats_and_downbeats/<file>.txt`)
- ✅ Segment annotations (`dataset/segments/<file>.txt`)
- ✅ JAMS file (`dataset/jams/<file>.jams`)
- ✅ YouTube URL (`dataset/youtube_urls.csv`)
- ✅ Alignment score (`dataset/youtube_alignment_scores.csv`)

---

## Next Steps

1. **Download mel-spectrograms** for all tracks (or just priority 10)
2. **Alternatively, download audio from YouTube** using `yt-dlp`:
   ```bash
   yt-dlp -x --audio-format wav --audio-quality 0 <youtube_url>
   ```
3. **Run librosa beat tracking** on each track
4. **Compare detected beats to ground truth** using mir_eval
5. **Identify failure patterns**:
   - Tempo doubling/halving
   - Meter confusion
   - Onset detection failures (distorted guitar)
6. **Test madmom alternatives** on failed tracks
7. **Optimize hyperparameters** per genre

---

## Expected Failure Pattern (Based on Opiate)

For rock tracks at ~100-110 BPM:
- **librosa:** Likely detects 200-220 BPM (tempo doubling)
- **Reason:** Locks onto hi-hat 8th notes instead of kick quarter notes
- **Solution:** Constrain tempo search to 80-120 BPM, or use bass-only spectral flux, or use madmom RNN

---

## Tools Required

Install if not already in venv:
```bash
pip install mir_eval  # For beat tracking evaluation metrics
pip install yt-dlp    # For downloading YouTube audio
pip install jams      # For reading JAMS annotation format
```

Add to requirements.txt:
```
mir_eval>=0.7
yt-dlp>=2024.0.0
jams>=0.3.4
```
