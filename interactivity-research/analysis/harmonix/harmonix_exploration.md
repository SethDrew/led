# Harmonix Set Dataset Exploration

**Date:** 2026-02-06
**Dataset:** Harmonix Set (https://github.com/urinieto/harmonixset)
**Purpose:** Beat tracking validation and hyperparameter optimization for LED audio-reactive project

---

## Executive Summary

The Harmonix Set is a professional-quality beat tracking dataset containing **912 Western pop music tracks** with **human-verified beat annotations** from music game developers (Harmonix Music). The dataset provides:

- ✅ **Beat and downbeat timestamps** (tab-separated format)
- ✅ **Functional segment labels** (intro, verse, chorus, bridge, outro, etc.)
- ✅ **Comprehensive metadata** (genre, tempo, time signature, duration)
- ✅ **JAMS-format annotations** (unified annotation standard)
- ⚠️ **NO audio files** (copyright restrictions — annotations only)
- ✅ **Mel-spectrograms available** via Dropbox (~1.2GB, optional)
- ✅ **YouTube URLs and alignment scores** for sourcing audio

**Total Duration:** 56.2 hours of annotated music
**License:** MIT (permissive open source)

---

## Dataset Statistics

### Genre Distribution
| Genre | Count | Percentage |
|-------|-------|------------|
| Pop | 422 | 46.3% |
| Hip-Hop | 140 | 15.4% |
| Dance/Electronic | 129 | 14.1% |
| Country | 39 | 4.3% |
| Rock | 36 | 4.0% |
| Alternative | 33 | 3.6% |
| Metal | 32 | 3.5% |
| R&B | 25 | 2.7% |
| Classic Rock | 8 | 0.9% |
| Grunge | 6 | 0.7% |
| Prog | 5 | 0.5% |
| Others | 7 | 0.8% |

### Tempo Statistics
- **Range:** 57 - 237 BPM
- **Mean:** 113.1 BPM
- **Median:** 119.0 BPM
- **Standard Deviation:** 25.1 BPM

### Time Signature Distribution
- **4/4:** 890 tracks (97.6%)
- **6/8:** 14 tracks (1.5%)
- **3/4:** 4 tracks (0.4%)
- **6/4:** 3 tracks (0.3%)
- **2/4:** 1 track (0.1%)

### Duration Statistics
- **Range:** 101.8 - 647.5 seconds (1.7 - 10.8 minutes)
- **Mean:** 221.7 seconds (3.7 minutes)
- **Median:** 218.6 seconds (3.6 minutes)

---

## Relevance for Beat Tracking Validation

### Rock Tracks (117 total)
**Why rock is crucial for our project:**
- Our user's "Opiate" test case (Tool, psych rock) showed **librosa doubling the tempo** (161 BPM detected vs. ~105 BPM actual)
- Rock drums have **hi-hat subdivision ambiguity** — algorithms lock onto 8th-note hi-hats instead of quarter-note kicks
- Complex fills, syncopation, and distorted guitars confuse onset detection
- **Tempo variability** within songs (intro != groove != outro)

**Rock/Metal genre tempo analysis:**
- **Metal:** 59-237 BPM (mean 141.7 BPM, std 42.2) — huge variance!
- **Rock:** 65-152 BPM (mean ~110 BPM)
- **Alternative:** 61-150 BPM (mean 104.4 BPM)
- **Classic Rock:** 82-150 BPM (mean 114.3 BPM)

**Example rock tracks in dataset:**
- Metallica — "...and Justice for All" (172 BPM, 6/4 time, 9:52 duration)
- Metallica — "Blackened" (190 BPM, 4/4, 6:42)
- The Who — "Baba O'Riley" (118 BPM, 4/4, 5:14)
- Jimi Hendrix — "Are You Experienced?" (82 BPM, 4/4, 4:18)
- Dream Theater — "Constant Motion" (180 BPM, Prog)
- Rush — "The Camera Eye" (166 BPM, Prog, 10:47)

### Electronic Tracks (129 total)
**Why electronic is useful as baseline:**
- **Metronomic precision** — good for establishing algorithm baseline accuracy
- **Spectral complexity in bass** — tests frequency-specific detection
- **Synthesized attacks** — different envelope profiles than acoustic drums

**Electronic tempo analysis:**
- **Range:** 77-152 BPM
- **Mean:** 121.6 BPM
- **Median:** 125.0 BPM
- **Standard Deviation:** 12.3 BPM (much tighter than rock!)

**Example electronic tracks:**
- Daft Punk — "Around The World" (121 BPM)
- Lady Gaga — "Bad Romance" remixes (119-130 BPM)
- Duck Sauce — "Barbra Streisand" (127 BPM)

---

## Annotation Format

### Beat Annotations (`beats_and_downbeats/*.txt`)
Tab-separated format with 3 columns:
```
<timestamp> <beat_position> <bar_number>
0.0         1               1
0.530973    2               1
1.061946    3               1
1.592919    4               1
2.123892    1               2
...
```

- **Column 1:** Beat timestamp in seconds
- **Column 2:** Beat position within bar (1 = downbeat)
- **Column 3:** Bar number
- **Downbeats:** When column 2 = 1

### Segment Annotations (`segments/*.txt`)
Tab-separated format with 2 columns:
```
<timestamp> <label>
0.0         intro
8.495568    verse
25.486704   chorus
42.475328   verse
...
```

Common labels: `intro`, `verse`, `chorus`, `bridge`, `solo`, `outro`, `end`

### JAMS Files (`jams/*.jams`)
Unified JSON format combining all annotations + metadata (JAMS v0.3.3 standard)

---

## Usage for Our Project

### ✅ What We Can Do Now
1. **Use annotations as ground truth** for beat tracking algorithm validation
2. **Test tempo detection** against 912 human-verified BPM values
3. **Analyze genre-specific performance** (rock vs. electronic vs. hip-hop)
4. **Optimize hyperparameters** (librosa tempo search range, madmom settings, etc.)
5. **Identify problematic track characteristics** (tempo doubling/halving patterns)

### ⚠️ What We Need to Get Audio
**The dataset does NOT include audio files (copyright).** To get actual audio:

1. **Option 1: YouTube URLs** (provided in `youtube_urls.csv`)
   - 912 YouTube URLs included
   - Alignment scores available (`youtube_alignment_scores.csv`)
   - Use `yt-dlp` or similar to download audio
   - Note: YouTube audio may differ from original annotations (check alignment scores)

2. **Option 2: MusicBrainz/AcoustID matching**
   - MusicBrainz IDs provided for most tracks
   - Match against user's existing music library
   - Use AcoustID for audio fingerprint matching

3. **Option 3: Mel-spectrograms** (partial audio representation)
   - Download from Dropbox: https://www.dropbox.com/s/zxnqlx0hxz0lsyc/Harmonix_melspecs.tgz (~1.2GB)
   - Includes pre-computed mel-scale spectrograms
   - Sufficient for some beat tracking algorithms
   - NOT full audio (can't play back or do full spectral analysis)

### 🎯 Recommended Next Steps

1. **Download mel-spectrograms** (~1.2GB) — this gives us partial audio data without copyright issues
2. **Test librosa/madmom on spectrograms** — validate tempo detection against ground truth
3. **Analyze rock track failures** — find patterns in tempo doubling (e.g., tracks with 2:1 BPM errors)
4. **Identify 10-20 key validation tracks** — cover range of tempos and genres
5. **For those key tracks, source full audio** via YouTube URLs (for development/testing only)
6. **Build validation pipeline** — automated beat tracking accuracy scoring

---

## Rock Track Deep Dive

Since our user's issue is **rock beat tracking**, here are high-priority validation candidates:

### Metallica Tracks (Metal, complex rhythms)
- "...and Justice for All" — 172 BPM, 6/4 time, 9:52 duration
- "Blackened" — 190 BPM, 4/4, 6:42
- Both have **polyrhythmic sections** and **tempo variations**

### Progressive Rock (tempo changes, odd meters)
- Rush — "The Camera Eye" — 166 BPM, 4/4, 10:47
- Dream Theater — "Constant Motion" — 180 BPM, 4/4, 7:02
- **89-93% bars in 4/4** (meaning some bars deviate)

### Classic Rock (groove-based, subdivision ambiguity)
- The Who — "Baba O'Riley" — 118 BPM, 4/4, 5:14
- Jimi Hendrix — "Are You Experienced?" — 82 BPM, 4/4, 4:18
- Tom Petty — "A Thing About You" — 152 BPM, 4/4, 5:00

---

## Dataset Files Structure

```
harmonix/
├── dataset/
│   ├── beats_and_downbeats/     # 912 .txt files (beat timestamps)
│   ├── segments/                # 912 .txt files (structural segments)
│   ├── jams/                    # 912 .jams files (unified format)
│   ├── metadata.csv             # Track metadata (genre, BPM, etc.)
│   ├── youtube_urls.csv         # YouTube video URLs
│   └── youtube_alignment_scores.csv  # Audio alignment quality scores
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── Dataset Analysis.ipynb
│   ├── JAMS Creation.ipynb
│   └── Audio Alignment.ipynb
├── results/                     # Benchmark results
│   └── segmentation/
│       ├── annot_beats.csv      # Using human annotations
│       ├── korz_beats.csv       # Using madmom beats
│       └── librosa_beats.csv    # Using librosa beats
├── src/                         # Source code for dataset creation
└── README.md
```

---

## Comparison to Our Existing Test Segments

| Track | Genre | BPM | Duration | Status |
|-------|-------|-----|----------|--------|
| **Our test segments:** | | | | |
| Opiate Intro (Tool) | Psych Rock | ~105 | 40s | ✅ Has user tap annotations |
| Ambient (Fred again..) | Electronic Ambient | ? | 30s | ❌ No annotations yet |
| Electronic Beat (Fred again..) | Electronic | ? | 50s | ❌ No annotations yet |
| **Harmonix Set:** | | | | |
| 912 tracks | Various | 57-237 | 56 hours | ✅ Professional annotations |
| 117 rock/metal tracks | Rock/Metal/Alt/Prog | 59-237 | ~8 hours | ✅ Perfect for testing |
| 129 electronic tracks | Dance/Electronic | 77-152 | ~8 hours | ✅ Baseline validation |

---

## Genre-Specific Tempo Insights

### Metal (32 tracks)
- **Widest tempo range:** 59-237 BPM (178 BPM spread!)
- **Highest mean tempo:** 141.7 BPM
- **Highest variance:** std 42.2 BPM
- **Implication:** Need wide tempo search range for metal

### Dance/Electronic (129 tracks)
- **Tightest distribution:** std 12.3 BPM
- **Mean:** 121.6 BPM
- **Range:** 77-152 BPM
- **Implication:** Narrower search range OK for electronic

### Hip-Hop (140 tracks)
- **Mean:** 101.2 BPM
- **Range:** 65-155 BPM
- **Implication:** Lower tempo range than electronic

### Rock (36 tracks) + Alternative (33 tracks)
- **Combined 69 tracks**
- **Mean:** ~107 BPM
- **Range:** 61-152 BPM
- **Moderate variance**

---

## Action Items

### Immediate (< 1 hour)
- [x] Clone Harmonix Set repository
- [x] Explore dataset structure
- [x] Compute statistics
- [x] Identify rock/electronic tracks
- [x] Document findings

### Short-term (< 1 week)
- [ ] Download mel-spectrograms from Dropbox (~1.2GB)
- [ ] Write validation script to test librosa beat tracking against ground truth
- [ ] Measure tempo detection accuracy (overall + by genre)
- [ ] Identify tempo doubling/halving failure patterns
- [ ] Test madmom RNN beat tracker on same subset

### Medium-term (1-2 weeks)
- [ ] Identify 10-20 key validation tracks across genres
- [ ] Source full audio for those tracks (YouTube URLs)
- [ ] Build automated validation pipeline
- [ ] Optimize hyperparameters for rock genre specifically
- [ ] Document best practices for tempo/beat detection by genre

---

## References

- **Paper:** Nieto et al., "The Harmonix Set: Beats, Downbeats, and Functional Segment Annotations of Western Popular Music", ISMIR 2019
  - PDF: https://ccrma.stanford.edu/~urinieto/MARL/publications/ISMIR2019-Nieto-Harmonix.pdf
- **Repository:** https://github.com/urinieto/harmonixset
- **Mel-spectrograms:** https://www.dropbox.com/s/zxnqlx0hxz0lsyc/Harmonix_melspecs.tgz (1.2GB)
- **License:** MIT

---

## Summary

The Harmonix Set is **exactly what we need** for validating beat tracking algorithms. It provides:

1. **Human-verified ground truth** from professional music game developers
2. **Large dataset** (912 tracks, 56 hours) for statistical validation
3. **Genre diversity** including 117 rock/metal tracks (perfect for our Opiate problem)
4. **Professional annotations** (beats, downbeats, segments)
5. **Multiple formats** (tab-separated, JAMS)
6. **Sourcing options** (YouTube URLs, MusicBrainz IDs, mel-spectrograms)

**Next step:** Download mel-spectrograms and build validation pipeline to measure librosa/madmom accuracy on rock tracks.
