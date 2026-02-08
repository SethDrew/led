# Existing Datasets for Audio-Reactive LED Systems

Research findings on music annotation, crowd reaction, and dance datasets that could be used for training audio-reactive LED systems.

**Research Date:** 2026-02-06
**Purpose:** Identify existing datasets for mapping music features to human feelings and LED behaviors

---

## 1. Music Structure & Annotation Datasets

### SALAMI (Structural Analysis of Large Amounts of Music Information)
- **What it contains:** Hierarchical structural segmentation annotations with timestamps
- **Size:** ~2,200 songs across multiple genres
- **Annotations:**
  - Structural boundaries (verse, chorus, bridge, intro, outro)
  - Multi-level hierarchical segmentation
  - Text file format with timestamps and labels
- **Availability:** Freely available (annotations only, not audio due to copyright)
- **Where:** https://github.com/DDMAL/salami-data-public
- **Relevance:** HIGH - Direct mapping of song structure to LED scene changes
- **Limitation:** Doesn't include "feeling" annotations (tension, release, energy)

### Harmonix Set
- **What it contains:** Beat, downbeat, and structural annotations from Harmonix Music Systems
- **Size:** ~900 tracks from various genres including rock and electronic
- **Annotations:**
  - Beat-level timestamps (human-verified, not algorithmic)
  - Downbeat positions
  - Structural segments with labels
  - Tempo changes
- **Availability:** Freely available for research
- **Where:** https://github.com/urinieto/harmonixset
- **Relevance:** VERY HIGH - Human-annotated beats solve our "librosa doubles tempo on rock" problem
- **Note:** Created by professional music game developers, highly accurate

### RWC Music Database
- **What it contains:** Multi-faceted annotations for Japanese and Western music
- **Size:** 100 popular music tracks, 50 jazz, 50 classical, 35 genres
- **Annotations:**
  - Beat positions
  - Chord progressions
  - Melody lines
  - Musical structure
  - Instrumentation
- **Availability:** Available for research (requires application)
- **Where:** https://staff.aist.go.jp/m.goto/RWC-MDB/
- **Relevance:** MEDIUM - More focused on musicological analysis than "feeling"

### MIREX Datasets
- **What it contains:** Various evaluation datasets for Music Information Retrieval tasks
- **Tasks include:**
  - Beat tracking
  - Structural segmentation
  - Onset detection
  - Audio chord estimation
- **Size:** Varies by task (typically 100-1000 tracks per task)
- **Availability:** Available for MIREX participants
- **Where:** https://www.music-ir.org/mirex/
- **Relevance:** MEDIUM - Ground truth for beat detection, but focused on algorithm evaluation

### Isophonics Reference Annotations
- **What it contains:** High-quality annotations for Beatles, Queen, Carole King, etc.
- **Annotations:**
  - Beat positions
  - Chord sequences
  - Structural segmentation
- **Size:** ~300 tracks (classic rock focus)
- **Availability:** Freely available
- **Where:** https://isophonics.net/
- **Relevance:** HIGH for rock music - our use case includes Tool (psych rock)

---

## 2. Music Emotion & Feeling Datasets

### DEAM (MediaEval Database for Emotional Analysis in Music)
- **What it contains:** Continuous valence/arousal annotations over time
- **Size:** 1,802 songs (45-second excerpts)
- **Annotations:**
  - Dynamic valence (happy/sad) per second
  - Dynamic arousal (calm/energetic) per second
  - Static emotional labels
- **Availability:** Freely available
- **Where:** https://cvml.unige.ch/databases/DEAM/
- **Relevance:** HIGH - Time-varying emotional annotations could map to LED color/intensity
- **Limitation:** Only 2D (valence/arousal), not specific feelings like "airy" or "heavy"

### MTG-Jamendo Dataset
- **What it contains:** Music tagging with mood, genre, and instrument labels
- **Size:** 55,000 tracks with Creative Commons licenses
- **Annotations:**
  - Genre tags
  - Mood tags (happy, sad, dark, epic, energetic, etc.)
  - Instrument tags
  - Audio available for download
- **Availability:** Freely available (includes audio)
- **Where:** https://github.com/MTG/mtg-jamendo-dataset
- **Relevance:** MEDIUM-HIGH - Mood tags could correlate with LED palettes
- **Limitation:** Track-level tags, not time-varying annotations

### MagnaTagATune (MTAT)
- **What it contains:** Crowd-sourced music tags from TagATune game
- **Size:** ~25,863 clips (29 seconds each) from ~5,500 songs
- **Annotations:**
  - 188 diverse tags (genre, instrument, mood, tempo, vocal quality)
  - Tags like "ambient", "heavy", "dark", "bright", "fast", "slow"
  - Multiple annotators per clip
- **Availability:** Freely available (clips provided)
- **Where:** http://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
- **Relevance:** HIGH - Tags overlap with "feeling layer" vocabulary (heavy, bright, dark)
- **Limitation:** Clip-level, not timestamped within song

### AcousticBrainz
- **What it contains:** Crowd-sourced audio feature database
- **Size:** 2+ million tracks (pre-computed features)
- **Features:**
  - Low-level spectral features
  - High-level predictions (mood, genre, danceability)
  - All computed via Essentia
- **Availability:** Was freely available (project discontinued in 2022)
- **Where:** https://acousticbrainz.org/ (archived)
- **Relevance:** LOW - Pre-computed features, no time-varying annotations
- **Note:** Could be mined for feature→mood correlations

### AudioSet
- **What it contains:** Large-scale audio event dataset from YouTube
- **Size:** 2,084,320 10-second clips with 527 labels
- **Music annotations:**
  - Genre labels (rock, electronic, hip hop, jazz, etc.)
  - Instrument labels
  - Crowd sounds: cheering (4,380), applause (2,247), crowd (10,403)
- **Availability:** Freely available (YouTube URLs + timestamps)
- **Where:** https://research.google.com/audioset/
- **Relevance:** MEDIUM - Could extract crowd energy from concert clips
- **Limitation:** 10-second granularity, not full songs

---

## 3. Dance Motion Datasets

### AIST++ Dance Motion Database
- **What it contains:** 3D dance motion capture synchronized to music
- **Size:** 1,408 dance sequences, 10,108,015 frames
- **Features:**
  - 10 dance genres (hip hop, house, ballet, krumping, etc.)
  - 3D keypoint data (full body)
  - 9-camera multi-view recordings
  - Music audio included
- **Availability:** Freely available
- **Where:** https://google.github.io/aistplusplus_dataset/
- **Relevance:** VERY HIGH - Dance motion = visual proxy for music energy
- **Use case:** Extract movement intensity/speed from mocap → map to LED speed/intensity
- **Note:** Mostly hip hop/house, less rock, but motion energy principles generalize

### Kinetics Dataset (Video Action Recognition)
- **What it contains:** 10-second video clips of human actions
- **Size:** 650,000 clips covering 700 action classes
- **Dance-related classes:**
  - "dancing ballet"
  - "dancing charleston"
  - "dancing gangnam style"
  - "dancing macarena"
  - "breakdancing"
  - "zumba"
  - "tap dancing"
  - "robot dancing"
  - "salsa dancing"
- **Availability:** YouTube URLs provided (some videos now unavailable)
- **Where:** https://deepmind.google/research/open-source/kinetics
- **Relevance:** MEDIUM - Could analyze crowd motion energy via optical flow
- **Limitation:** Not synchronized to music analysis, just action labels

### Music4Dance
- **What it contains:** Dance motion-to-music generation dataset
- **Size:** Unknown (research project)
- **Features:** Motion capture + music synchronization
- **Availability:** Not publicly confirmed
- **Relevance:** HIGH if accessible - direct motion→music mapping
- **Note:** Check recent ISMIR/CVPR papers for availability

---

## 4. Crowd/Audience Reaction Data

### SoundCloud Timestamped Comments
- **What it contains:** User comments with timestamps on tracks
- **Potential data:**
  - "drop" markers where crowd would react
  - Emoji reactions at specific points
  - Text comments describing feelings
- **Size:** Millions of tracks with comments
- **Availability:** Via SoundCloud API (deprecated 2021, limited access)
- **Where:** https://developers.soundcloud.com/
- **Relevance:** HIGH - Real human reactions at specific song timestamps
- **Limitation:** API access restricted, scraping against ToS
- **Note:** Could manually annotate popular electronic tracks with existing comments

### YouTube Concert/Festival Videos (Boiler Room, etc.)
- **What it contains:** Raw concert footage with crowd visible
- **Potential extraction:**
  - Optical flow analysis of crowd movement
  - Pose estimation (OpenPose, MediaPipe) for energy level
  - Audio-visual synchronization
- **Size:** Unlimited (YouTube)
- **Availability:** Publicly viewable, scraping in legal gray area
- **Relevance:** HIGH - Direct visual measure of audience energy
- **Research precedent:**
  - Papers on crowd flow analysis exist
  - Music festival research (e.g., "Analysis of Crowd Movement in Electronic Dance Music Events")
- **DIY approach:** Download Boiler Room set, run pose estimation, correlate body movement with audio features

### Concert Wristband LED Data
- **What it contains:** Some concerts use synchronized LED wristbands that change with music
- **Examples:** Coldplay, Taylor Swift tours with Xylobands
- **Data:** Likely proprietary LED control sequences
- **Availability:** NOT available publicly
- **Relevance:** VERY HIGH - Professional audio→LED mappings
- **Limitation:** Commercial IP, impossible to obtain

---

## 5. Specialized Rhythm & Beat Datasets

### Ballroom Dance Dataset
- **What it contains:** Dance music with beat annotations
- **Size:** 698 30-second excerpts
- **Genres:** Waltz, tango, foxtrot, quickstep, samba, etc.
- **Annotations:** Beat positions, tempo, genre
- **Availability:** Freely available
- **Where:** http://mtg.upf.edu/ismir2004/contest/tempoContest/node5.html
- **Relevance:** LOW - Limited to ballroom music, not rock/electronic

### GTZAN Tempo and Beat Tracking
- **What it contains:** Beat annotations for genre recognition dataset
- **Size:** 1,000 tracks (30 seconds each), 10 genres
- **Annotations:** Beat positions, tempo, genre
- **Availability:** Freely available
- **Where:** http://marsyas.info/downloads/datasets.html
- **Relevance:** LOW-MEDIUM - Beat ground truth, but limited size

### Groove MIDI Dataset (Magenta)
- **What it contains:** MIDI with human timing and velocity
- **Size:** ~1,150 MIDI files (drums)
- **Features:**
  - Human "groove" timing variations
  - Velocity (intensity) annotations
  - Style labels (jazz, rock, funk, Latin)
- **Availability:** Freely available
- **Where:** https://magenta.tensorflow.org/datasets/groove
- **Relevance:** MEDIUM - Velocity/intensity could map to LED brightness
- **Limitation:** MIDI only, not full music audio

---

## 6. What's Missing (Gaps in Existing Datasets)

### The "Feeling Layer" - Not Solved
NO existing dataset provides:
- Time-varying annotations for feelings like "airy", "heavy", "tense", "release"
- Multi-dimensional emotional space beyond valence/arousal
- Genre-specific feeling vocabularies (rock tension ≠ EDM buildup)
- Direct audio feature→LED behavior mappings

**Why:** This mapping is:
- Subjective and cultural
- Context-dependent (same audio features = different feelings in different genres)
- Multi-modal (combines rhythm, harmony, timbre, structure)
- Not academically standardized

**Solution:** User-in-the-loop annotation tool (we already built annotate_segment.py)

### Crowd Energy Extraction - Limited Research
Some papers exist but no standard dataset:
- "Automatic Analysis of Music Festival Audiences" (2019) - optical flow
- "Dance Movement Recognition Using Motion Capture" (2018)
- BUT: No publicly available processed crowd energy datasets

**DIY approach:**
1. Download Boiler Room set (YouTube)
2. Run pose estimation frame-by-frame
3. Extract movement speed, density, coordination
4. Correlate with audio features from same track
5. Train regression model: audio features → expected crowd energy

### SoundCloud Reactions - Access Blocked
SoundCloud had perfect data (timestamped emojis: 🔥💀😭😮) but:
- API deprecated in 2021
- Web scraping against ToS
- Could manually collect from popular tracks (labor-intensive)

---

## 7. Recommendations for Our Use Case

### Immediately Useful
1. **Harmonix Set** - Fix beat detection on rock music (solves Tool tempo problem)
2. **AIST++** - Extract motion energy patterns, map to LED animation speed
3. **MagnaTagATune** - Mine for "heavy", "dark", "bright" tags → correlate with features
4. **DEAM** - Time-varying arousal → LED intensity/speed modulation

### Worth Investigating
5. **SALAMI** - Song structure → LED scene transitions (verse/chorus changes)
6. **AudioSet** - Extract "crowd" and "cheering" clips → analyze what audio preceded them
7. **Isophonics** - Beat annotations for classic rock → validate on similar genres

### DIY Projects
8. **Boiler Room Analysis** - Download videos, run pose estimation, extract crowd energy
9. **SoundCloud Manual Annotation** - Pick 10 popular tracks, manually log emoji timestamps
10. **User Annotation** - Use our annotate_segment.py tool to build custom dataset

### Not Worth Pursuing
- Million Song Dataset (no audio, just pre-computed features)
- GTZAN (too small, low quality)
- Proprietary concert LED data (impossible to obtain)

---

## 8. Key Insight: No Dataset Solves Our Core Problem

**The Challenge:** Mapping audio features → human feelings → LED behaviors is:
- **Subjective** - "Heavy" means different things to different people
- **Contextual** - Tool's "heavy" ≠ Aphex Twin's "heavy"
- **Multi-dimensional** - Not reducible to valence/arousal
- **Domain-specific** - Rock/electronic need different feature weightings

**Why No Dataset Exists:**
- Academic MIR focuses on objective tasks (beat tracking, genre classification)
- Emotion recognition uses simplified 2D models (valence/arousal)
- Commercial systems (Spotify, Shazam) keep their feeling→feature mappings proprietary
- Audio-reactive LEDs are a niche application (no research incentive)

**Our Advantage:**
- We control the vocabulary (define what "airy" means to us)
- We have the annotation tool (annotate_segment.py)
- We have diverse test tracks (Tool, Fred again.., more to come)
- User is in the loop (taste/art project, not objective ground truth)

**Path Forward:**
1. Use Harmonix Set for beat tracking validation
2. Use AIST++ for motion→energy inspiration
3. Build our own "feeling" dataset via user annotation
4. Train simple models on user data: features → feelings → LED params
5. Iterate with user feedback

---

## 9. Additional Resources

### Dataset Aggregators
- **mirdata** - Python library with loaders for 40+ MIR datasets
  - https://github.com/mir-dataset-loaders/mirdata
  - Includes SALAMI, GTZAN, MagnaTagATune, and more

### Research Communities
- **ISMIR** (International Society for Music Information Retrieval)
  - Annual conference proceedings: https://ismir.net/
  - Papers often release datasets

- **MIREX** (Music Information Retrieval Evaluation eXchange)
  - Annual benchmarking with datasets
  - https://www.music-ir.org/mirex/

### Visualization Research
- Search "music visualization" + "dance" on arXiv
- Check NIME (New Interfaces for Musical Expression) conference
- Look for "audio-reactive" papers in ACM SIGGRAPH

---

## 10. Conclusion

**Good news:** Multiple datasets exist for beat tracking, structure analysis, and emotion recognition.

**Reality:** None directly solve the "feeling layer" problem for audio-reactive LEDs.

**Best path:**
1. Use Harmonix Set beats + AIST++ motion patterns as foundation
2. Build custom "feeling" annotations with our tools
3. Train lightweight models on our annotated segments
4. Keep user in creative loop (this is an art project, not pure ML)

**The "feeling layer" is our competitive advantage** - no one else has solved this mapping, and it's inherently personal. Our annotation tool is the right approach.
