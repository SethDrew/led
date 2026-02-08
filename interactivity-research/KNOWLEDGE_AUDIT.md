# Knowledge Transfer Audit: STATUS.md Completeness Review

*Audit date: 2026-02-07*
*Auditor: Claude Opus 4.6*
*Target: `/interactivity-research/STATUS.md`*
*Source files reviewed: 20+ documents across analysis/, research/, tools/, led-tree/, and MEMORY.md*

---

## Phase 1: Fresh Eyes (STATUS.md Only)

### Understanding of the Project

This project is building audio-reactive LED effects for physical installations (an LED tree with 197 LEDs and a nebula strip with 150 LEDs). The approach centers on understanding what makes music *feel* a certain way -- rather than just pulsing to volume, the goal is to detect structural and emotional qualities like builds, drops, airiness, and flourishes. The project has generated a small but rich dataset of human annotations on two songs (Tool and Fred again..), extracted several general principles from that data, built two real-time beat detection prototypes that stream over serial to Arduino-based hardware, and validated those detectors against the Harmonix dataset. A core tension in the document is honest self-criticism: the custom algorithms score poorly on standard metrics, and no comparison against off-the-shelf solutions (WLED Sound Reactive) has been done.

### Questions STATUS.md Doesn't Fully Answer

1. **What is BlackHole 2ch, exactly?** Mentioned as audio capture but not explained. How does system audio get routed to Python?
2. **What does the serial protocol look like in practice?** Two start bytes + RGB data is stated, but what happens when bytes are corrupted? Is there any error detection?
3. **Why was the frame rate reduced from 60 to 30 FPS?** Mentioned as "preventing serial overflow" but no detail on the failure mode or the math behind why 60 was too fast.
4. **Why is brightness capped at 3%?** Is this a power concern, a testing convenience, or an aesthetic choice?
5. **How does the tree topology map nodes to strips?** STATUS.md says "3 strips (pins 11/12/13)" but doesn't explain how 197 nodes are distributed or how the streaming receiver handles mapping.
6. **What is the annotation tool's workflow?** "Tap-to-annotate with named layers" is mentioned, but how does the user interact? Is it keyboard-based? Does it play back audio?
7. **What are the keyboard repeat artifacts?** The beat_tap_analysis found 5 key repeat candidates. Is this a known source of noise? How are they handled?
8. **What hyperparameter values were actually used in the working prototypes?** STATUS.md lists settings in tables, but the actual values from the tools (threshold_multiplier=1.5, attack_alpha=0.95, decay_alpha=0.12, etc.) aren't all surfaced.
9. **How does the full-spectrum onset detector differ from the bass flux detector in implementation?** The A/B test tool exists, but the specifics (mel bands=40, fmin=20, fmax=8000, threshold_mult=2.0) aren't in STATUS.md.
10. **What is the Harmonix dataset structure?** STATUS.md says "912 tracks, human beat annotations" but doesn't explain the JAMS format, the beat annotation structure, or how the validation pipeline works.
11. **What happened with the `fred_drop_1_br.wav` vs `fa_br_drop1.wav` confusion?** The catalog has both entries pointing to the same Fred again.. track with different durations.
12. **What does the latency budget actually look like end-to-end?** STATUS.md mentions 50ms target but doesn't break down audio capture, processing, serial transmission, and Arduino update times.

### Areas That Feel Vague or Under-Specified

- **"Interpreter Registry (pluggable detectors) -- Designed"**: What does "designed" mean? Is there a spec? Code? Or just conversation notes?
- **"5-Layer Real-Time Pipeline -- Designed"**: Similar vagueness. The REALTIME_CONSTRAINTS.md has extensive detail on these 5 layers, but STATUS.md reduces this to a one-liner.
- **"Known Issues: LED glitching from irregular serial timing (fix applied)"**: What was the fix, specifically? This is arguably the most important architectural learning in the project.
- **"Don't fork scottlawsonbc/LedFx/WLED -- our pipeline is cleaner"**: This is an opinion stated without evidence. The streaming-audio directory actually contains a "scottlawsonbc-spectrum" subdirectory, suggesting some code was borrowed.
- **The build taxonomy (primer/sustained/bridge/drop)**: Listed with one-line descriptions, but the actual analysis (`fa_br_drop1_analysis.md`) has precise timestamps, feature values, and detection strategies that are much richer.

### Potential Contradictions

- STATUS.md says "Full-Spectrum Onset F1 = 0.252 on rock" and notes it's "60% better than bass flux on rock." But bass flux F1 on rock from the Harmonix optimization is 0.269, not lower than 0.252. The comparison seems to be against different baselines (Harmonix rock tracks for bass flux vs. Opiate user taps for onset).
- STATUS.md says the tree has "3 strips (pins 11/12/13)" -- this is correct, but the streaming README says "device 5 (BlackHole 2ch input)" while MEMORY.md says "device 3, not hardcoded." These are different, possibly reflecting different machine configurations at different times.
- The LEARNINGS.md references a ground truth tempo of "~101-109 BPM" for Opiate, while later analysis finds consistent-beat taps at 82.1 BPM, and madmom estimates 83.3 BPM. The library comparison document says ground truth is 61.9 BPM. These are different measurements of different things (user taps, tempo estimation) but STATUS.md doesn't clarify which one is "right."

---

## Phase 2: Deep Dive Discoveries

### Important Things Found in Source Files

1. **The WS2812B timing constraint is THE architectural decision** (`streaming-audio/README.md` lines 79-97). The entire reason audio callbacks must be decoupled from LED updates is that WS2812B bit-banging requires precise timing on the data line. STATUS.md mentions this but buries it in the "Hardware Constraints" section at the bottom, when it should be front-and-center as the foundational architecture principle.

2. **The streaming receiver reads directly into strip pixels with no frame buffer** (`streaming_receiver.cpp` line 9). The 30-byte rolling buffer approach is a RAM-saving design for ATmega328's 2KB limit. This is an important implementation detail not captured anywhere in STATUS.md.

3. **Tree topology is not just 3 strips -- it has specific branch structure** (`TreeTopology.h`). Strip 1 (pin 13) has 92 LEDs with a trunk splitting into branches A and B at depth 38. Strip 2 (pin 12) has just 6 LEDs on a side branch at depth 25-30. Strip 3 (pin 11) has 99 LEDs with an upper trunk and two more branches. The depth information (max depth 70) is used for effects. None of this is in STATUS.md.

4. **The onset detector tool has specific, different hyperparameters** (`realtime_onset_led.py`): 40 mel bands, fmin=20 Hz, fmax=8000 Hz, onset threshold multiplier=2.0, minimum interval=0.1s (faster than bass flux's 0.3s). These differ meaningfully from the bass flux detector and explain why it performs differently on electronic music.

5. **Keyboard repeat artifacts are a concrete data quality issue** (`beat_tap_analysis.md` lines 133-141). Five taps had intervals of exactly 83-85ms, which is a characteristic keyboard auto-repeat rate, not human tapping. These are concentrated at 29.3-29.6s and 32.2s. STATUS.md doesn't mention this at all.

6. **The airiness analysis produced quantitative effect sizes** (`air_feature_analysis.md`). The top discriminating feature was "novelty" (effect size -0.604), followed by spectral contrast (-0.420), not spectral centroid as you might infer from STATUS.md's description. The centroid effect size was only -0.247.

7. **The Harmonix optimization tested 126 combinations** (`OPTIMIZATION_SUMMARY.md`) across 7 tracks and the conclusion was unambiguous: "bass-band spectral flux does not work for music" with F1 = 0.06-0.34. The key finding is that electronic music failed HARDEST (F1=0.06) because synthesized kicks have slow attack/release, producing no spectral flux in the bass band. STATUS.md mentions F1 scores but understates how definitively this algorithm failed.

8. **madmom has a specific paradox** (`library_comparison.md`): tempo estimation is excellent (83.3 BPM, correct) but beat POSITIONS are still at doubled rate (~162 BPM). This is because the DBN post-processor doesn't properly use the tempo estimate to constrain beat placement. STATUS.md says madmom "Could not install" in the LEARNINGS section but was actually tested in the library comparison.

9. **The REALTIME_CONSTRAINTS.md is essentially the architecture specification** for the entire system. It has 5 layers, concrete code examples for every detector, specific latency budgets (18-58ms end-to-end breakdown), memory estimates (17KB per detector), and a phased implementation plan. STATUS.md reduces all of this to a 3-row table.

10. **The electronic drop analysis found that the drop ISN'T the loudest moment** (`fa_br_drop1_analysis.md`). The bridge section (113-121s) actually had higher RMS (0.3156) than the drop (0.2638). The drop is about *sustained* intensity, not peak intensity. This nuance is lost in STATUS.md.

11. **The annotation tool plays back audio while the user taps keys** (`LEARNINGS.md` Session 1). Each key creates a timestamped tap in a named layer. Multiple layers can be created by re-running with different layer names. This is how the user created the "beat," "changes," "air," "consistent-beat," "beat-flourish," and "beat-flourish2" layers.

12. **The golden copy preservation strategy** is mentioned in STATUS.md ("Golden copy at `audio-segments/golden/`") but there's no explanation of why this matters or what prompted it. The MEMORY.md mentions it as "Preserved original user input" -- this implies annotations were modified during analysis and the originals needed to be preserved.

13. **The EXISTING_DATASETS.md is much richer than STATUS.md's table** suggests. It covers 15+ datasets with detailed descriptions, availability, relevance ratings, and access URLs. It also identifies specific gaps (no time-varying feeling annotations exist anywhere) and proposes DIY approaches (Boiler Room video analysis, SoundCloud timestamp scraping).

14. **The ESP32 report identifies WLED Sound Reactive's specific capabilities** (`ESP32_AUDIO_DSP.md`): FFT (256 or 512 bins), frequency band extraction, simple beat detection, volume reactive effects, AGC, 40-60 FPS. This is the closest thing in the project to a competitive analysis, but STATUS.md says it was "NOT EVALUATED."

15. **Two separate audio streaming implementations existed before the current tools** (`streaming-audio/README.md`): a "basic-audio-reactive" bass detector and a "scottlawsonbc-spectrum" analyzer with mel-scale FFT. The `realtime_beat_led.py` and `realtime_onset_led.py` tools in the interactivity-research directory appear to be successors to these.

---

## Phase 3: Gap Inventory

### Point-by-Point Evaluation

#### 1. WS2812B timing constraint and architectural implications
**STATUS.md coverage:** Mentioned in "Hardware Constraints" section with a good explanation.
**Gap:** The explanation is buried at the bottom of the document under a section header that suggests it's a minor constraint. In reality, this IS the architecture. The fix (decoupling audio callback from frame send, using absolute time targets, serial flush) shaped every real-time tool. **Suggested fix:** Move this to the top of the document, perhaps right after "The Four Pillars," as "Foundational Architecture Constraint."

#### 2. Serial protocol (start bytes, data format, baud rate)
**STATUS.md coverage:** Mentioned as `[0xFF][0xAA][RGB x N]` at 1Mbps.
**Gap:** Adequate for reference. Missing: no error detection/correction exists. If bytes are lost, the frame is silently dropped by the receiver after a 100ms timeout. The receiver reads in 30-byte chunks to save RAM. These are implementation details that matter for debugging but aren't critical for STATUS.md. **Rating: Mostly adequate.**

#### 3. BlackHole audio capture setup and limitations
**STATUS.md coverage:** Mentioned as "BlackHole 2ch -> Python."
**Gap:** Significant. Someone reading STATUS.md would not know: (a) BlackHole is a macOS virtual audio driver that creates a loopback device, (b) you must create a "Multi-Output Device" in Audio MIDI Setup combining BlackHole + speakers, (c) you set this as system output, (d) Python reads from the BlackHole *input* device. The streaming README has setup instructions. **Suggested fix:** Add a 3-line explanation under Hardware Setup or a "How Audio Routing Works" subsection.

#### 4. Specific hyperparameter values
**STATUS.md coverage:** Partially. Lists some in tables and references key files.
**Gap:** The critical values from the actual working tools are:
- Bass flux: BASS_LOW=20Hz, BASS_HIGH=250Hz, FLUX_HISTORY=3.0s, MIN_BEAT_INTERVAL=0.3s, THRESHOLD_MULT=1.5, ATTACK_ALPHA=0.95, DECAY_ALPHA=0.12
- Onset: N_MELS=40, FMIN=20Hz, FMAX=8000Hz, ONSET_HISTORY=3.0s, ONSET_MIN_INTERVAL=0.1s, ONSET_THRESHOLD_MULT=2.0
- From LEARNINGS.md: attack alpha=0.7-0.99, decay alpha=0.1-0.3, gamma=2.2, mel bins 24-64
These are scattered across files. **Suggested fix:** Add a "Key Hyperparameters" reference table to the Working Prototypes section.

#### 5. Tree topology (3 strips, pin assignments, 197 LEDs)
**STATUS.md coverage:** Says "197 LEDs, 3 strips (pins 11/12/13), Arduino Nano."
**Gap:** Missing the distribution: Strip 1 (pin 13) = 92 LEDs (lower trunk + 2 branches), Strip 2 (pin 12) = 6 LEDs (side branch), Strip 3 (pin 11) = 99 LEDs (upper trunk + 2 branches). The max depth is 70. The tree has specific branch points (depth 38, depth 25, depth 43). Effects use depth for gradient calculations. **Suggested fix:** Add strip distribution to the Hardware Setup table.

#### 6. Annotation tool workflow
**STATUS.md coverage:** "Tap-to-annotate with named layers" in the tools table.
**Gap:** Missing HOW it works: user runs `annotate_segment.py <audio_file> <layer_name>`, audio plays back, user presses a key at moments they want to mark, timestamps are saved to a YAML file with the layer name. Multiple passes with different layer names create multi-layer annotations. This is a unique asset of the project. **Suggested fix:** Add 2-3 sentences describing the workflow.

#### 7. Nebula strip vs tree
**STATUS.md coverage:** Mentioned in the Hardware Setup table (tree = 197, nebula = 150).
**Gap:** The tools support both: `--port /dev/cu.usbserial-11240` for tree, `--port /dev/cu.usbserial-11230 --leds 150` for nebula. Both use the same protocol. The nebula is a single strip; the tree is 3 strips with topology mapping. The existing streaming pipeline (in `streaming/single-strip/`) is the predecessor for the nebula. **Rating: Adequate for STATUS.md's level of detail.**

#### 8. Frame rate reduction from 60 to 30 FPS
**STATUS.md coverage:** "30 FPS (reduced from 60 to prevent serial overflow)."
**Gap:** The math: at 1Mbps baud, 197 LEDs x 3 bytes + 2 start bytes = 593 bytes/frame = 4744 bits/frame. At 60 FPS = 284,640 bits/sec, which is well under 1Mbps, so raw bandwidth isn't the issue. The problem was serial buffer overflow on the Arduino side (ATmega328's 64-byte hardware serial buffer fills faster than the firmware can drain it). The real fix was decoupling audio callbacks from frame sends, not just reducing FPS. The 30 FPS may be more conservative than necessary. **Suggested fix:** Clarify that the reduction was a pragmatic fix for serial buffer overflow, not a bandwidth limitation.

#### 9. 3% brightness cap
**STATUS.md coverage:** "Capped at 3% for testing."
**Gap:** This is purely a testing convenience to avoid blinding brightness during development and to limit power draw from USB. It's not a design constraint. **Rating: Adequate.**

#### 10. What BlackHole 2ch is and how audio routing works
**STATUS.md coverage:** Not explained. Just named.
**Gap:** Significant for anyone trying to reproduce the setup. BlackHole is a macOS virtual audio driver that creates a kernel extension with input and output channels. System audio is routed through a Multi-Output Device to both speakers and BlackHole. Python reads from BlackHole's input side using sounddevice. **Suggested fix:** Add a brief explanation or link to the streaming-audio README's setup instructions.

#### 11. Specific failure modes of each beat detection algorithm
**STATUS.md coverage:** Table showing F1 scores with brief notes ("Misses electronic (continuous sub-bass)").
**Gap:** The source material is much richer:
- **Bass spectral flux on electronic**: F1=0.06 because synthesized kicks have slow attack/release, no spectral flux in 20-250Hz band. The bass IS there, but it's continuous, not transient.
- **Bass spectral flux on rock**: F1=0.27 because it catches kick drums (physical drum has sharp attack) but misses all snare/hi-hat beats. Detects ~25% of beats.
- **Bass spectral flux on ambient**: CV=0.791, reacts to pad swells and texture changes instead of rhythmic events.
- **Bass spectral flux tempo errors**: Consistently detects every other beat on fast tracks (190 BPM -> ~129 BPM).
- **librosa beat_track**: Doubles tempo on syncopated rock. `start_bpm` constraint is IGNORED.
- **madmom**: Correct tempo estimate but doubled beat placement due to DBN post-processor.
**Suggested fix:** Add 1-2 sentences per algorithm explaining the specific failure mechanism, not just the F1 score.

#### 12. What madmom can vs cannot do in real-time
**STATUS.md coverage:** "Onset detection: YES. Beat tracking: NO (needs full file)." And a note that RNNBeatProcessor produces frame-by-frame activation.
**Gap:** The library comparison has more nuance: RNNBeatProcessor IS real-time (produces beat activation probability per frame), but DBNBeatTrackingProcessor needs the full activation buffer. The practical hybrid is: use RNN activation + simple threshold-based peak picking in real-time, buffer activation for periodic tempo estimation. Processing is 2.4s for 40s audio (not real-time for the full pipeline). **Rating: Adequate but could be more precise.**

#### 13. User's specific taste preferences
**STATUS.md coverage:** Good. The "User Taste Inputs" section has 7 bullet points of direct quotes and paraphrases.
**Gap:** Missing one important preference from MEMORY.md: "Prefers Sonnet/Haiku for background tasks, Opus for conversation/architecture" and "This is a taste/art project -- user wants to be in the loop for creative decisions." The first is irrelevant for STATUS.md, but the second is a design philosophy that should be preserved. Also missing: the user's observation that "the first time something like this happens, ESPECIALLY if it's alone on the audio, the more airy it feels" -- novelty as a distinct concept from deviation-from-context. **Rating: Mostly good, minor gap on novelty vs deviation.**

#### 14. Build taxonomy detail
**STATUS.md coverage:** Four types listed with one-line descriptions.
**Gap:** The fa_br_drop1_analysis.md has precise timestamps, audio feature values for each section, feature changes at boundaries, 5 detected cyclic builds within the tease section, and the crucial insight that the bridge has HIGHER RMS than the drop. STATUS.md's one-liners are too thin to implement from. **Suggested fix:** Add key quantitative markers for each phase (e.g., "Tease: bass drops 26%, 5 cyclic builds detected" or "Bridge: spectral flatness spikes, RMS peaks at 0.32").

#### 15. How the streaming receiver handles multi-strip topology
**STATUS.md coverage:** Not mentioned.
**Gap:** The receiver uses `TreeTopology.h` which maps sequential node indices (0-196) to specific strip/index pairs via a `CompactNode` struct (stripId, stripIndex, depth). The sender sends nodes in sequential order; the receiver maps them to the correct physical LED. This is transparent to the sender (just send 197 RGB values in order), but essential for understanding the firmware. **Suggested fix:** One sentence: "Streaming receiver maps sequential node indices to physical strip positions via TreeTopology.h."

#### 16. Golden copy annotation preservation strategy
**STATUS.md coverage:** Mentioned as "Golden copy at `audio-segments/golden/`" in the annotations table.
**Gap:** No explanation of why it exists. Annotations were likely modified during analysis (trimming, re-timing, adding computed layers), so the raw user input needed to be preserved as ground truth. **Suggested fix:** Add one sentence: "Golden copies preserve the user's original raw tap data before any analytical modifications."

#### 17. Keyboard repeat artifacts in tap data
**STATUS.md coverage:** Not mentioned.
**Gap:** The beat_tap_analysis found 5 taps with intervals of exactly 83-85ms (characteristic of keyboard auto-repeat) at 29.3-29.6s and 32.2s. These are potential data quality issues that could affect any analysis using those specific taps. **Suggested fix:** Add to Known Issues or the annotations table: "5 potential keyboard repeat artifacts identified at 29.3-29.6s."

#### 18. Audio test files and their content
**STATUS.md coverage:** Partially. The annotations table lists "Tool - Opiate, 40s, psych rock" and "Fred again.., 129s, EDM." But other audio files (no_sound.wav, ambient.wav, electronic_beat.wav, fred_drop_1_br.wav) are not listed in STATUS.md.
**Gap:** The catalog has 6 entries. STATUS.md only references the 2 that have annotations. The other files were used for beat detector validation (ambient, electronic_beat) and are referenced in the validation reports. **Suggested fix:** Add a complete audio file listing or reference the catalog.

#### 19. Harmonix dataset structure
**STATUS.md coverage:** "912 tracks, human beat annotations, Downloaded, explored, partial optimization done."
**Gap:** Missing: annotations are in JAMS format (JSON with beat timestamps), audio must be downloaded separately (YouTube IDs provided), the validation pipeline uses mir_eval for F1/precision/recall scoring with a 70ms tolerance window, 7 tracks were actually downloaded and tested (Daft Punk, Lady Gaga, Metallica, Dream Theater, 2x Rush, Pantera). The optimization tested 18 hyperparameter combinations per track (126 total runs). **Suggested fix:** Add 2-3 sentences about the validation methodology.

#### 20. F1 scores -- what they mean vs what they don't
**STATUS.md coverage:** Good. There's a specific caveat: "F1 measured against ALL beats -- intentionally bass-only, so low recall expected" and "Low F1 against 'all beats' ground truth doesn't mean the algorithm is bad for LED purposes."
**Gap:** The OPTIMIZATION_SUMMARY.md is more specific: the 70ms tolerance window matters (standard for MIREX), and the scores are compared against human-annotated beats from professional music game developers. The F1=0.06 for electronic means 94% of beats were missed entirely. STATUS.md handles this adequately with its caveat but could note the tolerance window. **Rating: Adequate.**

---

## Contradictions Between STATUS.md and Source Files

### Contradiction 1: Onset Detector F1 Comparison
**STATUS.md says:** "Full-Spectrum Onset F1 = 0.252 on rock, 60% better than bass flux on rock."
**Source says:** Bass flux F1 on Harmonix rock = 0.269 (harmonix_optimization.md). The 0.252 onset F1 appears to be from a DIFFERENT evaluation (against user taps on Opiate, not Harmonix). So the "60% better" claim compares apples (onset on user-tapped rock) to oranges (bass flux on Harmonix rock). The onset detector is NOT actually 60% better when measured against the same ground truth.
**Impact:** Medium. This could lead to incorrect algorithm selection decisions.

### Contradiction 2: Opiate Tempo Ground Truth
**STATUS.md (Pillar 2 table):** Doesn't specify the "correct" tempo explicitly.
**LEARNINGS.md:** Says "~101-109 BPM (confirmed by user tap annotations)."
**Library comparison:** Says "ground truth ~62 BPM" (from consistent-beat layer, 61.9 BPM).
**Beat vs consistent analysis:** Shows consistent-beat layer median = 81.9 BPM, groove section = 82.1 BPM.
**Madmom:** Estimates 83.3 BPM.
**What's happening:** The "tempo" depends on what you're measuring. The free-form beat layer has a median of 111.3 BPM (pulled by flourishes). The consistent-beat layer in the groove section is 82.1 BPM. The 61.9 BPM appears to be a half-time feel measurement. The 101-109 BPM from LEARNINGS.md was an early estimate. None of these are "wrong," but LEARNINGS.md's initial estimate of 101-109 BPM is outdated.
**Impact:** Low (STATUS.md doesn't cite a specific tempo), but LEARNINGS.md should be updated.

### Contradiction 3: madmom Installation Status
**LEARNINGS.md says:** "madmom RNN: Could not install."
**Library comparison says:** madmom was successfully installed (version 0.17.dev0) and tested.
**STATUS.md says:** madmom "Tempo estimation: YES" -- implying it was tested.
**What happened:** madmom was initially not installable, then was installed in a later session.
**Impact:** Low. STATUS.md is correct; LEARNINGS.md is stale on this point.

### Contradiction 4: BlackHole Device Number
**Streaming README says:** "Python scripts use device 5 (BlackHole 2ch input)."
**MEMORY.md says:** "BlackHole 2ch for system audio capture (auto-detected as device 3, not hardcoded)."
**What's happening:** Device numbers change between system configurations. The tools auto-detect by name.
**Impact:** Low. Just a documentation inconsistency in different files.

### Contradiction 5: Existing Code Relationship
**STATUS.md says (via LEARNINGS.md reference):** "Don't fork scottlawsonbc/LedFx/WLED -- our pipeline is cleaner."
**Reality:** The streaming-audio directory contains `scottlawsonbc-spectrum/` which IS code inspired by/borrowed from scottlawsonbc. The LEARNINGS.md more precisely says "Steal algorithms... but not architecture."
**Impact:** Low. STATUS.md doesn't repeat this claim directly, but the nuance matters.

---

## Additional Findings

### Things in Source Files STATUS.md SHOULD Mention But Doesn't

1. **The latency budget breakdown** from REALTIME_CONSTRAINTS.md (audio capture 0-10ms, FFT 10-20ms, feature detection 1-5ms, LED command 1-5ms, serial transmission 5-15ms, Arduino processing 1-3ms, total 18-58ms). This is critical for making performance tradeoff decisions.

2. **The 5-layer pipeline architecture** from REALTIME_CONSTRAINTS.md is the actual proposed system design: Layer 1 (Immediate, <50ms), Layer 2 (Short-context, 50ms-2s), Layer 3 (Medium-context, 2-10s), Layer 4 (Reactive events), Layer 5 (Heuristic prediction). STATUS.md reduces this to a link.

3. **The phased implementation plan** (5 phases over 5 weeks) from REALTIME_CONSTRAINTS.md. This is a concrete roadmap for the project.

4. **Specific code patterns for each detector** (TempoTracker, BeatGrid, DeviationDetector, FlourishRatioTracker, BuildDetector, DropDetector, etc.) exist in REALTIME_CONSTRAINTS.md. These are designed but not built.

5. **The per-section feature tracking behavior** from beat_tap_analysis.md: user tracks centroid_peak in sections 0, 4, 5, 6, but flux_peak in sections 1, 2, 3. This mode-switching behavior is the empirical basis for the "mode selector" concept but isn't explained in STATUS.md.

6. **The high-confidence vs all-flourish discrepancy** from flourish_audio_properties.md: when looking only at high-confidence flourishes (2+ source agreement, N=19), the effect sizes nearly vanish (percussive_energy drops from -0.533 to -0.007). This dramatically weakens the "flourishes are quieter" finding.

7. **The electronic drop analysis's specific feature values at each section boundary** (e.g., bass drops 96.8% from tease to real build, then increases 4928% from build to bridge). These concrete numbers would help anyone trying to implement detection thresholds.

8. **The streaming-audio directory has TWO older implementations** (basic-audio-reactive and scottlawsonbc-spectrum) that preceded the interactivity-research tools. Understanding this lineage helps contextualize the project evolution.

### Opinions Stated as Facts

1. **"Our pipeline is cleaner" than WLED/LedFx** (from LEARNINGS.md, not STATUS.md). STATUS.md rightly flags this as unvalidated in its "Opinions" section.

2. **"Spectral flux is better than most existing ESP32 projects"** (ESP32_AUDIO_DSP.md). This is asserted without testing any existing projects.

3. STATUS.md does an excellent job of separating facts from opinions in its "Facts, Opinions, and Assumptions" section. This is the strongest part of the document.

### Nuances Lost in Summarization

1. **The drop isn't the loudest part.** The fa_br_drop1 analysis found the bridge (0.3156 RMS) was louder than the drop (0.2638). STATUS.md doesn't capture this, which matters for detection strategy (you can't detect drops by looking for max RMS).

2. **Flourish findings weaken with higher confidence thresholds.** The headline finding ("flourishes are 24.7% quieter in percussive energy") only holds for all flourishes. For high-confidence flourishes, the effect disappears. STATUS.md reports the headline without this caveat.

3. **The annotation tool's keyboard-based input creates specific artifacts.** The 83-85ms interval keyboard repeats are a systematic noise source that could bias any timing analysis. STATUS.md doesn't mention this.

4. **The real-time pipeline has cold-start latencies.** Tempo estimation needs 2-4 seconds to stabilize. Flourish ratio needs 5 seconds. Build detection needs 5-10 seconds. These cold starts mean the first few seconds of any song will have reduced capability. STATUS.md doesn't surface this operational limitation.

5. **The user's airiness annotation has 24 taps, not 25.** STATUS.md says 25 taps in the air layer; the air_feature_analysis.md says 24. Minor but incorrect.

---

## Recommendations (Prioritized)

### Priority 1: Structural Changes (High Impact)

**1.1 Elevate the WS2812B timing constraint.** Move from "Hardware Constraints" section to immediately after the project overview. This is the foundational architecture decision, not a footnote. Add the concrete pattern (audio callback stores results, fixed-rate loop sends frames).

**1.2 Add a "Latency Budget" subsection** under Working Prototypes. Include the end-to-end breakdown from REALTIME_CONSTRAINTS.md: audio capture (0-10ms) + FFT (10-20ms) + detection (1-5ms) + serial (5-15ms) + Arduino (1-3ms) = 18-58ms total. This is essential for performance decisions.

**1.3 Expand the 5-Layer Pipeline summary.** Currently a one-liner. Add at minimum the layer names and their latency categories: Immediate (<50ms), Short-context (50ms-2s), Medium-context (2-10s), Reactive events, Heuristic prediction. These categories are the intellectual framework for the project.

### Priority 2: Accuracy Corrections (Medium Impact)

**2.1 Fix the onset detector F1 comparison.** Clarify that the 0.252 onset F1 and 0.27 bass flux F1 were measured against DIFFERENT ground truths (user taps vs Harmonix annotations). Remove or qualify the "60% better" claim.

**2.2 Fix the air tap count.** Change 25 to 24 (or verify which is correct).

**2.3 Add high-confidence caveat to flourish findings.** Note that the "flourishes are quieter" finding weakens dramatically (effect size drops from -0.533 to -0.007) when only high-confidence flourishes are considered.

### Priority 3: Missing Knowledge (Medium Impact)

**3.1 Explain BlackHole audio routing.** Add 3-4 sentences: what BlackHole is, how Multi-Output Device works, that Python reads from BlackHole's input side via sounddevice. Without this, the setup is unreproducible.

**3.2 Add a hyperparameter reference table.** Consolidate the key values from both real-time tools into one table in the Working Prototypes section. Include bass/onset frequency ranges, threshold multipliers, attack/decay alphas, mel band count, and frame/FFT sizes.

**3.3 Document the tree topology distribution.** Add strip LED counts (92/6/99) and pin assignments to the Hardware Setup table. Note that effects use depth (0-70) for spatial mapping.

**3.4 Describe the annotation tool workflow.** Replace "Tap-to-annotate with named layers" with: "User runs annotate_segment.py with an audio file and layer name. Audio plays back while user presses keys at relevant moments, creating timestamped entries in a YAML file. Multiple passes with different layer names build multi-layer annotations."

**3.5 Add keyboard repeat artifacts as a known data quality issue.** Note the 5 candidates at 29.3-29.6s with 83-85ms intervals.

### Priority 4: Enrichment (Lower Impact)

**4.1 Add section boundary feature values for the build taxonomy.** At minimum: "Tease to Build boundary: bass drops 96.8%. Build to Bridge: bass increases 4928%, RMS jumps 74.6%. Bridge: highest spectral flatness. Drop: sustained (not peak) intensity."

**4.2 Note that the bridge is louder than the drop** in the Fred again.. analysis. This is counterintuitive and critical for detection strategy.

**4.3 Add the cold-start latency table.** Tempo: 2-4s. Flourish ratio: 5s. Build detection: 5-10s. Airiness baseline: 1-2s. This sets expectations for system behavior.

**4.4 Note that 7 specific Harmonix tracks were downloaded** (not the full 912). List them: Daft Punk - Around The World, Metallica - Blackened, Lady Gaga - Bad Romance, Dream Theater - Constant Motion, Rush - The Camera Eye, Rush - Limelight, Pantera - Floods. This sets the scope of validation.

**4.5 Add the electronic-specific failure explanation.** "Bass spectral flux fails on electronic music because synthesized kicks have slow attack/release, producing no spectral flux in the bass band. The bass IS there but it's continuous, not transient."

**4.6 Mention the pre-existing streaming implementations** (basic-audio-reactive and scottlawsonbc-spectrum) as predecessors, to document the project's evolution.

**4.7 Add golden copy rationale.** "Golden copies preserve the user's original raw tap data before any analytical modifications or computed layer additions."

### Priority 5: Nice-to-Have (Lowest Impact)

**5.1 List all 6 audio files from the catalog**, not just the 2 with annotations.

**5.2 Note the Harmonix validation methodology**: JAMS format, mir_eval scoring, 70ms tolerance window, 18 parameter combinations per track.

**5.3 Cross-reference the streaming-audio README** for the canonical WS2812B timing explanation.

**5.4 Note the SALAMI dataset size discrepancy**: STATUS.md says 1,359 songs, EXISTING_DATASETS.md says ~2,200. Verify which is correct.

---

## Summary Assessment

STATUS.md is a genuinely excellent document. The "Facts, Opinions, and Assumptions" section is particularly strong -- it demonstrates the kind of intellectual honesty that is rare in project documentation. The self-critical examination of implicit assumptions is valuable and should be preserved.

However, STATUS.md has two structural weaknesses:

1. **Architecture knowledge is buried.** The WS2812B timing constraint and the 5-layer pipeline design are the two most important technical decisions in the project, and both are underweighted in the document. The constraint is in a bottom section; the pipeline is a one-line table entry.

2. **Quantitative precision is lost.** The source analyses contain specific numbers (feature values at section boundaries, hyperparameter settings, latency breakdowns, cold-start times) that are essential for implementation but are reduced to qualitative descriptions in STATUS.md.

With the Priority 1-3 changes (approximately 15-20 additional lines of text), STATUS.md would go from "good knowledge capture with gaps" to "sufficient for a new contributor to make correct decisions about the project."
